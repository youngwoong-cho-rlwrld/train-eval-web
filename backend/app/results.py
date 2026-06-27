"""Eval result discovery and aggregation."""

from __future__ import annotations

import asyncio
import base64
import json
from typing import Any

from pydantic import BaseModel, Field

from . import variants
from .checkpoint_links import checkpoint_copy_links as _checkpoint_link_payload
from .clusters import load_cluster, list_clusters
from .mlxp_config import get_settings as get_mlxp_settings
from .mlxp_data_pod import ensure_listing_pod
from .paths import CLUSTER_STAGING_REL
from .ssh import ssh_run


class ResultCell(BaseModel):
    eval_set: str
    mean_success_rate: float | None
    std_success_rate: float | None
    per_run_success_rate: list[float]
    success_counts: list[int | None] = Field(default_factory=list)
    episode_counts: list[int | None] = Field(default_factory=list)
    completed_runs: int
    expected_runs: int | None = None
    source: str | None = None


class ResultTask(BaseModel):
    task: str
    task_name: str | None = None
    instruction: str | None = None
    eval_sets: list[ResultCell] = Field(default_factory=list)


class ResultVariant(BaseModel):
    cluster: str
    job_id: str | None = None
    job_name: str | None = None
    job_state: str | None = None
    checkpoint_job_cluster: str | None = None
    checkpoint_job_id: str | None = None
    checkpoint_job_name: str | None = None
    variant: str
    experiment: str | None = None
    model_version: str | None = None
    note: str | None = None
    checkpoint: str | None = None
    n_episodes: int | None = None
    n_runs: int | None = None
    num_envs_per_gpu: int | None = None
    total_num_envs: int | None = None
    source: str | None = None
    # Epoch seconds of the newest results.json mtime — when the eval finished.
    completed_at: float | None = None
    tasks: list[ResultTask] = Field(default_factory=list)


class ClusterResultError(BaseModel):
    cluster: str
    error: str


class ResultsResponse(BaseModel):
    clusters: list[str]
    variants: list[ResultVariant]
    errors: list[ClusterResultError] = Field(default_factory=list)


async def list_results(cluster: str | None = None) -> ResultsResponse:
    target_clusters = [cluster] if cluster else list_clusters()
    payload = await _variant_payload()
    checkpoint_links = await _checkpoint_link_payload()

    async def _one(c: str) -> tuple[list[ResultVariant], ClusterResultError | None]:
        try:
            if c == "mlxp":
                raw = await _read_mlxp_results(payload, checkpoint_links)
                return [ResultVariant.model_validate(x) for x in raw], None
            env = await load_cluster(c)
            raw = await _read_cluster_results(env.ssh_alias, c, payload, checkpoint_links)
            return [ResultVariant.model_validate(x) for x in raw], None
        except Exception as e:
            # asyncio.TimeoutError (and some others) stringify to "" — fall back
            # to the type name so the UI never shows a blank error.
            return [], ClusterResultError(cluster=c, error=str(e) or type(e).__name__)

    groups = await asyncio.gather(*(_one(c) for c in target_clusters))
    out: list[ResultVariant] = []
    errors: list[ClusterResultError] = []
    for rows, err in groups:
        out.extend(rows)
        if err:
            errors.append(err)

    await _attach_mlxp_result_states(out)
    out.sort(key=lambda r: (r.cluster, r.model_version or "", r.variant))
    return ResultsResponse(clusters=target_clusters, variants=out, errors=errors)


async def _attach_mlxp_result_states(rows: list[ResultVariant]) -> None:
    mlxp_rows = [r for r in rows if r.cluster == "mlxp" and r.job_id and not r.job_state]
    if not mlxp_rows:
        return
    try:
        from . import mlxp_jobs
        jobs = await mlxp_jobs.list_jobs()
    except Exception:
        return
    state_by_id = {j.job_id: j.state for j in jobs}
    for row in mlxp_rows:
        state = state_by_id.get(row.job_id or "")
        if state:
            row.job_state = state


async def _variant_payload() -> list[dict[str, Any]]:
    names = variants.list_variants()
    loaded = await asyncio.gather(*(variants.load_variant(n) for n in names))
    payload: list[dict[str, Any]] = []
    for v in loaded:
        tasks = _tasks_for_variant(v)
        payload.append(
            {
                "variant": v.name,
                "model_version": v.vars.get("MODEL_ID") or v.vars.get("MODEL_VERSION"),
                "note": v.vars.get("TRAIN_NOTE"),
                "eval_sets": v.arrays.get("EVAL_SETS", []),
                "n_runs": _int_or_none(v.vars.get("N_RUNS")),
                "n_episodes": _int_or_none(v.vars.get("N_EPISODES")),
                "tasks": tasks,
            }
        )
    return payload


def _tasks_for_variant(v: variants.Variant) -> list[dict[str, str | None]]:
    out: list[dict[str, str | None]] = []
    for entry in v.arrays.get("TASKS", []):
        short, task_name, instruction = _split_task_entry(entry)
        if short:
            out.append(
                {
                    "short": short,
                    "task_name": task_name,
                    "instruction": instruction,
                }
            )
    if out:
        return out
    return [
        {
            "short": v.name,
            "task_name": v.vars.get("TASK_NAME") or v.name,
            "instruction": v.vars.get("INSTRUCTION"),
        }
    ]


def _split_task_entry(entry: str) -> tuple[str | None, str | None, str | None]:
    parts = entry.split("|", 2)
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    if len(parts) == 2:
        return parts[0], parts[1], None
    if len(parts) == 1:
        return parts[0], parts[0], None
    return None, None, None


def _int_or_none(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _remote_program(env_vars: dict[str, str]) -> str:
    """Env setup + scan script, fed to a remote `python3 -` over stdin."""
    lines = ["import os"]
    for key, value in env_vars.items():
        lines.append(f"os.environ[{key!r}] = {value!r}")
    return "\n".join(lines) + "\n" + _REMOTE_SCRIPT


# Result scans walk every experiment dir over the cluster's network filesystem;
# on slow mounts (e.g. skt /fsx) the single-pass threaded scan can take well
# over a minute. The Results page fetches per cluster, so a generous ceiling
# here only delays the one slow column rather than the whole page.
_SCAN_TIMEOUT = 180.0


async def _read_cluster_results(
    host: str,
    cluster: str,
    payload: list[dict[str, Any]],
    checkpoint_links: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    # The whole program (env setup + script + payload) travels over stdin:
    # in argv it would hit Linux's 128KiB per-argument cap as results grow.
    program = _remote_program(
        {
            "RESULTS_PAYLOAD_B64": base64.b64encode(json.dumps(payload).encode()).decode(),
            "RESULTS_CHECKPOINT_LINKS_B64": base64.b64encode(json.dumps(checkpoint_links).encode()).decode(),
            "RESULTS_CLUSTER": cluster,
            "RESULTS_STAGING_REL": CLUSTER_STAGING_REL,
        }
    )
    r = await ssh_run(host, "python3 -", timeout=_SCAN_TIMEOUT, input_text=program)
    if r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout).strip() or f"ssh command failed on {cluster}")
    try:
        return json.loads(r.stdout or "[]")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"could not parse results from {cluster}: {e}: {r.stdout[:500]}")


async def _read_mlxp_results(
    payload: list[dict[str, Any]],
    checkpoint_links: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    settings = get_mlxp_settings()
    pod = await ensure_listing_pod()
    program = _remote_program(
        {
            "RESULTS_PAYLOAD_B64": base64.b64encode(json.dumps(payload).encode()).decode(),
            "RESULTS_CHECKPOINT_LINKS_B64": base64.b64encode(json.dumps(checkpoint_links).encode()).decode(),
            "RESULTS_CLUSTER": "mlxp",
            "RESULTS_EXPERIMENTS_ROOT": settings.experiments_dir,
        }
    )
    proc = await asyncio.create_subprocess_exec(
        "kubectl", "exec", "-i", "-n", settings.namespace, pod, "--", "python3", "-",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(proc.communicate(program.encode()), timeout=_SCAN_TIMEOUT)
    if proc.returncode != 0:
        raise RuntimeError(
            stderr.decode(errors="replace").strip()
            or stdout.decode(errors="replace").strip()
            or "kubectl exec failed on mlxp"
        )
    out = stdout.decode(errors="replace")
    try:
        return json.loads(out or "[]")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"could not parse results from mlxp: {e}: {out[:500]}")


_REMOTE_SCRIPT = r'''
import base64
import json
import os
import statistics
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


# Network filesystems (FSx Lustre on skt) charge a round trip per metadata
# or read op; overlapping them in threads is what makes the scan fast. The
# pool is only ever used from the main thread, never from its own workers.
_IO_POOL = ThreadPoolExecutor(max_workers=16)

payload = json.loads(base64.b64decode(os.environ["RESULTS_PAYLOAD_B64"]).decode())
checkpoint_links = json.loads(
    base64.b64decode(os.environ.get("RESULTS_CHECKPOINT_LINKS_B64", "W10=")).decode()
)
cluster = os.environ["RESULTS_CLUSTER"]
staging_rel = os.environ.get("RESULTS_STAGING_REL", ".train-eval-web")
experiments_root = Path(os.environ["RESULTS_EXPERIMENTS_ROOT"]) if os.environ.get("RESULTS_EXPERIMENTS_ROOT") else Path.home() / staging_rel / "experiments"


def int_or_none(value):
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def float_or_none(value):
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


_json_cache = {}


def _load_json_outcome(path):
    try:
        with open(path) as f:
            return (True, json.load(f))
    except Exception as e:
        return (False, e)


def read_json(path):
    key = str(path)
    outcome = _json_cache.get(key)
    if outcome is None:
        outcome = _load_json_outcome(key)
        _json_cache[key] = outcome
    ok, value = outcome
    if not ok:
        raise value
    return value


def prefetch_json(paths):
    fresh = [str(p) for p in paths]
    fresh = [p for p in fresh if p not in _json_cache]
    for path, outcome in zip(fresh, _IO_POOL.map(_load_json_outcome, fresh)):
        _json_cache[path] = outcome


def eval_set_key(eval_sets):
    order = {v: i for i, v in enumerate(eval_sets or [])}
    return lambda item: (order.get(item[0], len(order)), item[0])


def rate_from_run(data):
    summary = data.get("summary") or {}
    rate = float_or_none(summary.get("success_rate"))
    success_count = summary.get("success_count")
    total = summary.get("total_episodes") or summary.get("episode_count")
    if rate is not None:
        return rate, int_or_none(success_count), int_or_none(total)
    success = data.get("success")
    if isinstance(success, list) and success:
        count = sum(1 for v in success if v)
        return count / len(success), count, len(success)
    return None, int_or_none(success_count), int_or_none(total)


def cell_from_rates(eval_set, rates, success_counts=None, episode_counts=None, expected_runs=None, source=None):
    episode_counts = episode_counts or []
    if not rates and not any((v or 0) > 0 for v in episode_counts):
        return None
    return {
        "eval_set": eval_set,
        "mean_success_rate": statistics.mean(rates) if rates else None,
        "std_success_rate": (statistics.pstdev(rates) if len(rates) > 1 else 0.0) if rates else None,
        "per_run_success_rate": rates,
        "success_counts": success_counts or [],
        "episode_counts": episode_counts,
        "completed_runs": len(rates),
        "expected_runs": expected_runs,
        "source": source,
    }


def cells_from_aggregate(eval_sets_obj, expected_runs, configured_eval_sets, source):
    cells = []
    for eval_set, data in sorted((eval_sets_obj or {}).items(), key=eval_set_key(configured_eval_sets)):
        runs = data.get("per_run_success_rate") or data.get("run_success_rates") or []
        runs = [float(v) for v in runs]
        mean = float_or_none(data.get("mean_success_rate"))
        std = float_or_none(data.get("std_success_rate"))
        if mean is None and runs:
            mean = statistics.mean(runs)
        if std is None and runs:
            std = statistics.pstdev(runs) if len(runs) > 1 else 0.0
        if mean is None:
            continue
        cells.append({
            "eval_set": eval_set,
            "mean_success_rate": mean,
            "std_success_rate": std or 0.0,
            "per_run_success_rate": runs,
            "success_counts": [],
            "episode_counts": [],
            "completed_runs": len(runs) if runs else expected_runs or 0,
            "expected_runs": expected_runs,
            "source": source,
        })
    return cells


class DirNode:
    __slots__ = ("path", "name", "children", "has_results", "results_mtime")

    def __init__(self, path, name):
        self.path = path
        self.name = name
        self.children = {}
        self.has_results = False
        self.results_mtime = None


def build_tree(root):
    """Single-pass scandir snapshot of an eval_results tree.

    Every consumer below used to re-walk the same tree with its own
    recursive glob (completion mtime, run metadata, per-task cells, plus
    the per-variant legacy pass); on metadata-slow filesystems (FSx
    Lustre) those repeated walks dominated the whole fetch. `videos/`
    run-artifact dirs are listed but not descended into, and symlinked
    dirs are listed but not followed (matching pathlib `**` semantics).
    """
    root = Path(root)
    if not root.is_dir():
        return None
    top = DirNode(str(root), root.name)
    level = [top]
    while level:
        next_level = []
        for node, entries in zip(level, _IO_POOL.map(_scan_dir, [n.path for n in level])):
            for name, path, is_dir, descend, mtime in entries:
                if is_dir:
                    child = DirNode(path, name)
                    node.children[name] = child
                    if descend and name != "videos":
                        next_level.append(child)
                else:
                    node.has_results = True
                    node.results_mtime = mtime
        level = next_level
    return top


def _scan_dir(path):
    out = []
    try:
        with os.scandir(path) as it:
            for entry in it:
                try:
                    is_real_dir = entry.is_dir(follow_symlinks=False)
                    if is_real_dir or entry.is_dir():
                        out.append((entry.name, entry.path, True, is_real_dir, None))
                    elif entry.name == "results.json" and entry.is_file():
                        try:
                            mtime = entry.stat().st_mtime
                        except OSError:
                            mtime = None
                        out.append((entry.name, entry.path, False, False, mtime))
                except OSError:
                    continue
    except OSError:
        pass
    return out


def iter_result_json_paths(root_node):
    out = []
    stack = [root_node]
    while stack:
        node = stack.pop()
        if node.has_results:
            out.append(os.path.join(node.path, "results.json"))
        stack.extend(node.children.values())
    return out


def iter_run_result_nodes(root_node):
    """Nodes matching `**/run_*/results.json` strictly below root_node."""
    out = []
    stack = [root_node]
    while stack:
        node = stack.pop()
        for child in node.children.values():
            if child.name.startswith("run_") and child.has_results:
                out.append(child)
            stack.append(child)
    return out


def scan_eval_sets(node, configured_eval_sets):
    found = list(node.children) if node is not None else []
    ordered = []
    for eval_set in configured_eval_sets or []:
        if eval_set in found:
            ordered.append(eval_set)
    ordered.extend(sorted(v for v in found if v not in set(ordered)))
    return ordered


def cells_from_runs(task_node, configured_eval_sets, expected_runs):
    cells = []
    if task_node is None:
        return cells
    for eval_set in scan_eval_sets(task_node, configured_eval_sets):
        eval_node = task_node.children[eval_set]
        run_nodes = sorted(
            (c for c in eval_node.children.values() if c.name.startswith("run_")),
            key=lambda n: run_name_key(n.name),
        )
        vals = []
        completed = set()
        for node in run_nodes:
            if not node.has_results:
                continue
            try:
                rate, success_count, total = rate_from_run(read_json(os.path.join(node.path, "results.json")))
            except Exception:
                continue
            if rate is not None:
                vals.append((rate, success_count, total))
                completed.add(node.name)
        partial_episode_counts = []
        for node in run_nodes:
            if node.name in completed:
                continue
            try:
                count = sum(1 for _ in Path(node.path, "videos").glob("ep*.mp4"))
            except Exception:
                count = 0
            if count > 0:
                partial_episode_counts.append(count)
        rates = [v[0] for v in vals]
        cell = cell_from_rates(
            eval_set,
            rates,
            [v[1] for v in vals] + [None for _ in partial_episode_counts],
            [v[2] for v in vals] + partial_episode_counts,
            expected_runs,
            eval_node.path,
        )
        if cell:
            cells.append(cell)
    return cells


def metadata_from_runs(task_nodes):
    for task_node in task_nodes:
        if task_node is None:
            continue
        for node in sorted(iter_run_result_nodes(task_node), key=lambda n: n.path):
            try:
                data = read_json(os.path.join(node.path, "results.json"))
            except Exception:
                continue
            config = data.get("config") or {}
            summary = data.get("summary") or {}
            return {
                "checkpoint": (
                    config.get("checkpoint")
                    or config.get("checkpoint_path")
                    or config.get("model_path")
                ),
                "n_episodes": int_or_none(
                    config.get("n_episodes")
                    or summary.get("total_episodes")
                    or summary.get("episode_count")
                ),
            }
    return {}


def path_key(value):
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    home = str(Path.home())
    if s.startswith("$HOME/"):
        s = home + s[len("$HOME"):]
    elif s == "$HOME":
        s = home
    elif s.startswith("~/"):
        s = home + s[1:]
    return s.rstrip("/")


def is_checkpoint_step_leaf(name):
    return str(name or "").startswith("checkpoint-")


def add_job_index_entry(index, key, info):
    key = path_key(key)
    if not key:
        return
    index[key] = info
    if key.endswith("/results.json"):
        index[key[: -len("/results.json")]] = info


def add_job_name_index(index, info):
    job_name = (info.get("job_name") or "").strip()
    if job_name:
        index[job_name] = info


def add_checkpoint_index_entry(index, key, info, *, overwrite=True):
    key = path_key(key)
    if not key:
        return
    if overwrite or key not in index:
        index[key] = info
    leaf = Path(key).name
    if leaf and not is_checkpoint_step_leaf(leaf) and (overwrite or leaf not in index):
        index[leaf] = info


def job_info_from_meta(job_id, meta):
    return {
        "cluster": meta.get("cluster") or cluster,
        "job_id": str(job_id),
        "job_name": meta.get("job_name") or str(job_id),
        "job_state": meta.get("state"),
        "train_note": meta.get("train_note"),
    }


def add_eval_job_indexes(path_index, name_index, info, meta):
    add_job_name_index(name_index, info)
    add_job_index_entry(path_index, meta.get("eval_dir"), info)
    add_job_index_entry(path_index, meta.get("results_path"), info)
    if meta.get("variant") and meta.get("output_namespace"):
        add_job_index_entry(
            path_index,
            experiments_root / meta["variant"] / "eval_results" / meta["output_namespace"],
            info,
        )


def add_checkpoint_job_indexes(checkpoint_index, info, meta):
    add_checkpoint_index_entry(checkpoint_index, meta.get("checkpoint_dir"), info)
    add_checkpoint_index_entry(checkpoint_index, meta.get("output_namespace"), info)


def add_job_indexes(path_index, name_index, checkpoint_index, info, meta):
    phase = meta.get("phase")
    if phase in ("train", "resume"):
        add_checkpoint_job_indexes(checkpoint_index, info, meta)
    elif phase == "eval":
        add_eval_job_indexes(path_index, name_index, info, meta)


def _read_text_or_none(path):
    try:
        return Path(path).read_text()
    except Exception:
        return None


def parse_sidecar_meta_text(text):
    out = {}
    if text is None:
        return out
    for line in text.splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def job_meta_sort_key(path):
    # Numeric-aware order: jobs sharing a results namespace (resume chains)
    # collide in the last-writer-wins index, so the newest job id must be
    # processed last. Lexicographic order gets this wrong across digit
    # counts ("100034" < "94097").
    stem = path.stem
    return (0, int(stem)) if stem.isdigit() else (1, stem)


def load_result_job_index():
    path_index = {}
    name_index = {}
    checkpoint_index = {}

    job_meta_root = Path.home() / staging_rel / "jobs"
    job_meta_paths = sorted(job_meta_root.glob("*.meta"), key=job_meta_sort_key) if job_meta_root.exists() else []
    texts = list(_IO_POOL.map(_read_text_or_none, job_meta_paths))
    for path, text in zip(job_meta_paths, texts):
        meta = parse_sidecar_meta_text(text)
        info = job_info_from_meta(path.stem, meta)
        add_job_indexes(path_index, name_index, checkpoint_index, info, meta)

    snapshot_meta_paths = sorted(experiments_root.glob("*/config_*.meta.json")) if experiments_root.exists() else []
    prefetch_json(snapshot_meta_paths)
    for path in snapshot_meta_paths:
        try:
            meta = read_json(path)
        except Exception:
            continue
        if not meta.get("job_id"):
            continue
        info = job_info_from_meta(meta["job_id"], meta)
        add_job_indexes(path_index, name_index, checkpoint_index, info, meta)

    return path_index, name_index, checkpoint_index


def iter_unique_job_infos(*indexes):
    seen = set()
    for index in indexes:
        for info in index.values():
            ident = id(info)
            if ident in seen:
                continue
            seen.add(ident)
            yield info


def slurm_states(job_ids):
    job_ids = [str(j) for j in dict.fromkeys(job_ids) if str(j).strip()]
    if not job_ids:
        return {}
    joined = ",".join(job_ids)
    out = {}

    def run(args):
        try:
            return subprocess.run(
                args,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=10,
            ).stdout
        except Exception:
            return ""

    for line in run(["sacct", "-X", "-j", joined, "-P", "-n", "-o", "JobID,State"]).splitlines():
        parts = line.split("|")
        if len(parts) >= 2 and parts[0] and parts[1]:
            out[parts[0]] = parts[1].split(" ", 1)[0]

    # Prefer squeue for active jobs because it is fresher than sacct.
    for line in run(["squeue", "-h", "-j", joined, "-o", "%i|%T"]).splitlines():
        parts = line.split("|")
        if len(parts) >= 2 and parts[0] and parts[1]:
            out[parts[0]] = parts[1]
    return out


def add_current_job_states(path_index, name_index):
    infos = list(iter_unique_job_infos(path_index, name_index))
    states = slurm_states(info.get("job_id") for info in infos)
    for info in infos:
        state = states.get(str(info.get("job_id")))
        if state:
            info["job_state"] = state


def add_copied_checkpoint_indexes(checkpoint_index):
    for item in checkpoint_links or []:
        if not isinstance(item, dict):
            continue
        info = item.get("info")
        key = item.get("key")
        if not isinstance(info, dict) or not key:
            continue
        if not info.get("cluster") or not info.get("job_id"):
            continue
        add_checkpoint_index_entry(checkpoint_index, key, info, overwrite=False)


def result_job_info(eval_root, top_path, job_name=None):
    for candidate in (top_path, eval_root):
        info = job_path_index.get(path_key(candidate))
        if info:
            return info
    if job_name:
        info = job_name_index.get(job_name)
        if info:
            return info
    return None


def apply_job_info(result, info):
    if not info:
        return
    result["job_id"] = info.get("job_id")
    result["job_name"] = info.get("job_name")
    result["job_state"] = info.get("job_state")
    train_note = str(info.get("train_note") or "").strip()
    if train_note:
        result["note"] = train_note


def checkpoint_job_info(checkpoint):
    checkpoint = path_key(checkpoint)
    if not checkpoint:
        return None
    path = Path(checkpoint)
    candidates = [checkpoint]
    if is_checkpoint_step_leaf(path.name):
        parent = path.parent
        candidates.extend([str(parent), parent.name])
    else:
        candidates.append(path.name)
    for candidate in candidates:
        info = checkpoint_index.get(path_key(candidate) or candidate)
        if info:
            return info
    return None


def apply_checkpoint_job_info(result):
    info = checkpoint_job_info(result.get("checkpoint"))
    if not info:
        return
    result["checkpoint_job_cluster"] = info.get("cluster")
    result["checkpoint_job_id"] = info.get("job_id")
    result["checkpoint_job_name"] = info.get("job_name")


def finalize_result(result):
    apply_checkpoint_job_info(result)
    return result


def run_name_key(name):
    # Numeric-aware ordering of run_<N> directory names.
    try:
        return (0, int(name.rsplit("_", 1)[-1]))
    except Exception:
        return (1, name)


def newest_result_mtime(eval_node, top_mtime):
    """Completion time: newest mtime among the aggregate and per-run results.json."""
    times = []
    if top_mtime is not None:
        times.append(top_mtime)
    if eval_node is not None:
        for node in iter_run_result_nodes(eval_node):
            if node.results_mtime is not None:
                times.append(node.results_mtime)
    return max(times) if times else None


def task_name_for(short, configured_tasks, fallback=None):
    for task in configured_tasks:
        if task.get("short") == short:
            return task.get("task_name") or fallback
    return fallback


def instruction_for(short, configured_tasks, fallback=None):
    for task in configured_tasks:
        if task.get("short") == short:
            return task.get("instruction") or fallback
    return fallback


def build_variant_from_root(meta, eval_node, eval_root, top_path, top_exists, top_mtime):
    variant = meta["variant"]
    configured_tasks = meta.get("tasks") or []
    configured_eval_sets = meta.get("eval_sets") or []
    expected_runs = int_or_none(meta.get("n_runs"))
    top = None
    aggregate_tasks = []
    if top_exists:
        try:
            top = read_json(top_path)
        except Exception:
            top = None

    result = {
        "cluster": cluster,
        "job_id": None,
        "job_name": None,
        "job_state": None,
        "checkpoint_job_cluster": None,
        "checkpoint_job_id": None,
        "checkpoint_job_name": None,
        "variant": variant,
        "experiment": None,
        "model_version": meta.get("model_version"),
        "note": meta.get("note"),
        "checkpoint": None,
        "n_episodes": int_or_none(meta.get("n_episodes")),
        "n_runs": expected_runs,
        "num_envs_per_gpu": None,
        "total_num_envs": None,
        "source": str(top_path) if top_exists else str(eval_root),
        "completed_at": newest_result_mtime(eval_node, top_mtime),
        "tasks": [],
    }
    job_info = result_job_info(eval_root, top_path)
    apply_job_info(result, job_info)
    if top:
        result.update({
            "experiment": top.get("experiment") or top.get("experiment_name"),
            "model_version": top.get("model_version") or result["model_version"],
            "note": top.get("note") or result["note"],
            "checkpoint": top.get("checkpoint") or top.get("checkpoint_path"),
            "n_episodes": int_or_none(top.get("n_episodes")) or result["n_episodes"],
            "n_runs": int_or_none(top.get("n_runs")) or result["n_runs"],
            "num_envs_per_gpu": int_or_none(top.get("num_envs_per_gpu")),
            "total_num_envs": int_or_none(top.get("total_num_envs")),
        })
        apply_job_info(result, job_info)
        expected_runs = result["n_runs"]
        if not result["job_id"]:
            job_info = result_job_info(eval_root, top_path, result.get("experiment"))
            apply_job_info(result, job_info)
        if isinstance(top.get("tasks"), dict):
            for short, task_data in sorted(top["tasks"].items()):
                cells = cells_from_aggregate(
                    task_data.get("eval_sets") or {},
                    expected_runs,
                    configured_eval_sets,
                    str(top_path),
                )
                if cells:
                    aggregate_tasks.append({
                        "task": short,
                        "task_name": task_data.get("task_name") or task_name_for(short, configured_tasks, short),
                        "instruction": task_data.get("instruction") or instruction_for(short, configured_tasks),
                        "eval_sets": cells,
                    })
        elif isinstance(top.get("eval_sets"), dict):
            short = configured_tasks[0].get("short") if configured_tasks else variant
            cells = cells_from_aggregate(
                top.get("eval_sets") or {},
                expected_runs,
                configured_eval_sets,
                str(top_path),
            )
            if cells:
                aggregate_tasks.append({
                    "task": short,
                    "task_name": top.get("task_name") or task_name_for(short, configured_tasks, short),
                    "instruction": instruction_for(short, configured_tasks),
                    "eval_sets": cells,
                })

    if eval_node is None:
        if aggregate_tasks:
            result["tasks"] = aggregate_tasks
            return finalize_result(result)
        return None

    if len(configured_tasks) > 1:
        tasks = configured_tasks
        task_nodes = [eval_node.children.get(task.get("short") or variant) for task in tasks]
    else:
        tasks = configured_tasks or [{"short": variant, "task_name": variant, "instruction": None}]
        task_nodes = [eval_node]

    fallback_meta = metadata_from_runs(task_nodes)
    if fallback_meta.get("checkpoint") and not result["checkpoint"]:
        result["checkpoint"] = fallback_meta["checkpoint"]
    if fallback_meta.get("n_episodes") and not result["n_episodes"]:
        result["n_episodes"] = fallback_meta["n_episodes"]

    for task in tasks:
        short = task.get("short") or variant
        task_node = eval_node.children.get(short) if len(tasks) > 1 else eval_node
        cells = cells_from_runs(task_node, configured_eval_sets, expected_runs)
        if cells:
            result["tasks"].append({
                "task": short,
                "task_name": task.get("task_name") or short,
                "instruction": task.get("instruction"),
                "eval_sets": cells,
            })

    if result["tasks"]:
        result["source"] = str(eval_root)
        return finalize_result(result)
    if aggregate_tasks:
        result["tasks"] = aggregate_tasks
        return finalize_result(result)
    return None


def build_variant(meta):
    variant = meta["variant"]
    exp_dir = experiments_root / variant
    if not exp_dir.exists():
        return []

    eval_root = exp_dir / "eval_results"
    tree = build_tree(eval_root)
    if tree is not None:
        prefetch_json(iter_result_json_paths(tree))
    rows = []

    # New layout: one immutable output namespace per eval submission:
    #   <variant>/eval_results/<output_namespace>/results.json
    if tree is not None:
        for name in sorted(tree.children):
            node = tree.children[name]
            run_root = eval_root / name
            item = build_variant_from_root(
                meta, node, run_root, run_root / "results.json",
                node.has_results, node.results_mtime,
            )
            if item:
                rows.append(item)

    # Legacy layout: all eval runs directly under <variant>/eval_results and
    # aggregate at <variant>/results.json.
    legacy_top = exp_dir / "results.json"
    legacy_exists = legacy_top.exists()
    legacy_mtime = None
    if legacy_exists:
        try:
            legacy_mtime = legacy_top.stat().st_mtime
        except OSError:
            pass
    legacy_item = build_variant_from_root(meta, tree, eval_root, legacy_top, legacy_exists, legacy_mtime)
    if legacy_item:
        legacy_source = legacy_item.get("source") or ""
        duplicate_new_source = "/eval_results/" in legacy_source and legacy_source.endswith("/results.json")
        if not duplicate_new_source:
            rows.append(legacy_item)
    return rows


rows = []
job_path_index, job_name_index, checkpoint_index = load_result_job_index()
add_current_job_states(job_path_index, job_name_index)
add_copied_checkpoint_indexes(checkpoint_index)
for meta in payload:
    rows.extend(build_variant(meta))
print(json.dumps(rows))
'''
