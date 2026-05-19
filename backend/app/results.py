"""Eval result discovery and aggregation."""

from __future__ import annotations

import asyncio
import base64
import json
import shlex
from typing import Any

from pydantic import BaseModel

from . import variants
from .clusters import load_cluster, list_clusters
from .paths import CLUSTER_STAGING_REL
from .ssh import ssh_run


class ResultCell(BaseModel):
    eval_set: str
    mean_success_rate: float
    std_success_rate: float
    per_run_success_rate: list[float]
    success_counts: list[int | None] = []
    episode_counts: list[int | None] = []
    completed_runs: int
    expected_runs: int | None = None
    source: str | None = None


class ResultTask(BaseModel):
    task: str
    task_name: str | None = None
    instruction: str | None = None
    eval_sets: list[ResultCell]


class ResultVariant(BaseModel):
    cluster: str
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
    tasks: list[ResultTask]


class ClusterResultError(BaseModel):
    cluster: str
    error: str


class ResultsResponse(BaseModel):
    clusters: list[str]
    variants: list[ResultVariant]
    errors: list[ClusterResultError] = []


async def list_results(cluster: str | None = None) -> ResultsResponse:
    target_clusters = [cluster] if cluster else [c for c in list_clusters() if c != "mlxp"]
    payload = await _variant_payload()

    async def _one(c: str) -> tuple[list[ResultVariant], ClusterResultError | None]:
        try:
            env = await load_cluster(c)
            raw = await _read_cluster_results(env.ssh_alias, c, payload)
            return [ResultVariant.model_validate(x) for x in raw], None
        except Exception as e:
            return [], ClusterResultError(cluster=c, error=str(e))

    groups = await asyncio.gather(*(_one(c) for c in target_clusters))
    out: list[ResultVariant] = []
    errors: list[ClusterResultError] = []
    for rows, err in groups:
        out.extend(rows)
        if err:
            errors.append(err)

    out.sort(key=lambda r: (r.cluster, r.model_version or "", r.variant))
    return ResultsResponse(clusters=target_clusters, variants=out, errors=errors)


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


async def _read_cluster_results(host: str, cluster: str, payload: list[dict[str, Any]]) -> list[dict[str, Any]]:
    payload_b64 = base64.b64encode(json.dumps(payload).encode()).decode()
    rel = shlex.quote(CLUSTER_STAGING_REL)
    cmd = (
        f"RESULTS_PAYLOAD_B64={shlex.quote(payload_b64)} "
        f"RESULTS_CLUSTER={shlex.quote(cluster)} "
        f"RESULTS_STAGING_REL={rel} "
        "python3 - <<'PY'\n"
        + _REMOTE_SCRIPT
        + "\nPY"
    )
    r = await ssh_run(host, cmd, timeout=45.0)
    if r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout).strip() or f"ssh command failed on {cluster}")
    try:
        return json.loads(r.stdout or "[]")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"could not parse results from {cluster}: {e}: {r.stdout[:500]}")


_REMOTE_SCRIPT = r'''
import base64
import json
import os
import statistics
from pathlib import Path


payload = json.loads(base64.b64decode(os.environ["RESULTS_PAYLOAD_B64"]).decode())
cluster = os.environ["RESULTS_CLUSTER"]
staging_rel = os.environ.get("RESULTS_STAGING_REL", ".train-eval-web")
experiments_root = Path.home() / staging_rel / "experiments"


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


def read_json(path):
    with open(path) as f:
        return json.load(f)


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
    if not rates:
        return None
    return {
        "eval_set": eval_set,
        "mean_success_rate": statistics.mean(rates),
        "std_success_rate": statistics.pstdev(rates) if len(rates) > 1 else 0.0,
        "per_run_success_rate": rates,
        "success_counts": success_counts or [],
        "episode_counts": episode_counts or [],
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


def scan_eval_sets(root, configured_eval_sets):
    found = [p.name for p in root.iterdir() if p.is_dir()] if root.is_dir() else []
    ordered = []
    for eval_set in configured_eval_sets or []:
        if eval_set in found:
            ordered.append(eval_set)
    ordered.extend(sorted(v for v in found if v not in set(ordered)))
    return ordered


def cells_from_runs(task_root, configured_eval_sets, expected_runs):
    cells = []
    for eval_set in scan_eval_sets(task_root, configured_eval_sets):
        eval_root = task_root / eval_set
        vals = []
        for path in sorted(eval_root.glob("run_*/results.json"), key=run_path_key):
            try:
                rate, success_count, total = rate_from_run(read_json(path))
            except Exception:
                continue
            if rate is not None:
                vals.append((rate, success_count, total, str(path)))
        rates = [v[0] for v in vals]
        cell = cell_from_rates(
            eval_set,
            rates,
            [v[1] for v in vals],
            [v[2] for v in vals],
            expected_runs,
            str(eval_root),
        )
        if cell:
            cells.append(cell)
    return cells


def metadata_from_runs(task_roots):
    for task_root in task_roots:
        for path in sorted(task_root.glob("**/run_*/results.json"), key=lambda p: str(p)):
            try:
                data = read_json(path)
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


def run_path_key(path):
    try:
        return (0, int(path.parent.name.rsplit("_", 1)[-1]))
    except Exception:
        return (1, path.parent.name)


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


def build_variant(meta):
    variant = meta["variant"]
    exp_dir = experiments_root / variant
    if not exp_dir.exists():
        return None

    configured_tasks = meta.get("tasks") or []
    configured_eval_sets = meta.get("eval_sets") or []
    expected_runs = int_or_none(meta.get("n_runs"))
    top_path = exp_dir / "results.json"
    top = None
    aggregate_tasks = []
    if top_path.exists():
        try:
            top = read_json(top_path)
        except Exception:
            top = None

    result = {
        "cluster": cluster,
        "variant": variant,
        "experiment": None,
        "model_version": meta.get("model_version"),
        "note": meta.get("note"),
        "checkpoint": None,
        "n_episodes": int_or_none(meta.get("n_episodes")),
        "n_runs": expected_runs,
        "num_envs_per_gpu": None,
        "total_num_envs": None,
        "source": str(top_path) if top_path.exists() else str(exp_dir / "eval_results"),
        "tasks": [],
    }
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
        expected_runs = result["n_runs"]
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

    eval_root = exp_dir / "eval_results"
    if not eval_root.exists():
        if aggregate_tasks:
            result["tasks"] = aggregate_tasks
            return result
        return None

    if len(configured_tasks) > 1:
        tasks = configured_tasks
        task_roots = [eval_root / (task.get("short") or variant) for task in tasks]
    else:
        tasks = configured_tasks or [{"short": variant, "task_name": variant, "instruction": None}]
        task_roots = [eval_root]

    fallback_meta = metadata_from_runs(task_roots)
    if fallback_meta.get("checkpoint") and not result["checkpoint"]:
        result["checkpoint"] = fallback_meta["checkpoint"]
    if fallback_meta.get("n_episodes") and not result["n_episodes"]:
        result["n_episodes"] = fallback_meta["n_episodes"]

    for task in tasks:
        short = task.get("short") or variant
        task_root = eval_root / short if len(tasks) > 1 else eval_root
        cells = cells_from_runs(task_root, configured_eval_sets, expected_runs)
        if cells:
            result["tasks"].append({
                "task": short,
                "task_name": task.get("task_name") or short,
                "instruction": task.get("instruction"),
                "eval_sets": cells,
            })

    if result["tasks"]:
        result["source"] = str(eval_root)
        return result
    if aggregate_tasks:
        result["tasks"] = aggregate_tasks
        return result
    return None


rows = []
for meta in payload:
    item = build_variant(meta)
    if item:
        rows.append(item)
print(json.dumps(rows))
'''
