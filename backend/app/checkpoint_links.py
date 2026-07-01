"""Resolve checkpoints back to the training jobs that produced them."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import shlex
from typing import Any

from .clusters import list_clusters, load_cluster
from .mlxp_config import get_settings as get_mlxp_settings
from .paths import CHECKPOINT_COPY_HISTORY_REL, CLUSTER_STAGING_REL
from .remote_paths import _kubectl_bash_lc
from .ssh import ssh_run


def checkpoint_leaf(path: Any) -> str | None:
    path = str(path or "").rstrip("/")
    if not path:
        return None
    return path.rsplit("/", 1)[-1]


def is_checkpoint_step_leaf(name: Any) -> bool:
    return str(name or "").startswith("checkpoint-")


def checkpoint_parent_if_step(path: Any) -> str | None:
    path = str(path or "").rstrip("/")
    leaf = checkpoint_leaf(path)
    if not leaf or not is_checkpoint_step_leaf(leaf) or "/" not in path:
        return None
    return path.rsplit("/", 1)[0]


def _dedupe(keys: list[str]) -> list[str]:
    """Order-preserving dedupe, dropping empty keys."""
    return list(dict.fromkeys(k for k in keys if k))


def checkpoint_lookup_keys(path: Any) -> list[str]:
    path = str(path or "").strip().rstrip("/")
    if not path:
        return []
    keys = [path]
    leaf = checkpoint_leaf(path)
    parent = checkpoint_parent_if_step(path)
    if parent:
        keys.extend([parent, checkpoint_leaf(parent) or ""])
    elif leaf:
        keys.append(leaf)
    return _dedupe(keys)


async def checkpoint_copy_links() -> list[dict[str, Any]]:
    """Return copied-checkpoint aliases keyed by source and destination paths."""

    async def _one(cluster_name: str) -> list[dict[str, Any]]:
        try:
            if cluster_name == "mlxp":
                return await _mlxp_copy_history_records()
            env = await load_cluster(cluster_name)
            return await _slurm_copy_history_records(env.ssh_alias)
        except Exception:
            return []

    groups = await asyncio.gather(*(_one(c) for c in list_clusters()))
    links: list[dict[str, Any]] = []
    seen: set[tuple[str, str | None, str | None]] = set()
    for records in groups:
        for record in records:
            info = _copy_record_job_info(record)
            if not info:
                continue
            for key in _copy_record_checkpoint_keys(record):
                dedupe_key = (key, info.get("cluster"), info.get("job_id"))
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                links.append({"key": key, "info": info})
    return links


async def find_training_job_for_checkpoint(
    cluster: str,
    checkpoint: str | None,
) -> dict[str, Any] | None:
    """Best-effort training job lookup for an eval checkpoint path."""
    if not checkpoint:
        return None

    local_task = asyncio.create_task(_local_checkpoint_job(cluster, checkpoint))
    links = await checkpoint_copy_links()

    # Exact copied-checkpoint history should win for cross-cluster evals, but
    # leaf-only copy aliases are weaker than local sidecar metadata.
    info = find_checkpoint_info_from_links(
        links,
        checkpoint,
        keys=checkpoint_strong_lookup_keys(checkpoint),
    )
    if info:
        local_task.cancel()
        with contextlib.suppress(BaseException):
            await local_task
        return info

    try:
        info = await local_task
    except Exception:
        info = None
    if info:
        return info

    return find_checkpoint_info_from_links(links, checkpoint)


def find_checkpoint_info_from_links(
    checkpoint_links: list[dict[str, Any]],
    checkpoint: str | None,
    *,
    keys: list[str] | None = None,
) -> dict[str, Any] | None:
    if not checkpoint:
        return None
    by_key: dict[str, dict[str, Any]] = {}
    for item in checkpoint_links:
        if not isinstance(item, dict):
            continue
        key = str(item.get("key") or "").strip().rstrip("/")
        info = item.get("info")
        if not key or not isinstance(info, dict):
            continue
        if not info.get("cluster") or not info.get("job_id"):
            continue
        by_key.setdefault(key, info)

    for key in keys or checkpoint_lookup_keys(checkpoint):
        info = by_key.get(key)
        if info:
            return info
    return None


def checkpoint_strong_lookup_keys(path: Any) -> list[str]:
    path = str(path or "").strip().rstrip("/")
    if not path:
        return []
    keys = [path]
    parent = checkpoint_parent_if_step(path)
    if parent:
        keys.append(parent)
    return _dedupe(keys)


async def _slurm_copy_history_records(host: str) -> list[dict[str, Any]]:
    path = f"$HOME/{CHECKPOINT_COPY_HISTORY_REL}/*.jsonl"
    r = await ssh_run(host, f"cat {path} 2>/dev/null || true", timeout=15.0)
    if r.returncode != 0:
        return []
    return _parse_jsonl_records(r.stdout)


async def _mlxp_copy_history_records() -> list[dict[str, Any]]:
    settings = get_mlxp_settings()
    hist_dir = shlex.quote(f"{settings.experiments_dir}/{CHECKPOINT_COPY_HISTORY_REL}")
    rc, out, _err = await _kubectl_bash_lc(f"cat {hist_dir}/*.jsonl 2>/dev/null || true", 15.0)
    if rc != 0:
        return []
    return _parse_jsonl_records(out)


def _parse_jsonl_records(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _copy_record_job_info(record: dict[str, Any]) -> dict[str, Any] | None:
    source_cluster = str(record.get("source_cluster") or "").strip()
    source_job = str(record.get("source_job") or "").strip()
    if not source_cluster or not source_job:
        return None
    source_path = record.get("source_path")
    source_root = checkpoint_parent_if_step(source_path) or source_path
    return {
        "cluster": source_cluster,
        "job_id": source_job,
        "job_name": checkpoint_leaf(source_root) or source_job,
    }


def _copy_record_checkpoint_keys(record: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    for field in ("dest_path", "source_path"):
        path = str(record.get(field) or "").strip()
        if path:
            keys.extend(checkpoint_lookup_keys(path))
    return _dedupe(keys)


async def _local_checkpoint_job(cluster: str, checkpoint: str) -> dict[str, Any] | None:
    checkpoint_b64 = base64.b64encode(checkpoint.encode()).decode()
    if cluster == "mlxp":
        settings = get_mlxp_settings()
        env_assignments = [
            f"CHECKPOINT_LOOKUP_B64={shlex.quote(checkpoint_b64)}",
            "CHECKPOINT_LOOKUP_CLUSTER=mlxp",
            f"CHECKPOINT_LOOKUP_EXPERIMENTS_ROOT={shlex.quote(settings.experiments_dir)}",
        ]
    else:
        env_assignments = [
            f"CHECKPOINT_LOOKUP_B64={shlex.quote(checkpoint_b64)}",
            f"CHECKPOINT_LOOKUP_CLUSTER={shlex.quote(cluster)}",
            f"CHECKPOINT_LOOKUP_STAGING_REL={shlex.quote(CLUSTER_STAGING_REL)}",
        ]
    return await _run_remote_lookup(cluster, env_assignments)


async def _run_remote_lookup(
    cluster: str,
    env_assignments: list[str],
) -> dict[str, Any] | None:
    """Run the checkpoint-lookup script over ssh (slurm) or kubectl (mlxp).

    Both transports assemble the same `<env> python3 - <<PY ... PY` heredoc and
    decode the JSON payload; only the env vars and the transport differ.
    """
    cmd = (
        " ".join(env_assignments)
        + " python3 - <<'PY'\n"
        + _REMOTE_CHECKPOINT_LOOKUP_SCRIPT
        + "\nPY"
    )
    if cluster == "mlxp":
        rc, out, _err = await _kubectl_bash_lc(cmd, 20.0)
        if rc != 0:
            return None
        return _decode_lookup_payload(out)

    env = await load_cluster(cluster)
    r = await ssh_run(env.ssh_alias, cmd, timeout=20.0)
    if r.returncode != 0:
        return None
    return _decode_lookup_payload(r.stdout)


def _decode_lookup_payload(text: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(text or "null")
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    if not payload.get("cluster") or not payload.get("job_id"):
        return None
    return payload


_REMOTE_CHECKPOINT_LOOKUP_SCRIPT = r'''
import base64
import json
import os
from pathlib import Path


checkpoint = base64.b64decode(os.environ["CHECKPOINT_LOOKUP_B64"]).decode()
cluster = os.environ["CHECKPOINT_LOOKUP_CLUSTER"]
staging_rel = os.environ.get("CHECKPOINT_LOOKUP_STAGING_REL", ".train-eval-web")
experiments_root = Path(os.environ["CHECKPOINT_LOOKUP_EXPERIMENTS_ROOT"]) if os.environ.get("CHECKPOINT_LOOKUP_EXPERIMENTS_ROOT") else Path.home() / staging_rel / "experiments"


def read_json(path):
    with open(path) as f:
        return json.load(f)


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
    }


def add_checkpoint_job_indexes(checkpoint_index, info, meta):
    add_checkpoint_index_entry(checkpoint_index, meta.get("checkpoint_dir"), info)
    add_checkpoint_index_entry(checkpoint_index, meta.get("output_namespace"), info)


def add_job_indexes(checkpoint_index, info, meta):
    if meta.get("phase") in ("train", "resume"):
        add_checkpoint_job_indexes(checkpoint_index, info, meta)


def parse_sidecar_meta(path):
    out = {}
    try:
        for line in path.read_text().splitlines():
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    except Exception:
        return {}
    return out


def job_meta_sort_key(path):
    # Numeric-aware order: jobs sharing an output namespace (resume chains)
    # collide in the last-writer-wins index, so the newest job id must be
    # processed last. Lexicographic order gets this wrong across digit
    # counts ("100034" < "94097").
    stem = path.stem
    return (0, int(stem)) if stem.isdigit() else (1, stem)


def load_checkpoint_index():
    checkpoint_index = {}

    job_meta_root = Path.home() / staging_rel / "jobs"
    job_meta_paths = sorted(job_meta_root.glob("*.meta"), key=job_meta_sort_key) if job_meta_root.exists() else []
    for path in job_meta_paths:
        meta = parse_sidecar_meta(path)
        info = job_info_from_meta(path.stem, meta)
        add_job_indexes(checkpoint_index, info, meta)

    snapshot_meta_paths = sorted(experiments_root.glob("*/config_*.meta.json")) if experiments_root.exists() else []
    for path in snapshot_meta_paths:
        try:
            meta = read_json(path)
        except Exception:
            continue
        if not meta.get("job_id"):
            continue
        info = job_info_from_meta(meta["job_id"], meta)
        add_job_indexes(checkpoint_index, info, meta)

    return checkpoint_index


def checkpoint_job_info(checkpoint, checkpoint_index):
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


print(json.dumps(checkpoint_job_info(checkpoint, load_checkpoint_index())))
'''
