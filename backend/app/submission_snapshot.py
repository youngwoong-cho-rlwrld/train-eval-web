"""Training submission snapshots.

The variant `config.sh` is only the starting point. A real training
submission may override datasets, GPU count, batch size, step counts, and
extra scheduler/job args. This module renders the effective config record and
captures the train-eval-web git revision used for the submission.
"""

from __future__ import annotations

import asyncio
import json
import re
import shlex
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from .paths import REPO_ROOT


class GitStatus(BaseModel):
    commit: str | None
    short_commit: str | None
    dirty: bool
    files: list[str] = Field(default_factory=list)
    error: str | None = None


@dataclass(frozen=True)
class SubmitGitInfo:
    commit: str | None
    dirty_before: bool
    committed_dirty: bool
    dirty_files: list[str]
    commit_message: str | None = None


async def _git(*args: str, timeout: float = 20.0) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        "git",
        *args,
        cwd=REPO_ROOT,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise RuntimeError(f"git {' '.join(args)} timed out after {timeout:g}s")
    return (
        proc.returncode,
        stdout.decode(errors="replace"),
        stderr.decode(errors="replace"),
    )


def _short(commit: str | None) -> str | None:
    return commit[:12] if commit else None


async def git_status() -> GitStatus:
    rc, head, err = await _git("rev-parse", "HEAD")
    if rc != 0:
        return GitStatus(
            commit=None,
            short_commit=None,
            dirty=True,
            files=[],
            error=(err or head).strip() or "git rev-parse failed",
        )
    rc, status, err = await _git("status", "--short")
    if rc != 0:
        return GitStatus(
            commit=head.strip(),
            short_commit=_short(head.strip()),
            dirty=True,
            files=[],
            error=(err or status).strip() or "git status failed",
        )
    files = [line for line in status.splitlines() if line.strip()]
    commit = head.strip()
    return GitStatus(
        commit=commit,
        short_commit=_short(commit),
        dirty=bool(files),
        files=files,
    )


async def prepare_training_git(
    job_name: str,
    commit_dirty_changes: bool,
    *,
    require_clean: bool = True,
) -> SubmitGitInfo:
    status = await git_status()
    if status.error:
        raise RuntimeError(status.error)
    if not status.dirty:
        return SubmitGitInfo(
            commit=status.commit,
            dirty_before=False,
            committed_dirty=False,
            dirty_files=[],
        )
    if not require_clean:
        return SubmitGitInfo(
            commit=status.commit,
            dirty_before=True,
            committed_dirty=False,
            dirty_files=status.files,
        )
    if not commit_dirty_changes:
        raise ValueError(
            "working tree has uncommitted changes; approve the training "
            "snapshot commit before submitting"
        )

    message = f"chore(training): snapshot state for {job_name}"
    rc, _, err = await _git("add", "-A", timeout=30.0)
    if rc != 0:
        raise RuntimeError(err.strip() or "git add failed")
    rc, out, err = await _git("commit", "-m", message, timeout=60.0)
    if rc != 0:
        raise RuntimeError((err or out).strip() or "git commit failed")
    clean = await git_status()
    if clean.error:
        raise RuntimeError(clean.error)
    return SubmitGitInfo(
        commit=clean.commit,
        dirty_before=True,
        committed_dirty=True,
        dirty_files=status.files,
        commit_message=message,
    )


def snapshot_suffix(job_name: str) -> str:
    matches = re.findall(r"\d{8}_\d{6}", job_name)
    if matches:
        return matches[-1]
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def apply_dataset_override(config_text: str, override: str | list[str]) -> str:
    """Rewrite a variant config.sh to use the requested dataset(s)."""
    if isinstance(override, str):
        new_line = f"DATASET_NAME={shlex.quote(override)}"
        return re.sub(
            r"^(\s*(?:export\s+)?)DATASET_NAME=.*$",
            lambda m: (m.group(1) or "") + new_line,
            config_text,
            flags=re.MULTILINE,
        )

    is_names_only = all("|" not in e for e in override)
    block_name = "TRAIN_DATASET_NAMES" if is_names_only else "DATASETS"
    new_block_lines = [f"{block_name}=("]
    new_block_lines.extend(f"    {shlex.quote(entry)}" for entry in override)
    new_block_lines.append(")")
    new_block = "\n".join(new_block_lines)
    pattern = rf"^{block_name}=\(.*?^\)\s*$"
    return re.sub(
        pattern,
        new_block,
        count=1,
        flags=re.MULTILINE | re.DOTALL,
    )


def _set_scalar(config_text: str, name: str, value: int | str) -> str:
    rendered = f"{name}={shlex.quote(str(value))}"
    pattern = rf"^(\s*(?:export\s+)?)({re.escape(name)})=.*$"
    if re.search(pattern, config_text, flags=re.MULTILINE):
        return re.sub(
            pattern,
            lambda m: f"{m.group(1) or ''}{rendered}",
            config_text,
            count=1,
            flags=re.MULTILINE,
        )
    suffix = "" if config_text.endswith("\n") else "\n"
    return f"{config_text}{suffix}{rendered}\n"


def _array_assignment(name: str, values: list[str]) -> str:
    lines = [f"{name}=("]
    lines.extend(f"    {shlex.quote(v)}" for v in values)
    lines.append(")")
    return "\n".join(lines)


def render_training_config_snapshot(
    *,
    base_config: str,
    variant: str,
    model: str,
    job_name: str,
    cluster: str,
    partition: str | None = None,
    node: str | None = None,
    dataset_override: str | list[str] | None = None,
    extra_args: list[str] | None = None,
    train_num_gpus: int,
    train_global_batch_size: int | None,
    train_max_steps: int,
    train_save_steps: int,
    git: SubmitGitInfo | None = None,
) -> str:
    text = base_config
    if dataset_override is not None:
        text = apply_dataset_override(text, dataset_override)

    text = _set_scalar(text, "TRAIN_NUM_GPUS", train_num_gpus)
    text = _set_scalar(text, "MAX_STEPS", train_max_steps)
    text = _set_scalar(text, "SAVE_STEPS", train_save_steps)
    if train_global_batch_size is not None:
        text = _set_scalar(text, "TRAIN_GLOBAL_BATCH_SIZE", train_global_batch_size)
        if model == "n1.5" and train_num_gpus > 0:
            text = _set_scalar(text, "TRAIN_BATCH_SIZE", train_global_batch_size // train_num_gpus)

    footer = [
        "",
        "# ---- train-eval-web submission snapshot ----",
        f"SUBMIT_JOB_NAME={shlex.quote(job_name)}",
        f"SUBMIT_VARIANT={shlex.quote(variant)}",
        f"SUBMIT_CLUSTER={shlex.quote(cluster)}",
    ]
    if partition:
        footer.append(f"SUBMIT_PARTITION={shlex.quote(partition)}")
    if node:
        footer.append(f"SUBMIT_NODE={shlex.quote(node)}")
    if git:
        if git.commit:
            footer.append(f"SUBMIT_GIT_COMMIT={shlex.quote(git.commit)}")
        footer.append(f"SUBMIT_GIT_DIRTY_AT_SUBMIT={'1' if git.dirty_before else '0'}")
        footer.append(f"SUBMIT_GIT_COMMITTED_DIRTY={'1' if git.committed_dirty else '0'}")
    if dataset_override is not None:
        footer.append(
            "SUBMIT_DATASET_OVERRIDE_JSON="
            + shlex.quote(json.dumps(dataset_override, ensure_ascii=True))
        )
    if extra_args:
        footer.append(_array_assignment("SUBMIT_EXTRA_ARGS", extra_args))
    footer.append("# -------------------------------------------")

    suffix = "" if text.endswith("\n") else "\n"
    return f"{text}{suffix}" + "\n".join(footer) + "\n"


def snapshot_metadata(
    *,
    job_name: str,
    cluster: str,
    variant: str,
    path: str,
    meta_path: str,
    phase: str = "train",
    job_id: str | None = None,
    partition: str | None = None,
    node: str | None = None,
    dataset_override: str | list[str] | None = None,
    extra_args: list[str] | None = None,
    train_num_gpus: int | None = None,
    train_global_batch_size: int | None = None,
    train_max_steps: int | None = None,
    train_save_steps: int | None = None,
    git: SubmitGitInfo | None = None,
) -> dict[str, Any]:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cluster": cluster,
        "phase": phase,
        "variant": variant,
        "job_id": job_id,
        "job_name": job_name,
        "partition": partition,
        "node": node,
        "config_snapshot_path": path,
        "config_snapshot_meta_path": meta_path,
        "train": {
            "num_gpus": train_num_gpus,
            "global_batch_size": train_global_batch_size,
            "max_steps": train_max_steps,
            "save_steps": train_save_steps,
        },
        "dataset_override": dataset_override,
        "extra_args": extra_args or [],
        "git": {
            "commit": git.commit if git else None,
            "dirty_at_submit": git.dirty_before if git else None,
            "committed_dirty": git.committed_dirty if git else None,
            "dirty_files": git.dirty_files if git else [],
            "commit_message": git.commit_message if git else None,
        },
    }


def metadata_json(meta: dict[str, Any]) -> str:
    return json.dumps(meta, indent=2, sort_keys=True) + "\n"
