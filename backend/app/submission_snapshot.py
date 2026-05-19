"""Training submission snapshots.

The variant `config.sh` is only the starting point. A real training
submission may override datasets, GPU count, batch size, step counts, and
extra scheduler/job args. This module renders the effective config record and
captures the actual model-code git revision used for the submission.
"""

from __future__ import annotations

import asyncio
import json
import re
import shlex
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from pydantic import BaseModel, Field

from .ssh import ssh_run
from .training_models import (
    TrainingModel,
    load_training_model,
    slurm_repo_path,
)


class GitStatus(BaseModel):
    repo_path: str | None = None
    repo_label: str | None = None
    commit: str | None
    short_commit: str | None
    dirty: bool
    files: list[str] = Field(default_factory=list)
    error: str | None = None


@dataclass(frozen=True)
class SubmitGitInfo:
    repo_path: str
    repo_label: str
    commit: str | None
    dirty_before: bool
    committed_dirty: bool
    dirty_files: list[str]
    commit_message: str | None = None


GitRunner = Callable[[str, float], Awaitable[tuple[int, str, str]]]


def _short(commit: str | None) -> str | None:
    return commit[:12] if commit else None


def _coerce_model(model: str | TrainingModel) -> TrainingModel:
    return model if isinstance(model, TrainingModel) else load_training_model(model)


def training_repo_var(model: str | TrainingModel) -> str:
    resolved = _coerce_model(model)
    if not resolved.slurm_repo_var:
        raise ValueError(f"model {resolved.id} missing SLURM_REPO_VAR")
    return resolved.slurm_repo_var


def training_repo_label(model: str | TrainingModel) -> str:
    return _coerce_model(model).label


def slurm_training_repo_path(
    cluster_vars: dict[str, str],
    model: str | TrainingModel,
) -> str:
    return slurm_repo_path(cluster_vars, _coerce_model(model))


async def _git_status(
    run: GitRunner,
    *,
    repo_path: str,
    repo_label: str,
) -> GitStatus:
    repo = shlex.quote(repo_path)
    rc, head, err = await run(f"cd {repo} && git rev-parse HEAD", 20.0)
    if rc != 0:
        return GitStatus(
            repo_path=repo_path,
            repo_label=repo_label,
            commit=None,
            short_commit=None,
            dirty=True,
            files=[],
            error=(err or head).strip() or "git rev-parse failed",
        )
    rc, status, err = await run(f"cd {repo} && git status --short", 20.0)
    if rc != 0:
        commit = head.strip()
        return GitStatus(
            repo_path=repo_path,
            repo_label=repo_label,
            commit=commit,
            short_commit=_short(commit),
            dirty=True,
            files=[],
            error=(err or status).strip() or "git status failed",
        )
    files = [line for line in status.splitlines() if line.strip()]
    commit = head.strip()
    return GitStatus(
        repo_path=repo_path,
        repo_label=repo_label,
        commit=commit,
        short_commit=_short(commit),
        dirty=bool(files),
        files=files,
    )


async def _prepare_training_git(
    run: GitRunner,
    *,
    repo_path: str,
    repo_label: str,
    job_name: str,
    commit_dirty_changes: bool,
    require_clean: bool = True,
) -> SubmitGitInfo:
    status = await _git_status(run, repo_path=repo_path, repo_label=repo_label)
    if status.error:
        raise RuntimeError(status.error)
    if not status.dirty:
        return SubmitGitInfo(
            repo_path=repo_path,
            repo_label=repo_label,
            commit=status.commit,
            dirty_before=False,
            committed_dirty=False,
            dirty_files=[],
        )
    if not require_clean:
        return SubmitGitInfo(
            repo_path=repo_path,
            repo_label=repo_label,
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
    repo = shlex.quote(repo_path)
    rc, _, err = await run(f"cd {repo} && git add -A", 30.0)
    if rc != 0:
        raise RuntimeError(err.strip() or "git add failed")
    rc, out, err = await run(f"cd {repo} && git commit -m {shlex.quote(message)}", 60.0)
    if rc != 0:
        raise RuntimeError((err or out).strip() or "git commit failed")
    clean = await _git_status(run, repo_path=repo_path, repo_label=repo_label)
    if clean.error:
        raise RuntimeError(clean.error)
    return SubmitGitInfo(
        repo_path=repo_path,
        repo_label=repo_label,
        commit=clean.commit,
        dirty_before=True,
        committed_dirty=True,
        dirty_files=status.files,
        commit_message=message,
    )


async def _slurm_git_run(host: str, cmd: str, timeout: float) -> tuple[int, str, str]:
    r = await ssh_run(host, cmd, timeout=timeout)
    return r.returncode, r.stdout, r.stderr


async def slurm_git_status(*, host: str, repo_path: str, repo_label: str) -> GitStatus:
    return await _git_status(
        lambda cmd, timeout: _slurm_git_run(host, cmd, timeout),
        repo_path=repo_path,
        repo_label=repo_label,
    )


async def prepare_slurm_training_git(
    *,
    host: str,
    repo_path: str,
    repo_label: str,
    job_name: str,
    commit_dirty_changes: bool,
    require_clean: bool = True,
) -> SubmitGitInfo:
    return await _prepare_training_git(
        lambda cmd, timeout: _slurm_git_run(host, cmd, timeout),
        repo_path=repo_path,
        repo_label=repo_label,
        job_name=job_name,
        commit_dirty_changes=commit_dirty_changes,
        require_clean=require_clean,
    )


async def _mlxp_git_run(cmd: str, timeout: float) -> tuple[int, str, str]:
    import shutil

    from .mlxp_config import NAMESPACE
    from .mlxp_data_pod import ensure_listing_pod

    if shutil.which("kubectl") is None:
        raise RuntimeError("kubectl not found on PATH")

    pod = await ensure_listing_pod()
    proc = await asyncio.create_subprocess_exec(
        "kubectl",
        "exec",
        "-n",
        NAMESPACE,
        pod,
        "--",
        "bash",
        "-lc",
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise RuntimeError(f"MLXP git command timed out after {timeout:g}s")
    return (
        proc.returncode or 0,
        stdout.decode(errors="replace"),
        stderr.decode(errors="replace"),
    )


async def mlxp_git_status(*, repo_path: str, repo_label: str) -> GitStatus:
    return await _git_status(
        _mlxp_git_run,
        repo_path=repo_path,
        repo_label=repo_label,
    )


async def prepare_mlxp_training_git(
    *,
    repo_path: str,
    repo_label: str,
    job_name: str,
    commit_dirty_changes: bool,
    require_clean: bool = True,
) -> SubmitGitInfo:
    return await _prepare_training_git(
        _mlxp_git_run,
        repo_path=repo_path,
        repo_label=repo_label,
        job_name=job_name,
        commit_dirty_changes=commit_dirty_changes,
        require_clean=require_clean,
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


def _set_array(config_text: str, name: str, values: list[str]) -> str:
    rendered = _array_assignment(name, values)
    pattern = rf"^{re.escape(name)}=\(.*?^\)\s*$"
    if re.search(pattern, config_text, flags=re.MULTILINE | re.DOTALL):
        return re.sub(
            pattern,
            rendered,
            count=1,
            flags=re.MULTILINE | re.DOTALL,
        )
    suffix = "" if config_text.endswith("\n") else "\n"
    return f"{config_text}{suffix}{rendered}\n"


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
        footer.append(f"SUBMIT_GIT_REPO_LABEL={shlex.quote(git.repo_label)}")
        footer.append(f"SUBMIT_GIT_REPO_PATH={shlex.quote(git.repo_path)}")
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


def render_eval_config_preview(
    *,
    base_config: str,
    variant: str,
    job_name: str,
    cluster: str,
    partition: str | None = None,
    node: str | None = None,
    dataset_override: str | list[str] | None = None,
    eval_n_episodes: int | None = None,
    eval_n_runs: int | None = None,
    eval_sets: list[str] | None = None,
    eval_overwrite_results: bool = False,
    checkpoint_path: str | None = None,
    extra_args: list[str] | None = None,
) -> str:
    text = base_config
    if dataset_override is not None:
        text = apply_dataset_override(text, dataset_override)
    if eval_n_episodes is not None:
        text = _set_scalar(text, "N_EPISODES", eval_n_episodes)
    if eval_n_runs is not None:
        text = _set_scalar(text, "N_RUNS", eval_n_runs)
    if eval_sets is not None:
        text = _set_array(text, "EVAL_SETS", eval_sets)

    footer = [
        "",
        "# ---- train-eval-web eval submission preview ----",
        f"SUBMIT_JOB_NAME={shlex.quote(job_name)}",
        f"SUBMIT_VARIANT={shlex.quote(variant)}",
        f"SUBMIT_CLUSTER={shlex.quote(cluster)}",
    ]
    if partition:
        footer.append(f"SUBMIT_PARTITION={shlex.quote(partition)}")
    if node:
        footer.append(f"SUBMIT_NODE={shlex.quote(node)}")
    if checkpoint_path:
        footer.append(f"SUBMIT_EVAL_CHECKPOINT={shlex.quote(checkpoint_path)}")
    if eval_overwrite_results:
        footer.append("SUBMIT_EVAL_OVERWRITE_RESULTS=1")
    if extra_args:
        footer.append(_array_assignment("SUBMIT_EXTRA_ARGS", extra_args))
    footer.append("# -------------------------------------------------")

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
            "repo_path": git.repo_path if git else None,
            "repo_label": git.repo_label if git else None,
            "commit": git.commit if git else None,
            "dirty_at_submit": git.dirty_before if git else None,
            "committed_dirty": git.committed_dirty if git else None,
            "dirty_files": git.dirty_files if git else [],
            "commit_message": git.commit_message if git else None,
        },
    }


def metadata_json(meta: dict[str, Any]) -> str:
    return json.dumps(meta, indent=2, sort_keys=True) + "\n"
