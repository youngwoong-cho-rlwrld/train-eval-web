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
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from pydantic import BaseModel, Field

from .kubectl_errors import is_kubectl_exec_transport_error
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
    branch: str | None = None
    dirty: bool
    files: list[str] = Field(default_factory=list)
    error: str | None = None


@dataclass(frozen=True)
class SubmitGitInfo:
    repo_path: str
    repo_label: str
    commit: str | None
    branch: str | None
    dirty_before: bool
    committed_dirty: bool
    dirty_files: list[str]
    commit_message: str | None = None


GitRunner = Callable[[str, float], Awaitable[tuple[int, str, str]]]


def _short(commit: str | None) -> str | None:
    return commit[:12] if commit else None


def _clean_branch(branch: str | None) -> str | None:
    branch = (branch or "").strip()
    return branch or None


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


def _safe_git(repo_path: str) -> str:
    return f"git -c {shlex.quote(f'safe.directory={repo_path}')}"


async def _git_status(
    run: GitRunner,
    *,
    repo_path: str,
    repo_label: str,
) -> GitStatus:
    repo = shlex.quote(repo_path)
    git = _safe_git(repo_path)
    rc, head, err = await run(f"cd {repo} && {git} rev-parse HEAD", 20.0)
    if rc != 0:
        return GitStatus(
            repo_path=repo_path,
            repo_label=repo_label,
            commit=None,
            short_commit=None,
            branch=None,
            dirty=True,
            files=[],
            error=(err or head).strip() or "git rev-parse failed",
        )
    rc, branch, _ = await run(f"cd {repo} && {git} branch --show-current", 20.0)
    branch = branch if rc == 0 else ""
    rc, status, err = await run(f"cd {repo} && {git} status --short", 20.0)
    if rc != 0:
        commit = head.strip()
        return GitStatus(
            repo_path=repo_path,
            repo_label=repo_label,
            commit=commit,
            short_commit=_short(commit),
            branch=_clean_branch(branch),
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
        branch=_clean_branch(branch),
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
            branch=status.branch,
            dirty_before=False,
            committed_dirty=False,
            dirty_files=[],
        )
    if not require_clean:
        return SubmitGitInfo(
            repo_path=repo_path,
            repo_label=repo_label,
            commit=status.commit,
            branch=status.branch,
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
    git = _safe_git(repo_path)
    rc, _, err = await run(f"cd {repo} && {git} add -A", 30.0)
    if rc != 0:
        raise RuntimeError(err.strip() or "git add failed")
    rc, out, err = await run(f"cd {repo} && {git} commit -m {shlex.quote(message)}", 60.0)
    if rc != 0:
        raise RuntimeError((err or out).strip() or "git commit failed")
    clean = await _git_status(run, repo_path=repo_path, repo_label=repo_label)
    if clean.error:
        raise RuntimeError(clean.error)
    return SubmitGitInfo(
        repo_path=repo_path,
        repo_label=repo_label,
        commit=clean.commit,
        branch=clean.branch,
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

    from .mlxp_config import get_settings
    from .mlxp_data_pod import ensure_listing_pod

    if shutil.which("kubectl") is None:
        raise RuntimeError("kubectl not found on PATH")
    settings = get_settings()

    try:
        pod = await ensure_listing_pod()
        proc = await asyncio.create_subprocess_exec(
            "kubectl",
            "exec",
            "-n",
            settings.namespace,
            pod,
            "--",
            "bash",
            "-lc",
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return await _mlxp_git_run_with_pod(cmd, timeout)
    except RuntimeError as e:
        if is_kubectl_exec_transport_error(str(e)):
            return await _mlxp_git_run_with_pod(cmd, timeout)
        raise

    out = stdout.decode(errors="replace")
    err = stderr.decode(errors="replace")
    if (proc.returncode or 0) != 0 and is_kubectl_exec_transport_error(err or out):
        return await _mlxp_git_run_with_pod(cmd, timeout)
    return proc.returncode or 0, out, err


def is_mlxp_transport_error(message: str) -> bool:
    return is_kubectl_exec_transport_error(message)


async def _mlxp_git_run_with_pod(cmd: str, timeout: float) -> tuple[int, str, str]:
    """Run git on MLXP DDN without kubectl exec.

    MLXP's exec path occasionally fails at the apiserver/kubelet proxy layer
    while pod creation and logs still work. A short-lived pod gives the submit
    preflight the same filesystem view without depending on exec.
    """
    from .mlxp_config import get_settings

    settings = get_settings()
    pod_name = f"tew-git-{uuid.uuid4().hex[:10]}"
    marker = f"__TRAIN_EVAL_WEB_RC_{uuid.uuid4().hex}__"
    wrapped_cmd = (
        "set +e\n"
        f"{cmd} >/tmp/tew-git.out 2>/tmp/tew-git.err\n"
        "rc=$?\n"
        "TEW_RC=\"$rc\" python3 - <<'PY'\n"
        "import json, os\n"
        "\n"
        "LIMIT = 3200\n"
        "\n"
        "def read_text(path):\n"
        "    try:\n"
        "        with open(path, 'r', encoding='utf-8', errors='replace') as f:\n"
        "            return f.read()\n"
        "    except FileNotFoundError:\n"
        "        return ''\n"
        "\n"
        "stdout = read_text('/tmp/tew-git.out')\n"
        "stderr = read_text('/tmp/tew-git.err')\n"
        "payload = {\n"
        "    'rc': int(os.environ.get('TEW_RC') or '1'),\n"
        "    'stdout': stdout,\n"
        "    'stderr': stderr,\n"
        "    'truncated': False,\n"
        "}\n"
        "\n"
        "def encode(data):\n"
        "    return json.dumps(data, ensure_ascii=True, separators=(',', ':'))\n"
        "\n"
        "encoded = encode(payload)\n"
        "if len(encoded) > LIMIT:\n"
        "    payload['truncated'] = True\n"
        "    budget = max(256, (LIMIT - 200) // 2)\n"
        "    payload['stdout'] = stdout[:budget]\n"
        "    payload['stderr'] = stderr[:budget]\n"
        "    encoded = encode(payload)\n"
        "\n"
        "with open('/dev/termination-log', 'w', encoding='utf-8') as f:\n"
        "    f.write(encoded[:LIMIT])\n"
        "PY\n"
        "exit \"$rc\"\n"
    )
    spec = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": pod_name,
            "namespace": settings.namespace,
            "annotations": {
                "mlx.navercorp.com/zone": settings.zone,
                "sidecar.istio.io/inject": "false",
            },
            "labels": {
                "owner": settings.owner_label,
                "tool": settings.tool_label,
            },
        },
        "spec": {
            "restartPolicy": "Never",
            "imagePullSecrets": [{"name": settings.image_pull_secret}],
            "volumes": [
                {
                    "name": "ddn",
                    "persistentVolumeClaim": {"claimName": settings.ddn_pvc},
                }
            ],
            "affinity": {
                "nodeAffinity": {
                    "requiredDuringSchedulingIgnoredDuringExecution": {
                        "nodeSelectorTerms": [
                            {
                                "matchExpressions": [
                                    {
                                        "key": "kubernetes.io/hostname",
                                        "operator": "In",
                                        "values": [settings.default_node],
                                    }
                                ]
                            }
                        ]
                    }
                }
            },
            "containers": [
                {
                    "name": "main",
                    "image": settings.image,
                    "command": ["bash", "-lc"],
                    "args": [wrapped_cmd],
                    "terminationMessagePath": "/dev/termination-log",
                    "terminationMessagePolicy": "File",
                    "env": [{"name": "NVIDIA_VISIBLE_DEVICES", "value": "none"}],
                    "resources": {
                        # Keep requests tiny so this can schedule even when
                        # all GPUs are occupied by training jobs on the node.
                        "requests": {"cpu": "10m", "memory": "128Mi"},
                        "limits": {"cpu": "1", "memory": "512Mi"},
                    },
                    "volumeMounts": [{"name": "ddn", "mountPath": settings.ddn_mount}],
                }
            ],
        },
    }

    async def kubectl(*args: str, stdin: bytes | None = None, deadline: float = 30.0) -> tuple[int, str, str]:
        proc = await asyncio.create_subprocess_exec(
            "kubectl",
            *args,
            stdin=asyncio.subprocess.PIPE if stdin is not None else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(input=stdin), timeout=deadline)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return 124, "", f"kubectl {' '.join(args)} timed out after {deadline:g}s"
        return (
            proc.returncode or 0,
            stdout.decode(errors="replace"),
            stderr.decode(errors="replace"),
        )

    try:
        create_rc = 1
        create_err = ""
        for attempt in range(4):
            create_rc, _, create_err = await kubectl(
                "create",
                "-f",
                "-",
                "--validate=false",
                "-n",
                settings.namespace,
                stdin=json.dumps(spec).encode(),
                deadline=30.0,
            )
            if create_rc == 0:
                break
            if not is_kubectl_exec_transport_error(create_err) or attempt == 3:
                return create_rc, "", create_err.strip() or "kubectl create git pod failed"
            await asyncio.sleep(1.5 * (attempt + 1))

        deadline_at = asyncio.get_event_loop().time() + timeout
        phase = ""
        exit_code: int | None = None
        termination_message = ""
        while asyncio.get_event_loop().time() < deadline_at:
            rc, out, err = await kubectl(
                "get", "pod", pod_name, "-n", settings.namespace, "-o", "json", deadline=15.0
            )
            if rc != 0:
                return rc, "", err.strip() or out.strip() or "kubectl get git pod failed"
            try:
                pod = json.loads(out)
            except json.JSONDecodeError:
                return 1, "", "kubectl get git pod returned invalid JSON"
            phase = (pod.get("status") or {}).get("phase") or ""
            statuses = (pod.get("status") or {}).get("containerStatuses") or []
            if statuses:
                state = statuses[0].get("state") or {}
                terminated = state.get("terminated")
                if terminated:
                    exit_code = terminated.get("exitCode")
                    termination_message = terminated.get("message") or ""
            if phase in ("Succeeded", "Failed") or exit_code is not None:
                break
            await asyncio.sleep(0.5)

        if not phase:
            return 124, "", f"MLXP git pod {pod_name} did not start"
        if phase not in ("Succeeded", "Failed") and exit_code is None:
            return 124, "", f"MLXP git pod {pod_name} timed out after {timeout:g}s"

        if termination_message:
            parsed = _parse_mlxp_git_termination_message(termination_message)
            if parsed is not None:
                return parsed

        logs_rc, logs, logs_err = await kubectl(
            "logs", pod_name, "-n", settings.namespace, deadline=30.0
        )
        if logs_rc != 0:
            return logs_rc, "", logs_err.strip() or "kubectl logs git pod failed"
        parsed_rc = exit_code if exit_code is not None else (0 if phase == "Succeeded" else 1)
        output = logs
        marker_match = re.search(rf"\n?{re.escape(marker)}(\d+)\s*$", output)
        if marker_match:
            parsed_rc = int(marker_match.group(1))
            output = output[: marker_match.start()].rstrip("\n")
            if output:
                output += "\n"
        return parsed_rc, output, ""
    finally:
        await kubectl(
            "delete", "pod", pod_name, "-n", settings.namespace, "--wait=false", deadline=15.0
        )


def _parse_mlxp_git_termination_message(message: str) -> tuple[int, str, str] | None:
    try:
        payload = json.loads(message)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    rc = payload.get("rc")
    if not isinstance(rc, int):
        return None
    stdout = payload.get("stdout")
    stderr = payload.get("stderr")
    if not isinstance(stdout, str) or not isinstance(stderr, str):
        return None
    if payload.get("truncated"):
        suffix = "\n[train-eval-web: output truncated from MLXP git helper]\n"
        if stderr:
            stderr = stderr.rstrip("\n") + suffix
        else:
            stdout = stdout.rstrip("\n") + suffix
    return rc, stdout, stderr


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
    updated = re.sub(
        pattern,
        new_block,
        config_text,
        count=1,
        flags=re.MULTILINE | re.DOTALL,
    )
    if updated == config_text:
        raise ValueError(f"config has no {block_name} array to override")
    return updated


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


def shell_array_assignment(name: str, values: list[str]) -> str:
    lines = [f"{name}=("]
    lines.extend(f"    {shlex.quote(v)}" for v in values)
    lines.append(")")
    return "\n".join(lines)


def _set_array(config_text: str, name: str, values: list[str]) -> str:
    rendered = shell_array_assignment(name, values)
    pattern = rf"^{re.escape(name)}=\(.*?^\)\s*$"
    if re.search(pattern, config_text, flags=re.MULTILINE | re.DOTALL):
        return re.sub(
            pattern,
            rendered,
            config_text,
            count=1,
            flags=re.MULTILINE | re.DOTALL,
        )
    suffix = "" if config_text.endswith("\n") else "\n"
    return f"{config_text}{suffix}{rendered}\n"


def _apply_submission_config_overrides(
    config_text: str,
    *,
    dataset_override: str | list[str] | None = None,
    train_note: str | None = None,
    data_dir: str | None = None,
) -> str:
    text = config_text
    if dataset_override is not None:
        text = apply_dataset_override(text, dataset_override)
    if train_note is not None:
        text = _set_scalar(text, "TRAIN_NOTE", train_note)
    if data_dir:
        text = _set_scalar(text, "DATA_DIR", data_dir)
    return text


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
    train_action_horizon: int | None = None,
    train_note: str | None = None,
    wandb_project: str | None = None,
    git: SubmitGitInfo | None = None,
) -> str:
    text = _apply_submission_config_overrides(
        base_config,
        dataset_override=dataset_override,
        train_note=train_note,
    )

    text = _set_scalar(text, "TRAIN_NUM_GPUS", train_num_gpus)
    text = _set_scalar(text, "MAX_STEPS", train_max_steps)
    text = _set_scalar(text, "SAVE_STEPS", train_save_steps)
    if train_action_horizon is not None:
        text = _set_scalar(text, "TRAIN_ACTION_HORIZON", train_action_horizon)
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
    if wandb_project:
        footer.append(f"SUBMIT_WANDB_PROJECT={shlex.quote(wandb_project)}")
    if partition:
        footer.append(f"SUBMIT_PARTITION={shlex.quote(partition)}")
    if node:
        footer.append(f"SUBMIT_NODE={shlex.quote(node)}")
    if train_action_horizon is not None:
        footer.append(f"SUBMIT_TRAIN_ACTION_HORIZON={train_action_horizon}")
    if git:
        footer.append(f"SUBMIT_GIT_REPO_LABEL={shlex.quote(git.repo_label)}")
        footer.append(f"SUBMIT_GIT_REPO_PATH={shlex.quote(git.repo_path)}")
        if git.branch:
            footer.append(f"SUBMIT_GIT_BRANCH={shlex.quote(git.branch)}")
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
        footer.append(shell_array_assignment("SUBMIT_EXTRA_ARGS", extra_args))
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
    data_dir: str | None = None,
    train_note: str | None = None,
) -> str:
    text = _apply_submission_config_overrides(
        base_config,
        dataset_override=dataset_override,
        train_note=train_note,
        data_dir=data_dir,
    )
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
        footer.append(shell_array_assignment("SUBMIT_EXTRA_ARGS", extra_args))
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
    train_action_horizon: int | None = None,
    train_note: str | None = None,
    wandb_project: str | None = None,
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
        "train_note": train_note,
        "train": {
            "num_gpus": train_num_gpus,
            "global_batch_size": train_global_batch_size,
            "max_steps": train_max_steps,
            "save_steps": train_save_steps,
            "action_horizon": train_action_horizon,
            "wandb_project": wandb_project,
        },
        "dataset_override": dataset_override,
        "extra_args": extra_args or [],
        "git": {
            "repo_path": git.repo_path if git else None,
            "repo_label": git.repo_label if git else None,
            "branch": git.branch if git else None,
            "commit": git.commit if git else None,
            "dirty_at_submit": git.dirty_before if git else None,
            "committed_dirty": git.committed_dirty if git else None,
            "dirty_files": git.dirty_files if git else [],
            "commit_message": git.commit_message if git else None,
        },
    }


def metadata_json(meta: dict[str, Any]) -> str:
    return json.dumps(meta, indent=2, sort_keys=True) + "\n"
