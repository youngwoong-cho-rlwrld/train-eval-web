"""Per-job extended details: phase, paths, wandb url, progress.

Parses metadata out of the job_name (shape `{train|resume|eval}_{variant}_{YYYYMMDD}_{HHMMSS}`,
identical across slurm and mlxp), reads variant config locally, and asks
the cluster a few small questions over SSH (slurm) or kubectl (mlxp) to
compute progress.
"""

import os
import re
import shlex
from typing import Any

from pydantic import BaseModel, Field

from .clusters import load_cluster
from .eval_completion import eval_job_completed
from .job_identity import (
    parse_comment_metadata,
    parse_phase_and_variant,
    phase_variant_from_meta,
    resolve_phase_and_variant,
)
from .jobs import get_job
from .paths import CLUSTER_STAGING_REL
from .ssh import ssh_run
from .variants import load_variant


from .wandb_config import get_project

# Wandb config:
#   - run id: WANDB_RUN_ID pinned by submit.py (slurm) / body script
#     (mlxp) to job_name. Already in hand here.
#   - entity: wandb.Api().default_entity after `wandb login` on this
#     laptop. Resolved lazily in _wandb_entity.
#   - project: configurable in Settings (persisted via wandb_config),
#     since launch_finetune.py / gr00t_finetune.py override our exported
#     WANDB_PROJECT internally — no submission-side signal reveals which
#     project the run actually lands in.
WANDB_ENTITY_OVERRIDE = os.environ.get("TRAIN_EVAL_WEB_WANDB_ENTITY")


class Paths(BaseModel):
    stdout: str
    stderr: str
    exp_dir: str
    ckpt_dir: str | None = None
    eval_checkpoint: str | None = None
    eval_dir: str | None = None
    isaac_logs_glob: str | None = None


class Progress(BaseModel):
    phase: str
    # train: current step / max steps
    current_step: int | None = None
    max_steps: int | None = None
    # eval: completed runs / total runs
    completed_runs: int | None = None
    total_runs: int | None = None
    current_label: str | None = None     # e.g. "0cm / run 2/3"
    percent: float | None = None         # 0..100


class GpuDeviceUsage(BaseModel):
    index: int
    name: str | None = None
    utilization_gpu_percent: int | None = None
    used_gb: float
    total_gb: float
    used_mib: int
    total_mib: int


class GpuUsage(BaseModel):
    node: str | None = None
    utilization_gpu_percent: float | None = None
    used_gb: float | None = None
    total_gb: float | None = None
    devices: list[GpuDeviceUsage] = Field(default_factory=list)
    error: str | None = None


class JobDetails(BaseModel):
    cluster: str
    job_id: str
    job_name: str
    phase: str            # "train" | "resume" | "eval" | "unknown"
    variant: str | None
    state: str
    elapsed: str
    wandb_url: str | None
    paths: Paths
    progress: Progress
    gpu: GpuUsage | None = None


async def _read_slurm_meta(host: str, job_id: str) -> dict[str, str]:
    """Read the sidecar written at submit time.

    `~/.train-eval-web/jobs/<job_id>.meta` shape: lines `key=value`."""
    r = await ssh_run(
        host,
        f"cat $HOME/.train-eval-web/jobs/{job_id}.meta 2>/dev/null",
        timeout=10.0,
    )
    if r.returncode != 0 or not r.stdout.strip():
        return {}
    fields: dict[str, str] = {}
    for line in r.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            fields[k.strip()] = v.strip()
    return fields


async def _read_slurm_scontrol_comment(host: str, job_id: str) -> str | None:
    """Pull the Comment field out of `scontrol show job` (the live
    controller's view — sacct doesn't archive Comment on this cluster)."""
    r = await ssh_run(
        host,
        f"scontrol show job {job_id} 2>/dev/null | tr ' ' '\\n' | grep -m1 '^Comment='",
        timeout=10.0,
    )
    if r.returncode != 0:
        return None
    line = r.stdout.strip()
    if not line.startswith("Comment="):
        return None
    return line[len("Comment="):]


async def _resolve_slurm_log_paths(
    host: str,
    log_dir: str,
    job_name: str,
    job_id: str,
) -> tuple[str, str, str | None]:
    """Return stdout/stderr paths and the job-name portion from real logs.

    Historical jobs may not have sidecar metadata, and Slurm accounting can be
    less reliable than the actual log filenames. The logs all end in
    `_<job_id>.out|err`, regardless of the job-name convention.
    """
    default_stdout = f"{log_dir}/{job_name}_{job_id}.out"
    default_stderr = f"{log_dir}/{job_name}_{job_id}.err"
    r = await ssh_run(
        host,
        f"ls -1 {shlex.quote(log_dir)}/*_{shlex.quote(job_id)}.out 2>/dev/null | head -1",
        timeout=10.0,
    )
    stdout = r.stdout.strip().splitlines()[0] if r.stdout.strip() else default_stdout
    if not stdout.endswith(f"_{job_id}.out"):
        return default_stdout, default_stderr, None

    stderr = f"{stdout[:-4]}.err"
    leaf = stdout.rsplit("/", 1)[-1]
    log_job_name = leaf[: -len(f"_{job_id}.out")]
    return stdout, stderr, log_job_name or None


async def _resolve_runtime_exp_dir(
    host: str,
    stdout_path: str,
    phase: str,
) -> str | None:
    """Recover the experiment directory the job actually used from stdout."""
    if phase in ("train", "resume"):
        patterns = r"Output:[[:space:]]+|output_dir:[[:space:]]+|--output-dir[[:space:]]+"
    elif phase == "eval":
        patterns = (
            r"DONE[[:space:]]+|Saved to .*/results\.json|"
            r"Running eval -> |SKIP .*results\.json already exists"
        )
    else:
        return None

    r = await ssh_run(
        host,
        f"grep -E {shlex.quote(patterns)} {shlex.quote(stdout_path)} 2>/dev/null | tail -20",
        timeout=10.0,
    )
    for line in reversed(r.stdout.splitlines()):
        if phase in ("train", "resume"):
            m = re.search(r"Output:\s*(\S+)|output_dir:\s*(\S+)|--output-dir\s+(\S+)", line)
            if not m:
                continue
            ckpt_dir = next((g for g in m.groups() if g), "").strip("'\"")
            if "/checkpoints" in ckpt_dir:
                return ckpt_dir.split("/checkpoints", 1)[0]
        else:
            m = re.search(r"(?:DONE\s+|Saved to )(\S+)/results\.json", line)
            if m:
                return m.group(1)
            m = re.search(r"(/\S+?)/eval_results(?:/|\s|$)", line)
            if m:
                return m.group(1)
    return None


async def get_details(cluster: str, job_id: str, include_gpu: bool = False) -> JobDetails:
    sacct = await get_job(cluster, job_id)
    job_name = sacct.get("JobName", "")
    state = sacct.get("State", "")
    elapsed = sacct.get("Elapsed", "")

    phase, variant = resolve_phase_and_variant(job_name, sacct)

    if cluster == "mlxp":
        return await _mlxp_details(job_id, job_name, state, elapsed, phase, variant)

    env = await load_cluster(cluster)
    slurm_meta: dict[str, str] = {}

    # Slurm: if sacct didn't return Comment (slurmdbd doesn't archive it on
    # this cluster), check scontrol (works for jobs still in the
    # controller) and then the on-disk .meta sidecar (permanent).
    if not variant:
        scontrol_comment = await _read_slurm_scontrol_comment(env.ssh_alias, job_id)
        if scontrol_comment:
            p, v = parse_comment_metadata(scontrol_comment)
            if p and v:
                phase, variant = p, v
    if not variant:
        slurm_meta = await _read_slurm_meta(env.ssh_alias, job_id)
        p, v = phase_variant_from_meta(slurm_meta)
        if p and v:
            phase, variant = p, v
    elif phase == "eval":
        # Eval details may need checkpoint_path even when phase/variant were
        # already recovered from the job name or sacct comment.
        slurm_meta = await _read_slurm_meta(env.ssh_alias, job_id)
    log_dir = env.vars["LOG_DIR"]
    stdout_path, stderr_path, log_job_name = await _resolve_slurm_log_paths(
        env.ssh_alias,
        log_dir,
        job_name,
        job_id,
    )
    if not variant and log_job_name:
        p, v = parse_phase_and_variant(log_job_name)
        if p != "unknown" and v:
            phase, variant = p, v

    # The per-variant experiment dir on the cluster depends on who submitted:
    # web-submitted jobs use ~/.train-eval-web/experiments/<variant>; jobs
    # launched via the bash `./submit` use ~/train-eval-scripts/experiments/<variant>.
    # Probe both, prefer one that actually exists.
    exp_dir_remote = await _resolve_exp_dir(env.ssh_alias, job_id, variant) if variant else f"$HOME/{CLUSTER_STAGING_REL}/experiments"
    runtime_exp_dir = await _resolve_runtime_exp_dir(env.ssh_alias, stdout_path, phase)
    if runtime_exp_dir:
        exp_dir_remote = runtime_exp_dir
    ckpt_dir = f"{exp_dir_remote}/checkpoints" if phase in ("train", "resume") else None
    eval_dir = f"{exp_dir_remote}/eval_results" if phase == "eval" else None
    eval_checkpoint = (
        await _resolve_eval_checkpoint(
            env.ssh_alias,
            stdout_path,
            exp_dir_remote,
            slurm_meta.get("checkpoint_path"),
        )
        if phase == "eval" and variant
        else None
    )
    isaac_logs_glob = f"{exp_dir_remote}/logs/{job_id}/server_*.log" if phase == "eval" else None
    paths = Paths(
        stdout=stdout_path,
        stderr=stderr_path,
        exp_dir=exp_dir_remote,
        ckpt_dir=ckpt_dir,
        eval_checkpoint=eval_checkpoint,
        eval_dir=eval_dir,
        isaac_logs_glob=isaac_logs_glob,
    )

    wandb_url: str | None = None
    if phase in ("train", "resume"):
        # submit.py pins WANDB_RUN_ID = job_name via sbatch --export.
        entity = await _wandb_entity()
        if entity:
            wandb_url = f"https://wandb.ai/{entity}/{get_project()}/runs/{job_name}"

    progress = await _compute_progress(cluster, job_id, phase, variant, stdout_path, stderr_path, ckpt_dir, eval_dir)
    if phase == "eval" and variant and eval_dir:
        try:
            if await eval_job_completed(env.ssh_alias, stdout_path, eval_dir, variant):
                state = "COMPLETED"
        except Exception:
            pass
    gpu = (
        await _slurm_gpu_usage(
            env.ssh_alias,
            job_id,
            sacct.get("NodeList") or "",
            state,
        )
        if include_gpu
        else None
    )

    return JobDetails(
        cluster=cluster, job_id=job_id, job_name=job_name,
        phase=phase, variant=variant, state=state, elapsed=elapsed,
        wandb_url=wandb_url, paths=paths, progress=progress, gpu=gpu,
    )


async def _mlxp_details(job_id: str, job_name: str, state: str, elapsed: str,
                        phase: str, variant: str | None) -> JobDetails:
    """MLXP runs train via `kubectl apply` on a pod. All paths live on DDN."""
    exp_dir = f"/data/youngwoong/experiments/{variant}" if variant else "/data/youngwoong/experiments"
    ckpt_dir = f"{exp_dir}/checkpoints" if phase in ("train", "resume") else None
    paths = Paths(
        stdout=f"kubectl logs -n p-rlwrld -l job-name={job_id}",
        stderr=f"kubectl logs -n p-rlwrld -l job-name={job_id}  (k8s merges stdout+stderr)",
        exp_dir=exp_dir,
        ckpt_dir=ckpt_dir,
        eval_dir=None,
        isaac_logs_glob=None,
    )
    # mlxp body pins WANDB_RUN_ID = job_name (resolved from the Job's
    # display-name annotation by get_job). The k8s job_id has no wandb
    # run behind it.
    entity = await _wandb_entity()
    wandb_url = (
        f"https://wandb.ai/{entity}/{get_project()}/runs/{job_name}"
        if entity else None
    )

    progress = await _mlxp_progress(job_name, variant, phase)

    return JobDetails(
        cluster="mlxp", job_id=job_id, job_name=job_name,
        phase=phase, variant=variant, state=state, elapsed=elapsed,
        wandb_url=wandb_url, paths=paths, progress=progress,
    )


def _parse_gpu_usage(stdout: str, node: str | None) -> GpuUsage | None:
    devices: list[GpuDeviceUsage] = []
    for line in stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            index = int(parts[0])
            if len(parts) >= 5:
                name = parts[1] or None
                utilization = int(parts[2])
                used_mib = int(parts[3])
                total_mib = int(parts[4])
            elif len(parts) >= 4:
                name = None
                utilization = int(parts[1])
                used_mib = int(parts[2])
                total_mib = int(parts[3])
            else:
                name = None
                utilization = None
                used_mib = int(parts[1])
                total_mib = int(parts[2])
        except ValueError:
            continue
        devices.append(
            GpuDeviceUsage(
                index=index,
                name=name,
                utilization_gpu_percent=utilization,
                used_mib=used_mib,
                total_mib=total_mib,
                used_gb=round(used_mib / 1024, 1),
                total_gb=round(total_mib / 1024, 1),
            )
        )
    if not devices:
        return None
    used_mib_total = sum(d.used_mib for d in devices)
    total_mib_total = sum(d.total_mib for d in devices)
    utilization_values = [
        d.utilization_gpu_percent
        for d in devices
        if d.utilization_gpu_percent is not None
    ]
    return GpuUsage(
        node=node,
        utilization_gpu_percent=(
            round(sum(utilization_values) / len(utilization_values), 1)
            if utilization_values else None
        ),
        used_gb=round(used_mib_total / 1024, 1),
        total_gb=round(total_mib_total / 1024, 1),
        devices=devices,
    )


async def _first_slurm_node(host: str, nodelist: str) -> str | None:
    node_expr = nodelist.strip()
    if not node_expr or node_expr in {"None assigned", "N/A", "(None)"}:
        return None
    r = await ssh_run(
        host,
        f"PATH=/opt/slurm/bin:$PATH scontrol show hostnames {shlex.quote(node_expr)} 2>/dev/null | head -1",
        timeout=8.0,
    )
    node = r.stdout.strip().splitlines()[0] if r.stdout.strip() else ""
    return node or node_expr


async def _slurm_gpu_usage(
    host: str,
    job_id: str,
    nodelist: str,
    state: str,
) -> GpuUsage:
    node_expr = nodelist.strip()
    fallback_node = None if node_expr in {"", "None assigned", "N/A", "(None)"} else node_expr
    if not state.upper().startswith(("RUNNING", "COMPLETING")):
        return GpuUsage(node=fallback_node, error="GPU memory is only available while the job is running")

    node = await _first_slurm_node(host, nodelist)

    query = (
        "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total "
        "--format=csv,noheader,nounits"
    )

    # Preferred: execute inside the job allocation. This works even when
    # compute-node SSH is not available directly from the login node.
    srun_cmd = (
        f"PATH=/opt/slurm/bin:$PATH timeout 8 "
        f"srun --jobid={shlex.quote(job_id)} --overlap --ntasks=1 "
        f"bash -lc {shlex.quote(query)} 2>&1"
    )
    r = await ssh_run(host, srun_cmd, timeout=10.0)
    usage = _parse_gpu_usage(r.stdout, node)
    if usage:
        return usage

    # Fallback for clusters where login -> compute SSH is permitted.
    if node:
        ssh_cmd = (
            "timeout 8 ssh -o BatchMode=yes -o ConnectTimeout=5 "
            "-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "
            f"{shlex.quote(node)} {shlex.quote(query)} 2>&1"
        )
        r = await ssh_run(host, ssh_cmd, timeout=10.0)
        usage = _parse_gpu_usage(r.stdout, node)
        if usage:
            return usage

    err = (r.stderr or r.stdout or "").strip().splitlines()
    message = err[-1] if err else "GPU memory is unavailable"
    return GpuUsage(node=node, error=message[:240])


async def _mlxp_progress(run_id: str, variant: str | None, phase: str) -> Progress:
    """Progress for an MLXP training job.

    Primary source: the run's wandb summary (its `_step` field is updated
    every logging tick — i.e. every 10 training steps for gr00t-n16).
    Fallback: highest `checkpoint-N` dir on DDN (SAVE_STEPS granularity).

    `run_id` is the wandb run id — the job_name (display name) for MLXP,
    not the k8s job_id which has no wandb run behind it.
    """
    import asyncio

    progress = Progress(phase=phase)
    if not variant:
        return progress

    try:
        v = await load_variant(variant)
        if "MAX_STEPS" in v.vars:
            progress.max_steps = int(v.vars["MAX_STEPS"])
    except Exception:
        pass

    if phase not in ("train", "resume"):
        return progress

    # 1. wandb — fine-grained (per logging tick).
    step = await _wandb_step(run_id)
    if step is not None:
        progress.current_step = step
        if progress.max_steps:
            progress.percent = round(100.0 * step / progress.max_steps, 1)
            progress.current_label = f"step {step:,}/{progress.max_steps:,}"
        else:
            progress.current_label = f"step {step:,}"
        return progress

    # 2. checkpoint dir count — coarse (SAVE_STEPS granularity).
    import shutil
    if shutil.which("kubectl") is None:
        return progress
    ckpt_dir = f"/data/youngwoong/experiments/{variant}/checkpoints"
    from .mlxp_data_pod import ensure_listing_pod, NAMESPACE
    try:
        pod = await ensure_listing_pod()
    except Exception:
        return progress

    cmd = (
        f"ls -d {ckpt_dir}/checkpoint-* 2>/dev/null "
        "| sed 's:.*checkpoint-::' | sort -n | tail -1"
    )
    proc = None
    try:
        proc = await asyncio.create_subprocess_exec(
            "kubectl", "exec", "-n", NAMESPACE, pod, "--", "bash", "-c", cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=15.0)
    except Exception:
        if proc is not None and proc.returncode is None:
            proc.kill()
            await proc.wait()
        return progress

    latest = stdout.decode(errors="replace").strip()
    if not latest.isdigit():
        return progress
    cur = int(latest)
    progress.current_step = cur
    if progress.max_steps:
        progress.percent = round(100.0 * cur / progress.max_steps, 1)
        progress.current_label = f"step {cur:,}/{progress.max_steps:,}"
    else:
        progress.current_label = f"step {cur:,}"
    return progress


_wandb_entity_cache: str | None = None


async def _wandb_entity() -> str | None:
    """Resolve the wandb entity once. Order: env override → API default."""
    global _wandb_entity_cache
    if WANDB_ENTITY_OVERRIDE:
        return WANDB_ENTITY_OVERRIDE
    if _wandb_entity_cache is not None:
        return _wandb_entity_cache or None
    import asyncio

    def _default() -> str:
        try:
            import wandb
            return wandb.Api(timeout=10).default_entity or ""
        except Exception:
            return ""

    _wandb_entity_cache = await asyncio.to_thread(_default)
    return _wandb_entity_cache or None


async def _wandb_step(run_id: str) -> int | None:
    """Return the run's latest step (None if API unreachable / run not found)."""
    import asyncio

    entity = await _wandb_entity()
    if not entity:
        return None

    def _query() -> int | None:
        try:
            import wandb
            api = wandb.Api(timeout=10)
            run = api.run(f"{entity}/{get_project()}/{run_id}")
            # train/global_step is the actual training-loop step. wandb's
            # built-in `_step` counts wandb.log() calls, which is
            # global_step / logging_steps — off by ~10× for gr00t-n16.
            s = run.summary.get("train/global_step")
            if s is None:
                s = run.summary.get("global_step")
            if s is None:
                s = run.summary.get("_step")
            return int(s) if s is not None else None
        except Exception:
            return None

    return await asyncio.to_thread(_query)


_BODY_SUFFIX_RE = re.compile(r"/lib/(train|eval)_body(?:_n16)?\.sh$")


def _exp_dir_rel_candidates(variant: str) -> list[str]:
    """Experiment roots relative to $HOME, in preferred fallback order."""
    return [
        f"{CLUSTER_STAGING_REL}/experiments/{variant}",
        f"train-eval-scripts/experiments/{variant}",
    ]


def _latest_logs_dir_script(variant: str) -> str:
    """Shell snippet that prints the candidate exp dir with newest logs/.

    Keep this as a loop instead of a shell pipeline built with `;`: without
    grouping, only the last command is piped, which can return
    "<mtime> <path>" instead of just the path when the first candidate wins.
    """
    rels = " ".join(shlex.quote(rel) for rel in _exp_dir_rel_candidates(variant))
    return (
        "best_mtime=-1; best_path=''; "
        f"for rel in {rels}; do "
        'c="$HOME/$rel"; '
        'if [ -d "$c/logs" ]; then '
        'm=$(stat -c %Y "$c/logs" 2>/dev/null || echo 0); '
        'case "$m" in ""|*[!0-9]*) m=0;; esac; '
        'if [ "$m" -gt "$best_mtime" ]; then '
        'best_mtime="$m"; best_path="$c"; '
        "fi; "
        "fi; "
        "done; "
        'printf "%s\\n" "$best_path"'
    )


def _existing_exp_dirs_script(variant: str) -> str:
    """Shell snippet that prints existing candidate exp dirs, in order."""
    rels = " ".join(shlex.quote(rel) for rel in _exp_dir_rel_candidates(variant))
    return (
        f"for rel in {rels}; do "
        'c="$HOME/$rel"; '
        'if [ -d "$c" ]; then printf "%s\\n" "$c"; fi; '
        "done"
    )


def _remote_path_expr(path: str) -> str:
    """Quote a remote path for shell use while preserving a leading $HOME."""
    return path if path.startswith("$HOME/") else shlex.quote(path)


async def _resolve_eval_checkpoint(
    host: str,
    stdout_path: str,
    exp_dir: str,
    submitted_checkpoint: str | None,
) -> str | None:
    """Return the checkpoint path for an eval job.

    Completed/running eval logs are the most accurate source because the body
    only logs `Checkpoint:` after verifying the directory exists. For pending
    or pre-log jobs, fall back to the submit sidecar's explicit checkpoint,
    then the same nested-then-flat auto-pick used by eval_body_n16.sh.
    """
    stdout_q = shlex.quote(stdout_path)
    r = await ssh_run(
        host,
        f"grep -E 'Checkpoint: ' {stdout_q} 2>/dev/null | tail -1",
        timeout=10.0,
    )
    line = r.stdout.strip()
    marker = "Checkpoint: "
    if marker in line:
        checkpoint = line.split(marker, 1)[1].strip()
        if checkpoint:
            return checkpoint

    if submitted_checkpoint:
        checkpoint = submitted_checkpoint.strip()
        if checkpoint:
            return checkpoint

    ckpt_dir = f"{exp_dir}/checkpoints"
    ckpt_dir_expr = _remote_path_expr(ckpt_dir)
    cmd = (
        f"D={ckpt_dir_expr}; "
        'p=$(ls -d "$D"/*/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1); '
        '[ -z "$p" ] && p=$(ls -d "$D"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1); '
        'printf "%s\\n" "$p"'
    )
    r = await ssh_run(host, cmd, timeout=15.0)
    checkpoint = r.stdout.strip()
    return checkpoint or None


async def _resolve_exp_dir(host: str, job_id: str, variant: str) -> str:
    """Find the variant's experiment dir on the cluster, per-job.

    Two parallel submission paths write to different roots:
      - web app  → `~/.train-eval-web/experiments/<v>`
      - bash CLI → `~/train-eval-scripts/experiments/<v>`

    Whichever the live job is actually using is the one with a fresh
    `logs/` subdir. We pick by mtime — that handles the case where the
    scontrol Command path doesn't match the running job's REPO_ROOT
    (the user can `bash ./submit` from train-eval-scripts but with a
    body script copied into .train-eval-web, and vice versa).
    """
    # Pick the candidate whose logs/ has the most recent mtime. Returns
    # one line: the winning candidate path, or empty if neither has logs/.
    r = await ssh_run(host, _latest_logs_dir_script(variant), timeout=10.0)
    chosen = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
    if chosen:
        return chosen

    # No logs/ in either candidate → derive from scontrol's Command path
    # (works while the job is in slurm's recent memory).
    r = await ssh_run(host, f"scontrol show job {job_id} 2>/dev/null | grep -m1 '^   Command='", timeout=10.0)
    if r.returncode == 0 and r.stdout.strip():
        line = r.stdout.strip()
        cmd_path = line.split("=", 1)[1].strip() if "=" in line else ""
        m = _BODY_SUFFIX_RE.search(cmd_path)
        if m:
            repo_root = cmd_path[: m.start()]
            return f"{repo_root}/experiments/{variant}"

    # Last resort: existence probe.
    r = await ssh_run(host, _existing_exp_dirs_script(variant), timeout=10.0)
    lines = r.stdout.strip().splitlines()
    if lines:
        return lines[0]
    return f"$HOME/{_exp_dir_rel_candidates(variant)[0]}"


_TQDM_STEP_RE = re.compile(r"(\d+)/(\d+)\s*\[")


async def _compute_progress(cluster: str, job_id: str, phase: str, variant: str | None,
                             stdout: str, stderr: str,
                             ckpt_dir: str | None, eval_dir: str | None) -> Progress:
    progress = Progress(phase=phase)
    if not variant:
        return progress

    if phase in ("train", "resume"):
        host = (await load_cluster(cluster)).ssh_alias
        # 1) Live-running jobs: parse latest tqdm step from stderr.
        r = await ssh_run(
            host,
            f"tail -c 4096 {stderr} 2>/dev/null | tr '\\r' '\\n' | grep -oE '[0-9]+/[0-9]+ \\[' | tail -1",
            timeout=10.0,
        )
        m = _TQDM_STEP_RE.search(r.stdout or "")
        if m:
            cur, total = int(m.group(1)), int(m.group(2))
            progress.current_step = cur
            progress.max_steps = total
            if total > 0:
                progress.percent = round(100.0 * cur / total, 1)
            progress.current_label = f"step {cur:,}/{total:,}"
            return progress

        # 2) Pending / between-runs / pre-tqdm: derive from the highest
        #    checkpoint dir on disk. Useful for jobs that were preempted and
        #    are queued for requeue — we still want to show how far they got.
        max_steps: int | None = None
        try:
            v = await load_variant(variant)
            if "MAX_STEPS" in v.vars:
                max_steps = int(v.vars["MAX_STEPS"])
        except Exception:
            pass
        progress.max_steps = max_steps

        if not ckpt_dir:
            return progress
        r = await ssh_run(
            host,
            f"ls -d {ckpt_dir}/checkpoint-* 2>/dev/null | sed 's:.*checkpoint-::' | sort -n | tail -1",
            timeout=10.0,
        )
        latest = r.stdout.strip()
        if latest.isdigit():
            cur = int(latest)
            progress.current_step = cur
            if max_steps:
                progress.percent = round(100.0 * cur / max_steps, 1)
            progress.current_label = (
                f"step {cur:,}/{max_steps:,}" if max_steps else f"step {cur:,}"
            )

    elif phase == "eval":
        v = await load_variant(variant)
        eval_sets = v.arrays.get("EVAL_SETS", [])
        n_runs = int(v.vars.get("N_RUNS", "0") or 0)
        n_eps = int(v.vars.get("N_EPISODES", "0") or 0)
        tasks = v.arrays.get("TASKS") or ["__single__"]
        total = max(len(tasks) * len(eval_sets) * n_runs, 0)
        progress.total_runs = total or None

        if not (eval_dir and total > 0):
            return progress

        host = (await load_cluster(cluster)).ssh_alias
        # Completed runs = number of results.json files written by the body.
        r = await ssh_run(
            host,
            f"find {eval_dir} -type f -name results.json 2>/dev/null | wc -l",
            timeout=10.0,
        )
        try:
            completed = int(r.stdout.strip())
        except ValueError:
            completed = 0
        progress.completed_runs = completed

        # Episode counter inside this job. Prefer client stdout because native
        # Isaac vectorization can complete multiple environments per server
        # reset; server reset counts are only a fallback.
        active_eps = 0
        active_envs = 0
        job_completed_runs = 0
        logs_dir = eval_dir.rsplit("/", 1)[0] + "/logs"
        if n_eps > 0:
            env = await load_cluster(cluster)
            log_dir_q = shlex.quote(env.vars["LOG_DIR"])
            job_id_q = shlex.quote(job_id)
            stdout_cmd = (
                f"pattern={log_dir_q}/*_{job_id_q}.out; "
                "episodes=$(grep -h '^Episode .* completed' $pattern 2>/dev/null | wc -l); "
                "runs=$(grep -h '^Results saved to:' $pattern 2>/dev/null | wc -l); "
                "echo \"$episodes $runs\""
            )
            r = await ssh_run(host, stdout_cmd, timeout=10.0)
            try:
                parts = r.stdout.strip().split()
                active_eps = int(parts[0]) if parts else 0
                job_completed_runs = int(parts[1]) if len(parts) > 1 else 0
            except (ValueError, IndexError):
                active_eps = 0
                job_completed_runs = 0

            logs_dir_q = shlex.quote(logs_dir)
            ep_cmd = (
                f"job_logs={logs_dir_q}/{job_id_q}; "
                "if [ -d \"$job_logs\" ]; then pattern=\"$job_logs/server_*.log\"; "
                "else echo '0 0'; exit 0; fi; "
                "total=0; envs_started=0; "
                "for f in $pattern; do "
                "[ -f \"$f\" ] || continue; "
                "envs=$(grep -m1 'Num envs:' \"$f\" 2>/dev/null | sed -E 's/.*Num envs: ([0-9]+).*/\\1/'); "
                "case \"$envs\" in ''|*[!0-9]*) envs=1 ;; esac; "
                "c=$(grep -c 'Resetting environment with seed:' \"$f\" 2>/dev/null || echo 0); "
                "if [ \"$c\" -gt 0 ]; then c=$((c - 1)); fi; "
                f"if [ \"$c\" -gt {n_eps} ]; then c={n_eps}; fi; "
                "total=$((total + c * envs)); envs_started=$((envs_started + envs)); "
                "done; "
                "echo \"$total $envs_started\""
            )
            r = await ssh_run(host, ep_cmd, timeout=10.0)
            try:
                parts = r.stdout.strip().split()
                active_eps = max(active_eps, int(parts[0]) if parts else 0)
                active_envs = int(parts[1]) if len(parts) > 1 else 0
            except (ValueError, IndexError):
                active_envs = 0

        # Promote eval into the unified step-based shape so the frontend
        # ETA + progress bar work the same way as training:
        #   current_step = completed_runs · N_EPISODES + incomplete episodes in this job
        #   max_steps    = total_runs    · N_EPISODES
        if n_eps > 0:
            progress.max_steps = total * n_eps
            active_incomplete_eps = max(0, active_eps - job_completed_runs * n_eps)
            progress.current_step = min(completed * n_eps + active_incomplete_eps, progress.max_steps)
            progress.percent = round(100.0 * progress.current_step / progress.max_steps, 1)
            episode_label = f"{progress.current_step}/{progress.max_steps} episodes"
            progress.current_label = f"{completed}/{total} runs · {episode_label}"
        else:
            progress.percent = round(100.0 * completed / total, 1)
            progress.current_label = f"{completed}/{total} runs"
    return progress
