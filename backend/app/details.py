"""Per-job extended details: phase, paths, wandb url, progress.

Parses metadata out of the job_name (shape `{train|resume|eval}_{variant}_{YYYYMMDD}_{HHMMSS}`,
identical across slurm and mlxp), reads variant config locally, and asks
the cluster a few small questions over SSH (slurm) or kubectl (mlxp) to
compute progress.
"""

import os
import re
from typing import Any

from pydantic import BaseModel

from .clusters import load_cluster
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


def parse_phase_and_variant(job_name: str) -> tuple[str, str | None]:
    """Pull (phase, variant) out of a display name.

    Shape (both clusters): `{train|eval|resume}_{variant}_{YYYYMMDD}_{HHMMSS}`.
    The variant itself contains underscores (e.g. `n16_cube_stack_3cm_right_480`),
    so we anchor on the trailing timestamp `_\\d{8}_\\d{6}` to find the boundary.
    """
    m = re.match(r"^(train|resume|eval)_(.+)_(\d{8}_\d{6})$", job_name)
    if m:
        return m.group(1), m.group(2)
    return "unknown", None


def _parse_comment_metadata(comment: str) -> tuple[str | None, str | None]:
    """Recover (phase, variant) from the sacct Comment / k8s annotation
    we set at submit time. Shape: 'phase=<p>;variant=<v>'."""
    if not comment:
        return None, None
    fields: dict[str, str] = {}
    for chunk in comment.split(";"):
        if "=" in chunk:
            k, v = chunk.split("=", 1)
            fields[k.strip()] = v.strip()
    phase = fields.get("phase")
    variant = fields.get("variant")
    if phase not in ("train", "resume", "eval"):
        phase = None
    return phase, variant


def resolve_phase_and_variant(job_name: str, sacct: dict | None = None) -> tuple[str, str | None]:
    """Prefer the explicit phase/variant we stashed at submit time (sacct
    Comment / scontrol Comment / sidecar .meta for slurm, k8s annotation
    surfaced under JobComment for mlxp). Fall back to parsing the
    job_name."""
    if sacct:
        comment = sacct.get("Comment") or sacct.get("JobComment") or ""
        phase, variant = _parse_comment_metadata(comment)
        if phase and variant:
            return phase, variant
    return parse_phase_and_variant(job_name)


async def _read_slurm_meta(host: str, job_id: str) -> tuple[str | None, str | None]:
    """Read the phase/variant sidecar written at submit time.
    `~/.train-eval-web/jobs/<job_id>.meta` shape: lines `key=value`."""
    r = await ssh_run(
        host,
        f"cat $HOME/.train-eval-web/jobs/{job_id}.meta 2>/dev/null",
        timeout=10.0,
    )
    if r.returncode != 0 or not r.stdout.strip():
        return None, None
    fields: dict[str, str] = {}
    for line in r.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            fields[k.strip()] = v.strip()
    phase = fields.get("phase")
    if phase not in ("train", "resume", "eval"):
        phase = None
    return phase, fields.get("variant")


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


async def get_details(cluster: str, job_id: str) -> JobDetails:
    sacct = await get_job(cluster, job_id)
    job_name = sacct.get("JobName", "")
    state = sacct.get("State", "")
    elapsed = sacct.get("Elapsed", "")

    phase, variant = resolve_phase_and_variant(job_name, sacct)

    if cluster == "mlxp":
        return await _mlxp_details(job_id, job_name, state, elapsed, phase, variant)

    env = await load_cluster(cluster)

    # Slurm: if sacct didn't return Comment (slurmdbd doesn't archive it on
    # this cluster), check scontrol (works for jobs still in the
    # controller) and then the on-disk .meta sidecar (permanent).
    if not variant:
        scontrol_comment = await _read_slurm_scontrol_comment(env.ssh_alias, job_id)
        if scontrol_comment:
            p, v = _parse_comment_metadata(scontrol_comment)
            if p and v:
                phase, variant = p, v
    if not variant:
        p, v = await _read_slurm_meta(env.ssh_alias, job_id)
        if p and v:
            phase, variant = p, v
    log_dir = env.vars["LOG_DIR"]
    stdout_path = f"{log_dir}/{job_name}_{job_id}.out"
    stderr_path = f"{log_dir}/{job_name}_{job_id}.err"

    # The per-variant experiment dir on the cluster depends on who submitted:
    # web-submitted jobs use ~/.train-eval-web/experiments/<variant>; jobs
    # launched via the bash `./submit` use ~/train-eval-scripts/experiments/<variant>.
    # Probe both, prefer one that actually exists.
    exp_dir_remote = await _resolve_exp_dir(env.ssh_alias, job_id, variant) if variant else f"$HOME/{CLUSTER_STAGING_REL}/experiments"
    ckpt_dir = f"{exp_dir_remote}/checkpoints" if phase in ("train", "resume") else None
    eval_dir = f"{exp_dir_remote}/eval_results" if phase == "eval" else None
    isaac_logs_glob = f"{exp_dir_remote}/logs/server_*.log" if phase == "eval" else None
    paths = Paths(
        stdout=stdout_path,
        stderr=stderr_path,
        exp_dir=exp_dir_remote,
        ckpt_dir=ckpt_dir,
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

    return JobDetails(
        cluster=cluster, job_id=job_id, job_name=job_name,
        phase=phase, variant=variant, state=state, elapsed=elapsed,
        wandb_url=wandb_url, paths=paths, progress=progress,
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
    candidates = [
        f"$HOME/{CLUSTER_STAGING_REL}/experiments/{variant}",
        f"$HOME/train-eval-scripts/experiments/{variant}",
    ]
    # Pick the candidate whose logs/ has the most recent mtime. Returns
    # one line: the winning candidate path, or empty if neither has logs/.
    script = " ; ".join(
        f'[ -d {c}/logs ] && echo "$(stat -c %Y {c}/logs 2>/dev/null) {c}"'
        for c in candidates
    ) + " | sort -n | tail -1 | awk '{print $2}'"
    r = await ssh_run(host, script, timeout=10.0)
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
    probe = " ; ".join(f"(test -d {c} && echo {c})" for c in candidates)
    r = await ssh_run(host, probe, timeout=10.0)
    lines = r.stdout.strip().splitlines()
    if lines:
        return lines[0]
    return candidates[0]


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

        # Episode counter inside the *active* server log (most recently
        # touched server_<set>_run<i>.log). Each "Resetting environment
        # with seed:" marks an episode boundary; the body's stabilization
        # pass adds one extra reset, so clamp to N_EPISODES.
        current_ep = 0
        logs_dir = eval_dir.rsplit("/", 1)[0] + "/logs"
        if n_eps > 0:
            ep_cmd = (
                f"latest=$(ls -t {logs_dir}/server_*.log 2>/dev/null | head -1); "
                f"if [ -n \"$latest\" ]; then "
                f"grep -c 'Resetting environment with seed:' \"$latest\" 2>/dev/null || echo 0; "
                f"else echo 0; fi"
            )
            r = await ssh_run(host, ep_cmd, timeout=10.0)
            try:
                current_ep = min(int(r.stdout.strip()), n_eps)
            except ValueError:
                current_ep = 0

        # Promote eval into the unified step-based shape so the frontend
        # ETA + progress bar work the same way as training:
        #   current_step = completed_runs · N_EPISODES + episodes_in_active_run
        #   max_steps    = total_runs    · N_EPISODES
        if n_eps > 0:
            progress.max_steps = total * n_eps
            # Don't double-count: if a run finished, current_ep is from a
            # log file that may still belong to it. Reset to 0 when no
            # new run has started yet (server log mtime older than newest
            # results.json), but for simplicity we just rely on the clamp
            # — overcount of n_eps gets absorbed when results.json lands.
            progress.current_step = min(completed * n_eps + current_ep, progress.max_steps)
            progress.percent = round(100.0 * progress.current_step / progress.max_steps, 1)
            progress.current_label = (
                f"{completed}/{total} runs · episode {current_ep}/{n_eps}"
            )
        else:
            progress.percent = round(100.0 * completed / total, 1)
            progress.current_label = f"{completed}/{total} runs"
    return progress
