"""Per-job extended details: phase, paths, wandb url, progress.

Parses metadata out of the slurm job_name (which is shaped like
`<phase>_<variant>_<cluster>_<partition>_<timestamp>`), reads variant
config locally, and asks the cluster a few small questions over SSH to
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


# Wandb entity defaults to current user; override with TRAIN_EVAL_WEB_WANDB_ENTITY.
WANDB_ENTITY = os.environ.get("TRAIN_EVAL_WEB_WANDB_ENTITY", "youngwoong")
WANDB_PROJECT = os.environ.get("TRAIN_EVAL_WEB_WANDB_PROJECT", "gr00t")

KNOWN_CLUSTERS = ("kakao", "skt")


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


def parse_phase_and_variant(job_name: str, cluster: str) -> tuple[str, str | None]:
    """Slurm job names: '<phase>_<variant>_<cluster>_<partition>_<timestamp>'.
    MLXP job names: 'youngwoong-<phase>-<variant-slug>-<timestamp>'.

    For MLXP the variant slug has hyphens substituted for underscores, so
    we walk the existing experiments dir to find the longest match.
    """
    if cluster == "mlxp":
        m = re.match(r"^youngwoong-(train|resume|eval)-(.+)-(\d{8}-\d{6})$", job_name)
        if not m:
            return "unknown", None
        phase = m.group(1)
        slug = m.group(2)
        # Try to recover the original underscore-bearing variant name by
        # scanning available variants.
        from .variants import list_variants
        try:
            for v in sorted(list_variants(), key=len, reverse=True):
                if v.lower().replace("_", "-") == slug.lower():
                    return phase, v
        except Exception:
            pass
        return phase, slug.replace("-", "_")

    phase_match = re.match(r"^(train|resume|eval)_", job_name)
    if not phase_match:
        return "unknown", None
    phase = phase_match.group(1)
    after_phase = job_name[len(phase) + 1:]
    # Find '_<cluster>_' and take what's before it as the variant.
    needle = f"_{cluster}_"
    idx = after_phase.find(needle)
    if idx < 0:
        return phase, None
    return phase, after_phase[:idx]


async def get_details(cluster: str, job_id: str) -> JobDetails:
    sacct = await get_job(cluster, job_id)
    job_name = sacct.get("JobName", "")
    state = sacct.get("State", "")
    elapsed = sacct.get("Elapsed", "")

    phase, variant = parse_phase_and_variant(job_name, cluster)

    if cluster == "mlxp":
        return await _mlxp_details(job_id, job_name, state, elapsed, phase, variant)

    env = await load_cluster(cluster)
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
        # train_body.sh exports WANDB_RUN_ID=slurm_$SLURM_JOB_ID with WANDB_RESUME=allow.
        wandb_url = f"https://wandb.ai/{WANDB_ENTITY}/{WANDB_PROJECT}/runs/slurm_{job_id}"

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
    # MLXP training writes WANDB_RUN_ID via env from the Job spec; not yet
    # pinned to job_id like slurm, so omit the wandb URL until we add that.
    wandb_url = None

    # Progress: count latest checkpoint dir on DDN. Re-uses the same logic
    # we have for pending slurm jobs but via kubectl exec into the running
    # pod. For MVP we just count checkpoints by listing the dir from a fresh
    # short-lived pod... too heavy. Skip MLXP progress for now and let the
    # log tail show tqdm directly.
    progress = Progress(phase=phase)
    if variant:
        try:
            v = await load_variant(variant)
            if "MAX_STEPS" in v.vars:
                progress.max_steps = int(v.vars["MAX_STEPS"])
        except Exception:
            pass

    return JobDetails(
        cluster="mlxp", job_id=job_id, job_name=job_name,
        phase=phase, variant=variant, state=state, elapsed=elapsed,
        wandb_url=wandb_url, paths=paths, progress=progress,
    )


_BODY_SUFFIX_RE = re.compile(r"/lib/(train|eval)_body(?:_n16)?\.sh$")


async def _resolve_exp_dir(host: str, job_id: str, variant: str) -> str:
    """Find the variant's experiment dir on the cluster, per-job.

    Different submission tools write to different roots: the legacy bash
    `./submit` uses `$REPO_ROOT/experiments/<v>` where `$REPO_ROOT` is
    `~/train-eval-scripts`; the web app uses `~/.train-eval-web`. Ask
    slurm which one this specific job actually ran from by parsing the
    Command field of `scontrol show job`.
    """
    # Try scontrol first (works while the job is in slurm's recent memory).
    r = await ssh_run(host, f"scontrol show job {job_id} 2>/dev/null | grep -m1 '^   Command='", timeout=10.0)
    if r.returncode == 0 and r.stdout.strip():
        line = r.stdout.strip()
        cmd_path = line.split("=", 1)[1].strip() if "=" in line else ""
        m = _BODY_SUFFIX_RE.search(cmd_path)
        if m:
            repo_root = cmd_path[: m.start()]
            return f"{repo_root}/experiments/{variant}"

    # Fallback for jobs that have aged out of scontrol — probe both known paths.
    candidates = [
        f"$HOME/train-eval-scripts/experiments/{variant}",
        f"$HOME/{CLUSTER_STAGING_REL}/experiments/{variant}",
    ]
    probe = " || ".join(f"(test -d {c} && echo {c})" for c in candidates)
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
        tasks = v.arrays.get("TASKS") or ["__single__"]
        total = max(len(tasks) * len(eval_sets) * n_runs, 0)
        progress.total_runs = total or None

        if eval_dir and total > 0:
            host = (await load_cluster(cluster)).ssh_alias
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
            progress.percent = round(100.0 * completed / total, 1)
            progress.current_label = f"{completed}/{total} runs"
    return progress
