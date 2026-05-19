"""Submit a job to a remote cluster.

Flow:
  1. rsync the local `configs/` + `lib/` into `~/.train-eval-web/` on the cluster
  2. Resolve cluster + variant configs (locally) to derive partition, time, GPUs, body script path.
  3. Build an `sbatch` command targeting the cluster-side body script.
  4. Run it over ssh, parse "Submitted batch job <id>" out of stdout.
"""

import re
import shlex
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from .clusters import load_cluster
from .paths import CLUSTER_STAGING_REL, CONFIGS_DIR, LIB_DIR
from .ssh import rsync_to, ssh_run
from .variants import load_variant
from .variant_values import variant_int


def _is_background_partition(name: str) -> bool:
    """Preemptible partitions (auto-add --requeue at submit time)."""
    return name == "background" or name.endswith("_background")


def _apply_dataset_override(config_text: str, override: str | list[str]) -> str:
    """Rewrite a variant config.sh to use the requested dataset(s).

    - String override → replaces DATASET_NAME=... (single-task).
    - List override → replaces whichever multi-dataset block the variant
      already uses: TRAIN_DATASET_NAMES (N1.6, name-only) or DATASETS
      (N1.5, "name|cfg|weight"). Block shape is detected from the entries.
    """
    if isinstance(override, str):
        new_line = f"DATASET_NAME={override}"
        return re.sub(
            r"^(\s*export\s+)?DATASET_NAME=.*$",
            lambda m: (m.group(1) or "") + new_line,
            config_text,
            flags=re.MULTILINE,
        )

    # N1.6 (name-only): entries have no `|` → TRAIN_DATASET_NAMES.
    is_names_only = all("|" not in e for e in override)
    block_name = "TRAIN_DATASET_NAMES" if is_names_only else "DATASETS"
    new_block_lines = [f"{block_name}=("]
    new_block_lines.extend(f'    "{entry}"' for entry in override)
    new_block_lines.append(")")
    new_block = "\n".join(new_block_lines)
    pattern = rf"^{block_name}=\(.*?^\)\s*$"
    return re.sub(
        pattern,
        new_block,
        config_text,
        count=1,
        flags=re.MULTILINE | re.DOTALL,
    )


class SubmitRequest(BaseModel):
    cluster: str
    variant: str
    phase: Literal["train", "eval"]
    # Internal job action: retry a timed-out training job from its existing
    # checkpoint. This is intentionally separate from the submit-page phase.
    resume: bool = False
    resume_of: str | None = None
    # Slurm-only: partition name (None → fall back to cluster.env default).
    partition: str | None = None
    # MLXP-only: which k8s node to pin via nodeAffinity (each rlwrld team
    # member is assigned a specific h200-03-w-XXXX in the GPU Resource
    # Schedule sheet). None falls back to mlxp_submit.DEFAULT_NODE.
    node: str | None = None
    # Per-submit dataset override. Two shapes accepted:
    #   - single string  → replaces DATASET_NAME in single-task variants
    #   - list of "name|cfg|weight" entries → replaces DATASETS array
    # None means "use whatever the variant config.sh says".
    dataset_override: str | list[str] | None = None
    extra_args: list[str] = Field(default_factory=list)
    # Train-only: per-submission overrides. None means "use config.sh".
    train_num_gpus: int | None = Field(default=None, ge=1)
    train_global_batch_size: int | None = Field(default=None, ge=1)
    train_max_steps: int | None = Field(default=None, ge=1)
    train_save_steps: int | None = Field(default=None, ge=1)
    # Eval-only: override Isaac's native vectorized env count per GPU.
    eval_num_envs_per_gpu: int | None = Field(default=None, ge=1)
    # Legacy request field accepted from older frontends/resume metadata.
    eval_parallel_sims_per_gpu: int | None = Field(default=None, ge=1)
    # Eval-only: per-submission overrides for eval_allex.py and eval matrix.
    eval_n_episodes: int | None = Field(default=None, ge=1)
    eval_n_runs: int | None = Field(default=None, ge=1)
    eval_sets: list[str] | None = None
    eval_overwrite_results: bool = False
    # Eval-only: absolute path to the checkpoint dir on the cluster. The
    # eval body uses this verbatim when set; otherwise it auto-picks.
    checkpoint_path: str | None = None
    # Internal eval resume: remote eval_results dir from the timed-out job.
    # Seed the staged eval_results before sbatch so completed runs are skipped.
    seed_eval_results_from: str | list[str] | None = None
    # Optional override for the auto-generated job_name. Must match
    # `{train|eval}_<anything>_<YYYYMMDD>_<HHMMSS>` so the parser
    # keeps working. None → server builds the default.
    job_name: str | None = None


def make_default_job_name(phase: str, variant: str) -> str:
    return f"{phase}_{variant}_{datetime.now():%Y%m%d_%H%M%S}"


def resolve_job_name(req_job_name: str | None, phase: str, variant: str) -> str:
    """Return user-provided job_name if non-empty, else build the default.

    No format validation — caller may pass any string. Note that names that
    don't match `{phase}_<slug>_<YYYYMMDD>_<HHMMSS>` will resolve to
    ("unknown", None) in parse_phase_and_variant, so phase/variant won't be
    derivable from the name.
    """
    if req_job_name is None:
        return make_default_job_name(phase, variant)
    name = req_job_name.strip()
    if not name:
        return make_default_job_name(phase, variant)
    return name


def _normalize_eval_sets(eval_sets: list[str] | None) -> list[str] | None:
    if eval_sets is None:
        return None
    out: list[str] = []
    seen: set[str] = set()
    for raw in eval_sets:
        item = raw.strip()
        if not item or item in seen:
            continue
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", item):
            raise ValueError(f"invalid eval set {item!r}; use letters, numbers, dot, underscore, or hyphen")
        seen.add(item)
        out.append(item)
    if not out:
        raise ValueError("eval_sets must contain at least one eval set")
    return out


class SubmitResponse(BaseModel):
    job_id: str
    job_name: str
    partition: str
    sbatch_cmd: str
    rsync_stdout: str
    sbatch_stdout: str


_BODY_BY_PHASE_MODEL = {
    ("train", "n1.5"): ("train_body.sh", "48:00:00"),
    ("train", "n1.6"): ("train_body_n16.sh", "48:00:00"),
    ("eval", "n1.5"):  ("eval_body.sh", "08:00:00"),
    ("eval", "n1.6"):  ("eval_body_n16.sh", "08:00:00"),
}

MAX_EVAL_NUM_ENVS_PER_GPU = 1


async def submit(req: SubmitRequest) -> SubmitResponse:
    if req.resume and req.phase != "train":
        raise ValueError("resume=true is only valid for phase=train")
    eval_num_envs_per_gpu = req.eval_num_envs_per_gpu or req.eval_parallel_sims_per_gpu
    eval_sets = _normalize_eval_sets(req.eval_sets)
    if req.phase != "eval" and any((
        eval_num_envs_per_gpu is not None,
        req.eval_n_episodes is not None,
        req.eval_n_runs is not None,
        eval_sets is not None,
        req.eval_overwrite_results,
    )):
        raise ValueError("eval overrides are only valid for phase=eval")
    if req.phase != "train" and any((
        req.train_num_gpus is not None,
        req.train_global_batch_size is not None,
        req.train_max_steps is not None,
        req.train_save_steps is not None,
    )):
        raise ValueError("train overrides are only valid for phase=train")
    if (
        req.phase == "eval"
        and eval_num_envs_per_gpu is not None
        and eval_num_envs_per_gpu > MAX_EVAL_NUM_ENVS_PER_GPU
    ):
        raise ValueError(
            "eval_num_envs_per_gpu > 1 is disabled: the ALLEX target reset "
            "path is not vector-env safe"
        )

    cluster = await load_cluster(req.cluster)
    variant = await load_variant(req.variant)

    # ── Resolve partition + body script + walltime ──
    model = variant.vars.get("MODEL_VERSION", "n1.5")
    body_walltime = _BODY_BY_PHASE_MODEL.get((req.phase, model))
    if body_walltime is None:
        raise ValueError(f"Unsupported (phase, model): ({req.phase}, {model})")
    body_script, walltime = body_walltime

    partition = req.partition or cluster.vars["PARTITION"]
    sbatch_flags: list[str] = []
    if _is_background_partition(partition):
        sbatch_flags.append("--requeue")
    exclude_nodes = (cluster.vars.get("SBATCH_EXCLUDE") or cluster.vars.get("SLURM_EXCLUDE_NODES") or "").strip()
    if exclude_nodes:
        sbatch_flags.append(f"--exclude={shlex.quote(exclude_nodes)}")

    train_num_gpus = req.train_num_gpus or variant_int(variant, "TRAIN_NUM_GPUS", 2)
    train_max_steps = req.train_max_steps or variant_int(variant, "MAX_STEPS", 30000)
    train_save_steps = req.train_save_steps or variant_int(variant, "SAVE_STEPS", 1000)
    if req.phase == "train" and req.train_global_batch_size is not None and model == "n1.5":
        if req.train_global_batch_size % train_num_gpus != 0:
            raise ValueError("train_global_batch_size must be divisible by train_num_gpus for n1.5 training")
    gpus = str(train_num_gpus)

    # ── Sync code to cluster staging ──
    # Body scripts expect $REPO_ROOT/{clusters,experiments,lib}/ at the staging
    # root, so flatten configs/ on the way out: configs/clusters → clusters/,
    # configs/experiments → experiments/.
    host = cluster.ssh_alias
    staging = f"$HOME/{CLUSTER_STAGING_REL}"
    mkdir_result = await ssh_run(host, f"mkdir -p {staging}/clusters {staging}/experiments {staging}/lib")
    if mkdir_result.returncode != 0:
        raise RuntimeError(f"mkdir on cluster failed: {mkdir_result.stderr}")

    rsync_results = []
    # (local source with trailing slash, remote target dir name)
    sync_targets = [
        (str(CONFIGS_DIR / "clusters") + "/",    "clusters"),
        (str(CONFIGS_DIR / "experiments") + "/", "experiments"),
        (str(LIB_DIR) + "/",                      "lib"),
    ]
    for local, remote_name in sync_targets:
        remote = f"{CLUSTER_STAGING_REL}/{remote_name}"
        excludes = (
            ["checkpoints/", "eval_results/", "logs/", "results.json"]
            if remote_name == "experiments"
            else None
        )
        r = await rsync_to(host, local, remote, delete=True, exclude=excludes)
        if r.returncode != 0:
            raise RuntimeError(f"rsync failed for {local}: {r.stderr}")
        rsync_results.append(r)

    # Apply dataset override to the staged config.sh, if requested.
    if req.dataset_override is not None:
        modified = _apply_dataset_override(variant.raw, req.dataset_override)
        if modified != variant.raw:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix="_config.sh", delete=False,
            ) as fp:
                fp.write(modified)
                tmp_path = fp.name
            try:
                remote_cfg = f"{CLUSTER_STAGING_REL}/experiments/{req.variant}/config.sh"
                r = await rsync_to(host, tmp_path, remote_cfg)
                if r.returncode != 0:
                    raise RuntimeError(f"rsync override failed: {r.stderr}")
                rsync_results.append(r)
            finally:
                Path(tmp_path).unlink(missing_ok=True)

    if req.phase == "eval" and req.seed_eval_results_from:
        dst_rel = f"{CLUSTER_STAGING_REL}/experiments/{req.variant}/eval_results"
        seed_sources = (
            [req.seed_eval_results_from]
            if isinstance(req.seed_eval_results_from, str)
            else req.seed_eval_results_from
        )
        for source in seed_sources:
            src = source.strip().rstrip("/")
            if not src:
                continue
            seed_cmd = (
                f"src={shlex.quote(src)}; dst=$HOME/{shlex.quote(dst_rel)}; "
                "mkdir -p \"$dst\" && "
                "if [ \"$src\" != \"$dst\" ] && [ -d \"$src\" ]; then "
                "rsync -a --ignore-existing \"$src\"/ \"$dst\"/; "
                "fi"
            )
            seeded = await ssh_run(host, seed_cmd, timeout=120.0)
            if seeded.returncode != 0:
                raise RuntimeError(f"seeding eval results failed: {seeded.stderr or seeded.stdout}")

    # ── Build sbatch command ──
    # Unified shape across slurm + MLXP. The cluster/partition were
    # cosmetic in the slurm filename — drop them; the table column shows
    # both, and `parse_phase_and_variant` now expects this exact format.
    job_name = resolve_job_name(req.job_name, req.phase, req.variant)
    log_dir = cluster.vars["LOG_DIR"]
    resume_expected = "1" if req.resume else "0"

    body_path = f"$HOME/{CLUSTER_STAGING_REL}/lib/{body_script}"
    repo_root_remote = f"$HOME/{CLUSTER_STAGING_REL}"

    # Persist phase+variant in sacct's Comment field so the details page can
    # recover them even when the user picked a custom job_name that doesn't
    # match the unified regex.
    comment = f"phase={req.phase};variant={req.variant}"
    if req.phase == "train":
        comment += (
            f";train_num_gpus={train_num_gpus}"
            f";train_max_steps={train_max_steps}"
            f";train_save_steps={train_save_steps}"
        )
        if req.train_global_batch_size is not None:
            comment += f";train_global_batch_size={req.train_global_batch_size}"

    sbatch_parts = [
        "/opt/slurm/bin/sbatch",
        f"--job-name={shlex.quote(job_name)}",
        f"--partition={shlex.quote(partition)}",
        "--nodes=1",
        f"--gpus-per-node={shlex.quote(gpus)}",
        f"--time={shlex.quote(walltime)}",
        f"--output={log_dir}/{job_name}_%j.out",
        f"--error={log_dir}/{job_name}_%j.err",
        f"--comment={shlex.quote(comment)}",
        f"--export=ALL,VARIANT={shlex.quote(req.variant)},CLUSTER={shlex.quote(req.cluster)},"
        f"REPO_ROOT={repo_root_remote},RESUME_EXPECTED={resume_expected},"
        f"SUBMIT_PARTITION={shlex.quote(partition)},"
        # Pin wandb run id to the slurm display name so the URL is stable
        # and matches MLXP's run-id format.
        f"WANDB_RUN_ID={shlex.quote(job_name)}"
        + (
            f",SUBMIT_TRAIN_NUM_GPUS={train_num_gpus},"
            f"SUBMIT_TRAIN_MAX_STEPS={train_max_steps},"
            f"SUBMIT_TRAIN_SAVE_STEPS={train_save_steps}"
            if req.phase == "train" else ""
        )
        + (
            f",SUBMIT_TRAIN_GLOBAL_BATCH_SIZE={req.train_global_batch_size}"
            if req.phase == "train" and req.train_global_batch_size is not None else ""
        )
        + (
            f",SUBMIT_EVAL_NUM_ENVS_PER_GPU={eval_num_envs_per_gpu}"
            if req.phase == "eval" and eval_num_envs_per_gpu is not None else ""
        )
        + (
            f",SUBMIT_EVAL_N_EPISODES={req.eval_n_episodes}"
            if req.phase == "eval" and req.eval_n_episodes is not None else ""
        )
        + (
            f",SUBMIT_EVAL_N_RUNS={req.eval_n_runs}"
            if req.phase == "eval" and req.eval_n_runs is not None else ""
        )
        + (
            f",SUBMIT_EVAL_SETS={shlex.quote(' '.join(eval_sets))}"
            if req.phase == "eval" and eval_sets is not None else ""
        )
        + (
            ",SUBMIT_EVAL_OVERWRITE_RESULTS=1"
            if req.phase == "eval" and req.eval_overwrite_results else ""
        )
        + (
            f",EVAL_CHECKPOINT={shlex.quote(req.checkpoint_path)}"
            if req.phase == "eval" and req.checkpoint_path else ""
        ),
        *sbatch_flags,
        *[shlex.quote(a) for a in req.extra_args],
        body_path,
    ]
    # Fallback to which-sbatch if /opt/slurm/bin/sbatch missing:
    sbatch_cmd = (
        "SBATCH_BIN=$(command -v sbatch 2>/dev/null || echo /opt/slurm/bin/sbatch); "
        + " ".join(sbatch_parts).replace("/opt/slurm/bin/sbatch", "$SBATCH_BIN", 1)
    )

    sb = await ssh_run(host, sbatch_cmd, timeout=30.0)
    if sb.returncode != 0:
        raise RuntimeError(f"sbatch failed: {sb.stderr or sb.stdout}")

    m = re.search(r"Submitted batch job (\d+)", sb.stdout)
    if not m:
        raise RuntimeError(f"could not parse sbatch output: {sb.stdout!r}")
    job_id = m.group(1)

    # Persistent sidecar so the details page can recover phase/variant for
    # this job_id forever. Slurm's --comment is unreliable: it's on the
    # live controller (scontrol) but most slurmdbd setups (kakao's
    # included) don't archive it to sacct.
    meta_dir = "$HOME/.train-eval-web/jobs"
    meta = (
        f"phase={req.phase}\n"
        f"variant={req.variant}\n"
        f"job_name={job_name}\n"
        + (
            f"resume_of={req.resume_of.strip()}\n"
            if req.resume_of and req.resume_of.strip()
            else ""
        )
        + ("resume=true\n" if req.resume else "")
        + (
            f"eval_num_envs_per_gpu={eval_num_envs_per_gpu}\n"
            if req.phase == "eval" and eval_num_envs_per_gpu is not None
            else ""
        )
        + (
            f"train_num_gpus={train_num_gpus}\n"
            f"train_max_steps={train_max_steps}\n"
            f"train_save_steps={train_save_steps}\n"
            if req.phase == "train"
            else ""
        )
        + (
            f"train_global_batch_size={req.train_global_batch_size}\n"
            if req.phase == "train" and req.train_global_batch_size is not None
            else ""
        )
        + (
            f"exclude_nodes={exclude_nodes}\n"
            if exclude_nodes
            else ""
        )
        + (
            f"eval_n_episodes={req.eval_n_episodes}\n"
            if req.phase == "eval" and req.eval_n_episodes is not None
            else ""
        )
        + (
            f"eval_n_runs={req.eval_n_runs}\n"
            if req.phase == "eval" and req.eval_n_runs is not None
            else ""
        )
        + (
            f"eval_sets={' '.join(eval_sets)}\n"
            if req.phase == "eval" and eval_sets is not None
            else ""
        )
        + (
            "eval_overwrite_results=true\n"
            if req.phase == "eval" and req.eval_overwrite_results
            else ""
        )
        + (
            f"checkpoint_path={req.checkpoint_path.strip()}\n"
            if req.phase == "eval" and req.checkpoint_path and req.checkpoint_path.strip()
            else ""
        )
    )
    meta_cmd = (
        f"mkdir -p {meta_dir} && "
        f"cat > {meta_dir}/{job_id}.meta <<'EOF'\n{meta}EOF"
    )
    await ssh_run(host, meta_cmd, timeout=15.0)

    return SubmitResponse(
        job_id=job_id,
        job_name=job_name,
        partition=partition,
        sbatch_cmd=sbatch_cmd,
        rsync_stdout="\n".join(r.stdout for r in rsync_results),
        sbatch_stdout=sb.stdout,
    )
