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

from pydantic import BaseModel

from .clusters import ClusterEnv, load_cluster
from .paths import CLUSTER_STAGING_REL, CLUSTERS_DIR, CONFIGS_DIR, EXPERIMENTS_DIR, LIB_DIR
from .ssh import SSHResult, rsync_to, ssh_run
from .variants import Variant, load_variant


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
    phase: str                  # "train" | "resume" | "eval"
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
    extra_args: list[str] = []
    # Eval-only: absolute path to the checkpoint dir on the cluster. The
    # eval body uses this verbatim when set; otherwise it auto-picks.
    checkpoint_path: str | None = None
    # Optional override for the auto-generated job_name. Must match
    # `{train|resume|eval}_<anything>_<YYYYMMDD>_<HHMMSS>` so the parser
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


async def submit(req: SubmitRequest) -> SubmitResponse:
    cluster = await load_cluster(req.cluster)
    variant = await load_variant(req.variant)

    # ── Resolve partition + body script + walltime ──
    model = variant.vars.get("MODEL_VERSION", "n1.5")
    route_phase = "train" if req.phase == "resume" else req.phase
    body_walltime = _BODY_BY_PHASE_MODEL.get((route_phase, model))
    if body_walltime is None:
        raise ValueError(f"Unsupported (phase, model): ({route_phase}, {model})")
    body_script, walltime = body_walltime

    partition = req.partition or cluster.vars["PARTITION"]
    sbatch_flags: list[str] = []
    if _is_background_partition(partition):
        sbatch_flags.append("--requeue")

    gpus = variant.vars.get("TRAIN_NUM_GPUS", "2")

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
        r = await rsync_to(host, local, remote, delete=True)
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

    # ── Build sbatch command ──
    # Unified shape across slurm + MLXP. The cluster/partition were
    # cosmetic in the slurm filename — drop them; the table column shows
    # both, and `parse_phase_and_variant` now expects this exact format.
    job_name = resolve_job_name(req.job_name, req.phase, req.variant)
    log_dir = cluster.vars["LOG_DIR"]
    resume_expected = "1" if req.phase == "resume" else "0"

    body_path = f"$HOME/{CLUSTER_STAGING_REL}/lib/{body_script}"
    repo_root_remote = f"$HOME/{CLUSTER_STAGING_REL}"

    # Persist phase+variant in sacct's Comment field so the details page can
    # recover them even when the user picked a custom job_name that doesn't
    # match the unified regex.
    comment = f"phase={req.phase};variant={req.variant}"

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
        # Pin wandb run id to the slurm display name so the URL is stable
        # and matches MLXP's run-id format.
        f"WANDB_RUN_ID={shlex.quote(job_name)}"
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
