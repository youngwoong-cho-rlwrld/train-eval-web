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

    - If `override` is a string, replace the DATASET_NAME=... line (single-task).
    - If `override` is a list of "name|cfg|weight" strings, replace the
      DATASETS=( ... ) block (multi-task).
    """
    if isinstance(override, str):
        new_line = f"DATASET_NAME={override}"
        return re.sub(
            r"^(\s*export\s+)?DATASET_NAME=.*$",
            lambda m: (m.group(1) or "") + new_line,
            config_text,
            flags=re.MULTILINE,
        )

    # Array override.
    new_block_lines = ["DATASETS=("]
    new_block_lines.extend(f'    "{entry}"' for entry in override)
    new_block_lines.append(")")
    new_block = "\n".join(new_block_lines)
    return re.sub(
        r"^DATASETS=\(.*?^\)\s*$",
        new_block,
        config_text,
        count=1,
        flags=re.MULTILINE | re.DOTALL,
    )


class SubmitRequest(BaseModel):
    cluster: str
    variant: str
    phase: str                  # "train" | "resume" | "eval"
    partition: str | None = None  # if None, fall back to cluster.env PARTITION
    # Per-submit dataset override. Two shapes accepted:
    #   - single string  → replaces DATASET_NAME in single-task variants
    #   - list of "name|cfg|weight" entries → replaces DATASETS array
    # None means "use whatever the variant config.sh says".
    dataset_override: str | list[str] | None = None
    extra_args: list[str] = []


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
    job_name = f"{req.phase}_{req.variant}_{req.cluster}_{partition}_{datetime.now():%Y%m%d_%H%M%S}"
    log_dir = cluster.vars["LOG_DIR"]
    resume_expected = "1" if req.phase == "resume" else "0"

    body_path = f"$HOME/{CLUSTER_STAGING_REL}/lib/{body_script}"
    repo_root_remote = f"$HOME/{CLUSTER_STAGING_REL}"

    sbatch_parts = [
        "/opt/slurm/bin/sbatch",
        f"--job-name={shlex.quote(job_name)}",
        f"--partition={shlex.quote(partition)}",
        "--nodes=1",
        f"--gpus-per-node={shlex.quote(gpus)}",
        f"--time={shlex.quote(walltime)}",
        f"--output={log_dir}/{job_name}_%j.out",
        f"--error={log_dir}/{job_name}_%j.err",
        f"--export=ALL,VARIANT={shlex.quote(req.variant)},CLUSTER={shlex.quote(req.cluster)},"
        f"REPO_ROOT={repo_root_remote},RESUME_EXPECTED={resume_expected}",
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

    return SubmitResponse(
        job_id=job_id,
        job_name=job_name,
        partition=partition,
        sbatch_cmd=sbatch_cmd,
        rsync_stdout="\n".join(r.stdout for r in rsync_results),
        sbatch_stdout=sb.stdout,
    )
