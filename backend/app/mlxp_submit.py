"""Render a k8s Job YAML for a gr00t training variant and `kubectl apply` it.

Mirrors the slurm submit flow conceptually:
  - load the variant config (DATASETS, MAX_STEPS, …)
  - render a Job YAML that runs gr00t_finetune.py against the user's MLXP DDN
  - apply it with `kubectl apply`, parse the returned Job name

Different from slurm:
  - no partition picker (k8s scheduler does its thing); user picks `num_gpus`
    and we map to CPU/memory per the Notion guide's table
  - the body script is inlined into the Job spec's `args` (no separate
    train_body.sh file synced to the cluster — DDN already has the gr00t repo)
  - logs/status come from `kubectl logs` / `kubectl get pod`, not slurm tools
"""

import asyncio
import json
import re
import shutil
from datetime import datetime

import yaml
from pydantic import BaseModel

from .variants import load_variant


# Per-GPU resource map (from the Notion MLXP guide section 3.1).
# Node total: CPU=112, memory=1760Gi, GPU=8.
_GPU_RESOURCES = {
    1: ("14",  "220Gi"),
    2: ("28",  "440Gi"),
    4: ("56",  "880Gi"),
    8: ("100", "1500Gi"),
}

SANCTIONED_NODE = "h200-03-w-3a18"
DDN_MOUNT = "/data"
USER_HOME_ON_DDN = "/data/youngwoong"
GR00T_DIR = f"{USER_HOME_ON_DDN}/workspace/gr00t"


class MlxpSubmitRequest(BaseModel):
    variant: str
    num_gpus: int = 2
    dataset_override: str | list[str] | None = None
    extra_args: list[str] = []
    wandb_secret: str = "youngwoong-wandb"


class MlxpSubmitResponse(BaseModel):
    job_name: str
    pod_name: str | None = None
    yaml: str
    apply_stdout: str


async def submit_mlxp(req: MlxpSubmitRequest) -> MlxpSubmitResponse:
    if shutil.which("kubectl") is None:
        raise RuntimeError("kubectl not found on PATH")
    if req.num_gpus not in _GPU_RESOURCES:
        raise ValueError(f"num_gpus must be one of {list(_GPU_RESOURCES)}, got {req.num_gpus}")

    variant = await load_variant(req.variant)
    cpu, mem = _GPU_RESOURCES[req.num_gpus]

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # k8s names: lowercase, alphanumeric + '-'. Replace '_' with '-'.
    safe_variant = re.sub(r"[^a-z0-9-]+", "-", req.variant.lower())
    job_name = f"youngwoong-train-{safe_variant}-{timestamp}"[:63]  # k8s name limit

    body_script = _render_body_script(variant, req)
    spec = _render_job_yaml(job_name, body_script, req.num_gpus, cpu, mem, req.wandb_secret)
    yaml_text = yaml.safe_dump(spec, sort_keys=False)

    proc = await asyncio.create_subprocess_exec(
        "kubectl", "apply", "-f", "-", "--validate=false", "-n", "p-rlwrld",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate(input=yaml_text.encode())
    if proc.returncode != 0:
        raise RuntimeError(f"kubectl apply failed: {stderr.decode(errors='replace').strip()}")

    return MlxpSubmitResponse(
        job_name=job_name,
        pod_name=None,
        yaml=yaml_text,
        apply_stdout=stdout.decode(errors="replace").strip(),
    )


def _render_body_script(variant, req: MlxpSubmitRequest) -> str:
    """Render the inline bash that the container will run.

    Renders a data_config.yaml on the fly from the variant's DATASETS array,
    then invokes torchrun → gr00t_finetune.py with the variant's hyperparams.
    """
    # Build the data_config rows.
    rows: list[str] = []
    datasets_decl = variant.arrays.get("DATASETS")
    if req.dataset_override is not None:
        # Override: list[str] for multi, str for single.
        if isinstance(req.dataset_override, list):
            datasets_decl = req.dataset_override
        else:
            datasets_decl = [
                f"{req.dataset_override}|{variant.vars.get('DATA_CONFIG', 'allex_thetwo_ck40_egostereo')}|1.0"
            ]
    if not datasets_decl:
        # Single-dataset variant: synthesize from DATASET_NAME + DATA_CONFIG.
        name = variant.vars.get("DATASET_NAME")
        cfg = variant.vars.get("DATA_CONFIG", "allex_thetwo_ck40_egostereo")
        if not name:
            raise ValueError(f"variant {variant.name} has no DATASETS and no DATASET_NAME")
        datasets_decl = [f"{name}|{cfg}|1.0"]

    yaml_rows = []
    for entry in datasets_decl:
        parts = entry.split("|", 2)
        if len(parts) != 3:
            raise ValueError(f"bad DATASETS entry (need name|cfg|weight): {entry!r}")
        name, cfg, weight = parts
        yaml_rows.append(
            f"    - path: {USER_HOME_ON_DDN}/datasets/{name}\n"
            f"      embodiment_tag: new_embodiment\n"
            f"      data_config: {cfg}\n"
            f"      weight: {weight}"
        )
    data_config_yaml = "train:\n  datasets:\n" + "\n".join(yaml_rows)

    max_steps = variant.vars.get("MAX_STEPS", "30000")
    save_steps = variant.vars.get("SAVE_STEPS", "1000")
    batch_size = variant.vars.get("TRAIN_BATCH_SIZE", "64")
    train_extra = " ".join(variant.arrays.get("TRAIN_EXTRA_ARGS") or [])
    user_extra = " ".join(req.extra_args)

    ckpt_dir = f"{USER_HOME_ON_DDN}/experiments/{variant.name}/checkpoints"

    # No leading indentation — keeps the embedded heredoc YAML well-formed.
    return f"""\
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export WANDB_PROJECT=gr00t
export NO_ALBUMENTATIONS_UPDATE=1
export TOKENIZERS_PARALLELISM=false

cd {GR00T_DIR}
source .venv/bin/activate

mkdir -p {ckpt_dir}

# Render data_config.yaml from variant config.
cat > /tmp/data_config.yaml <<'YAML_EOF'
{data_config_yaml}
YAML_EOF

# Auto-resume from latest checkpoint if any.
RESUME_FLAG=""
if compgen -G "{ckpt_dir}/checkpoint-*" > /dev/null; then
    echo "[mlxp] existing checkpoint detected — will resume"
    RESUME_FLAG="--resume"
fi

torchrun --nproc_per_node={req.num_gpus} scripts/gr00t_finetune.py \\
    --num-gpus {req.num_gpus} \\
    --batch-size {batch_size} \\
    --learning_rate 1e-4 \\
    --output-dir {ckpt_dir} \\
    --data-config /tmp/data_config.yaml \\
    --max-steps {max_steps} \\
    --save-steps {save_steps} \\
    --dataloader_num_workers 16 \\
    --dataloader-prefetch-factor 10 \\
    --video-backend torchcodec \\
    --report-to wandb \\
    --pin_memory \\
    --run_name "{variant.name}" \\
    --seed 42 \\
    $RESUME_FLAG {train_extra} {user_extra}
"""


def _render_job_yaml(job_name: str, body: str, num_gpus: int, cpu: str, mem: str, wandb_secret: str) -> dict:
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": job_name,
            "namespace": "p-rlwrld",
            "labels": {"owner": "youngwoong", "tool": "train-eval-web"},
        },
        "spec": {
            "ttlSecondsAfterFinished": 1800,
            "backoffLimit": 0,
            "template": {
                "metadata": {
                    "labels": {"owner": "youngwoong", "tool": "train-eval-web"},
                    "annotations": {
                        "mlx.navercorp.com/zone": "private-h200-rlwrld-0",
                        "sidecar.istio.io/inject": "false",
                    },
                },
                "spec": {
                    "restartPolicy": "Never",
                    "imagePullSecrets": [{"name": "mlxp-registry"}],
                    "volumes": [
                        {"name": "ddn", "persistentVolumeClaim": {"claimName": "ddn-rlwrld-shared"}},
                        {"name": "dshm", "emptyDir": {"medium": "Memory", "sizeLimit": "256Gi"}},
                    ],
                    "affinity": {
                        "nodeAffinity": {
                            "requiredDuringSchedulingIgnoredDuringExecution": {
                                "nodeSelectorTerms": [{
                                    "matchExpressions": [{
                                        "key": "kubernetes.io/hostname",
                                        "operator": "In",
                                        "values": [SANCTIONED_NODE],
                                    }],
                                }],
                            },
                        },
                    },
                    "containers": [{
                        "name": "main",
                        "image": "mlxp.kr.ncr.ntruss.com/rlwrld-gpu-base:latest",
                        "imagePullPolicy": "Always",
                        "command": ["/bin/bash", "-c"],
                        "args": [body],
                        "env": [{
                            "name": "WANDB_API_KEY",
                            "valueFrom": {
                                "secretKeyRef": {
                                    "name": wandb_secret,
                                    "key": "api-key",
                                    "optional": True,
                                },
                            },
                        }],
                        "volumeMounts": [
                            {"name": "ddn",  "mountPath": DDN_MOUNT},
                            {"name": "dshm", "mountPath": "/dev/shm"},
                        ],
                        "resources": {
                            "requests": {"cpu": cpu, "memory": mem, "nvidia.com/gpu": str(num_gpus)},
                            "limits":   {"cpu": cpu, "memory": mem, "nvidia.com/gpu": str(num_gpus)},
                        },
                    }],
                },
            },
        },
    }
