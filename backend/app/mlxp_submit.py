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

from .paths import EXPERIMENTS_DIR
from .variants import load_variant


# Per-GPU resource map (from the Notion MLXP guide section 3.1).
# Node total: CPU=112, memory=1760Gi, GPU=8.
_GPU_RESOURCES = {
    1: ("14",  "220Gi"),
    2: ("28",  "440Gi"),
    4: ("56",  "880Gi"),
    8: ("100", "1500Gi"),
}

DEFAULT_NODE = "h200-03-w-3a18"
DDN_MOUNT = "/data"
USER_HOME_ON_DDN = "/data/youngwoong"
GR00T_DIR = f"{USER_HOME_ON_DDN}/workspace/gr00t"
GR00T_N16_DIR = f"{USER_HOME_ON_DDN}/workspace/gr00t-n16"


class MlxpSubmitRequest(BaseModel):
    variant: str
    num_gpus: int = 2
    # The k8s node to pin via nodeAffinity. Each rlwrld team member is
    # sanctioned for a specific node (see the GPU Resource Schedule sheet);
    # using anyone else's node has triggered admin deletions in the past.
    # Leave None to fall back to DEFAULT_NODE.
    node: str | None = None
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

    node = req.node or DEFAULT_NODE
    body_script = _render_body_script(variant, req, job_name)
    spec = _render_job_yaml(job_name, body_script, req.num_gpus, cpu, mem, req.wandb_secret, node)
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


def _render_body_script(variant, req: MlxpSubmitRequest, job_name: str) -> str:
    """Render the inline bash the container runs.

    Resolves the variant's dataset list, then dispatches to the right gr00t
    entrypoint based on MODEL_VERSION:
      - n1.5 → gr00t_finetune.py with /tmp/data_config.yaml
      - n1.6 → launch_finetune.py with --dataset-path + --modality-config-path
    """
    model = (variant.vars.get("MODEL_VERSION") or "n1.5").strip()

    # ── Resolve dataset name list (model-agnostic) ──
    names: list[str] = []
    override = req.dataset_override
    if override is not None:
        if isinstance(override, list):
            # Either "name" or "name|cfg|weight" entries.
            names = [e.split("|", 1)[0] for e in override]
            override_full = override  # preserve N1.5 cfg/weight if present
        else:
            names = [override]
            override_full = [override]
    else:
        if variant.arrays.get("TRAIN_DATASET_NAMES"):
            names = list(variant.arrays["TRAIN_DATASET_NAMES"])
            override_full = None
        elif variant.arrays.get("DATASETS"):
            names = [e.split("|", 1)[0] for e in variant.arrays["DATASETS"]]
            override_full = None
        elif variant.vars.get("DATASET_NAME"):
            names = [variant.vars["DATASET_NAME"]]
            override_full = None
        else:
            raise ValueError(
                f"variant {variant.name} has no DATASET_NAME / DATASETS / TRAIN_DATASET_NAMES"
            )

    max_steps = variant.vars.get("MAX_STEPS", "30000")
    save_steps = variant.vars.get("SAVE_STEPS", "1000")
    batch_size = variant.vars.get("TRAIN_BATCH_SIZE", "64")
    train_extra = " ".join(variant.arrays.get("TRAIN_EXTRA_ARGS") or [])
    user_extra = " ".join(req.extra_args)

    ckpt_dir = f"{USER_HOME_ON_DDN}/experiments/{variant.name}/checkpoints"

    if model == "n1.6":
        return _render_body_n16(
            variant=variant, req=req, job_name=job_name, names=names,
            max_steps=max_steps, save_steps=save_steps, batch_size=batch_size,
            train_extra=train_extra, user_extra=user_extra, ckpt_dir=ckpt_dir,
        )

    # ── N1.5: build the data_config.yaml rows ──
    if override_full is not None and isinstance(override, list) and any("|" in e for e in override):
        datasets_decl = override_full
    elif override_full is not None and isinstance(override, str):
        cfg = variant.vars.get("DATA_CONFIG", "allex_thetwo_ck40_egostereo")
        datasets_decl = [f"{override}|{cfg}|1.0"]
    elif variant.arrays.get("DATASETS"):
        datasets_decl = variant.arrays["DATASETS"]
    else:
        cfg = variant.vars.get("DATA_CONFIG", "allex_thetwo_ck40_egostereo")
        datasets_decl = [f"{names[0]}|{cfg}|1.0"]

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

    # No leading indentation — keeps the embedded heredoc YAML well-formed.
    return f"""\
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export WANDB_PROJECT=gr00t
# Pin the wandb run-id to the k8s Job name so requeues continue the same
# run (HF Trainer otherwise spawns a fresh run on each container start).
export WANDB_RUN_ID="{job_name}"
export WANDB_RESUME=allow
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


def _render_body_n16(*, variant, req: MlxpSubmitRequest, job_name: str,
                     names: list[str], max_steps: str, save_steps: str,
                     batch_size: str, train_extra: str, user_extra: str,
                     ckpt_dir: str) -> str:
    """Body script for GR00T N1.6 (launch_finetune.py).

    Unlike N1.5, N1.6 takes --dataset-path (multiple) + --modality-config-path
    (a Python file). We inline the modality config from the local variant
    directory so MLXP doesn't need a rsync step.
    """
    modality_rel = variant.vars.get("TRAIN_MODALITY_CONFIG")
    if not modality_rel:
        raise ValueError(f"variant {variant.name}: TRAIN_MODALITY_CONFIG missing")
    modality_path = EXPERIMENTS_DIR / variant.name / modality_rel
    if not modality_path.is_file():
        raise FileNotFoundError(f"modality config not found: {modality_path}")
    modality_text = modality_path.read_text()

    dataset_paths_arg = " \\\n        ".join(
        f"{USER_HOME_ON_DDN}/datasets/{n}" for n in names
    )
    global_batch = int(batch_size) * req.num_gpus

    return f"""\
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export WANDB_PROJECT=gr00t
export WANDB_RUN_ID="{job_name}"
export WANDB_RESUME=allow
export NO_ALBUMENTATIONS_UPDATE=1
export TOKENIZERS_PARALLELISM=false
export OMNI_KIT_ACCEPT_EULA=Y

cd {GR00T_N16_DIR}

mkdir -p {ckpt_dir}

cat > /tmp/modality_config.py <<'PY_EOF'
{modality_text}
PY_EOF

RESUME_FLAG=""
if compgen -G "{ckpt_dir}/checkpoint-*" > /dev/null; then
    echo "[mlxp] existing checkpoint detected — will resume"
    RESUME_FLAG="--resume"
fi

uv run torchrun --nproc_per_node={req.num_gpus} gr00t/experiment/launch_finetune.py \\
    --base-model-path nvidia/GR00T-N1.6-3B \\
    --dataset-path \\
        {dataset_paths_arg} \\
    --embodiment-tag NEW_EMBODIMENT \\
    --modality-config-path /tmp/modality_config.py \\
    --num-gpus {req.num_gpus} \\
    --output-dir {ckpt_dir} \\
    --global-batch-size {global_batch} \\
    --learning-rate 1e-4 \\
    --max-steps {max_steps} \\
    --save-steps {save_steps} \\
    --save-total-limit 5 \\
    --dataloader-num-workers 8 \\
    --experiment-name "{job_name}" \\
    --use-wandb \\
    $RESUME_FLAG {train_extra} {user_extra}
"""


def _render_job_yaml(job_name: str, body: str, num_gpus: int, cpu: str, mem: str,
                     wandb_secret: str, node: str) -> dict:
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
                                        "values": [node],
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
