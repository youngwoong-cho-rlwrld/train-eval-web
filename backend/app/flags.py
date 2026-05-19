"""Render the exact flag list a job's training/eval entrypoint receives.

The body scripts in `lib/` are the source of truth — this module mirrors
their flag set so the UI can show the same list of flags the body would
actually emit for a given variant. If you change lib/train_body*.sh,
update the corresponding builder here.
"""

from typing import Any

from .variants import Variant
from .wandb_config import get_project


def flags_for(variant: Variant, cluster: str, phase: str) -> list[tuple[str, str]]:
    """Return [(flag, value), ...] for the variant's entrypoint.

    Pseudo-values like `<job-name>` mark fields the submitter fills in at
    submit time rather than reading from the variant.
    """
    model = (variant.vars.get("MODEL_VERSION") or "n1.5").strip()
    if phase in ("train", "resume"):
        if model == "n1.6":
            return _train_n16(variant, cluster)
        return _train_n15(variant, cluster)
    if phase == "eval":
        if model == "n1.6":
            return _eval_n16(variant, cluster)
        return _eval_n15(variant, cluster)
    return []


# ── train ─────────────────────────────────────────────────────────────

def _train_n15(v: Variant, cluster: str) -> list[tuple[str, str]]:
    """scripts/gr00t_finetune.py — N1.5 training, called from train_body.sh."""
    return [
        ("--num-gpus", v.vars.get("TRAIN_NUM_GPUS", "")),
        ("--batch-size", v.vars.get("TRAIN_BATCH_SIZE", "")),
        ("--learning_rate", "1e-4"),
        ("--output-dir", "$EXP_DIR/checkpoints"),
        ("--data-config", "$EXP_DIR/data_config.yaml"),
        ("--max-steps", v.vars.get("MAX_STEPS", "")),
        ("--save-steps", v.vars.get("SAVE_STEPS", "")),
        ("--dataloader_num_workers", "16"),
        ("--dataloader-prefetch-factor", "10"),
        ("--video-backend", "torchcodec"),
        ("--resume", "(if checkpoint exists)"),
        ("--report-to", "wandb"),
        ("--pin_memory", ""),
        ("--run_name", "<job-name>"),
        ("--seed", "42"),
        *[(a, "") for a in (v.arrays.get("TRAIN_EXTRA_ARGS") or [])],
    ]


def _train_n16(v: Variant, cluster: str) -> list[tuple[str, str]]:
    """gr00t/experiment/launch_finetune.py — N1.6 training."""
    try:
        nb = int(v.vars.get("TRAIN_NUM_GPUS", "0")) * int(v.vars.get("TRAIN_BATCH_SIZE", "0"))
        global_batch = str(nb) if nb > 0 else ""
    except ValueError:
        global_batch = ""
    names = v.arrays.get("TRAIN_DATASET_NAMES") or (
        [v.vars["DATASET_NAME"]] if "DATASET_NAME" in v.vars else []
    )
    dataset_paths = " ".join(f"$DATA_DIR/{n}" for n in names)
    modality_file = v.vars.get("TRAIN_MODALITY_CONFIG", "")
    out: list[tuple[str, str]] = [
        ("--base-model-path", "nvidia/GR00T-N1.6-3B"),
        ("--dataset-path", dataset_paths),
        ("--embodiment-tag", "NEW_EMBODIMENT"),
        ("--modality-config-path", f"$EXP_DIR/{modality_file}" if modality_file else ""),
        ("--num-gpus", v.vars.get("TRAIN_NUM_GPUS", "")),
        ("--output-dir", "$EXP_DIR/checkpoints"),
        ("--global-batch-size", global_batch),
        ("--learning-rate", "1e-4"),
        ("--max-steps", v.vars.get("MAX_STEPS", "")),
        ("--save-steps", v.vars.get("SAVE_STEPS", "")),
        ("--save-total-limit", "5"),
        ("--dataloader-num-workers", "8"),
        ("--experiment-name", "<job-name>"),
        ("--use-wandb", ""),
        ("--color-jitter-params", "brightness 0.2 contrast 0.2 saturation 0.2 hue 0.1"),
    ]
    if cluster == "mlxp":
        out.append(("--wandb-project", get_project()))
    out.extend((a, "") for a in (v.arrays.get("TRAIN_EXTRA_ARGS") or []))
    return out


# ── eval (currently slurm-only — mlxp eval not wired yet) ────────────

def _eval_n15(v: Variant, cluster: str) -> list[tuple[str, str]]:
    """Mirror lib/eval_body.sh — gr00t inference + isaac client run."""
    return [
        ("--task-name", v.vars.get("TASK_NAME", "")),
        ("--instruction", v.vars.get("INSTRUCTION", "")),
        ("--n-episodes", v.vars.get("N_EPISODES", "")),
        ("--n-runs", v.vars.get("N_RUNS", "")),
        ("EVAL_NUM_ENVS_PER_GPU", "1"),
        ("--execution-horizon", v.vars.get("EXECUTION_HORIZON", "")),
        ("--max-episode-steps", v.vars.get("MAX_EPISODE_STEPS", "")),
        ("(eval_sets)", " ".join(v.arrays.get("EVAL_SETS") or [])),
    ]


def _eval_n16(v: Variant, cluster: str) -> list[tuple[str, str]]:
    return _eval_n15(v, cluster)
