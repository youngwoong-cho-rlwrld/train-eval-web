# Sample variant config.
#
# Copy this directory to configs/experiments/<your-variant-name>/ and edit.
# Everything under configs/experiments/ is .gitignored EXCEPT _sample/, so
# your variant stays out of git. To share a variant with the team, commit it
# inside _sample/ as a reference, or check it into a separate repo.
#
# This file is sourced as bash; lines starting with `#` are comments.
# Variants that begin with `_` are hidden from the Submit dropdown.


# ───── model ─────
MODEL_VERSION=n1.5                  # n1.5 (gr00t train_body.sh) | n1.6 (train_body_n16.sh)
TRAIN_NOTE="N1.5 single-task @ 480x640 — sample variant, edit me"


# ───── datasets ─────
# Pick ONE of the two modes:
#   (a) Single-task → uncomment DATASET_NAME + DATA_CONFIG below.
#   (b) Multi-task  → uncomment the DATASETS array instead.
#
# DATA_DIR resolves to ~/datasets; keep each cluster's dataset symlink tree
# or DATA_DIR override pointed at the same logical dataset names.
export DATA_DIR="$HOME/datasets"

# (a) Single-task:
DATASET_NAME=v4_cube_box_5cm_left_480
DATA_CONFIG=allex_thetwo_ck40_egostereo

# (b) Multi-task — uncomment + drop DATASET_NAME above:
# DATASETS=(
#     "v4_cube_box_5cm_left_480|allex_thetwo_ck40_egostereo|1.0"
#     "v4_cube_stack_3cm_right_480|allex_thetwo_ck40_egostereo|1.0"
# )


# ───── task (eval-time policy prompt) ─────
TASK_NAME=task-Cube_Box-5cmLeft
INSTRUCTION="Pick up the cube with your left hand and place it in the box"


# ───── training ─────
MAX_STEPS=30000
SAVE_STEPS=1000                     # checkpoint frequency; smaller = lower preempt-loss
TRAIN_NUM_GPUS=2                    # for slurm. mlxp uses the Submit-page picker instead.
TRAIN_BATCH_SIZE=64                 # per-GPU
TRAIN_EXTRA_ARGS=()                 # e.g. (--tune-visual --random-diffusion)


# ───── eval ─────
N_EPISODES=70
EXECUTION_HORIZON=8
MAX_EPISODE_STEPS=300
N_RUNS=3                            # eval runs per (task, eval_set) combination
EVAL_NUM_ENVS_PER_GPU=1             # ALLEX eval currently runs one Isaac env per GPU
EVAL_PIN_CUDA_DEVICES=1
EVAL_UNSET_CUDA_VISIBLE_DEVICES_FOR_SERVER=0
EVAL_PIN_CLIENT_CUDA_DEVICES=1
EVAL_SETS=(0cm 1cm 3cm 5cm 7cm)
