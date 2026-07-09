# ============================================================================
# Sample variant config — copy this whole directory to start your own.
#
#   cp -r configs/experiments/_sample configs/experiments/<your-variant-name>
#   $EDITOR configs/experiments/<your-variant-name>/config.sh
#
# Everything under configs/experiments/ is .gitignored EXCEPT _sample/, so your
# variant stays local. Variants whose name starts with `_` are hidden from the
# Submit dropdown (that's why this one doesn't show up).
#
# This file is plain bash, sourced on the cluster by the body scripts that the
# selected model points at (configs/models/<MODEL_ID>.env). Lines starting with
# `#` are comments. Arrays use bash `NAME=(a b c)` syntax.
#
# The active (uncommented) settings below are a complete, runnable **DexJoCo
# n1.6 single-task** eval+train variant. Alternative shapes — Isaac Sim eval,
# n1.5, and multi-task — are shown as commented blocks. Uncomment the one you
# want and comment out the conflicting active lines.
#
# Legend:  [REQUIRED]  must be set     [n1.6]/[n1.5]  family-specific
#          [isaac]/[dexjoco]  eval-harness-specific     otherwise optional
# ============================================================================


# ───────────────────────── model ─────────────────────────
# MODEL_ID selects configs/models/<id>.env, which fixes the model family, the
# train/eval body scripts, and the action-horizon mode. Available today:
#   n1.5              GR00T n1.5,      Isaac eval
#   n1.6              GR00T n1.6,      Isaac eval
#   physixel          PhysiXel (n1.6), Isaac eval
#   dexjoco-n16       GR00T n1.6,      DexJoCo (MuJoCo) eval
#   dexjoco-physixel  PhysiXel (n1.6), DexJoCo (MuJoCo) eval
#   dexjoco-pi05      pi0.5 baseline,  DexJoCo eval (eval-only, no training)
MODEL_ID=dexjoco-n16                 # [REQUIRED]
MODEL_VERSION=n1.6                   # [REQUIRED] runtime family: n1.5 or n1.6
TRAIN_NOTE="DexJoCo <task> — n1.6 single-arm — sample variant, EDIT ME"  # [REQUIRED] shown in the Jobs table / wandb


# ───────────────────────── datasets ─────────────────────────
# DATA_DIR is the root that dataset names resolve under. It defaults to
# ~/datasets on the cluster; override it if your LeRobot datasets live elsewhere.
export DATA_DIR="$HOME/datasets"
#
# Choose ONE dataset mode for your model family:
#
#   n1.6 single-task : DATASET_NAME=<name>                 (dir under $DATA_DIR)
#   n1.6 multi-task  : TRAIN_DATASET_NAMES=(name1 name2 …)
#   n1.5 single-task : DATASET_NAME=<name> + DATA_CONFIG=<cfg>
#   n1.5 multi-task  : DATASETS=("name|data_config|weight" …)
#
# (n1.6 does not use DATA_CONFIG — the modality config below defines the data
#  layout instead. DATA_CONFIG is an n1.5-only knob.)

# --- active: n1.6 single-task ---
DATASET_NAME=my_task_dataset

# --- alt: n1.6 multi-task (comment out DATASET_NAME above) ---
# TRAIN_DATASET_NAMES=(my_task_a my_task_b)
# Optional per-dataset embodiment tags, parallel to TRAIN_DATASET_NAMES, for
# mixed-embodiment training. Length must match TRAIN_DATASET_NAMES exactly.
# TRAIN_DATASET_EMBODIMENT_TAGS=(new_embodiment new_embodiment)

# --- alt: n1.5 single-task ---
# DATASET_NAME=my_task_dataset
# DATA_CONFIG=allex_thetwo_ck40_egostereo    # [n1.5] a data_config known to the model repo

# --- alt: n1.5 multi-task (name|data_config|weight) ---
# DATASETS=(
#     "my_task_a|allex_thetwo_ck40_egostereo|1.0"
#     "my_task_b|allex_thetwo_ck40_egostereo|1.0"
# )


# ───────────────────────── modality config (n1.6 only) ─────────────────────────
# [n1.6][REQUIRED] Path (relative to this dir) of the Python modality config that
# declares the video/state/action/language keys and the action representation.
# Both training and Isaac eval load it. Edit modality_config.py for your
# embodiment/dataset. (n1.5 ignores this — it uses DATA_CONFIG instead.)
TRAIN_MODALITY_CONFIG=modality_config.py


# ───────────────────────── training ─────────────────────────
MAX_STEPS=10000
SAVE_STEPS=10000                     # checkpoint frequency; == MAX_STEPS means save only at the end
TRAIN_NUM_GPUS=4                     # slurm world size; on MLXP the Submit-page GPU picker overrides this
TRAIN_GLOBAL_BATCH_SIZE=128          # [n1.6] total batch across GPUs (must be divisible by TRAIN_NUM_GPUS)
# TRAIN_BATCH_SIZE=64                 # [n1.5] per-GPU batch (n1.5 uses this instead of GLOBAL)
TRAIN_ACTION_HORIZON=16              # [n1.6] number of future action steps predicted
ACTION_HORIZON_MODE=modality         # [n1.6] how the horizon is applied — must match the model:
                                     #   modality          → written into the modality config (n1.6, dexjoco-n16)
                                     #   modality_and_cli  → modality config + a --action-horizon CLI flag (physixel)
                                     #   cli               → CLI flag only
                                     #   none              → not tunable (e.g. dexjoco-pi05)
TRAIN_EXTRA_ARGS=()                  # extra flags passed verbatim to the trainer, e.g. (--tune-visual)
# TRAIN_GIT_COMMIT=abc1234           # optional: pin the model-code repo to an exact commit for reproducibility


# ───────────────────────── eval ─────────────────────────
# Pick the harness that matches your MODEL_ID:
#   isaac   (default) → Isaac Sim + eval_body.sh               (n1.5, n1.6, physixel)
#   dexjoco           → MuJoCo benchmark + eval_body_dexjoco.sh (dexjoco-*)
EVAL_HARNESS=dexjoco                  # [dexjoco] omit or set "isaac" for Isaac Sim variants

N_EPISODES=50                         # episodes per eval run
N_RUNS=1                              # eval runs per (task, eval_set); results are averaged across runs
EVAL_BASE_SEED=0                      # base RNG seed (isaac default 42, dexjoco default 0)

# --- DexJoCo eval knobs ([dexjoco] harness) ---
DEXJOCO_TASK=my_task                  # [dexjoco][REQUIRED single-task] config stem: $DEXJOCO_DIR/configs/<family>/<task>.yaml
DEXJOCO_SERVER_TYPE=groot             # groot (a GR00T/PhysiXel checkpoint) or openpi (pi0.5 baseline)
INSTRUCTION="Grasp the object and complete the task."   # policy-server language prompt
EVAL_SETS=(rand_obj)                  # DexJoCo config families to sweep: rand_obj rand_full multi_task ipad_reasoning
# DEXJOCO_IMAGE_SIZE=224              # optional: override the server's input image size (e.g. 224 vs 256)
# DEXJOCO_EMBODIMENT_TAG=new_embodiment          # embodiment tag handed to the groot server (single-arm tasks)
# DEXJOCO_EMBODIMENT_TAG_BIMANUAL=new_embodiment # tag used for bimanual_* tasks in a multi-task run

# --- alt: Isaac Sim eval ([isaac] harness — set EVAL_HARNESS=isaac or remove it) ---
# TASK_NAME=task-Cube_Box-5cmLeft                          # [isaac] eval task id
# INSTRUCTION="Pick up the cube and place it in the box"   # [isaac] policy prompt
# EXECUTION_HORIZON=8                                       # [isaac] action steps executed per inference
# MAX_EPISODE_STEPS=300                                     # [isaac] hard episode length cap
# EVAL_SETS=(0cm 1cm 3cm 5cm 7cm)                           # [isaac] object-offset variants to sweep
# EVAL_NUM_ENVS_PER_GPU=1            # ALLEX eval runs one Isaac env per GPU (leave at 1)
# EVAL_PIN_CUDA_DEVICES=1
# EVAL_PIN_CLIENT_CUDA_DEVICES=1
# EVAL_UNSET_CUDA_VISIBLE_DEVICES_FOR_SERVER=0

# --- alt: multi-task eval matrix (both harnesses) ---
# Runs every task in a single job. Format: "short_label|task_name|instruction".
# Drop DEXJOCO_TASK / TASK_NAME above when using this.
# TASKS=(
#     "water_plant|water_plant|Grasp the watering can and water the plant."
#     "click_mouse|click_mouse|Move the mouse to the pad and click the left button."
# )
