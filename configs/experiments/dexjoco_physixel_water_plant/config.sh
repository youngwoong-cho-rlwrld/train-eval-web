# Experiment: dexjoco_physixel_water_plant
# DexJoCo water_plant - physixel (single-arm). Train + eval; pick the phase at submit.

# ───── model ─────
MODEL_ID=dexjoco-physixel
MODEL_VERSION=n1.6
TRAIN_GIT_COMMIT=73f2aeb02220e430445af9e18051cadf6f2a9a9f
TRAIN_MODALITY_CONFIG=dexjoco_config.py            # n1.6 modality config (path relative to this dir)
TRAIN_ACTION_HORIZON=16
ACTION_HORIZON_MODE=modality
TRAIN_NOTE="DexJoCo water_plant - physixel 256x256 (single-arm) [physixel @73f2aeb]"

# ───── datasets ─────
export DATA_DIR="$HOME/workspace/dexjoco_n16/src_v30/dexjoco_lerobot_datasets"
DATASET_NAME=water_plant

# ───── task (eval-time policy prompt) ─────
DEXJOCO_TASK=water_plant
TASK_NAME=water_plant
INSTRUCTION="Grasp the watering can and apply water to the plant."

# ───── training ─────
MAX_STEPS=10000
SAVE_STEPS=10000                      # save only at the end
TRAIN_NUM_GPUS=4
TRAIN_GLOBAL_BATCH_SIZE=128
TRAIN_EXTRA_ARGS=()

# ───── eval (DexJoCo MuJoCo harness) ─────
EVAL_HARNESS=dexjoco
DEXJOCO_SERVER_TYPE=groot
DEXJOCO_IMAGE_SIZE=256
EVAL_NUM_GPUS=4
N_EPISODES=50
N_RUNS=1
EVAL_BASE_SEED=0
EVAL_SETS=(rand_obj)
# eval: supply EVAL_CHECKPOINT = the finetune output dir for water_plant
DEXJOCO_INFERENCE_MODE=blocking_overlap
DEXJOCO_ACTION_HORIZON=16
DEXJOCO_REPLAN_RATIO=0.5
