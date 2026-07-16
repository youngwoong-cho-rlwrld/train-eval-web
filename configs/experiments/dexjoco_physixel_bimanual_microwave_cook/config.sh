# Experiment: dexjoco_physixel_bimanual_microwave_cook
# DexJoCo bimanual_microwave_cook - physixel (dual-arm). Train + eval; pick the phase at submit.

# ───── model ─────
MODEL_ID=dexjoco-physixel
MODEL_VERSION=n1.6
DEXJOCO_GIT_COMMIT=6a6d1b2c28459aab6067b25bcd38003dfa491017
TRAIN_GIT_COMMIT=73f2aeb02220e430445af9e18051cadf6f2a9a9f
TRAIN_MODALITY_CONFIG=dexjoco_config_dual_arm.py            # n1.6 modality config (path relative to this dir)
TRAIN_ACTION_HORIZON=16
ACTION_HORIZON_MODE=modality
TRAIN_NOTE="DexJoCo bimanual_microwave_cook - physixel 256x256 (dual-arm) [physixel @73f2aeb]"

# ───── datasets ─────
export DATA_DIR="$HOME/workspace/dexjoco_n16/src_v30/dexjoco_lerobot_datasets"
DATASET_NAME=bimanual_microwave_cook

# ───── task (eval-time policy prompt) ─────
DEXJOCO_TASK=bimanual_microwave_cook
TASK_NAME=bimanual_microwave_cook
INSTRUCTION="Open the microwave door, place the food inside the microwave, close the door, and press the start button."

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
# eval: supply EVAL_CHECKPOINT = the finetune output dir for bimanual_microwave_cook
DEXJOCO_INFERENCE_MODE=blocking_overlap
DEXJOCO_ACTION_HORIZON=16
DEXJOCO_REPLAN_RATIO=0.5
