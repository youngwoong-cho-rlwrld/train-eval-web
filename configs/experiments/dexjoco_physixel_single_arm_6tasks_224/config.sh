# Experiment: dexjoco_physixel_single_arm_6tasks_224
# DexJoCo single-arm multitask PhysiXel/N1.6 fine-tune at 224x224.

# ----- model -----
MODEL_ID=dexjoco-physixel
MODEL_VERSION=n1.6
DEXJOCO_GIT_COMMIT=6a6d1b2c28459aab6067b25bcd38003dfa491017
TRAIN_GIT_COMMIT=9faf40b35770763f4c7650db2094b66cb4328918
TRAIN_MODALITY_CONFIG=dexjoco_config_front.py
TRAIN_ACTION_HORIZON=16
ACTION_HORIZON_MODE=modality
TRAIN_NOTE="DexJoCo 6-task single-arm multitask - physixel 224x224, state/action 23/22"

# ----- datasets -----
export DATA_DIR="$HOME/workspace/dexjoco_n16/src_v30/dexjoco_lerobot_datasets"

TRAIN_DATASET_NAMES=(
    click_mouse
    fold_glasses
    hammer_nail
    pick_bucket
    pinch_tongs
    water_plant
)

TRAIN_DATASET_EMBODIMENT_TAGS=(
    DEXJOCO_SINGLE_ARM
    DEXJOCO_SINGLE_ARM
    DEXJOCO_SINGLE_ARM
    DEXJOCO_SINGLE_ARM
    DEXJOCO_SINGLE_ARM
    DEXJOCO_SINGLE_ARM
)

# ----- tasks -----
TASKS=(
    "click_mouse|click_mouse|Move the mouse to the purple mouse pad and click the left mouse button."
    "fold_glasses|fold_glasses|Fold the glasses and place them into the case."
    "hammer_nail|hammer_nail|Use the hammer to drive the nail into the wooden board."
    "pick_bucket|pick_bucket|Place the boxed food into the bucket and then lift the bucket."
    "pinch_tongs|pinch_tongs|Grasp the tongs and perform three consecutive open-close motions."
    "water_plant|water_plant|Grasp the watering can and apply water to the plant."
)

DEXJOCO_TASK=water_plant
TASK_NAME=water_plant
INSTRUCTION="Grasp the watering can and apply water to the plant."

# ----- training -----
MAX_STEPS=10000
SAVE_STEPS=10000
TRAIN_NUM_GPUS=2
TRAIN_GLOBAL_BATCH_SIZE=128
TRAIN_EXTRA_ARGS=(--shortest-image-edge 224 --crop-fraction 1.0)

# ----- eval (DexJoCo MuJoCo harness) -----
EVAL_HARNESS=dexjoco
DEXJOCO_SERVER_TYPE=groot
DEXJOCO_IMAGE_SIZE=224
DEXJOCO_EMBODIMENT_TAG=dexjoco_single_arm
EVAL_NUM_GPUS=4
N_EPISODES=50
N_RUNS=3
EVAL_BASE_SEED=0
EVAL_SETS=(rand_obj)
EVAL_SIM_START_STAGGER_SECONDS=300
DEXJOCO_MAX_CONCURRENT_UNITS=8
DEXJOCO_INFERENCE_MODE=blocking_overlap
DEXJOCO_ACTION_HORIZON=16
DEXJOCO_REPLAN_RATIO=0.5
