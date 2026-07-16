# Experiment: dexjoco_gam_single_arm_6tasks_224
# DexJoCo single-arm multitask GAM fine-tune at 224x224.

# ----- model -----
MODEL_ID=dexjoco-gam
MODEL_VERSION=gam
TRAIN_GIT_COMMIT=69afa536658198a22750b5618322edf68fdea93a
# GAM training config (second file, owned by the GAM-port workstream). Lists
# datasets/weights/dims and the action chunk size; consumed via GAM_CONFIG_YAML.
TRAIN_MODALITY_CONFIG=gam_config.yaml
TRAIN_ACTION_HORIZON=16
TRAIN_NOTE="DexJoCo 6-task single-arm multitask - GAM 224x224, state/action 23/22"

# ----- datasets -----
export DATA_DIR="$HOME/workspace/dexjoco_n16/src_v30/dexjoco_lerobot_datasets"

# Informational for GAM: the authoritative dataset list/weights/dims live in
# gam_config.yaml. GAM's train wrapper does not read these bash arrays.
TRAIN_DATASET_NAMES=(
    click_mouse
    fold_glasses
    hammer_nail
    pick_bucket
    pinch_tongs
    water_plant
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
MAX_STEPS=30000
SAVE_STEPS=10000
TRAIN_NUM_GPUS=8
TRAIN_GLOBAL_BATCH_SIZE=512

# ----- eval (DexJoCo MuJoCo harness) -----
EVAL_HARNESS=dexjoco
DEXJOCO_SERVER_TYPE=gam
DEXJOCO_IMAGE_SIZE=224
DEXJOCO_EMBODIMENT_TAG=dexjoco_single_arm
EVAL_NUM_GPUS=4
N_EPISODES=50
N_RUNS=3
EVAL_BASE_SEED=0
EVAL_SETS=(rand_obj)
DEXJOCO_INFERENCE_MODE=blocking_overlap
DEXJOCO_ACTION_HORIZON=16
DEXJOCO_REPLAN_RATIO=0.5
