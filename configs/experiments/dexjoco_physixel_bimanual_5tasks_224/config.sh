# Experiment: dexjoco_physixel_bimanual_5tasks_224
# DexJoCo bimanual multitask PhysiXel/N1.6 fine-tune at 224x224.

# ----- model -----
MODEL_ID=dexjoco-physixel
MODEL_VERSION=n1.6
DEXJOCO_GIT_COMMIT=6a6d1b2c28459aab6067b25bcd38003dfa491017
TRAIN_GIT_COMMIT=9faf40b35770763f4c7650db2094b66cb4328918
TRAIN_MODALITY_CONFIG=dexjoco_config_front.py
TRAIN_ACTION_HORIZON=16
ACTION_HORIZON_MODE=modality
TRAIN_NOTE="DexJoCo 5-task bimanual multitask - physixel 224x224, state/action 46/44"

# ----- datasets -----
export DATA_DIR="$HOME/workspace/dexjoco_n16/src_v30/dexjoco_lerobot_datasets"

TRAIN_DATASET_NAMES=(
    bimanual_assembly
    bimanual_hanoi
    bimanual_microwave_cook
    bimanual_photograph
    bimanual_unlock_ipad
)

TRAIN_DATASET_EMBODIMENT_TAGS=(
    DEXJOCO_DUAL_ARM
    DEXJOCO_DUAL_ARM
    DEXJOCO_DUAL_ARM
    DEXJOCO_DUAL_ARM
    DEXJOCO_DUAL_ARM
)

# ----- tasks -----
TASKS=(
    "bimanual_assembly|bimanual_assembly|Grasp the tray with the left hand and the peg with the right hand, then insert the peg into the hole."
    "bimanual_hanoi|bimanual_hanoi|Execute the final two moves of the three-level Tower of Hanoi: move the medium disk from the middle peg to the right peg with the right hand, then move the small disk from the left peg to the right peg with the left hand."
    "bimanual_microwave_cook|bimanual_microwave_cook|Open the microwave door, place the food inside the microwave, close the door, and press the start button."
    "bimanual_photograph|bimanual_photograph|Grasp the camera with the left hand, align it with the logo, and press the shutter button with the right hand."
    "bimanual_unlock_ipad|bimanual_unlock_ipad|Grasp the iPad and enter the password 123 to unlock the device."
)

DEXJOCO_TASK=bimanual_assembly
TASK_NAME=bimanual_assembly
INSTRUCTION="Grasp the tray with the left hand and the peg with the right hand, then insert the peg into the hole."

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
DEXJOCO_EMBODIMENT_TAG=dexjoco_dual_arm
DEXJOCO_EMBODIMENT_TAG_BIMANUAL=dexjoco_dual_arm
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
