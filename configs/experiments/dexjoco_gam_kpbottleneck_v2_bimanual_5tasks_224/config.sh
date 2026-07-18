# Experiment: dexjoco_gam_kpbottleneck_v2_bimanual_5tasks_224
# DexJoCo bimanual multitask GAM fine-tune at 224x224 with the keypoint action
# bottleneck (commit aed7a32): action tokens predict a 126-D hand-keypoint
# trajectory (42 points x 3, front-camera frame) decoded to joint actions.
# Datasets are the keypoint-augmented copies (*_kp) of the 5 bimanual tasks.

# ----- model -----
MODEL_ID=dexjoco-gam
TRAIN_GIT_COMMIT=e47b919f75d9b3e4918f6d80ea2f029e00cb8779
# GAM training config (second file, owned by the GAM-port workstream). Lists
# datasets/weights/dims and the action chunk size; consumed via GAM_CONFIG_YAML.
TRAIN_MODALITY_CONFIG=gam_config.yaml
TRAIN_ACTION_HORIZON=16
TRAIN_NOTE="DexJoCo 5-task bimanual GAM + keypoint bottleneck v2 (normalized kp targets, non-detached co-adaptation lambda 0.25 w warmup, anchor 0.5; commit e47b919)"

# ----- datasets -----
export DATA_DIR="$HOME/workspace/dexjoco_n16/src_v30/dexjoco_lerobot_datasets_kp"

# Informational for GAM: the authoritative dataset list/weights/dims live in
# gam_config.yaml. GAM's train wrapper does not read these bash arrays.
TRAIN_DATASET_NAMES=(
    bimanual_assembly_kp
    bimanual_hanoi_kp
    bimanual_microwave_cook_kp
    bimanual_photograph_kp
    bimanual_unlock_ipad_kp
)

# ----- tasks -----
TASKS=(
    "bimanual_assembly|bimanual_assembly|Grasp the tray with the left hand and the peg with the right hand, then insert the peg into the hole."
    "bimanual_hanoi|bimanual_hanoi|Execute the final two moves of the three-level Tower of Hanoi: move the medium disk from the middle peg to the right peg with the right hand, then move the small disk from the left peg to the right peg with the left hand."
    "bimanual_microwave_cook|bimanual_microwave_cook|Open the microwave door, place the food inside the microwave, close the door, and press the start button."
    "bimanual_photograph|bimanual_photograph|Grasp the camera with the left hand, align it with the logo, and press the shutter button with the right hand."
    "bimanual_unlock_ipad|bimanual_unlock_ipad|Grasp the iPad and enter the password 123 to unlock the device."
)

# ----- training -----
MAX_STEPS=30000
SAVE_STEPS=10000
TRAIN_NUM_GPUS=4
TRAIN_GLOBAL_BATCH_SIZE=512

# ----- eval (DexJoCo MuJoCo harness) -----
DEXJOCO_SERVER_TYPE=gam
DEXJOCO_IMAGE_SIZE=224
DEXJOCO_EMBODIMENT_TAG=dexjoco_dual_arm
DEXJOCO_EMBODIMENT_TAG_BIMANUAL=dexjoco_dual_arm
EVAL_NUM_GPUS=4
N_EPISODES=50
N_RUNS=3
EVAL_SETS=(rand_obj)
DEXJOCO_INFERENCE_MODE=blocking_overlap
DEXJOCO_ACTION_HORIZON=16
DEXJOCO_REPLAN_RATIO=0.5
