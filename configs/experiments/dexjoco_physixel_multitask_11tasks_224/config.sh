# Experiment: dexjoco_physixel_multitask_11tasks_224
# DexJoCo all-task multitask PhysiXel/N1.6 fine-tune at 224x224.

# ----- model -----
MODEL_ID=dexjoco-physixel
MODEL_VERSION=n1.6
DEXJOCO_GIT_COMMIT=6a6d1b2c28459aab6067b25bcd38003dfa491017
TRAIN_GIT_COMMIT=9faf40b35770763f4c7650db2094b66cb4328918
TRAIN_MODALITY_CONFIG=dexjoco_config_front.py
TRAIN_ACTION_HORIZON=16
ACTION_HORIZON_MODE=modality
TRAIN_NOTE="DexJoCo 11-task multitask - physixel 224x224, multi-embodiment tags (dual_arm 46/44 + single_arm 23/22)"

# ----- datasets -----
export DATA_DIR="$HOME/workspace/dexjoco_n16/src_v30/dexjoco_lerobot_datasets"

TRAIN_DATASET_NAMES=(
    bimanual_assembly
    bimanual_hanoi
    bimanual_microwave_cook
    bimanual_photograph
    bimanual_unlock_ipad
    click_mouse
    fold_glasses
    hammer_nail
    pick_bucket
    pinch_tongs
    water_plant
)

# Per-dataset embodiment tags (enum names, parallel to TRAIN_DATASET_NAMES).
# Bimanual datasets are state[46]/action[44]; single-arm are state[23]/action[22].
TRAIN_DATASET_EMBODIMENT_TAGS=(
    DEXJOCO_DUAL_ARM
    DEXJOCO_DUAL_ARM
    DEXJOCO_DUAL_ARM
    DEXJOCO_DUAL_ARM
    DEXJOCO_DUAL_ARM
    DEXJOCO_SINGLE_ARM
    DEXJOCO_SINGLE_ARM
    DEXJOCO_SINGLE_ARM
    DEXJOCO_SINGLE_ARM
    DEXJOCO_SINGLE_ARM
    DEXJOCO_SINGLE_ARM
)

# ----- tasks -----
TASKS=(
    "bimanual_assembly|bimanual_assembly|Grasp the tray with the left hand and the peg with the right hand, then insert the peg into the hole."
    "bimanual_hanoi|bimanual_hanoi|Execute the final two moves of the three-level Tower of Hanoi: move the medium disk from the middle peg to the right peg with the right hand, then move the small disk from the left peg to the right peg with the left hand."
    "bimanual_microwave_cook|bimanual_microwave_cook|Open the microwave door, place the food inside the microwave, close the door, and press the start button."
    "bimanual_photograph|bimanual_photograph|Grasp the camera with the left hand, align it with the logo, and press the shutter button with the right hand."
    "bimanual_unlock_ipad|bimanual_unlock_ipad|Grasp the iPad and enter the password 123 to unlock the device."
    "click_mouse|click_mouse|Move the mouse to the purple mouse pad and click the left mouse button."
    "fold_glasses|fold_glasses|Fold the glasses and place them into the case."
    "hammer_nail|hammer_nail|Use the hammer to drive the nail into the wooden board."
    "pick_bucket|pick_bucket|Place the boxed food into the bucket and then lift the bucket."
    "pinch_tongs|pinch_tongs|Grasp the tongs and perform three consecutive open-close motions."
    "water_plant|water_plant|Grasp the watering can and apply water to the plant."
)

# DexJoCo eval still runs one task per submission. This default keeps the
# variant submittable; use the submit-time task override for per-task evals.
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
# Multi-embodiment checkpoint: single-arm tasks are served with the
# dexjoco_single_arm tag, bimanual_* tasks with dexjoco_dual_arm.
DEXJOCO_EMBODIMENT_TAG=dexjoco_single_arm
DEXJOCO_EMBODIMENT_TAG_BIMANUAL=dexjoco_dual_arm
EVAL_NUM_GPUS=4
N_EPISODES=50
N_RUNS=3
EVAL_BASE_SEED=0
EVAL_SETS=(rand_obj)
# The 2026-07-09/10 episode-0 "hangs" were all EGL renderers landing on GPU 0
# (fixed via MUJOCO_EGL_DEVICE_ID in eval_body_dexjoco.sh) — a big stagger is
# not needed; 15s just softens the NFS model-load burst. The eval-body
# watchdog retries any unit that still stalls.
# 5-min stagger serializes the ~15-min server compiles enough that the
# watchdog (40 min) never kills a unit that is merely waiting on compile.
EVAL_SIM_START_STAGGER_SECONDS=300
# 8-way concurrency deadlocked repeatedly during the 2026-07-10 incident
# window but the EGL stalls eased by morning; running at full width again
# with the watchdog+retry as the safety net.
DEXJOCO_MAX_CONCURRENT_UNITS=8
# A/B 2026-07-11: osmesa (CPU) rendering avoids the EGL driver stalls but is
# ~4x slower on the bimanual scene (~12 min/episode) — net loss. Stay on egl;
# the eval-body watchdog recycles the occasional fatal render stall.
DEXJOCO_INFERENCE_MODE=blocking_overlap
DEXJOCO_ACTION_HORIZON=16
DEXJOCO_REPLAN_RATIO=0.5
