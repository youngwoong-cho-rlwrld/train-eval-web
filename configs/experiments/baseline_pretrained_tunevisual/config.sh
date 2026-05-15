# Variant: baseline_pretrained_tunevisual
# --tune-visual, no --random-diffusion

# Model version: routes to lib/train_body.sh (n1.5) or lib/train_body_n16.sh (n1.6)
MODEL_VERSION=n1.5

# Shared eval scenario config (currently identical across all 4 variants).
# Override here if a variant ever needs a different task/dataset/instruction.
DATASET_NAME=v4_cube_box_5cm_left_100_100
DATA_CONFIG=allex_thetwo_ck40_egostereo

TASK_NAME=task-Cube_Box-5cmLeft
INSTRUCTION="Pick up the cube with your left hand and place it in the box"

# Training
MAX_STEPS=30000
SAVE_STEPS=10000

# Evaluation
N_EPISODES=70
EXECUTION_HORIZON=8
MAX_EPISODE_STEPS=300
N_RUNS=3
EVAL_SETS=(0cm 1cm 3cm 5cm 7cm)

TRAIN_NUM_GPUS=4
TRAIN_BATCH_SIZE=32
TRAIN_EXTRA_ARGS=(--tune-visual)
TRAIN_NOTE="--tune-visual, no --random-diffusion"
