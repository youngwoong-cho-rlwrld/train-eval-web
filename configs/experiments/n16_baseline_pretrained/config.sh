# Variant: n16_baseline_pretrained
# GR00T N1.6 + pretrained backbone, no --tune-visual

# Model version: routes to lib/train_body_n16.sh
MODEL_VERSION=n1.6

# Modality config: a per-variant Python file (path is RELATIVE to experiments/<variant>/)
TRAIN_MODALITY_CONFIG=allex_egostereo_ck40_config_absolute.py

# Shared eval scenario config (currently identical across all variants).
# Override here if a variant ever needs a different task/dataset/instruction.
DATASET_NAME=v4_cube_box_5cm_left_100_100
DATA_CONFIG=allex_thetwo_ck40_egostereo

TASK_NAME=task-Cube_Box-5cmLeft
INSTRUCTION="Pick up the cube with your left hand and place it in the box"

# Training
MAX_STEPS=30000
SAVE_STEPS=30000  # save only at the end

# Evaluation (eval support for n1.6 is TBD — train only for now)
N_EPISODES=70
EXECUTION_HORIZON=8
MAX_EPISODE_STEPS=300
N_RUNS=3
EVAL_SETS=(0cm 1cm 3cm 5cm 7cm)

TRAIN_NUM_GPUS=2
TRAIN_BATCH_SIZE=64
TRAIN_EXTRA_ARGS=()
TRAIN_NOTE="GR00T N1.6 baseline_pretrained"
