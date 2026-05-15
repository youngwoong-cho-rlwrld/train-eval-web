# Variant: n15_cylinder_tube_place_7cm_left
# GR00T N1.5 fine-tune on V4 sim dataset, no --random-diffusion, no --tune-visual.

MODEL_VERSION=n1.5

# Override DATA_DIR so DATASET_NAME resolves under ~/datasets/ (the V4 symlinks),
# instead of the kakao.env default of /rlwrld2/home/seungcheol/80_datasets/v4.
export DATA_DIR="$HOME/datasets"
DATASET_NAME=v4_cylinder_tube_place_7cm_left_224
DATA_CONFIG=allex_thetwo_ck40_egostereo

TASK_NAME=task-Cylinder_Tube_Place-T15cmC7cmLeft
INSTRUCTION="Lift the cylinder with your left hand and place it in the middle of the tube without touching the tube."

# Training
MAX_STEPS=30000
SAVE_STEPS=1000

# Evaluation
N_EPISODES=70
EXECUTION_HORIZON=8
MAX_EPISODE_STEPS=300
N_RUNS=3
EVAL_SETS=(0cm 1cm 3cm 5cm 7cm)

TRAIN_NUM_GPUS=2
TRAIN_BATCH_SIZE=64
TRAIN_EXTRA_ARGS=()
TRAIN_NOTE="N1.5 pretrained, no --random-diffusion, no --tune-visual"
