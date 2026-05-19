# Experiment: physixel_cube_stack_3cm_right_480
# PhysiXel fine-tune on V4 Cube_Stack-3cmRight at 480x640.

MODEL_ID=physixel
MODEL_VERSION=n1.6

# Reuse the N1.6 Cube Stack modality config; path is relative to this
# experiment directory after configs/experiments is staged.
TRAIN_MODALITY_CONFIG=../n16_cube_stack_3cm_right_480/allex_egostereo_ck40_config_absolute.py

export DATA_DIR="$HOME/datasets"
DATASET_NAME=v4_cube_stack_3cm_right_480
DATA_CONFIG=allex_thetwo_ck40_egostereo

TASK_NAME=task-Cube_Stack-3cmRight
INSTRUCTION="Pick the red cube with your right hand and stack it on the blue cube."

# Training
MAX_STEPS=30000
SAVE_STEPS=10000

# Evaluation
N_EPISODES=70
EXECUTION_HORIZON=8
MAX_EPISODE_STEPS=300
N_RUNS=3
EVAL_NUM_ENVS_PER_GPU=1
EVAL_PIN_CUDA_DEVICES=1
EVAL_UNSET_CUDA_VISIBLE_DEVICES_FOR_SERVER=0
EVAL_PIN_CLIENT_CUDA_DEVICES=1
EVAL_SETS=(0cm 1cm 3cm 5cm 7cm)

TRAIN_NUM_GPUS=2
TRAIN_BATCH_SIZE=64
TRAIN_EXTRA_ARGS=()
TRAIN_NOTE="PhysiXel n1.6-compatible @ 480x640"
