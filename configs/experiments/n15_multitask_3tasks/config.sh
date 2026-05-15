# Variant: n15_multitask_3tasks
# GR00T N1.5 multi-task fine-tune on 3 V4 sim datasets:
#   - Cube_Box-5cmLeft
#   - Cube_Stack-3cmRight
#   - Cylinder_Tube_Place-T15cmC7cmLeft
# Excludes Cylinder_Tube_Pick tasks (hypothesis: pick-only tasks hurt cotraining).

MODEL_VERSION=n1.5

export DATA_DIR="$HOME/datasets"

DATASETS=(
    "v4_cube_box_5cm_left_100_100|allex_thetwo_ck40_egostereo|1.0"
    "v4_cube_stack_3cm_right_224|allex_thetwo_ck40_egostereo|1.0"
    "v4_cylinder_tube_place_7cm_left_224|allex_thetwo_ck40_egostereo|1.0"
)
DATA_CONFIG=allex_thetwo_ck40_egostereo

TASKS=(
    "cube_box_5cm_left|task-Cube_Box-5cmLeft|Pick up the cube with your left hand and place it in the box"
    "cube_stack_3cm_right|task-Cube_Stack-3cmRight|Pick the red cube with your right hand and stack it on the blue cube."
    "cylinder_tube_place_7cm_left|task-Cylinder_Tube_Place-T15cmC7cmLeft|Lift the cylinder with your left hand and place it in the middle of the tube without touching the tube."
)

MAX_STEPS=30000
SAVE_STEPS=1000

N_EPISODES=70
EXECUTION_HORIZON=8
MAX_EPISODE_STEPS=300
N_RUNS=3
EVAL_SETS=(0cm 1cm 3cm 5cm 7cm)

TRAIN_NUM_GPUS=2
TRAIN_BATCH_SIZE=64
TRAIN_EXTRA_ARGS=()
TRAIN_NOTE="N1.5 multi-task on 3 sim datasets (cube_box, cube_stack, cylinder_tube_place_7cm_left), equal weight=1.0"
