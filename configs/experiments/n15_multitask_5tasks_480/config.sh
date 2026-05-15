# Variant: n15_multitask_5tasks_480
# GR00T N1.5 multi-task fine-tune on 5 V4 sim datasets at 480x640
# (cube_stack + 4 cylinder_tube tasks). Mirrors n15_multitask_5tasks
# but uses 480x640 source datasets.

MODEL_VERSION=n1.5

export DATA_DIR="$HOME/datasets"

DATASETS=(
    "v4_cube_stack_3cm_right_480|allex_thetwo_ck40_egostereo|1.0"
    "v4_cylinder_tube_pick_5cm_right_480|allex_thetwo_ck40_egostereo|1.0"
    "v4_cylinder_tube_pick_7cm_left_480|allex_thetwo_ck40_egostereo|1.0"
    "v4_cylinder_tube_place_5cm_right_480|allex_thetwo_ck40_egostereo|1.0"
    "v4_cylinder_tube_place_7cm_left_480|allex_thetwo_ck40_egostereo|1.0"
)
DATA_CONFIG=allex_thetwo_ck40_egostereo

TASKS=(
    "cube_stack_3cm_right|task-Cube_Stack-3cmRight|Pick the red cube with your right hand and stack it on the blue cube."
    "cylinder_tube_pick_5cm_right|task-Cylinder_Tube_Pick-T15cmC5cmRight|Lift up the cylinder with your right hand, making sure not to touch the tube."
    "cylinder_tube_pick_7cm_left|task-Cylinder_Tube_Pick-T15cmC7cmLeft|Lift up the cylinder with your left hand, making sure not to touch the tube."
    "cylinder_tube_place_5cm_right|task-Cylinder_Tube_Place-T15cmC5cmRight|Lift the cylinder with your right hand and place it in the middle of the tube without touching the tube."
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
TRAIN_NOTE="N1.5 multi-task @ 480x640 on 5 sim datasets, equal weight=1.0"
