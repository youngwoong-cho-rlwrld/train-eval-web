# Variant: n15_multitask_5tasks
# GR00T N1.5 multi-task fine-tune over 5 V4 sim datasets (cube_stack + 4 cylinder tube tasks).
# One checkpoint, evaluated separately on each task. Hypothesis: shared
# representations reduce per-task SR std across the 3 eval repeats vs the 5
# single-task baselines (n15_cube_stack_3cm_right, n15_cylinder_tube_*).

MODEL_VERSION=n1.5

# Override DATA_DIR so DATASETS entries resolve under ~/datasets/ (V4 symlinks),
# instead of the kakao.env default of /rlwrld2/home/seungcheol/80_datasets/v4.
export DATA_DIR="$HOME/datasets"

# Multi-dataset training. Format: "name|data_config|weight" (name is under $DATA_DIR).
# weight=1.0 across all 5 matches the existing cotrain-*.yaml convention; this
# upsamples the smaller Pick datasets (34 eps each) to roughly balance against
# Place/Stack (520 eps each). Revisit weighting if Pick over- or under-fits.
DATASETS=(
    "v4_cube_stack_3cm_right_224|allex_thetwo_ck40_egostereo|1.0"
    "v4_cylinder_tube_pick_5cm_right_224|allex_thetwo_ck40_egostereo|1.0"
    "v4_cylinder_tube_pick_7cm_left_224|allex_thetwo_ck40_egostereo|1.0"
    "v4_cylinder_tube_place_5cm_right_224|allex_thetwo_ck40_egostereo|1.0"
    "v4_cylinder_tube_place_7cm_left_224|allex_thetwo_ck40_egostereo|1.0"
)
# DATA_CONFIG is the shared key passed to eval_allex.py --data_config; all 5 share it.
DATA_CONFIG=allex_thetwo_ck40_egostereo

# Multi-task eval matrix. Format: "task_short|task_name|instruction".
# task_short is used as a subdir under eval_results/. task_name must exist in
# rlwrld_isaac/task_json/task_config.json. instruction is the exact line from
# the dataset's meta/tasks.jsonl.
TASKS=(
    "cube_stack_3cm_right|task-Cube_Stack-3cmRight|Pick the red cube with your right hand and stack it on the blue cube."
    "cylinder_tube_pick_5cm_right|task-Cylinder_Tube_Pick-T15cmC5cmRight|Lift up the cylinder with your right hand, making sure not to touch the tube."
    "cylinder_tube_pick_7cm_left|task-Cylinder_Tube_Pick-T15cmC7cmLeft|Lift up the cylinder with your left hand, making sure not to touch the tube."
    "cylinder_tube_place_5cm_right|task-Cylinder_Tube_Place-T15cmC5cmRight|Lift the cylinder with your right hand and place it in the middle of the tube without touching the tube."
    "cylinder_tube_place_7cm_left|task-Cylinder_Tube_Place-T15cmC7cmLeft|Lift the cylinder with your left hand and place it in the middle of the tube without touching the tube."
)

# Training
MAX_STEPS=30000
SAVE_STEPS=1000

# Evaluation (shared across all tasks in the matrix)
N_EPISODES=70
EXECUTION_HORIZON=8
MAX_EPISODE_STEPS=300
N_RUNS=3
EVAL_SETS=(0cm 1cm 3cm 5cm 7cm)

TRAIN_NUM_GPUS=2
TRAIN_BATCH_SIZE=64
TRAIN_EXTRA_ARGS=()
TRAIN_NOTE="N1.5 multi-task on 5 sim datasets, equal weight=1.0"
