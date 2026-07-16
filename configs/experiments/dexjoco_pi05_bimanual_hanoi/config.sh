# Experiment: dexjoco_pi05_bimanual_hanoi
# DexJoCo bimanual_hanoi - pi0.5 baseline (openpi serve_policy + MuJoCo client). Eval-only.

# ───── model ─────
MODEL_ID=dexjoco-pi05
MODEL_VERSION=n1.6
TRAIN_NOTE="DexJoCo bimanual_hanoi - pi0.5 baseline"

# ───── task (eval-time policy prompt) ─────
DEXJOCO_TASK=bimanual_hanoi
TASK_NAME=bimanual_hanoi
INSTRUCTION="Execute the final two moves of the three-level Tower of Hanoi: move the medium disk from the middle peg to the right peg with the right hand, then move the small disk from the left peg to the right peg with the left hand."

# ───── eval (DexJoCo MuJoCo harness) ─────
EVAL_HARNESS=dexjoco
DEXJOCO_SERVER_TYPE=openpi
N_EPISODES=50
N_RUNS=1
EVAL_BASE_SEED=0
EVAL_SETS=(rand_obj)
# Eval-only: submit phase=eval; EVAL_CHECKPOINT=~/workspace/dexjoco/checkpoints/pi05_dexjoco_ckpt/bimanual_hanoi
DEXJOCO_INFERENCE_MODE=blocking_overlap
DEXJOCO_ACTION_HORIZON=30
DEXJOCO_REPLAN_RATIO=0.8
