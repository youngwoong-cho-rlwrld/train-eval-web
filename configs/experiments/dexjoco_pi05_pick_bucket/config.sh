# Experiment: dexjoco_pi05_pick_bucket
# DexJoCo pick_bucket - pi0.5 baseline (openpi serve_policy + MuJoCo client). Eval-only.

# ───── model ─────
MODEL_ID=dexjoco-pi05
MODEL_VERSION=n1.6
TRAIN_NOTE="DexJoCo pick_bucket - pi0.5 baseline"

# ───── task (eval-time policy prompt) ─────
DEXJOCO_TASK=pick_bucket
TASK_NAME=pick_bucket
INSTRUCTION="Place the boxed food into the bucket and then lift the bucket."

# ───── eval (DexJoCo MuJoCo harness) ─────
EVAL_HARNESS=dexjoco
DEXJOCO_SERVER_TYPE=openpi
N_EPISODES=50
N_RUNS=1
EVAL_BASE_SEED=0
EVAL_SETS=(rand_obj)
# Eval-only: submit phase=eval; EVAL_CHECKPOINT=~/workspace/dexjoco/checkpoints/pi05_dexjoco_ckpt/pick_bucket
DEXJOCO_INFERENCE_MODE=blocking_overlap
DEXJOCO_ACTION_HORIZON=30
DEXJOCO_REPLAN_RATIO=0.8
