# Experiment: dexjoco_pi05_fold_glasses
# DexJoCo fold_glasses - pi0.5 baseline (openpi serve_policy + MuJoCo client). Eval-only.

# ───── model ─────
MODEL_ID=dexjoco-pi05
MODEL_VERSION=n1.6
DEXJOCO_GIT_COMMIT=6a6d1b2c28459aab6067b25bcd38003dfa491017
TRAIN_NOTE="DexJoCo fold_glasses - pi0.5 baseline"

# ───── task (eval-time policy prompt) ─────
DEXJOCO_TASK=fold_glasses
TASK_NAME=fold_glasses
INSTRUCTION="Fold the glasses and place them into the case."

# ───── eval (DexJoCo MuJoCo harness) ─────
EVAL_HARNESS=dexjoco
DEXJOCO_SERVER_TYPE=openpi
N_EPISODES=50
N_RUNS=1
EVAL_BASE_SEED=0
EVAL_SETS=(rand_obj)
# Eval-only: submit phase=eval; EVAL_CHECKPOINT=~/workspace/dexjoco/checkpoints/pi05_dexjoco_ckpt/fold_glasses
DEXJOCO_INFERENCE_MODE=blocking_overlap
DEXJOCO_ACTION_HORIZON=30
DEXJOCO_REPLAN_RATIO=0.8
