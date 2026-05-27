| Area | Field | md1e59 | mbbb90 | Difference |
|---|---|---|---|---|
| Model code | Origin repo | `RLWRLD/physixel` | `RLWRLD/gr00t-n16` | Different repo |
| Model code | Worktree | `/data/youngwoong/experiments/.worktrees/md1e59` | `/data/youngwoong/experiments/.worktrees/mbbb90` | Different run worktree |
| Model code | Commit | `73f2aeb02220e430445af9e18051cadf6f2a9a9f` | `73f2aeb02220e430445af9e18051cadf6f2a9a9f` | Same |
| Model code | Commit subject | `Update allex_coffeepot_ck40_abs.sh for rrc_eval env` | Same | Same |
| Model code | Source diff | `gr00t/`, `scripts/`, `configs/`, `pyproject.toml` match | Same | No tracked model/source-code diff detected |
| Model code | Runtime artifacts | `.venv`, `wandb/` differ | `.venv`, `wandb/` differ | Runtime/generated artifacts only |
| Config snapshot | Variant | `action_horizon_ablation_ah40` | `n16_multitask_3tasks_480` | Different experiment |
| Config snapshot | Job name | `train_action_horizon_ablation_ah40_20260526_213844` | `[re]train_n16_multitask_3tasks_480_20260524_153101` | Different |
| Config snapshot | Snapshot path | `/data/youngwoong/experiments/action_horizon_ablation_ah40/config_action_horizon_ablation_ah40_20260526_213844_83a5a7.sh` | `/data/youngwoong/experiments/n16_multitask_3tasks_480/config_20260524_153101_mbbb90.sh` | Different |
| Config snapshot | `MODEL_ID` | `physixel` | not set | Different |
| Config snapshot | `MODEL_VERSION` | `n1.6` | `n1.6` | Same |
| Config snapshot | Training repo path | `/data/youngwoong/workspace/physixel` | `/data/youngwoong/workspace/gr00t-n16` | Different |
| Config snapshot | Training branch | `feature/configurable-action-horizon` | `main` | Different |
| Config snapshot | Training commit | `73f2aeb02220e430445af9e18051cadf6f2a9a9f` | `73f2aeb02220e430445af9e18051cadf6f2a9a9f` | Same |
| Config snapshot | `TRAIN_GIT_COMMIT` | set to `73f2aeb...` | not set | Different snapshot schema |
| Config snapshot | `SUBMIT_GIT_COMMIT` | `73f2aeb...` | `73f2aeb...` | Same |
| Config snapshot | `TRAIN_ACTION_HORIZON` | `40` | not set | Different snapshot schema |
| Config snapshot | `ACTION_HORIZON_MODE` | `modality` | not set | Different snapshot schema |
| Config snapshot | `TRAIN_MODALITY_CONFIG` | `modality_action_horizon_ablation_ah40_20260526_213844_83a5a7.py` | `allex_egostereo_ck40_config_absolute.py` | Different file name |
| Config snapshot | Train note | `GR00T N1.6 Action horizon ablation test with action horizon 40` | `N1.6 multi-task @ 480x640 on 3 sim datasets` | Different |
| Config snapshot | Checkpoint output dir | `/data/youngwoong/experiments/action_horizon_ablation_ah40/checkpoints/action_horizon_ablation_ah40_20260526_213844_83a5a7` | `/data/youngwoong/experiments/n16_multitask_3tasks_480/checkpoints/[re]train_n16_multitask_3tasks_480_20260524_153101` | Different |
| Config snapshot | Core train flags | datasets, base model, max steps, save steps, GPUs, batch size, LR, workers all match | same | No core hyperparameter diff detected |
| Modality config | Source file | Generated snapshot modality file | Static variant modality file | Different file path/name |
| Modality config | Registered config name | `allex_egostereo_ck40_config_absolute` | `allex_egostereo_ck40_config_absolute` | Same |
| Modality config | Embodiment tag | `NEW_EMBODIMENT` | `NEW_EMBODIMENT` | Same |
| Modality config | Video delta | `[0]` | `[0]` | Same |
| Modality config | State delta | `[0]` | `[0]` | Same |
| Modality config | Action delta | `list(range(40))` | `list(range(40))` | Same |
| Modality config | Modality keys | right/left arm, right/left hand, neck, waist | same | Same |
| Modality config | Action representation | `ABSOLUTE`, `NON_EEF`, `DEFAULT` for all 6 groups | same | Same |
| Modality config | Semantic diff | none detected | none detected | Only formatting/comments/path differ |
