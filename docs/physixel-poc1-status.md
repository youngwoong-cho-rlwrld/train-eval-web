# PhysiXel PoC1 Status

Last updated: 2026-05-25 23:02:26 KST

## Current State

| Area | Status |
| --- | --- |
| Implementation scope | State tokenization only. Actions were not semantically partitioned. DiT/action encoder/decoder are intended to stay unchanged. |
| PoC1 random partitions | Done for `K=3,5,7,9,11`, each with `ps0/ps1/ps2`. Full permutations/groupings are recorded on the Notion page. |
| 1-token baseline | Added to the Notion page and local plot. This is the vanilla state-token setup under the same intended AH=16 setting. |
| Result trend | More state tokens generally helps, especially `K=11`. Random grouping matters: the same token count can vary substantially across `ps0/ps1/ps2`. |
| Validation | Partition size test, determinism test, saved-permutation replay, and checkpoint reload checks are implemented/passed. |
| Important caveat | Existing PoC1 checkpoints are not clean AH=16 ablations because saved model/processor configs still used horizon 50 internally. Results are useful as exploratory evidence, but should be rerun cleanly. |
| PoC2 semantic grouping | Explicit grouping support is implemented and documented. Semantic group definitions for `K=3/5/7/9/11` are on the Notion page. |

## Interpretation

The current PoC1 result mixes three randomness sources:

| Randomness source | Current control |
| --- | --- |
| Training randomness | Not fully controlled; needs repeated trainings. |
| Eval randomness | Eval seeds should be fixed and reused across all comparisons. |
| Token composition randomness | Controlled by recorded `ps0/ps1/ps2`, but each composition currently has only one training run. |

Treat `ps0/ps1/ps2` as different token-composition experiments, not just noise.

## Action Horizon Probe

Checked at: 2026-05-25 22:38:00 KST

Question: did the saved model/processor config really use horizon 50 even when the experiment intended a shorter action horizon?

Evidence from an existing old PoC1 checkpoint:

`/fsx/rlwrld/youngwoong_cho/.train-eval-web/experiments/physixel_multitask_3tasks_480_ah16/checkpoints/train_physixel_multitask_pt7_ps2_20260522_233405`

| Source | Observed value |
| --- | --- |
| `checkpoint-30000/experiment_cfg/conf.yaml` | `action_horizon: 16` |
| `checkpoint-30000/config.json` | `action_horizon: 50` |
| `checkpoint-30000/processor_config.json` | `processor_kwargs.max_action_horizon: 50` |
| Base checkpoint `nvidia/GR00T-N1.6-3B` config | `action_horizon: 50` |

Pre-fix dry probe:

| Check | Observed value |
| --- | --- |
| `FinetuneConfig` exposes `action_horizon` CLI arg | `False` |
| Current default `get_default_config().model.action_horizon` | `16` |
| Local `ck16` modality action delta length | `16` |
| Local `ck40` modality action delta length | `40` |

Conclusion: old PoC1 checkpoints did save/use model and processor horizon 50 despite the experiment config saying AH=16.

Follow-up fix implemented at 2026-05-25 22:53:28 KST:

- train-eval-web now carries a typed `train_action_horizon` train override.
- Slurm N1.6 jobs export `SUBMIT_TRAIN_ACTION_HORIZON` and pass `--action-horizon <N>`.
- MLXP N1.6 jobs render the same `--action-horizon <N>` argument.
- The submit UI shows an editable `--action-horizon` row for N1.6 training, defaulted from the modality config action delta length.
- PhysiXel `FinetuneConfig` now exposes `action_horizon`, and `launch_finetune.py` applies it to `config.model.action_horizon`.
- PhysiXel validates that `--action-horizon` matches the loaded modality action delta length, so mismatched `ck16`/`ck40` submissions fail instead of silently producing non-clean ablations.

Current validation after the fix:

| Check | Observed value |
| --- | --- |
| train-eval-web `physixel_multitask_3tasks_480_ah16` inferred train action horizon | `16` |
| Snapshot preview contains | `TRAIN_ACTION_HORIZON=16` |
| Snapshot preview flags contain | `--action-horizon 16` |
| Mismatched request `--action-horizon 40` with ck16 modality | rejected before submit |
| PhysiXel `ck16` modality with requested `16` | accepted |
| PhysiXel `ck40` modality with requested `40` | accepted |
| PhysiXel `ck16` modality with requested `40` | rejected |

## Action Items

1. Rerun clean AH=16 PoC1.
   - Retrain the 1-token baseline and `K=3/5/7/9/11`.
   - Use the fixed model/processor horizon override so saved configs actually say `16`.
   - Use dedicated clean-rerun variants or result destinations so checkpoint and eval output directories do not collide with old exploratory results.
   - On submit, confirm the N1.6 train flag table shows `--action-horizon 16`.
   - On the job detail config snapshot, confirm both `TRAIN_ACTION_HORIZON=16` and `SUBMIT_TRAIN_ACTION_HORIZON=16`.
   - In the training log, confirm PhysiXel prints `Using action_horizon=16`.
   - After checkpoint save, confirm `config.json` and `processor_config.json` both record horizon `16`.

2. Add repeated training runs.
   - For each token composition, run 2 more trainings.
   - Scope: current result + 2 repeats for the 1-token baseline and every `K=3/5/7/9/11 x ps0/ps1/ps2`.

3. Keep eval seeds identical.
   - Same eval seed schedule for every checkpoint.
   - Report per-run values, mean, and std.

4. Analyze token composition meaning.
   - Use dataset `info.json` to map each state dim to joint names.
   - For best/worst random groupings, check whether groups accidentally align with body parts, hands, arms, torso, etc.

5. Run PoC2 semantic partitions.
   - `K=3`: torso / left / right.
   - `K=5`: torso / left arm / right arm / left hand / right hand.
   - `K=7/9/11`: progressively split fingers as suggested in the feedback.
   - Train/eval with the same clean AH=16 setup.

6. Update Notion after reruns.
   - Mark old PoC1 results as exploratory / not clean AH=16.
   - Add clean rerun tables separately.
   - Keep random PoC1 and semantic PoC2 results clearly separated.
