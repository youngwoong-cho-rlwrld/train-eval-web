# PhysiXel AH40 Root-Cause Validation

Generated: 2026-05-26 17:28:51 KST

## Scope

Compared N1.6 commit `73f2aeb02220e430445af9e18051cadf6f2a9a9f` and PhysiXel commit `d930025adec40c3a6e2d5fccb098655b9f497b3f` for `physixel_multitask_3tasks_480_ah40` with state-part tokenization disabled.

## Artifacts

- Single-process root-cause probe: `/data/youngwoong/experiments/diagnostics/physixel_ah40_root_cause/20260526_150617`
- Eval-mode processor probe: `/data/youngwoong/experiments/diagnostics/physixel_ah40_eval_batch/20260526_155551`
- DDP N16-vs-PhysiXel probe: `/data/youngwoong/experiments/diagnostics/physixel_ah40_ddp_root_cause/20260526_155645`
- DDP repeat probe: `/data/youngwoong/experiments/diagnostics/physixel_ah40_ddp_repeat/20260526_162555`

## Results

| Check | Result |
| --- | --- |
| Worktree commits/import paths | Matched expected commits and worktree imports |
| Train-mode processor/collated batch | Exact match |
| Eval-mode processor/collated batch | Exact match |
| Initial trainable/action-head weights | Exact match |
| Step-0 forward outputs/loss | Exact match |
| Single-process one-step update | Small differences, also present in N16-vs-N16 repeat |
| Short DDP N16-vs-PhysiXel checkpoints | Checkpoint 1 exact, checkpoint 2 first diverges |
| Short DDP N16-vs-N16 repeat | Checkpoint 1 exact, checkpoint 2 first diverges |
| Short DDP PhysiXel-vs-PhysiXel repeat | Checkpoint 1 exact, checkpoint 2 first diverges |

## DDP Repeat Details

Checkpoint hash divergence counts for selected action-head tensors:

| Comparison | Ckpt 1 | Ckpt 2 | Ckpt 3 |
| --- | ---: | ---: | ---: |
| N16 vs N16 | 0 | 26 | 28 |
| PhysiXel vs PhysiXel | 0 | 24 | 28 |
| N16 vs PhysiXel A | 0 | 25 | 28 |
| N16 vs PhysiXel B | 0 | 26 | 28 |

Relative L2 magnitudes were comparable between repeat and cross comparisons. At checkpoint 2, mean relative L2 was approximately `2.1e-5` to `2.4e-5`; at checkpoint 3, approximately `1.2e-4` to `1.35e-4`.

## Conclusion

The first observed divergence under real DDP training is checkpoint 2, but the same divergence occurs within repeated N1.6 runs and within repeated PhysiXel runs. Therefore this validation does not identify a PhysiXel-specific processor, model-init, forward, or optimizer bug for AH40 with state-part tokenization disabled.

The persistent Cube Box performance gap is more likely due to long-run training variance/nondeterminism than an obvious deterministic code-path mismatch in `d930025`. More full retrain/eval replicates are needed to estimate whether the observed gap is statistically stable beyond run-to-run variance.
