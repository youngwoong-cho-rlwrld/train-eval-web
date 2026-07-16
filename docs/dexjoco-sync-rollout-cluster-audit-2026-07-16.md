# DexJoCo sync rollout and cross-cluster audit

Date: 2026-07-16 (Asia/Seoul)

## Executive summary

The initial audit found that the deployed rollout did not have the same sync
semantics as the `simon` reference implementation and that DexJoCo source and
runtime environments had drifted across Kakao, SKT, and MLXP. The baseline
sections below preserve that evidence. The remediation is now implemented and
deployed as described in **Remediation and deployment result**.

The old `--inference-mode=sync --action-horizon=16 --replan-ratio=0.5`
behavior was a blocking overlap rollout: it waited for the new chunk, but
requested it before the old chunk drained and blended the overlap. The new
contract names that behavior `blocking_overlap`; true `sync` now executes the
full chunk, waits only after the buffer drains, and has zero overlap.

The cancelled SKT GAM eval jobs 156609 and 156610 should not be used as valid
GAM results. In addition to the rollout-semantics mismatch, they pinned a clean
GAM commit that excluded required uncommitted DexJoCo inference changes.

No retraining is required to correct the rollout client, source pinning, or
cluster environment. PhysiXel checkpoints can therefore be evaluated again
after the eval stack is made reproducible. GAM is different: a separate audit
found that the completed GAM checkpoints were trained with consecutive
visual/proprio anchors while their action targets advanced in 16-action chunks.
Scientifically correct GAM results require retraining after fixing that temporal
alignment.

## Implementation status

The audit recommendations are being implemented on dedicated, reviewable
commits rather than in the previously dirty cluster checkouts:

- Canonical DexJoCo rollout client:
  `6a6d1b2c28459aab6067b25bcd38003dfa491017` on
  `RLWRLD/dexjoco:codex/canonical-dexjoco-rollout-client`.
- Canonical source matching the old GAM checkpoints:
  `9181c596c8b52efe7a7a73bcfab98abecd6e36f4`.
- Corrected GAM training source:
  `69afa536658198a22750b5618322edf68fdea93a`.
- train-eval-web implements `sync`, `async`, and `blocking_overlap`, validates
  their arguments, records their effective values in submission metadata, and
  can pin the external DexJoCo checkout by commit.
- All 29 DexJoCo experiment configs pin the canonical DexJoCo client. The three
  GAM configs pin the corrected GAM source, and the saved Kakao, SKT, and MLXP
  settings select the isolated canonical client and OpenPI environments.

The final rollout contract is:

- `sync`: request only after the buffer drains, block without advancing the
  simulator, never blend, and ignore `replan_ratio`. A numeric horizon caps the
  returned chunk; `auto` consumes the complete server chunk.
- `async`: request without blocking when
  `remaining <= replan_ratio * horizon`.
- `blocking_overlap`: use the same `<=` threshold, block for the replacement,
  and blend the overlapping suffix. With horizon 16 and ratio 0.5, replanning
  occurs after exactly eight actions.

## Remediation and deployment result

### Source and configuration

- Canonical DexJoCo commit `6a6d1b2c28459aab6067b25bcd38003dfa491017`
  is present in all three clusters and checked out in clean, stable worktrees:
  - Kakao: `/rlwrld2/home/youngwoong_cho/workspace/dexjoco-canonical-6a6d1b2c`
  - SKT: `/fsx/rlwrld/youngwoong_cho/workspace/dexjoco-canonical-6a6d1b2c`
  - MLXP: `/data/youngwoong/workspace/dexjoco-canonical-6a6d1b2c`
- Corrected GAM commit `69afa536658198a22750b5618322edf68fdea93a`
  is available to all clusters. Evaluation and training create a detached
  worktree at the configured commit, so mutable main-checkout changes cannot
  leak into a job.
- The SKT cluster template now exposes the same five DexJoCo environment fields
  as Kakao: repo, micromamba binary/root, client environment, and OpenPI server
  environment.
- GAM and PhysiXel use `blocking_overlap`, horizon 16, ratio 0.5. pi0.5 uses
  horizon 30 and ratio 0.8; N1.6 uses horizon 16 and ratio 0.8. True `sync`
  remains the default for a config that does not explicitly request overlap.

### Runtime environment parity

The path-independent package manifests are tracked under
[`docs/manifests`](manifests/README.md). Every row below was reproduced with the
same package count and SHA-256 on Kakao, SKT, and MLXP.

| Runtime | Environment/source | Packages | Manifest SHA-256 |
| --- | --- | ---: | --- |
| MuJoCo client | `dexjoco-canonical-6a6d1b2c` | 47 | `78b2329bae8d52538407ef86438b03b6dcad6af90cb33016852190b29d945913` |
| OpenPI policy server | `openpi-canonical-6a6d1b2c` | 175 | `ed6ed24cf77b6510b4c396cbabaf176105cedc0a32b377430645f06fc7ce6f74` |
| GAM policy runtime | GAM `.venv` with source `69afa536` | 149 | `bfcdc86fea8eb7e66a542ea78de35bab2a93d102768a441041f6509e9e5f97d3` |

The OpenPI environment does not install the incompatible `openpi-client`
distribution, which pins NumPy 1.26.4 while OpenPI requires NumPy greater than
2. It exposes the canonical client source through an absolute `.pth` and pins
the two required client runtime dependencies (`tree==0.2.4` and
`websockets==16.1`). `serve_policy.py --help`, representative single-arm and
bimanual policy configs, JAX import, module provenance, and source cleanliness
were verified independently on every cluster.

LeRobot 0.4.4 is intentionally installed with `--no-deps`, matching the source
repo's `install.bash`. Raw `pip check` therefore reports the same eleven
LeRobot-only metadata issues on every cluster; there are no other broken
requirements, and the imports used by the dataset and policy server pass.

The saved train-eval-web settings now select the canonical repo plus
`dexjoco-canonical-6a6d1b2c` and `openpi-canonical-6a6d1b2c` environments on all
three clusters. The settings were updated and read back through the web API.

### Job remediation state at 2026-07-16 13:45 KST

- Invalid/stale GAM evaluations, including 156609 and 156610, are cancelled.
- Corrected GAM retraining uses eight H200 GPUs, global batch size 512, 30k
  steps, and source `69afa536`. Single-arm job `...-93a00434` and bimanual job
  `...-d5f717f1` are producing healthy loss; multitask job `...-3ca6aff1` is
  pending only for eight-GPU capacity.
- Canonical PhysiXel evaluations 157772, 157774, and 157775 request SKT L40S,
  four GPUs, three runs, and the exact 11/6/5 task subsets. Their submission
  metadata records `blocking_overlap`, horizon 16, and ratio 0.5. They have not
  reached batch startup because SKT dynamic L40S nodes are repeatedly failing
  scheduler health checks; no rollout-code error or job stdout exists yet.

## Reference commits

- [`e540a879a7691ee28fa397a738e62470a4364054`](https://github.com/RLWRLD/dexjoco/commit/e540a879a7691ee28fa397a738e62470a4364054): implements synchronous action-chunk rollout and latency/overlap metrics.
- [`a73a3956d1f32887b3cee5ef1387ff85b8f4836b`](https://github.com/RLWRLD/dexjoco/commit/a73a3956d1f32887b3cee5ef1387ff85b8f4836b): makes RLDX scoring default to synchronous full-chunk rollout and advances the matching RLDX submodule.
- [`3343857cd37fcdd74d972b108ebaefb405172251`](https://github.com/RLWRLD/RLDX-1-mirror-private/commit/3343857cd37fcdd74d972b108ebaefb405172251): existing RLDX-1 RECAP launcher configuration on `origin/recap`.

The reference scoring defaults are:

```text
DEXJOCO_SYNC_ROLLOUT=1
DEXJOCO_ACTION_HORIZON=auto
DEXJOCO_REPLAN_RATIO=0.0
```

## Sync and async logic comparison

| Behavior | Simon reference | Current implementation |
| --- | --- | --- |
| Sync replan point | Request only after the action buffer is empty | Request whenever `len(buffer) < ratio * horizon`, including sync mode |
| `H=16`, `ratio=0.5` | Execute all 16 actions, then request; no overlap | Execute 9 actions, request and block, then blend the remaining 7 actions with the new chunk |
| Async threshold | `len(buffer) <= ratio * observed_horizon` | `len(buffer) < ratio * configured_horizon` |
| Horizon handling | Accepts `auto` or a positive integer; numeric sync horizon truncates the received chunk | Accepts an integer; it controls the threshold but does not truncate the received chunk |
| Simulation during inference | Sync mode does not advance simulation while waiting | Also blocks, but blocks at the early overlap-replan point |
| Missing-buffer behavior | Sync waits; async may issue a stay action | Same broad structure |
| Chunk merge | Timestamped buffer with geodesic SO(3) blending for overlap | Same core merge method |
| Metrics | Writes request latency, dropped prefix, overlap, stale chunk, append length, and sync wait metrics | No equivalent rollout metrics |

The strict `<` condition is important. With horizon 16 and ratio 0.5, the
current client does not replan after eight actions because eight remaining is
not less than eight. It replans after the ninth action, with seven actions
remaining. The incoming chunk is timestamp-aligned with those seven remaining
actions, so they are blended and a new tail is appended.

Therefore the current mode is more accurately described as
`blocking_overlap`, not reference-style full-chunk sync.

If reference sync semantics are adopted, `replan_ratio=0.5` has no effect in
sync mode. Ratio remains meaningful for async or an explicitly separate
blocking-overlap mode.

## Local train-eval-web configuration

The local harness defaults to:

```text
DEXJOCO_INFERENCE_MODE=sync
```

It forwards:

```text
--inference-mode
--action-horizon
--replan-ratio
```

The local experiment configuration is internally consistent with the earlier
requested values:

- GAM and PhysiXel: action horizon 16 and replan ratio 0.5.
- pi0.5: action horizon 30, with the client default replan ratio 0.8.
- N1.6 baseline: action horizon 16, with the client default replan ratio 0.8.
- All DexJoCo `config.sh` files and the harness pass `bash -n` validation.

The existing experiment configs explicitly select `blocking_overlap` to
preserve their intended ratio-based behavior. Unspecified future configs use
true full-chunk `sync`. All 29 DexJoCo configs and the three GAM YAMLs are now
tracked by Git, and GAM/PhysiXel use horizon 16 and ratio 0.5 while pi0.5 and
N1.6 retain ratio 0.8 with horizons 30 and 16 respectively.

## Audit baseline: DexJoCo source parity before remediation

All three DexJoCo checkouts currently report the same base revision:

```text
8bd79837c2ab997699ac7a5d30b25e158126070f
feature/sync-env
```

However, the active files imported by the editable environments are different.

| Component | Kakao | SKT | MLXP |
| --- | --- | --- | --- |
| Eval client SHA-256 prefix | `4c9135c3` | `434fcb8f` | `009f39d9` |
| `dexjoco_openpi_env.py` | Same | Same | Same |
| EGL render-timeout patch | Absent | Present | Present |
| Websocket keepalive disabled | Present | Absent | Absent |
| Episode-ID stale-chunk filtering | Present | Absent | Absent |
| `--start-episode` | Present | Present | Absent |

The environment imports `dexjoco` and `openpi_client` directly from each
cluster's mutable checkout, so these working-tree differences affect real jobs.

Other relevant differences:

- Kakao has websocket client/server changes that disable ping intervals and
  timeouts.
- SKT and MLXP have an EGL render watchdog/retry patch that Kakao lacks.
- SKT and MLXP have similar episode-drain fixes, while Kakao additionally tags
  observations and action chunks with an episode ID.
- Backup files do not execute, but all three active clients are uncommitted.
- The installed checkouts use `brave-eai/dexjoco` as `origin`; the supplied
  `simon` commits live in `RLWRLD/dexjoco` and are not part of the installed
  canonical history.

## Audit baseline: train-eval-web staging parity before remediation

| Component | Local | Kakao staging | SKT staging | MLXP |
| --- | --- | --- | --- | --- |
| `eval_body_dexjoco.sh` SHA-256 prefix | `8121f573` | `52fb8d8e` | `8121f573` | Generated from local source at submission |
| GAM adapter | `6e24f814` | Same | Same | Generated from local source |
| GR00T adapter | `6f526302` | Same | Same | Generated from local source |
| Current DexJoCo experiment configs | Canonical local copies | Stale | Exact local match | Submission snapshot |

Kakao's staged harness does not pass `--inference-mode` or
`--action-horizon`. Its staged GAM and PhysiXel configs also lack the new
action-horizon and replan-ratio values. If that staging tree is invoked as-is,
the client falls back to async mode, horizon 30, and ratio 0.8.

The three GAM `gam_config.yaml` files are byte-identical between local, Kakao,
and SKT. The drift is in the shell configuration and harness rather than those
training YAMLs.

## Audit baseline: DexJoCo evaluation environment parity before remediation

The key client runtime versions match:

```text
Python 3.11.15
NumPy 1.26.4
SciPy 1.17.1
MuJoCo 3.4.0
Tyro 1.0.15
ImageIO 2.37.3
```

The full environments do not match:

| Cluster | Package count | Notable difference |
| --- | ---: | --- |
| Kakao | 220 | Additional Hugging Face/HTTP CLI packages; `websockets==16.0` |
| SKT | 204 | `websockets==16.1`; several newer codec/system packages |
| MLXP | 205 | `websockets==16.0`; includes `py-spy` |

Only Kakao currently has a micromamba `openpi` environment. SKT has neither the
saved `DEXJOCO_OPENPI_ENV` setting nor the environment, and MLXP's defaults name
an `openpi` environment that is absent. pi0.5 evaluation is therefore not
portable across the three clusters in the present state.

## Audit baseline: GAM source and environment parity before remediation

### Checkpoint provenance

All three completed GAM checkpoints were trained from:

```text
ffc96674bd866e0c55371a8eaf1b9579916cbfd2
```

The bimanual training submission created that snapshot commit, and the
multitask and single-arm runs reused it.

### Training-time temporal alignment defect

At `ffc96674`, the DexJoCo dataset used visual and proprio anchors
`t, t+1, ..., t+7`, but sliced 128 low-level actions and reshaped them into
eight non-overlapping 16-action chunks. Thus chunk 1 represented `t...t+15`,
chunk 2 represented `t+16...t+31`, and so on, while the corresponding geometry
anchors advanced by only one frame.

The established GAM LIBERO data path and `DEXJOCO_INTEGRATION.md` both define
geometry anchors at `chunk_size * temporal_stride`. Commit `69afa536` fixes the
DexJoCo path to use anchors `t, t+16, ..., t+112` at stride 1, applies temporal
stride consistently to both action rows and anchors, and includes regression
tests for both stride 1 and stride 2. This changes the training targets and
requires retraining all three GAM variants.

### SKT eval pinning problem

The cancelled SKT jobs 156609 and 156610 recorded:

```text
git_commit=6b71f21862382293e2edb597334125a13f9aada9
git_dirty_at_submit=true
git_committed_dirty=false
```

The eval harness creates a detached clean worktree at the recorded commit.
Consequently, it excludes the dirty SKT changes that add:

- raw DexJoCo proprioception passthrough;
- 23/46-D proprio handling;
- 22/44-D action-history handling;
- the DexJoCo LeRobot dataset registration;
- the fixed GAM result layout.

The GAM server imports `load_stage1_policy` from that pinned worktree. The clean
`6b71f218` source lacks the required non-LIBERO dimensional handling, so those
jobs were not valid GAM evaluations even before considering the rollout-mode
mismatch.

SKT does not currently contain the `ffc96674` Git object. Kakao has no configured
`GAM_DIR` and no GAM checkout at the expected path.

### SKT versus MLXP GAM integration

The core model/eval source files otherwise have matching content, but two
integration files differ:

- MLXP's training wrapper adds `--reset-schedule` for finetuning from
  `pretrained-gam.pt` and adds optional W&B wiring; SKT lacks these changes.
- MLXP's dataset reader selects the sole non-wrist camera, including
  `observation.images.ego_right` for `click_mouse`, and supplies
  `effective_fps`/`max_stride`; SKT lacks these changes.

The GAM runtime core versions match on SKT and MLXP:

```text
Python 3.12.13
PyTorch 2.5.1+cu124
NumPy 1.26.4
```

Their package sets are close but not identical: SKT additionally contains
`depth-anything-3` and `opencv-python`.

## PhysiXel source and environment parity

The three main PhysiXel checkouts point at different current revisions, but all
three contain the experiment-pinned commit:

```text
9faf40b35770763f4c7650db2094b66cb4328918
```

Therefore the job worktree pin can make the PhysiXel source reproducible.
However, the `.venv` package manifests differ: Kakao and MLXP contain 123
packages, while SKT contains 125. Core versions match at Python 3.10.12,
PyTorch 2.7.0+cu126, and NumPy 1.26.4.

## Impact on evaluations

- Jobs 156609 and 156610 were correctly cancelled and should not contribute to
  GAM scores.
- Any results produced with the current ratio-0.5 sync mode are blocking-overlap
  results, not Simon-style full-chunk sync results.
- Direct use of Kakao's current staging would run different client arguments
  from SKT.
- MLXP resume attempts can fail because its client lacks `--start-episode` while
  the current harness may supply it.
- PhysiXel checkpoints do not need retraining for the rollout corrections.
- The three completed GAM checkpoints do need retraining because of the
  independently confirmed training-time anchor/chunk misalignment.

## Recommended synchronization sequence

1. Define `sync` as Simon-style full-chunk rollout: request only when the
   buffer drains, wait without advancing simulation, and execute without
   overlap.
2. Preserve the current ratio-based blocking behavior under a separate name,
   such as `blocking_overlap`, if it is still needed experimentally.
3. Port the reference horizon handling and rollout metrics into one canonical
   DexJoCo commit while retaining the useful episode-ID, resume, websocket, and
   render-watchdog fixes.
4. Use one comparison operator and document it. The reference uses `<=` for
   async replanning.
5. Commit and push the canonical DexJoCo source. Stop executing mutable dirty
   clients and pin the DexJoCo Git revision in submission metadata.
6. Create a canonical GAM commit containing the MLXP `ffc96674` inference
   support plus the current training-wrapper and camera-selection fixes, then a
   separate commit correcting chunk-aligned geometry/proprio anchors.
7. Transfer the corrected GAM commit to SKT and Kakao, configure `GAM_DIR`, add
   an explicit `TRAIN_GIT_COMMIT` to every GAM experiment, and retrain all three
   GAM variants.
8. Recreate the DexJoCo, GAM, and PhysiXel environments from checked-in lock
   files and verify normalized package manifests.
9. Restage train-eval-web to Kakao and SKT and verify that MLXP renders the same
   harness/config snapshot.
10. Add submission preflight checks for client SHA, accepted CLI flags, model
    commit availability, clean worktrees, action chunk size, and environment
    fingerprints.
11. Resubmit PhysiXel with the existing checkpoints and GAM only with the new
    corrected-training checkpoints.

## Recommended rollout-mode contract

```text
sync
  Full returned chunk; no overlap; simulation waits at an empty buffer.
  replan_ratio is ignored.

async
  Latency-hiding threshold replan; simulation continues; expired action prefix
  may be dropped; overlap may be blended.

blocking_overlap (optional)
  Threshold replan with overlap blending, but pause simulation while waiting
  for the replacement chunk. This describes the current sync implementation.
```
