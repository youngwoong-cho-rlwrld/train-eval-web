# _sample — variant template

Reference variant for starting your own. To create one:

```bash
cp -r configs/experiments/_sample configs/experiments/<your-variant-name>
$EDITOR configs/experiments/<your-variant-name>/config.sh
```

Your new variant directory is automatically picked up by the Submit page (via `/api/variants`) — no restart needed. It stays out of git because `configs/experiments/*` is `.gitignore`'d except for this `_sample` dir.

## Files in a variant directory

| File | When needed | Purpose |
|---|---|---|
| `config.sh` | always | The variant definition (plain bash, sourced on the cluster). |
| `modality_config.py` | n1.6 models only | Declares the video/state/action/language keys + action representation. `TRAIN_MODALITY_CONFIG` points at it. n1.5 variants don't use it and can delete it. |
| `dexjoco_config.py` / `dexjoco_config_dual_arm.py` | DexJoCo models only | Modality-config templates for single-arm / bimanual DexJoCo variants. Keep the one `TRAIN_MODALITY_CONFIG` points at; delete both for non-DexJoCo variants. |

`config.sh` is heavily commented — read it top to bottom; it's the real documentation. Highlights below.

## Model families

`MODEL_ID` selects `configs/models/<id>.env`, which fixes the model family, the train/eval body scripts, and the action-horizon mode:

| MODEL_ID | Family | Eval harness | Notes |
|---|---|---|---|
| `n1.5` | n1.5 | Isaac Sim | uses `DATA_CONFIG` + per-GPU `TRAIN_BATCH_SIZE` |
| `n1.6` | n1.6 | Isaac Sim | needs `modality_config.py` + `TRAIN_GLOBAL_BATCH_SIZE` |
| `physixel` | n1.6 | Isaac Sim | action horizon via `modality_and_cli` |
| `dexjoco-n16` | n1.6 | DexJoCo (MuJoCo) | |
| `dexjoco-physixel` | n1.6 | DexJoCo (MuJoCo) | |
| `dexjoco-pi05` | pi0.5 | DexJoCo (MuJoCo) | eval-only baseline, no training |

## Datasets

Set `DATA_DIR` to the root your LeRobot datasets live under (defaults to `~/datasets`). Then pick one mode:

- **n1.6 single-task** — `DATASET_NAME=<name>`
- **n1.6 multi-task** — `TRAIN_DATASET_NAMES=(a b …)` (optionally `TRAIN_DATASET_EMBODIMENT_TAGS=(…)` in parallel)
- **n1.5 single-task** — `DATASET_NAME=<name>` + `DATA_CONFIG=<cfg>`
- **n1.5 multi-task** — `DATASETS=("name|data_config|weight" …)`

`DATA_CONFIG` names (n1.5 only) must be data configs the model repo knows; copy one from an existing n1.5 variant or the model repo's `data` configs.

## Eval harnesses

`EVAL_HARNESS` (default `isaac`) picks how eval runs:

- **`isaac`** — Isaac Sim. Set `TASK_NAME` + `INSTRUCTION` (or a `TASKS` matrix), `EXECUTION_HORIZON`, `MAX_EPISODE_STEPS`, and `EVAL_SETS` (object-offset variants like `0cm 1cm 3cm 5cm 7cm`).
- **`dexjoco`** — MuJoCo benchmark. Set `DEXJOCO_TASK` (a config stem under `$DEXJOCO_DIR/configs/<family>/<task>.yaml`), `DEXJOCO_SERVER_TYPE` (`groot` or `openpi`), and `EVAL_SETS` = config families (`rand_obj`, `rand_full`, `multi_task`, `ipad_reasoning`). The Submit page's DexJoCo task picker lists the available tasks per cluster.

Either harness can run a **multi-task matrix** with `TASKS=("short|task_name|instruction" …)` — every entry is evaluated in one job.

## Submitting

1. Open `http://localhost:3000/submit`
2. Pick **cluster** (kakao / skt / mlxp)
3. Pick your **experiment** from the dropdown
4. (slurm) pick a **partition**; (mlxp) pick a **node** + **GPU count**
5. (optional) override the **dataset(s)** or eval knobs for this run only
6. Hit **Submit**

## Sharing

To share a variant with teammates, commit it under `_sample/` (so it survives the gitignore) and reference it as your starting point. Don't commit personal variants directly — keep them local.
