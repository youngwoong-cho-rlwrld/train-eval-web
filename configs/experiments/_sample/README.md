# _sample — variant template

Reference variant for starting your own. To create one:

```bash
cp -r configs/experiments/_sample configs/experiments/<your-variant-name>
$EDITOR configs/experiments/<your-variant-name>/config.sh
```

Your new variant directory is automatically picked up by the Submit page (via `/api/variants`) — no restart needed. It stays out of git because `configs/experiments/*` is `.gitignore`'d except for this `_sample` dir.

## What `config.sh` actually does

It's plain bash, sourced by the model's configured body scripts on the cluster. Set `MODEL_ID` to one of `configs/models/<model>.env`. Two key dataset modes:

- **Single-task** — set `DATASET_NAME` (one dataset under `$DATA_DIR`).
- **Multi-task** — set `DATASETS=("name|data_config|weight" ...)` instead (no `DATASET_NAME`).

The Submit page reads these at job-submission time, renders a `data_config.yaml` from them, and either `sbatch`'s a slurm body or `kubectl apply`'s a k8s Job depending on the selected cluster.

## Submitting

1. Open `http://localhost:3000/submit`
2. Pick **cluster** (kakao / skt / mlxp)
3. Pick your **experiment** from the dropdown
4. (slurm) pick a **partition**; (mlxp) pick a **node** + **GPU count**
5. (optional) override the **dataset(s)** for this run only
6. Hit **Submit**

## Sharing

To share a variant with teammates, commit it under `_sample/` (so it survives the gitignore) and reference it as your starting point. Don't commit personal variants directly — keep them local.
