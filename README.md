# train-eval-web

Local-first web UI for orchestrating GR00T training & evaluation across multiple clusters: the slurm clusters **kakao** and **skt** (over SSH), and **Naver MLXP** (Kubernetes, over `kubectl`).

Runs on your Mac, talking to the slurm clusters over SSH and to MLXP over `kubectl`. Configs live in this repo (`configs/experiments/<experiment>/config.sh`); on each slurm submit they're pushed to the cluster, while MLXP inlines the body script into the k8s Job — no train-eval-scripts dependency on the cluster side.

## Layout

```
train-eval-web/
├── configs/              # source of truth for experiments
│   ├── clusters/         # kakao.env, skt.env (slurm), mlxp.env (Naver k8s)
│   ├── models/           # <model>.env: repo/body-script/runtime definitions
│   └── experiments/      # <experiment>/config.sh, editable in UI
├── lib/                  # body scripts (train_body.sh, eval_body.sh, *_n16.sh, _common.sh)
├── backend/              # FastAPI; shells out to system ssh/rsync + kubectl
├── frontend/             # Next.js + shadcn/ui
└── scripts/run.sh        # boots both in dev mode
```

## Quick start

Prerequisites on your Mac:

- `node` ≥ 20, `npm`, `uv`, and `bash` ≥ 4 — `brew install node uv bash` (macOS's built-in bash 3.2 is too old; the backend sources cluster envs with bash ≥ 4)
- SSH access to `kakao-login-1` and `skt` in your `~/.ssh/config` (for the slurm clusters)
- `kubectl` configured for the Naver MLXP cluster (for MLXP)

```bash
./scripts/run.sh
```

Opens `http://localhost:3000`.

## Submit flow

### Slurm (kakao, skt)

1. Edit / create an experiment in the UI → writes `configs/experiments/<name>/config.sh` locally
2. On submit, backend rsyncs `configs/` + `lib/` to `~/.train-eval-web/` on the chosen cluster
3. `sbatch ~/.train-eval-web/lib/train_body.sh` runs on the cluster
4. UI tails logs / shows status over SSH

The cluster copy at `~/.train-eval-web/` is a transient mirror — it gets overwritten on every submit. Source of truth is always this repo.

### MLXP (Naver, Kubernetes)

1. Configure your user-specific MLXP settings (namespace, DDN mount, image, …) on the **Settings** page — these fill `configs/clusters/mlxp.env`.
2. On submit, backend renders a k8s Job YAML with the body script inlined into the Job spec (no file sync — the gr00t repo already lives on the MLXP DDN) and `kubectl apply`s it. You pick `num_gpus` instead of a partition.
3. UI shows logs / status via `kubectl logs` / `kubectl get pod`.

## Model-code changes and training snapshots

Experiment configs and the web UI live in this repo, but model-code changes live in the actual training repos. The mapping is data-driven:

- Experiments select a model with `MODEL_ID=<model>` in `configs/experiments/<experiment>/config.sh`.
- Models are defined in `configs/models/<model>.env`.
- Slurm repo paths come from the model's `SLURM_REPO_VAR`, resolved against `configs/clusters/<cluster>.env`.
- MLXP repo paths come from the model's `MLXP_REPO_DIR`.

When you submit a training job, the webapp checks the selected model repo for the chosen experiment and backend. For example, an experiment with `MODEL_ID=n1.6` checks `gr00t-n16`, not `train-eval-web`.

If the selected model repo is clean, the job submits immediately and the job detail page records that model repo commit in the Submission Snapshot. If the selected model repo has uncommitted changes, the submit UI opens a confirmation modal showing the dirty files and repo path. Clicking **Commit and submit** commits those model-code changes in the training repo, then submits the job and records the new commit hash.

So the normal workflow for changing N1.6 training code is:

1. Edit the N1.6 code in `gr00t-n16`.
2. Edit experiment config or submission options in `train-eval-web` if needed.
3. Submit an N1.6 training job from the UI.
4. If prompted, review the dirty `gr00t-n16` files and click **Commit and submit**.
5. Check the job detail page's Submission Snapshot for the effective config, code repo, repo path, training commit, and dirty-state record.

To add a new N1.6-compatible model repo, add `configs/models/<model>.env`, point `SLURM_REPO_VAR` at a cluster env variable, add that variable to the relevant `configs/clusters/<cluster>.env`, then set `MODEL_ID=<model>` in the experiment config.
