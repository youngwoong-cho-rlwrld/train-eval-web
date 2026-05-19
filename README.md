# train-eval-web

Local-first web UI for orchestrating GR00T training & evaluation across multi-slurm clusters (kakao + skt).

Runs on your Mac, talks to clusters over SSH. Configs live in this repo (`configs/experiments/<variant>/config.sh`) and are pushed to clusters on each submit — no train-eval-scripts dependency on the cluster side.

## Layout

```
train-eval-web/
├── configs/              # source of truth for experiments
│   ├── clusters/         # kakao.env, skt.env
│   └── experiments/      # <variant>/config.sh, editable in UI
├── lib/                  # body scripts run by sbatch (train_body.sh, eval_body.sh, ...)
├── backend/              # FastAPI + asyncssh
├── frontend/             # Next.js + shadcn/ui
└── scripts/run.sh        # boots both in dev mode
```

## Quick start

Prerequisites on your Mac: `node` ≥ 20, `npm`, `uv` (`brew install node uv`), and SSH access to `kakao-login-1` and `skt` in your `~/.ssh/config`.

```bash
./scripts/run.sh
```

Opens `http://localhost:3000`.

## Submit flow

1. Edit / create an experiment in the UI → writes `configs/experiments/<name>/config.sh` locally
2. On submit, backend rsyncs `configs/` + `lib/` to `~/.train-eval-web/` on the chosen cluster
3. `sbatch ~/.train-eval-web/lib/train_body.sh` runs on the cluster
4. UI tails logs / shows status over SSH

The cluster copy at `~/.train-eval-web/` is a transient mirror — it gets overwritten on every submit. Source of truth is always this repo.

## Model-code changes and training snapshots

Experiment configs and the web UI live in this repo, but GR00T model-code changes live in the actual training repos:

- N1.5 variants use the configured `gr00t` repo.
- N1.6 variants use the configured `gr00t-n16` repo.
- Slurm repo paths come from `configs/clusters/<cluster>.env` via `GROOT_DIR` and `GROOT_N16_DIR`.
- MLXP repo paths are the DDN workspaces used by the MLXP submitter.

When you submit a training job, the webapp checks the selected model repo for the chosen variant and backend. For example, an N1.6 training submission checks `gr00t-n16`, not `train-eval-web`.

If the selected model repo is clean, the job submits immediately and the job detail page records that model repo commit in the Submission Snapshot. If the selected model repo has uncommitted changes, the submit UI opens a confirmation modal showing the dirty files and repo path. Clicking **Commit and submit** commits those model-code changes in the training repo, then submits the job and records the new commit hash.

So the normal workflow for changing N1.6 training code is:

1. Edit the N1.6 code in `gr00t-n16`.
2. Edit experiment config or submission options in `train-eval-web` if needed.
3. Submit an N1.6 training job from the UI.
4. If prompted, review the dirty `gr00t-n16` files and click **Commit and submit**.
5. Check the job detail page's Submission Snapshot for the effective config, code repo, repo path, training commit, and dirty-state record.
