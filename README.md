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
