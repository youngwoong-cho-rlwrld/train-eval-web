# MLXP Isaac Eval Image

Builds the GR00T + IsaacLab/ALLEX evaluation runtime described in the
GR00T Training & Inference Guide.

The image keeps runtime software in `/workspace`:

- `/workspace/gr00t`
- `/workspace/IsaacLab`
- `/workspace/rlwrld_isaac`

Datasets, checkpoints, eval results, and large 3D assets should stay on MLXP
DDN storage under `/data`.

Build and push:

```bash
docker buildx build \
  --platform linux/amd64 \
  --ssh default=$HOME/.ssh/id_ed25519 \
  -t mlxp.kr.ncr.ntruss.com/youngwoong/train-eval-web-eval-isaac:20260528 \
  --push \
  docker/mlxp-eval-isaac
```
