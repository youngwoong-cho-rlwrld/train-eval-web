# DexJoCo runtime manifests

These path-independent package inventories record the runtime environments
validated on Kakao, SKT, and MLXP on 2026-07-16. They are audit evidence, not
standalone installation requirement files: `dexjoco`, `openpi`, and the GAM
source are editable or path-injected packages pinned separately by Git commit.

| Runtime | Packages | SHA-256 |
| --- | ---: | --- |
| DexJoCo MuJoCo client (`6a6d1b2c`) | 47 | `78b2329bae8d52538407ef86438b03b6dcad6af90cb33016852190b29d945913` |
| OpenPI policy server (`6a6d1b2c`) | 175 | `ed6ed24cf77b6510b4c396cbabaf176105cedc0a32b377430645f06fc7ce6f74` |
| GAM policy runtime (`69afa536`) | 149 | `bfcdc86fea8eb7e66a542ea78de35bab2a93d102768a441041f6509e9e5f97d3` |

The client and OpenPI manifests are sorted output from
`python -m pip list --format=freeze`. The GAM manifest is the equivalent sorted
`importlib.metadata` inventory because that uv-managed environment does not
install the `pip` module.

The OpenPI server deliberately exposes `openpi_client` from the canonical
source tree with an absolute `.pth` and does not install the `openpi-client`
distribution. That distribution pins NumPy 1.26.4, while the OpenPI server
requires NumPy greater than 2. The server environment instead installs only the
client runtime dependencies it uses (`tree==0.2.4` and `websockets==16.1`).

OpenPI's own `install.bash` installs LeRobot 0.4.4 with `--no-deps` because its
full dependency set conflicts with OpenPI. Consequently, raw `pip check`
reports eleven LeRobot-only metadata issues: seven intentionally omitted
full-stack/hardware packages and four version preferences. Validation requires
that there are no failures outside those known LeRobot lines, and independently
checks the dataset imports and `serve_policy.py --help`.
