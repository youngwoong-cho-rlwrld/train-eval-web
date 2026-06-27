"""DexJoCo task/family enumeration.

Lists the DexJoCo eval config families and their tasks by scanning
`$DEXJOCO_DIR/configs/<family>/*.yaml` on the chosen cluster over SSH.

Each subdirectory of `configs/` is a config family (rand_obj, rand_full,
multi_task, ipad_reasoning, ...). Each `*.yaml` stem inside a family is a
task (env_name) selectable from the submit page's task picker.

`DEXJOCO_DIR` is resolved from the cluster's saved env (see
configs/clusters/<cluster>.env). When it's unset we return empty lists so
the picker just shows nothing rather than erroring.
"""

from pydantic import BaseModel

from .clusters import load_cluster
from .ssh import ssh_run


class DexjocoFamily(BaseModel):
    family: str
    tasks: list[str]


class DexjocoTasks(BaseModel):
    families: list[DexjocoFamily]
    tasks: list[str]            # sorted union of all yaml stems across families


# A single python -c is more robust than a bash loop for walking the config
# tree and avoiding quoting hell over ssh — same approach as datasets.py.
def _list_py(dexjoco_dir: str) -> str:
    # `dexjoco_dir` is interpolated as a quoted string literal inside python.
    # Escape backslashes and single quotes so the embedded string is safe.
    safe = dexjoco_dir.replace("\\", "\\\\").replace("'", "\\'")
    return rf"""
import os, glob
base = os.path.expanduser('{safe}')
cfg = os.path.join(base, 'configs')
for d in sorted(glob.glob(os.path.join(cfg, '*'))):
    if not os.path.isdir(d):
        continue
    family = os.path.basename(d.rstrip('/'))
    stems = sorted({{
        os.path.splitext(os.path.basename(p))[0]
        for p in glob.glob(os.path.join(d, '*.yaml'))
    }})
    print('|'.join([family] + stems))
"""


def _shell_quote(s: str) -> str:
    """Wrap an arbitrary string in single quotes for inline shell use."""
    return "'" + s.replace("'", "'\"'\"'") + "'"


def _parse_lines(text: str) -> DexjocoTasks:
    families: list[DexjocoFamily] = []
    union: set[str] = set()
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split("|")
        family = parts[0]
        tasks = [t for t in parts[1:] if t]
        families.append(DexjocoFamily(family=family, tasks=tasks))
        union.update(tasks)
    return DexjocoTasks(families=families, tasks=sorted(union))


async def list_dexjoco_tasks(cluster: str) -> DexjocoTasks:
    """List DexJoCo config families + tasks for `cluster` over SSH.

    Returns empty lists when DEXJOCO_DIR is unset for the cluster.
    """
    env = await load_cluster(cluster)
    dexjoco_dir = (env.vars.get("DEXJOCO_DIR") or "").strip()
    if not dexjoco_dir:
        return DexjocoTasks(families=[], tasks=[])
    script = _list_py(dexjoco_dir)
    r = await ssh_run(env.ssh_alias, f"python3 -c {_shell_quote(script)}", timeout=30.0)
    if r.returncode != 0:
        raise RuntimeError(f"list_dexjoco_tasks({cluster}, {dexjoco_dir}) failed: {r.stderr}")
    return _parse_lines(r.stdout)
