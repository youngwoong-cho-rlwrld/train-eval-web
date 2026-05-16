"""Move a job's final checkpoint to another server.

Source layouts:
- mlxp:   /data/youngwoong/experiments/<variant>/checkpoints/<job_name>/checkpoint-N
- slurm:  $EXP_DIR/checkpoints/checkpoint-N

Destination layouts: same shape as the source per cluster (no nested
job-name dir on slurm, because the slurm body never created one).

Transfer mechanism:
- slurm → slurm: rsync over ssh (host A → host B).
- slurm → mlxp:  ssh tar | kubectl exec tar x on a DDN-mounted pod.
- mlxp  → slurm: kubectl exec tar | ssh tar x on the dest host.
- mlxp  → mlxp:  cp inside one pod (same DDN PVC).
"""

import asyncio
import shlex
import time
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .clusters import load_cluster
from .details import parse_phase_and_variant
from .jobs import get_job
from .mlxp_data_pod import ensure_listing_pod, NAMESPACE as MLXP_NS
from .ssh import ssh_run, _CM_OPTS


class MoveCheckpointRequest(BaseModel):
    dest_cluster: str
    # Absolute directory on dest. Each source's basename is appended.
    # None → mirrors source layout per cluster.
    dest_path_root: str | None = None
    # Explicit list of source checkpoint paths to copy. Empty → fall back to
    # auto-picking the latest under this job's dir.
    sources: list[str] = []
    delete_source: bool = False


class CheckpointEntry(BaseModel):
    path: str
    job_name: str
    step: int


class MoveResult(BaseModel):
    source: str
    dest: str


class MoveCheckpointResponse(BaseModel):
    dest_cluster: str
    moves: list[MoveResult]
    stdout: str = ""


class MoveCheckpointStartResponse(BaseModel):
    move_id: str


class MoveJobStatus(BaseModel):
    move_id: str
    status: str  # "running" | "done" | "error"
    error: str | None = None
    moves_total: int
    moves_done: int
    current_source: str | None = None
    current_dest: str | None = None
    src_size_bytes: int | None = None
    dest_size_bytes: int | None = None
    started_at: float
    finished_at: float | None = None


# In-memory registry of background transfer tasks. Keyed by move_id.
# Lost on uvicorn restart — fine for the user's solo workflow.
_MOVE_JOBS: dict[str, MoveJobStatus] = {}
# Parallel registry for runtime handles (asyncio Task + current subprocess).
# Lets `cancel_move` interrupt an in-flight transfer.
_MOVE_HANDLES: dict[str, dict] = {}


def _track_proc(move_id: str, proc: asyncio.subprocess.Process) -> None:
    h = _MOVE_HANDLES.get(move_id)
    if h is not None:
        h["proc"] = proc


def cancel_move(move_id: str) -> bool:
    """Kill the in-flight subprocess (if any) and cancel the asyncio task.
    Returns True if a running move was found."""
    h = _MOVE_HANDLES.get(move_id)
    if not h:
        return False
    proc = h.get("proc")
    if proc is not None and proc.returncode is None:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
    task = h.get("task")
    if task is not None and not task.done():
        task.cancel()
    state = _MOVE_JOBS.get(move_id)
    if state and state.status == "running":
        state.status = "error"
        state.error = "cancelled"
        state.finished_at = time.time()
    return True


# ── list checkpoints ────────────────────────────────────────────────────

async def list_checkpoints(cluster: str, job_id: str) -> list[CheckpointEntry]:
    """One entry per run folder.

    MLXP runs nest under `<variant>/checkpoints/<job_name>/checkpoint-N` —
    each `<job_name>/` is one run folder, and we tag it with its latest
    checkpoint step. Slurm flat layout has `<variant>/checkpoints/checkpoint-N`
    with no per-run nesting; we surface each `checkpoint-N` as its own folder.
    """
    sacct = await get_job(cluster, job_id)
    _, variant = parse_phase_and_variant(sacct.get("JobName") or job_id, cluster)
    if not variant:
        return []

    if cluster == "mlxp":
        pod = await ensure_listing_pod()
        # Emit `<run_dir>|<latest_step>` for every per-job dir that has at
        # least one checkpoint-N inside.
        script = (
            r"""
shopt -s nullglob
for d in /data/youngwoong/experiments/""" + shlex.quote(variant) + r"""/checkpoints/*/; do
    matches=( "$d"checkpoint-* )
    [ ${#matches[@]} -eq 0 ] && continue
    latest=$(for m in "${matches[@]}"; do basename "$m" | sed 's:^checkpoint-::'; done | sort -n | tail -1)
    printf '%s|%s\n' "${d%/}" "$latest"
done
""")
        proc = await asyncio.create_subprocess_exec(
            "kubectl", "exec", "-n", MLXP_NS, pod, "--", "bash", "-c", script,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=15.0)
        result: list[CheckpointEntry] = []
        for line in out.decode().splitlines():
            parts = line.split("|")
            if len(parts) != 2 or not parts[1].isdigit():
                continue
            path, step = parts
            result.append(CheckpointEntry(
                path=path, job_name=path.rsplit("/", 1)[-1], step=int(step),
            ))
        return result

    # slurm — probe both known exp roots; merge as flat checkpoint-N entries.
    env = await load_cluster(cluster)
    roots = [
        f"$HOME/train-eval-scripts/experiments/{variant}/checkpoints",
        f"$HOME/.train-eval-web/experiments/{variant}/checkpoints",
    ]
    cmds = " ; ".join(f"ls -d {r}/checkpoint-* 2>/dev/null" for r in roots)
    r = await ssh_run(env.ssh_alias, cmds, timeout=15.0)
    return _parse_paths(r.stdout, nested=False, fallback_job=job_id)


def _parse_paths(stdout: str, *, nested: bool, fallback_job: str = "") -> list[CheckpointEntry]:
    out: list[CheckpointEntry] = []
    for line in stdout.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        # Path looks like .../checkpoint-N. Step = N.
        leaf = line.rsplit("/", 1)[-1]
        if not leaf.startswith("checkpoint-"):
            continue
        try:
            step = int(leaf.split("-", 1)[1])
        except ValueError:
            continue
        job_name = line.rsplit("/", 2)[1] if nested else fallback_job
        out.append(CheckpointEntry(path=line, job_name=job_name, step=step))
    return out


# ── transfer ────────────────────────────────────────────────────────────

async def start_move(
    src_cluster: str, src_job: str, dest_cluster: str,
    sources: list[str], dest_path_root: str | None = None,
    delete_source: bool = False,
) -> str:
    """Kick the transfer off in a background task. Returns move_id; the
    client polls /api/move-jobs/<id> for progress."""
    if not sources:
        raise ValueError("no checkpoints selected")

    sacct = await get_job(src_cluster, src_job)
    _, variant = parse_phase_and_variant(sacct.get("JobName") or src_job, src_cluster)
    if not variant:
        raise ValueError("could not resolve variant from job name")

    if not dest_path_root:
        if dest_cluster == "mlxp":
            dest_path_root = f"/data/youngwoong/experiments/{variant}/checkpoints"
        else:
            dest_path_root = f"$HOME/.train-eval-web/experiments/{variant}/checkpoints"

    move_id = uuid.uuid4().hex[:12]
    _MOVE_JOBS[move_id] = MoveJobStatus(
        move_id=move_id, status="running",
        moves_total=len(sources), moves_done=0,
        started_at=time.time(),
    )
    _MOVE_HANDLES[move_id] = {"task": None, "proc": None}
    task = asyncio.create_task(
        _run_move(
            move_id, src_cluster, dest_cluster, sources, dest_path_root, delete_source,
        )
    )
    _MOVE_HANDLES[move_id]["task"] = task
    return move_id


def get_move_status(move_id: str) -> MoveJobStatus | None:
    return _MOVE_JOBS.get(move_id)


async def _children_of(cluster: str, path: str) -> list[str]:
    """ls -1 names of immediate children under `path`. [] if missing."""
    cmd = f"ls -1 {shlex.quote(path)} 2>/dev/null"
    if cluster == "mlxp":
        pod = await ensure_listing_pod()
        proc = await asyncio.create_subprocess_exec(
            "kubectl", "exec", "-n", MLXP_NS, pod, "--", "bash", "-c", cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=15.0)
        return [s for s in out.decode().splitlines() if s]
    env = await load_cluster(cluster)
    r = await ssh_run(env.ssh_alias, cmd, timeout=15.0)
    return [s for s in r.stdout.splitlines() if s]


async def _resolve_run_layout(cluster: str, src_path: str) -> tuple[bool, str | None]:
    """Return (is_run_folder, latest_checkpoint_basename).

    True when `src_path` contains at least one `checkpoint-N` child — we
    treat it as a per-run folder and copy top-level files + the highest
    `checkpoint-N` only. Otherwise it's the checkpoint dir itself.
    """
    names = await _children_of(cluster, src_path)
    steps: list[tuple[int, str]] = []
    for n in names:
        if not n.startswith("checkpoint-"):
            continue
        try:
            steps.append((int(n.split("-", 1)[1]), n))
        except ValueError:
            continue
    if not steps:
        return False, None
    steps.sort()
    return True, steps[-1][1]


def _selective_tar_files(parent_basename: str, names: list[str], latest: str) -> list[str]:
    """Build the tar arg list for a per-run folder: every non-`checkpoint-*`
    child + the chosen `checkpoint-N`."""
    items = [
        f"{parent_basename}/{n}"
        for n in names
        if not n.startswith("checkpoint-")
    ]
    items.append(f"{parent_basename}/{latest}")
    return items


async def _run_move(
    move_id: str, src_cluster: str, dest_cluster: str,
    sources: list[str], dest_path_root: str, delete_source: bool,
) -> None:
    state = _MOVE_JOBS[move_id]
    try:
        for i, src_path in enumerate(sources):
            leaf = Path(src_path).name
            dest_path = f"{dest_path_root.rstrip('/')}/{leaf}"
            state.current_source = src_path
            state.current_dest = dest_path
            state.dest_size_bytes = 0

            # Per-run folder (has checkpoint-* children) → copy top-level
            # files + the latest checkpoint-N only. Otherwise copy as-is.
            is_run, latest = await _resolve_run_layout(src_cluster, src_path)
            include_only = None
            if is_run and latest:
                names = await _children_of(src_cluster, src_path)
                include_only = _selective_tar_files(leaf, names, latest)
                # Sum sizes of just the picks for the progress denominator.
                picks = [n for n in names if not n.startswith("checkpoint-")] + [latest]
                sizes = await asyncio.gather(*[
                    _size(src_cluster, f"{src_path}/{name}") for name in picks
                ])
                state.src_size_bytes = sum(sizes)
            else:
                state.src_size_bytes = await _size(src_cluster, src_path)

            poll_task = asyncio.create_task(
                _poll_dest_size(state, dest_cluster, dest_path)
            )
            try:
                if src_cluster == "mlxp" and dest_cluster == "mlxp":
                    await _mlxp_to_mlxp(move_id, src_path, dest_path, include_only)
                elif src_cluster == "mlxp":
                    await _mlxp_to_slurm(move_id, src_path, dest_cluster, dest_path, include_only)
                elif dest_cluster == "mlxp":
                    await _slurm_to_mlxp(move_id, src_cluster, src_path, dest_path, include_only)
                else:
                    await _slurm_to_slurm(move_id, src_cluster, src_path, dest_cluster, dest_path, include_only)
            finally:
                poll_task.cancel()

            # Snapshot final size before the next source overwrites state.
            state.dest_size_bytes = state.src_size_bytes
            state.moves_done = i + 1

            if delete_source:
                await _delete_source(src_cluster, src_path)

        state.status = "done"
        state.finished_at = time.time()
    except asyncio.CancelledError:
        # cancel_move set status to error/cancelled already; just exit.
        return
    except Exception as e:
        state.status = "error"
        state.error = str(e)
        state.finished_at = time.time()
    finally:
        _MOVE_HANDLES.pop(move_id, None)


async def _poll_dest_size(state: MoveJobStatus, cluster: str, path: str) -> None:
    """Stat the destination periodically while the transfer runs."""
    try:
        while True:
            try:
                size = await _size(cluster, path)
                state.dest_size_bytes = size
            except Exception:
                pass
            await asyncio.sleep(2.0)
    except asyncio.CancelledError:
        pass


async def _size(cluster: str, path: str) -> int:
    """Return the byte size of a directory (or file). 0 if missing."""
    cmd = f"du -sb {shlex.quote(path)} 2>/dev/null | awk '{{print $1}}'"
    if cluster == "mlxp":
        pod = await ensure_listing_pod()
        proc = await asyncio.create_subprocess_exec(
            "kubectl", "exec", "-n", MLXP_NS, pod, "--", "bash", "-c", cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=15.0)
        s = out.decode().strip()
    else:
        env = await load_cluster(cluster)
        r = await ssh_run(env.ssh_alias, cmd, timeout=15.0)
        s = r.stdout.strip()
    try:
        return int(s)
    except ValueError:
        return 0


async def _mlxp_to_mlxp(move_id: str, src: str, dest: str,
                         include_only: list[str] | None = None) -> str:
    pod = await ensure_listing_pod()
    src_parent = str(Path(src).parent)
    leaf = Path(src).name
    dest_parent = str(Path(dest).parent)
    if include_only:
        # Reproduce just the selected children under dest/.
        copies = " && ".join(
            f"cp -r {shlex.quote(src_parent)}/{shlex.quote(item)} "
            f"{shlex.quote(dest_parent)}/{shlex.quote(item)}"
            for item in include_only
        )
        cmd = (
            f"mkdir -p {shlex.quote(dest)} && "
            + copies
        )
    else:
        cmd = (
            f"mkdir -p {shlex.quote(dest_parent)} && "
            f"cp -r {shlex.quote(src)} {shlex.quote(dest)}"
        )
    proc = await asyncio.create_subprocess_exec(
        "kubectl", "exec", "-n", MLXP_NS, pod, "--", "bash", "-c", cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _track_proc(move_id, proc)
    out, err = await asyncio.wait_for(proc.communicate(), timeout=3600.0)
    if proc.returncode != 0:
        raise RuntimeError(f"cp failed: {err.decode(errors='replace').strip()}")
    return out.decode(errors="replace")


async def _slurm_to_slurm(move_id: str, src_cluster: str, src_path: str,
                           dest_cluster: str, dest_path: str,
                           include_only: list[str] | None = None) -> str:
    src_env = await load_cluster(src_cluster)
    dest_env = await load_cluster(dest_cluster)
    dest_alias = shlex.quote(dest_env.ssh_alias)
    if include_only:
        # Two-step: top-level files (exclude every other checkpoint-*), then
        # the chosen checkpoint-N as a single rsync.
        leaf = Path(src_path).name
        wanted = {it.split("/", 1)[1] for it in include_only}
        latest_ck = next((n for n in wanted if n.startswith("checkpoint-")), None)
        cmd = (
            f"mkdir -p {shlex.quote(dest_path)} >/dev/null 2>&1 && "
            f"rsync -az --exclude='checkpoint-*' "
            f"{shlex.quote(src_path)}/ {dest_alias}:{shlex.quote(dest_path)}/"
        )
        if latest_ck:
            cmd += (
                f" && rsync -az {shlex.quote(src_path)}/{shlex.quote(latest_ck)} "
                f"{dest_alias}:{shlex.quote(dest_path)}/"
            )
    else:
        cmd = (
            f"mkdir -p {shlex.quote(str(Path(dest_path).parent.as_posix()))} >/dev/null 2>&1; "
            f"rsync -az {shlex.quote(src_path)}/ "
            f"{dest_alias}:{shlex.quote(dest_path)}/"
        )
    r = await ssh_run(src_env.ssh_alias, cmd, timeout=3600.0)
    if r.returncode != 0:
        raise RuntimeError(f"rsync failed: {r.stderr.strip() or r.stdout.strip()}")
    return r.stdout


async def _mlxp_to_slurm(move_id: str, src_path: str, dest_cluster: str, dest_path: str,
                          include_only: list[str] | None = None) -> str:
    pod = await ensure_listing_pod()
    dest_env = await load_cluster(dest_cluster)
    src_parent = str(Path(src_path).parent)
    src_leaf = Path(src_path).name
    dest_parent = str(Path(dest_path).parent.as_posix())

    tar_items = " ".join(shlex.quote(it) for it in (include_only or [src_leaf]))
    ssh_opts = " ".join(_CM_OPTS)
    pipeline = (
        f"kubectl exec -i -n {shlex.quote(MLXP_NS)} {shlex.quote(pod)} -- "
        f"tar c -C {shlex.quote(src_parent)} {tar_items} | "
        f"ssh -o BatchMode=yes {ssh_opts} {shlex.quote(dest_env.ssh_alias)} "
        f"\"mkdir -p {shlex.quote(dest_parent)} && tar x -C {shlex.quote(dest_parent)}\""
    )
    proc = await asyncio.create_subprocess_shell(
        f"set -o pipefail; {pipeline}",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _track_proc(move_id, proc)
    _, err = await asyncio.wait_for(proc.communicate(), timeout=3600.0)
    if proc.returncode != 0:
        raise RuntimeError(
            f"mlxp→slurm tar pipe failed (rc={proc.returncode}): "
            f"{err.decode(errors='replace').strip()}"
        )
    return f"copied {src_path} → {dest_env.ssh_alias}:{dest_path}"


async def _slurm_to_mlxp(move_id: str, src_cluster: str, src_path: str, dest_path: str,
                          include_only: list[str] | None = None) -> str:
    src_env = await load_cluster(src_cluster)
    pod = await ensure_listing_pod()
    src_parent = str(Path(src_path).parent.as_posix())
    src_leaf = Path(src_path).name
    dest_parent = str(Path(dest_path).parent)

    tar_items = " ".join(shlex.quote(it) for it in (include_only or [src_leaf]))
    ssh_opts = " ".join(_CM_OPTS)
    pipeline = (
        f"ssh -o BatchMode=yes {ssh_opts} {shlex.quote(src_env.ssh_alias)} "
        f"\"tar c -C {shlex.quote(src_parent)} {tar_items}\" | "
        f"kubectl exec -i -n {shlex.quote(MLXP_NS)} {shlex.quote(pod)} -- "
        f"bash -c \"mkdir -p {shlex.quote(dest_parent)} && tar x -C {shlex.quote(dest_parent)}\""
    )
    proc = await asyncio.create_subprocess_shell(
        f"set -o pipefail; {pipeline}",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _track_proc(move_id, proc)
    _, err = await asyncio.wait_for(proc.communicate(), timeout=3600.0)
    if proc.returncode != 0:
        raise RuntimeError(
            f"slurm→mlxp tar pipe failed (rc={proc.returncode}): "
            f"{err.decode(errors='replace').strip()}"
        )
    return f"copied {src_env.ssh_alias}:{src_path} → mlxp:{dest_path}"


async def _delete_source(src_cluster: str, src_path: str) -> None:
    if src_cluster == "mlxp":
        pod = await ensure_listing_pod()
        proc = await asyncio.create_subprocess_exec(
            "kubectl", "exec", "-n", MLXP_NS, pod, "--",
            "rm", "-rf", src_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await asyncio.wait_for(proc.communicate(), timeout=300.0)
        return
    env = await load_cluster(src_cluster)
    await ssh_run(env.ssh_alias, f"rm -rf {shlex.quote(src_path)}", timeout=300.0)
