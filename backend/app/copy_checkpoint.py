"""Copy a job's final checkpoint to another server.

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
from .details import get_details, resolve_phase_and_variant
from .jobs import get_job
from .mlxp_data_pod import ensure_listing_pod, NAMESPACE as MLXP_NS
from .ssh import ssh_run, _CM_OPTS


class CopyCheckpointRequest(BaseModel):
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


class CopyResult(BaseModel):
    source: str
    dest: str


class CopyCheckpointResponse(BaseModel):
    dest_cluster: str
    copies: list[CopyResult]
    stdout: str = ""


class CopyCheckpointStartResponse(BaseModel):
    copy_id: str


class CopyJobStatus(BaseModel):
    copy_id: str
    status: str  # "running" | "done" | "error"
    error: str | None = None
    copies_total: int
    copies_done: int
    current_source: str | None = None
    current_dest: str | None = None
    src_size_bytes: int | None = None
    dest_size_bytes: int | None = None
    started_at: float
    finished_at: float | None = None


# In-memory registry of background transfer tasks. Keyed by copy_id.
# Lost on uvicorn restart — fine for the user's solo workflow.
_COPY_JOBS: dict[str, CopyJobStatus] = {}
# Parallel registry for runtime handles (asyncio Task + current subprocess).
# Lets `cancel_copy` interrupt an in-flight transfer.
_COPY_HANDLES: dict[str, dict] = {}


def _track_proc(copy_id: str, proc: asyncio.subprocess.Process) -> None:
    h = _COPY_HANDLES.get(copy_id)
    if h is not None:
        h["proc"] = proc


def cancel_copy(copy_id: str) -> bool:
    """Kill the in-flight subprocess (if any) and cancel the asyncio task.
    Returns True if a running copy was found."""
    h = _COPY_HANDLES.get(copy_id)
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
    state = _COPY_JOBS.get(copy_id)
    if state and state.status == "running":
        state.status = "error"
        state.error = "cancelled"
        state.finished_at = time.time()
    return True


# ── list checkpoints ────────────────────────────────────────────────────

async def list_checkpoints(cluster: str, job_id: str) -> list[CheckpointEntry]:
    """One entry per run folder.

    MLXP runs nest under `<variant>/checkpoints/<job_name>/checkpoint-N`.
    Slurm jobs may use either `<variant>/checkpoints/checkpoint-N` or a
    one-level nested launcher directory. We surface every `checkpoint-N`.
    """
    if cluster == "mlxp":
        sacct = await get_job(cluster, job_id)
        _, variant = resolve_phase_and_variant(sacct.get("JobName") or job_id, sacct)
        if not variant:
            return []
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

    # slurm — prefer the exact path resolved for this job, then probe both
    # known roots for the normalized variant. Include one-level nested
    # checkpoint layouts used by newer launchers.
    env = await load_cluster(cluster)
    det = await get_details(cluster, job_id)
    variant = det.variant
    if not variant:
        return []
    roots = [
        det.paths.ckpt_dir,
        f"$HOME/train-eval-scripts/experiments/{variant}/checkpoints",
        f"$HOME/.train-eval-web/experiments/{variant}/checkpoints",
    ]
    deduped_roots = list(dict.fromkeys(r for r in roots if r))

    def path_expr(path: str) -> str:
        return path if path.startswith("$HOME/") else shlex.quote(path)

    cmds = " ; ".join(
        f"ls -d {path_expr(r)}/checkpoint-* {path_expr(r)}/*/checkpoint-* 2>/dev/null"
        for r in deduped_roots
    )
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
        parent = line.rsplit("/", 2)[1] if "/" in line else ""
        job_name = parent if (nested or parent != "checkpoints") else fallback_job
        out.append(CheckpointEntry(path=line, job_name=job_name, step=step))
    return out


# ── transfer ────────────────────────────────────────────────────────────

async def start_copy(
    src_cluster: str, src_job: str, dest_cluster: str,
    sources: list[str], dest_path_root: str | None = None,
    delete_source: bool = False,
) -> str:
    """Kick the transfer off in a background task. Returns copy_id; the
    client polls /api/copy-jobs/<id> for progress."""
    if not sources:
        raise ValueError("no checkpoints selected")

    if src_cluster == "mlxp":
        sacct = await get_job(src_cluster, src_job)
        _, variant = resolve_phase_and_variant(sacct.get("JobName") or src_job, sacct)
    else:
        variant = (await get_details(src_cluster, src_job)).variant
    if not variant:
        raise ValueError("could not resolve variant from job name")

    if not dest_path_root:
        if dest_cluster == "mlxp":
            dest_path_root = f"/data/youngwoong/experiments/{variant}/checkpoints"
        else:
            dest_path_root = f"$HOME/.train-eval-web/experiments/{variant}/checkpoints"

    copy_id = uuid.uuid4().hex[:12]
    _COPY_JOBS[copy_id] = CopyJobStatus(
        copy_id=copy_id, status="running",
        copies_total=len(sources), copies_done=0,
        started_at=time.time(),
    )
    _COPY_HANDLES[copy_id] = {"task": None, "proc": None}
    task = asyncio.create_task(
        _run_copy(
            copy_id, src_cluster, dest_cluster, sources, dest_path_root, delete_source,
        )
    )
    _COPY_HANDLES[copy_id]["task"] = task
    return copy_id


def get_copy_status(copy_id: str) -> CopyJobStatus | None:
    return _COPY_JOBS.get(copy_id)


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


async def _run_copy(
    copy_id: str, src_cluster: str, dest_cluster: str,
    sources: list[str], dest_path_root: str, delete_source: bool,
) -> None:
    state = _COPY_JOBS[copy_id]
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

            # Tar-pipe transports update dest_size_bytes directly from the
            # Python pump (accurate "bytes streamed this run"). Skip the
            # du-based poller for those — otherwise it'd overwrite the live
            # counter with the destination's apparent size (which includes
            # stale bytes from a prior killed run).
            is_tar_pipe = (
                (src_cluster == "mlxp") != (dest_cluster == "mlxp")
            )
            poll_task = None if is_tar_pipe else asyncio.create_task(
                _poll_dest_size(state, dest_cluster, dest_path)
            )
            try:
                if src_cluster == "mlxp" and dest_cluster == "mlxp":
                    await _mlxp_to_mlxp(copy_id, src_path, dest_path, include_only)
                elif src_cluster == "mlxp":
                    await _mlxp_to_slurm(copy_id, src_path, dest_cluster, dest_path, include_only)
                elif dest_cluster == "mlxp":
                    await _slurm_to_mlxp(copy_id, src_cluster, src_path, dest_path, include_only)
                else:
                    await _slurm_to_slurm(copy_id, src_cluster, src_path, dest_cluster, dest_path, include_only)
            finally:
                if poll_task is not None:
                    poll_task.cancel()

            # Snapshot final size before the next source overwrites state.
            state.dest_size_bytes = state.src_size_bytes
            state.copies_done = i + 1

            if delete_source:
                await _delete_source(src_cluster, src_path)

        state.status = "done"
        state.finished_at = time.time()
    except asyncio.CancelledError:
        # cancel_copy set status to error/cancelled already; just exit.
        return
    except Exception as e:
        state.status = "error"
        msg = str(e).strip()
        # Many asyncio exceptions (TimeoutError, CancelledError) stringify
        # to "" — always include the type so the UI shows *something*.
        state.error = f"{type(e).__name__}: {msg}" if msg else type(e).__name__
        state.finished_at = time.time()
    finally:
        _COPY_HANDLES.pop(copy_id, None)


async def _pump_bytes(copy_id: str, src: asyncio.subprocess.Process,
                       dst: asyncio.subprocess.Process) -> None:
    """Pipe src.stdout → dst.stdin, counting bytes into state.dest_size_bytes.
    That gives an accurate 'bytes shipped this run' counter instead of
    `du` on the destination (which sees leftover bytes from prior runs)."""
    state = _COPY_JOBS.get(copy_id)
    assert src.stdout is not None and dst.stdin is not None
    sent = 0
    try:
        while True:
            chunk = await src.stdout.read(65536)
            if not chunk:
                break
            dst.stdin.write(chunk)
            try:
                await dst.stdin.drain()
            except (BrokenPipeError, ConnectionResetError):
                break
            sent += len(chunk)
            if state is not None:
                state.dest_size_bytes = sent
    finally:
        try:
            dst.stdin.close()
        except Exception:
            pass
        await asyncio.gather(src.wait(), dst.wait(), return_exceptions=True)


async def _poll_dest_size(state: CopyJobStatus, cluster: str, path: str) -> None:
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
    """Return the byte size of a directory (or file).
    Returns 0 on any error — size is only used for progress display, so a
    flaky probe must NOT kill the actual transfer."""
    cmd = f"du -sb {shlex.quote(path)} 2>/dev/null | awk '{{print $1}}'"
    try:
        if cluster == "mlxp":
            pod = await ensure_listing_pod()
            proc = await asyncio.create_subprocess_exec(
                "kubectl", "exec", "-n", MLXP_NS, pod, "--", "bash", "-c", cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                out, _ = await asyncio.wait_for(proc.communicate(), timeout=15.0)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return 0
            s = out.decode().strip()
        else:
            env = await load_cluster(cluster)
            r = await ssh_run(env.ssh_alias, cmd, timeout=15.0)
            s = r.stdout.strip()
    except Exception:
        return 0
    try:
        return int(s)
    except ValueError:
        return 0


async def _mlxp_to_mlxp(copy_id: str, src: str, dest: str,
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
    _track_proc(copy_id, proc)
    out, err = await asyncio.wait_for(proc.communicate(), timeout=3600.0)
    if proc.returncode != 0:
        raise RuntimeError(f"cp failed: {err.decode(errors='replace').strip()}")
    return out.decode(errors="replace")


async def _slurm_to_slurm(copy_id: str, src_cluster: str, src_path: str,
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


async def _mlxp_to_slurm(copy_id: str, src_path: str, dest_cluster: str, dest_path: str,
                          include_only: list[str] | None = None) -> str:
    pod = await ensure_listing_pod()
    dest_env = await load_cluster(dest_cluster)
    src_parent = str(Path(src_path).parent)
    src_leaf = Path(src_path).name
    dest_parent = str(Path(dest_path).parent.as_posix())
    tar_args = list(include_only or [src_leaf])

    src_proc = await asyncio.create_subprocess_exec(
        "kubectl", "exec", "-i", "-n", MLXP_NS, pod, "--",
        "tar", "c", "-C", src_parent, *tar_args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        limit=1 << 22,
    )
    dst_proc = await asyncio.create_subprocess_exec(
        "ssh", "-o", "BatchMode=yes", *_CM_OPTS, dest_env.ssh_alias,
        f"mkdir -p {shlex.quote(dest_parent)} && tar x -C {shlex.quote(dest_parent)}",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        limit=1 << 22,
    )
    _track_proc(copy_id, src_proc)
    try:
        await _pump_bytes(copy_id, src_proc, dst_proc)
    finally:
        if src_proc.returncode is None:
            src_proc.kill()
            await src_proc.wait()
        if dst_proc.returncode is None:
            await dst_proc.wait()
    if src_proc.returncode != 0 or dst_proc.returncode != 0:
        src_err = (await src_proc.stderr.read()).decode(errors="replace").strip() if src_proc.stderr else ""
        dst_err = (await dst_proc.stderr.read()).decode(errors="replace").strip() if dst_proc.stderr else ""
        raise RuntimeError(
            f"mlxp→slurm tar pipe failed "
            f"(src rc={src_proc.returncode}, dst rc={dst_proc.returncode}): "
            f"{src_err or dst_err or 'no stderr'}"
        )
    return f"copied {src_path} → {dest_env.ssh_alias}:{dest_path}"


async def _slurm_to_mlxp(copy_id: str, src_cluster: str, src_path: str, dest_path: str,
                          include_only: list[str] | None = None) -> str:
    src_env = await load_cluster(src_cluster)
    pod = await ensure_listing_pod()
    src_parent = str(Path(src_path).parent.as_posix())
    src_leaf = Path(src_path).name
    dest_parent = str(Path(dest_path).parent)
    tar_args = " ".join(shlex.quote(it) for it in (include_only or [src_leaf]))

    src_proc = await asyncio.create_subprocess_exec(
        "ssh", "-o", "BatchMode=yes", *_CM_OPTS, src_env.ssh_alias,
        f"tar c -C {shlex.quote(src_parent)} {tar_args}",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        limit=1 << 22,
    )
    dst_proc = await asyncio.create_subprocess_exec(
        "kubectl", "exec", "-i", "-n", MLXP_NS, pod, "--",
        "bash", "-c",
        f"mkdir -p {shlex.quote(dest_parent)} && tar x -C {shlex.quote(dest_parent)}",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        limit=1 << 22,
    )
    _track_proc(copy_id, src_proc)
    try:
        await _pump_bytes(copy_id, src_proc, dst_proc)
    finally:
        if src_proc.returncode is None:
            src_proc.kill()
            await src_proc.wait()
        if dst_proc.returncode is None:
            await dst_proc.wait()
    if src_proc.returncode != 0 or dst_proc.returncode != 0:
        src_err = (await src_proc.stderr.read()).decode(errors="replace").strip() if src_proc.stderr else ""
        dst_err = (await dst_proc.stderr.read()).decode(errors="replace").strip() if dst_proc.stderr else ""
        raise RuntimeError(
            f"slurm→mlxp tar pipe failed "
            f"(src rc={src_proc.returncode}, dst rc={dst_proc.returncode}): "
            f"{src_err or dst_err or 'no stderr'}"
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
