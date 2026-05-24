"""Copy a job's final checkpoint to another server.

Source layouts:
- mlxp:   <MLXP experiments dir>/<variant>/checkpoints/<job_name>/checkpoint-N
- slurm:  $EXP_DIR/checkpoints/checkpoint-N

Destination layouts: same shape as the source per cluster (no nested
job-name dir on slurm, because the slurm body never created one).

Transfer mechanism:
- slurm → slurm: rsync over ssh (host A → host B).
- slurm → mlxp:  ssh tar | kubectl exec tar x on a DDN-mounted pod.
- mlxp  → slurm: kubectl cp each model artifact to local staging, then
  rsync local staging to the dest host and atomically promote it.
- mlxp  → mlxp:  cp inside one pod (same DDN PVC).
"""

import asyncio
import contextlib
import os
import shutil
import shlex
import tempfile
import time
import uuid
from pathlib import Path

from pydantic import BaseModel

from .clusters import load_cluster
from .details import get_details, resolve_phase_and_variant
from .jobs import get_job
from .kubectl_errors import is_completed_pod_exec_error
from .mlxp_config import get_settings
from .mlxp_data_pod import ensure_listing_pod, invalidate_pods_cache
from .ssh import ssh_run, _CM_OPTS
from .submission_snapshot import is_mlxp_transport_error


_MLXP_COPY_STREAMS = max(1, int(os.environ.get("TRAIN_EVAL_MLXP_COPY_STREAMS", "1")))
_MLXP_KUBECTL_CP_ATTEMPTS = max(1, int(os.environ.get("TRAIN_EVAL_MLXP_KUBECTL_CP_ATTEMPTS", "10")))
_MLXP_KUBECTL_CP_RETRIES = max(0, int(os.environ.get("TRAIN_EVAL_MLXP_KUBECTL_CP_RETRIES", "10")))
_MLXP_KUBECTL_CP_TIMEOUT = float(os.environ.get("TRAIN_EVAL_MLXP_KUBECTL_CP_TIMEOUT", "3600"))
_MLXP_LOCAL_RSYNC_ATTEMPTS = max(1, int(os.environ.get("TRAIN_EVAL_MLXP_LOCAL_RSYNC_ATTEMPTS", "3")))
_MLXP_LOCAL_RSYNC_TIMEOUT = float(os.environ.get("TRAIN_EVAL_MLXP_LOCAL_RSYNC_TIMEOUT", "7200"))
_KUBECTL_EXEC_ATTEMPTS = 3

_MODEL_ARTIFACT_EXCLUDES = (
    "checkpoint-*",
    "*/checkpoint-*",
    "optimizer*",
    "*/optimizer*",
    "scheduler.pt",
    "*/scheduler.pt",
    "rng_state_*.pth",
    "*/rng_state_*.pth",
    "trainer_state.json",
    "*/trainer_state.json",
    "global_step*",
    "*/global_step*",
    "latest",
    "*/latest",
    "zero_to_fp32.py",
    "*/zero_to_fp32.py",
    "logs",
    "logs/*",
    "*/logs",
    "*/logs/*",
)


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


class CopyCheckpointStartResponse(BaseModel):
    copy_id: str


class CopyJobStatus(BaseModel):
    copy_id: str
    status: str  # "running" | "done" | "error"
    error: str | None = None
    phase: str | None = None
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
_MLXP_STREAM_SEMAPHORE = asyncio.Semaphore(_MLXP_COPY_STREAMS)


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

async def _kubectl_exec_text(
    pod: str,
    *command: str,
    timeout: float = 15.0,
    attempts: int = _KUBECTL_EXEC_ATTEMPTS,
) -> str:
    """Run a short kubectl exec command and return stdout.

    MLXP's API proxy occasionally returns transient transport errors. Keep
    retries local to Kubernetes reads so callers don't copy/paste retry loops.
    """
    last_error = "kubectl exec failed"
    current_pod = pod
    for attempt in range(1, attempts + 1):
        settings = get_settings()
        proc = await asyncio.create_subprocess_exec(
            "kubectl", "exec", "-n", settings.namespace, current_pod, "--", *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            last_error = "kubectl exec timed out"
        else:
            if proc.returncode == 0:
                return out.decode(errors="replace")
            last_error = err.decode(errors="replace").strip() or "kubectl exec failed"
            if is_completed_pod_exec_error(last_error):
                invalidate_pods_cache()
                try:
                    current_pod = await ensure_listing_pod()
                except Exception:
                    pass
        if attempt == attempts or not is_mlxp_transport_error(last_error):
            break
        await asyncio.sleep(0.5 * attempt)
    raise RuntimeError(last_error)

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
        # Emit one row for each per-job checkpoint dir. Also include the
        # historical direct-root layout used by older MLXP jobs.
        experiments_root = shlex.quote(get_settings().experiments_dir)
        quoted_variant = shlex.quote(variant)
        script = (
            r"""
shopt -s nullglob
root=__EXPERIMENTS_ROOT__/__VARIANT__/checkpoints
for d in "$root"/ "$root"/*/ "$root"/*/*/; do
    matches=( "$d"checkpoint-* )
    [ ${#matches[@]} -eq 0 ] && continue
    latest=$(for m in "${matches[@]}"; do basename "$m" | sed 's:^checkpoint-::'; done | sort -n | tail -1)
    printf '%s|%s\n' "${d%/}" "$latest"
done
""".replace("__EXPERIMENTS_ROOT__", experiments_root).replace("__VARIANT__", quoted_variant))
        out_text = await _kubectl_exec_text(pod, "bash", "-c", script)
        result: list[CheckpointEntry] = []
        for line in out_text.splitlines():
            parts = line.split("|")
            if len(parts) != 2 or not parts[1].isdigit():
                continue
            path, step = parts
            job_name = path.rsplit("/", 1)[-1]
            if job_name == "checkpoints":
                job_name = sacct.get("JobName") or job_id
            result.append(CheckpointEntry(
                path=path, job_name=job_name, step=int(step),
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
            dest_path_root = f"{get_settings().experiments_dir}/{variant}/checkpoints"
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
    cmd = f"ls -1 {shlex.quote(path)} 2>/dev/null || true"
    if cluster == "mlxp":
        pod = await ensure_listing_pod()
        out = await _kubectl_exec_text(pod, "bash", "-c", cmd)
        return [s for s in out.splitlines() if s]
    env = await load_cluster(cluster)
    r = await ssh_run(env.ssh_alias, cmd, timeout=15.0)
    return [s for s in r.stdout.splitlines() if s]


async def _resolve_model_source_path(cluster: str, src_path: str) -> str:
    """Return the directory that holds deployable model artifacts.

    MLXP training output sometimes wraps the actual model directory one level
    deeper as `<run>/<run>/...`, with logs next to it. Prefer that inner
    directory so the copy target is the model artifact folder, not the wrapper.
    """
    names = await _children_of(cluster, src_path)
    leaf = Path(src_path).name
    if leaf in names:
        nested = f"{src_path.rstrip('/')}/{leaf}"
        nested_names = await _children_of(cluster, nested)
        if nested_names:
            return nested
    return src_path


def _should_copy_model_artifact(name: str) -> bool:
    lower = name.lower()
    if name.startswith("checkpoint-"):
        return False
    if name.startswith("global_step"):
        return False
    if name.startswith("rng_state_") and name.endswith(".pth"):
        return False
    if "optimizer" in lower:
        return False
    return name not in {
        "scheduler.pt",
        "trainer_state.json",
        "latest",
        "zero_to_fp32.py",
        "logs",
    }


def _select_model_artifact_files(parent_basename: str, names: list[str]) -> list[str]:
    """Build deployable model-artifact paths relative to the source parent.

    This excludes nested Hugging Face/DeepSpeed trainer checkpoints, optimizer
    state, scheduler/RNG state, and logs. It keeps top-level model shards,
    config files, processor files, and experiment metadata.
    """
    return [
        f"{parent_basename}/{n}"
        for n in names
        if _should_copy_model_artifact(n)
    ]


async def _run_copy(
    copy_id: str, src_cluster: str, dest_cluster: str,
    sources: list[str], dest_path_root: str, delete_source: bool,
) -> None:
    state = _COPY_JOBS[copy_id]
    try:
        for i, src_path in enumerate(sources):
            src_path = await _resolve_model_source_path(src_cluster, src_path)
            leaf = Path(src_path).name
            dest_path = f"{dest_path_root.rstrip('/')}/{leaf}"
            state.current_source = src_path
            state.current_dest = dest_path
            state.phase = "preparing"
            state.dest_size_bytes = 0

            # Copy deployable model artifacts only. Exclude nested trainer
            # checkpoints and optimizer/scheduler/RNG state.
            names = await _children_of(src_cluster, src_path)
            include_only = None
            if names:
                include_only = _select_model_artifact_files(leaf, names)
                picks = [item.split("/", 1)[1] for item in include_only]
                sizes = await asyncio.gather(*[
                    _size(src_cluster, f"{src_path}/{name}") for name in picks
                ])
                state.src_size_bytes = sum(sizes)
            else:
                state.src_size_bytes = await _size(src_cluster, src_path)

            # MLXP boundary transports report progress themselves: slurm→mlxp
            # via the Python tar pump, mlxp→slurm via local staging size. Skip
            # the du-based destination poller so it does not overwrite that
            # live counter with stale bytes from a prior killed run.
            self_reports_size = (
                (src_cluster == "mlxp") != (dest_cluster == "mlxp")
            )
            poll_task = None if self_reports_size else asyncio.create_task(
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
                await _cancel_task(poll_task)

            # Snapshot final size before the next source overwrites state.
            state.dest_size_bytes = state.src_size_bytes
            state.copies_done = i + 1

            if delete_source:
                await _delete_source(src_cluster, src_path)

        state.phase = None
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


async def _cancel_task(task: asyncio.Task | None) -> None:
    if task is None:
        return
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


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
                if state.src_size_bytes is not None and sent > state.src_size_bytes:
                    state.src_size_bytes = sent
    finally:
        try:
            dst.stdin.close()
        except Exception:
            pass
        await asyncio.gather(src.wait(), dst.wait(), return_exceptions=True)


async def _pump_and_check_tar_pipe(
    copy_id: str,
    label: str,
    src_proc: asyncio.subprocess.Process,
    dst_proc: asyncio.subprocess.Process,
) -> None:
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
            f"{label} tar pipe failed "
            f"(src rc={src_proc.returncode}, dst rc={dst_proc.returncode}): "
            f"{src_err or dst_err or 'no stderr'}"
        )


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
            s = (await _kubectl_exec_text(pod, "bash", "-c", cmd)).strip()
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


def _remove_local_path(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _local_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file() or path.is_symlink():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    total = 0
    for root, dirs, files in os.walk(path):
        root_path = Path(root)
        try:
            total += root_path.stat().st_size
        except OSError:
            pass
        for name in dirs + files:
            try:
                total += (root_path / name).stat().st_size
            except OSError:
                pass
    return total


async def _poll_local_size(state: CopyJobStatus, path: Path) -> None:
    try:
        while True:
            state.dest_size_bytes = await asyncio.to_thread(_local_size_bytes, path)
            await asyncio.sleep(2.0)
    except asyncio.CancelledError:
        pass


def _model_artifact_relpaths(src_leaf: str, include_only: list[str] | None) -> list[str]:
    if include_only is None:
        return [""]
    prefix = f"{src_leaf}/"
    relpaths: list[str] = []
    for item in include_only:
        if item == src_leaf:
            relpaths.append("")
        elif item.startswith(prefix):
            relpaths.append(item[len(prefix):])
        else:
            relpaths.append(Path(item).name)
    return relpaths


def _tar_artifact_args(src_leaf: str, include_only: list[str] | None) -> list[str]:
    if include_only is None:
        return [src_leaf]
    if not include_only:
        raise RuntimeError(f"no deployable model artifacts found under {src_leaf}")
    return list(include_only)


async def _run_checked_exec(
    copy_id: str,
    args: list[str],
    *,
    label: str,
    timeout: float,
) -> str:
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        limit=1 << 22,
    )
    _track_proc(copy_id, proc)
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.CancelledError:
        proc.kill()
        await proc.wait()
        raise
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise RuntimeError(f"{label} timed out after {timeout:g}s")
    stdout = out.decode(errors="replace")
    stderr = err.decode(errors="replace")
    if proc.returncode != 0:
        msg = (stderr or stdout).strip() or "no stderr"
        raise RuntimeError(f"{label} failed rc={proc.returncode}: {msg}")
    return stdout


async def _run_checked_exec_with_retries(
    copy_id: str,
    args: list[str],
    *,
    label: str,
    attempts: int,
    timeout: float,
) -> str:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return await _run_checked_exec(
                copy_id,
                args,
                label=f"{label} attempt {attempt}",
                timeout=timeout,
            )
        except Exception as e:
            last_error = e
            if attempt == attempts:
                raise
            await asyncio.sleep(min(60.0, 5.0 * attempt))
    assert last_error is not None
    raise last_error


async def _kubectl_cp_model_artifact(
    copy_id: str,
    pod: str,
    remote_path: str,
    local_path: Path,
    *,
    label: str,
) -> None:
    last_error: Exception | None = None
    for attempt in range(1, _MLXP_KUBECTL_CP_ATTEMPTS + 1):
        _remove_local_path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        settings = get_settings()
        proc = await asyncio.create_subprocess_exec(
            "kubectl", "cp", "-n", settings.namespace,
            f"{pod}:{remote_path}", str(local_path),
            f"--retries={_MLXP_KUBECTL_CP_RETRIES}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=1 << 22,
        )
        _track_proc(copy_id, proc)
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=_MLXP_KUBECTL_CP_TIMEOUT)
        except asyncio.CancelledError:
            proc.kill()
            await proc.wait()
            raise
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            last_error = RuntimeError(
                f"kubectl cp {label} attempt {attempt} timed out after "
                f"{_MLXP_KUBECTL_CP_TIMEOUT:g}s"
            )
        else:
            stdout = out.decode(errors="replace")
            stderr = err.decode(errors="replace")
            if proc.returncode == 0:
                return
            message = (stderr or stdout).strip() or "kubectl cp failed"
            last_error = RuntimeError(
                f"kubectl cp {label} attempt {attempt} failed rc={proc.returncode}: {message}"
            )
            lower = message.lower()
            if "no such file" in lower or "cannot stat" in lower:
                break
            if is_mlxp_transport_error(message):
                invalidate_pods_cache()
                try:
                    pod = await ensure_listing_pod()
                except Exception:
                    pass
        if attempt < _MLXP_KUBECTL_CP_ATTEMPTS:
            await asyncio.sleep(min(60.0, 5.0 * attempt))
    assert last_error is not None
    raise last_error


async def _mlxp_to_mlxp(copy_id: str, src: str, dest: str,
                         include_only: list[str] | None = None) -> str:
    pod = await ensure_listing_pod()
    src_parent = str(Path(src).parent)
    src_leaf = Path(src).name
    dest_parent = str(Path(dest).parent)
    tar_args = _tar_artifact_args(src_leaf, include_only)
    cmd = (
        f"mkdir -p {shlex.quote(dest_parent)} && "
        f"{_mlxp_tar_create_cmd(src_parent, tar_args)} | "
        f"tar x -C {shlex.quote(dest_parent)}"
    )
    settings = get_settings()
    proc = await asyncio.create_subprocess_exec(
        "kubectl", "exec", "-n", settings.namespace, pod, "--", "bash", "-c", cmd,
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
    excludes = " ".join(
        f"--exclude={shlex.quote(pattern)}"
        for pattern in _MODEL_ARTIFACT_EXCLUDES
    )
    cmd = (
        f"mkdir -p {shlex.quote(dest_path)} >/dev/null 2>&1 && "
        f"rsync -az {excludes} {shlex.quote(src_path)}/ "
        f"{dest_alias}:{shlex.quote(dest_path)}/"
    )
    r = await ssh_run(src_env.ssh_alias, cmd, timeout=3600.0)
    if r.returncode != 0:
        raise RuntimeError(f"rsync failed: {r.stderr.strip() or r.stdout.strip()}")
    return r.stdout


def _remote_shell_path(path: str) -> str:
    """Quote a remote path while preserving an intentional leading $HOME."""
    if path == "$HOME":
        return "$HOME"
    if path.startswith("$HOME/"):
        rest = path[len("$HOME/"):]
        return "$HOME/" + shlex.quote(rest)
    return shlex.quote(path)


def _mlxp_tar_create_cmd(src_parent: str, tar_args: list[str]) -> str:
    quoted_args = " ".join(shlex.quote(arg) for arg in tar_args)
    excludes = " ".join(
        f"--exclude={shlex.quote(pattern)}"
        for pattern in _MODEL_ARTIFACT_EXCLUDES
    )
    return f"tar c -C {shlex.quote(src_parent)} {excludes} {quoted_args}"


def _slurm_tar_extract_cmd(dest_parent: str, dest_path: str, leaf: str, copy_id: str) -> str:
    """Extract into a temporary sibling, then atomically promote to dest.

    A failed tar pipe should not leave a half-written checkpoint at the final
    path, especially because retries may follow immediately.
    """
    tmp_template = f".{leaf}.copy-{copy_id}.XXXXXX"
    dest_parent_q = _remote_shell_path(dest_parent)
    dest_path_q = _remote_shell_path(dest_path)
    leaf_q = shlex.quote(leaf)
    return (
        f"mkdir -p {dest_parent_q} && "
        f"tmp_dir=$(mktemp -d {dest_parent_q}/{shlex.quote(tmp_template)}) && "
        f"cleanup() {{ rm -rf \"$tmp_dir\"; }} && "
        f"trap cleanup EXIT && "
        f"tar x -C \"$tmp_dir\" && "
        f"test -e \"$tmp_dir\"/{leaf_q} && "
        f"rm -rf {dest_path_q} && "
        f"mv \"$tmp_dir\"/{leaf_q} {dest_path_q} && "
        f"trap - EXIT && "
        f"rm -rf \"$tmp_dir\""
    )


async def _copy_mlxp_artifacts_to_local(
    copy_id: str,
    pod: str,
    src_path: str,
    local_leaf: Path,
    include_only: list[str] | None,
) -> None:
    src_leaf = Path(src_path).name
    relpaths = _model_artifact_relpaths(src_leaf, include_only)
    if not relpaths:
        raise RuntimeError(f"no deployable model artifacts found under {src_leaf}")
    for relpath in relpaths:
        if relpath:
            remote_item = f"{src_path.rstrip('/')}/{relpath}"
            local_item = local_leaf / relpath
            label = f"{src_leaf}/{relpath}"
        else:
            remote_item = src_path
            local_item = local_leaf
            label = src_leaf
        await _kubectl_cp_model_artifact(
            copy_id, pod, remote_item, local_item, label=label,
        )


async def _create_remote_checkpoint_staging(
    dest_alias: str,
    dest_parent: str,
    src_leaf: str,
    copy_id: str,
) -> str:
    dest_parent_q = _remote_shell_path(dest_parent)
    tmp_template = f".{src_leaf}.copy-{copy_id}.XXXXXX"
    mktemp = await ssh_run(
        dest_alias,
        (
            f"mkdir -p {dest_parent_q} && "
            f"mktemp -d {dest_parent_q}/{shlex.quote(tmp_template)}"
        ),
        timeout=60.0,
    )
    if mktemp.returncode != 0:
        raise RuntimeError(f"remote temp dir creation failed: {mktemp.stderr or mktemp.stdout}")
    remote_tmp = mktemp.stdout.strip().splitlines()[-1]
    if not remote_tmp:
        raise RuntimeError("remote temp dir creation returned an empty path")

    mkdir_leaf = await ssh_run(
        dest_alias,
        f"mkdir -p {shlex.quote(remote_tmp)}/{shlex.quote(src_leaf)}",
        timeout=60.0,
    )
    if mkdir_leaf.returncode != 0:
        raise RuntimeError(f"remote staging mkdir failed: {mkdir_leaf.stderr or mkdir_leaf.stdout}")
    return remote_tmp


async def _rsync_local_checkpoint_to_remote(
    copy_id: str,
    dest_alias: str,
    local_leaf: Path,
    remote_tmp: str,
    src_leaf: str,
) -> None:
    ssh_e = "ssh -o BatchMode=yes " + " ".join(_CM_OPTS)
    rsync_args = [
        "rsync", "-az", "--delete", "--partial", "-e", ssh_e,
        f"{local_leaf}/",
        f"{dest_alias}:{remote_tmp}/{src_leaf}/",
    ]
    await _run_checked_exec_with_retries(
        copy_id,
        rsync_args,
        label=f"rsync staged {src_leaf}",
        attempts=_MLXP_LOCAL_RSYNC_ATTEMPTS,
        timeout=_MLXP_LOCAL_RSYNC_TIMEOUT,
    )


async def _promote_remote_checkpoint(
    dest_alias: str,
    remote_tmp: str,
    src_leaf: str,
    dest_path: str,
) -> None:
    leaf_q = shlex.quote(src_leaf)
    remote_tmp_q = shlex.quote(remote_tmp)
    dest_path_q = _remote_shell_path(dest_path)
    promote = await ssh_run(
        dest_alias,
        (
            f"test -d {remote_tmp_q}/{leaf_q} && "
            f"rm -rf {dest_path_q} && "
            f"mv {remote_tmp_q}/{leaf_q} {dest_path_q} && "
            f"rmdir {remote_tmp_q}"
        ),
        timeout=300.0,
    )
    if promote.returncode != 0:
        raise RuntimeError(f"remote promote failed: {promote.stderr or promote.stdout}")


async def _mlxp_to_slurm_once(
    copy_id: str,
    pod: str,
    src_path: str,
    dest_cluster: str,
    dest_path: str,
    include_only: list[str] | None = None,
) -> str:
    dest_env = await load_cluster(dest_cluster)
    dest_alias = dest_env.ssh_alias
    src_leaf = Path(src_path).name
    dest_parent = str(Path(dest_path).parent.as_posix())
    staging_parent = os.environ.get("TRAIN_EVAL_MLXP_LOCAL_STAGING_DIR") or None
    with tempfile.TemporaryDirectory(
        prefix=f"train-eval-web-{src_leaf}-",
        dir=staging_parent,
    ) as staging_root:
        local_leaf = Path(staging_root) / src_leaf
        state = _COPY_JOBS.get(copy_id)
        if state is not None:
            state.phase = "staging from MLXP"
        poll_task = (
            asyncio.create_task(_poll_local_size(state, local_leaf))
            if state is not None else None
        )
        try:
            await _copy_mlxp_artifacts_to_local(
                copy_id, pod, src_path, local_leaf, include_only,
            )
            if state is not None:
                state.dest_size_bytes = await asyncio.to_thread(_local_size_bytes, local_leaf)
        finally:
            await _cancel_task(poll_task)

        remote_tmp: str | None = None
        remote_poll_task: asyncio.Task | None = None
        try:
            remote_tmp = await _create_remote_checkpoint_staging(
                dest_alias, dest_parent, src_leaf, copy_id,
            )
            if state is not None:
                state.phase = f"uploading to {dest_cluster}"
                state.dest_size_bytes = 0
                remote_leaf = f"{remote_tmp.rstrip('/')}/{src_leaf}"
                remote_poll_task = asyncio.create_task(
                    _poll_dest_size(state, dest_cluster, remote_leaf)
                )

            await _rsync_local_checkpoint_to_remote(
                copy_id, dest_alias, local_leaf, remote_tmp, src_leaf,
            )
            await _cancel_task(remote_poll_task)
            remote_poll_task = None
            if state is not None:
                state.dest_size_bytes = state.src_size_bytes
                state.phase = "promoting"

            await _promote_remote_checkpoint(
                dest_alias, remote_tmp, src_leaf, dest_path,
            )
        except BaseException:
            await _cancel_task(remote_poll_task)
            if remote_tmp:
                with contextlib.suppress(Exception):
                    await ssh_run(dest_alias, f"rm -rf {shlex.quote(remote_tmp)}", timeout=300.0)
            raise

    return (
        f"copied {src_path} → {dest_alias}:{dest_path} "
        "via kubectl cp local staging"
    )


async def _mlxp_to_slurm(copy_id: str, src_path: str, dest_cluster: str, dest_path: str,
                          include_only: list[str] | None = None) -> str:
    pod = await ensure_listing_pod()
    async with _MLXP_STREAM_SEMAPHORE:
        return await _mlxp_to_slurm_once(
            copy_id, pod, src_path, dest_cluster, dest_path, include_only
        )


async def _slurm_to_mlxp(copy_id: str, src_cluster: str, src_path: str, dest_path: str,
                          include_only: list[str] | None = None) -> str:
    src_env = await load_cluster(src_cluster)
    pod = await ensure_listing_pod()
    src_parent = str(Path(src_path).parent.as_posix())
    src_leaf = Path(src_path).name
    dest_parent = str(Path(dest_path).parent)
    tar_args = _tar_artifact_args(src_leaf, include_only)

    async with _MLXP_STREAM_SEMAPHORE:
        settings = get_settings()
        src_proc = await asyncio.create_subprocess_exec(
            "ssh", "-o", "BatchMode=yes", *_CM_OPTS, src_env.ssh_alias,
            _mlxp_tar_create_cmd(src_parent, tar_args),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=1 << 22,
        )
        dst_proc = await asyncio.create_subprocess_exec(
            "kubectl", "exec", "-i", "-n", settings.namespace, pod, "--",
            "bash", "-c",
            f"mkdir -p {shlex.quote(dest_parent)} && tar x -C {shlex.quote(dest_parent)}",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=1 << 22,
        )
        await _pump_and_check_tar_pipe(copy_id, "slurm→mlxp", src_proc, dst_proc)
    return f"copied {src_env.ssh_alias}:{src_path} → mlxp:{dest_path}"


async def _delete_source(src_cluster: str, src_path: str) -> None:
    if src_cluster == "mlxp":
        pod = await ensure_listing_pod()
        settings = get_settings()
        proc = await asyncio.create_subprocess_exec(
            "kubectl", "exec", "-n", settings.namespace, pod, "--",
            "rm", "-rf", src_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await asyncio.wait_for(proc.communicate(), timeout=300.0)
        return
    env = await load_cluster(src_cluster)
    await ssh_run(env.ssh_alias, f"rm -rf {shlex.quote(src_path)}", timeout=300.0)
