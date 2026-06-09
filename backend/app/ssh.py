"""SSH command wrappers.

We shell out to the system `ssh` and `rsync` binaries instead of using
asyncssh directly. That way we inherit the user's `~/.ssh/config` (host
aliases, key paths) for free — same behavior as running
`ssh kakao-login-1 ...` from a terminal.

We add `ControlMaster=auto` ourselves so the second-through-Nth call to
the same host reuses an existing TCP+SSH session (sub-second). Without
it each call is its own handshake, which on slow links (skt) was costing
5-12s per command — the dominant factor in /api/jobs latency.
"""

import asyncio
import os
from dataclasses import dataclass


@dataclass
class SSHResult:
    returncode: int
    stdout: str
    stderr: str


# Some clusters (skt) don't have slurm tools on the default non-login-shell PATH.
# Prepend the common install location so sinfo/squeue/sacct/scancel/sbatch all work.
_SLURM_PATH_PREFIX = "export PATH=/opt/slurm/bin:$PATH; "


# ── SSH connection multiplexing ───────────────────────────────────────
# %C is a hash of (host, port, user, local user) — uniquely identifies
# a connection. ControlPersist keeps the master alive for 10 min after
# the last client exits, so polling endpoints (jobs/monitor) reuse it.
_CM_DIR = os.path.expanduser("~/.train-eval-web/ssh-cm")
os.makedirs(_CM_DIR, exist_ok=True)
_CM_OPTS = (
    "-o", "ControlMaster=auto",
    "-o", f"ControlPath={_CM_DIR}/%C",
    "-o", "ControlPersist=600",
)

_CM_FAILURE_MARKERS = (
    "Control socket connect",
    "mux_client",
    "Connection to master",
)


async def ssh_run(host: str, cmd: str, timeout: float = 60.0) -> SSHResult:
    """Run `cmd` on `host` over ssh and return its output."""
    result = await _ssh_run_once(host, cmd, timeout=timeout, use_control_master=True)
    if result.returncode == 255 and any(marker in result.stderr for marker in _CM_FAILURE_MARKERS):
        return await _ssh_run_once(host, cmd, timeout=timeout, use_control_master=False)
    return result


async def _ssh_run_once(
    host: str,
    cmd: str,
    *,
    timeout: float,
    use_control_master: bool,
) -> SSHResult:
    opts = _CM_OPTS if use_control_master else ()
    proc = await asyncio.create_subprocess_exec(
        "ssh",
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=10",
        *opts,
        host,
        _SLURM_PATH_PREFIX + cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise
    return SSHResult(
        returncode=proc.returncode or 0,
        stdout=stdout.decode(errors="replace"),
        stderr=stderr.decode(errors="replace"),
    )


def rsync_ssh_transport() -> str:
    """The `-e` transport string for rsync, reusing the multiplexed ssh master."""
    return "ssh -o BatchMode=yes " + " ".join(_CM_OPTS)


async def rsync_to(
    host: str,
    local_path: str,
    remote_path: str,
    *,
    delete: bool = False,
    exclude: list[str] | None = None,
    timeout: float = 120.0,
) -> SSHResult:
    """rsync local_path to host:remote_path. Creates parent dirs."""
    # Reuse the multiplexed ssh master rather than negotiating a new session.
    ssh_e = rsync_ssh_transport()
    args = ["rsync", "-az", "-e", ssh_e]
    if delete:
        args.append("--delete")
    for pattern in exclude or []:
        args.append(f"--exclude={pattern}")
    args.extend([local_path, f"{host}:{remote_path}"])
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise
    return SSHResult(
        returncode=proc.returncode or 0,
        stdout=stdout.decode(errors="replace"),
        stderr=stderr.decode(errors="replace"),
    )


async def ssh_tail_lines(host: str, remote_pattern: str, start_line: int = 1):
    """Async generator yielding `tail -F` lines from a remote file glob.

    `remote_pattern` may contain globs (`*`); we explicitly avoid shlex.quote
    so the remote login shell expands them. The path is constructed from
    cluster config + job id, not user input, so injection isn't a risk.

    Wraps in a small bash retry loop: if the file doesn't exist yet (PENDING
    job, eval mid-startup) the loop sleeps and retries until it does, then
    `tail -F` follows-by-name forever.
    """
    # Stream the entire file from the start, then follow forever. Frontend
    # owns the scrollback policy.
    safe_start = max(1, int(start_line))
    cmd = (
        f'while ! ls {remote_pattern} >/dev/null 2>&1; do sleep 2; done; '
        f'exec tail -n +{safe_start} -F {remote_pattern}'
    )
    # 1MB stream limit + split on both \n and \r so tqdm progress lines
    # don't overflow asyncio's default 64KB readline buffer.
    proc = await asyncio.create_subprocess_exec(
        "ssh", "-o", "BatchMode=yes", *_CM_OPTS, host, cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        limit=1 << 20,
    )
    assert proc.stdout is not None
    try:
        buf = b""
        while True:
            try:
                chunk = await proc.stdout.read(8192)
            except Exception:
                break
            if not chunk:
                if buf:
                    yield buf.decode(errors="replace")
                break
            buf += chunk
            while True:
                i_n = buf.find(b"\n")
                i_r = buf.find(b"\r")
                idx = (
                    min(i for i in (i_n, i_r) if i != -1)
                    if (i_n != -1 or i_r != -1)
                    else -1
                )
                if idx == -1:
                    if len(buf) > (1 << 19):
                        yield buf.decode(errors="replace")
                        buf = b""
                    break
                line = buf[:idx]
                buf = buf[idx + 1:]
                yield line.decode(errors="replace")
    finally:
        if proc.returncode is None:
            proc.kill()
            await proc.wait()
