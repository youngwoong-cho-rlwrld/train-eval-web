"""SSH command wrappers.

We shell out to the system `ssh` and `rsync` binaries instead of using
asyncssh directly. That way we inherit the user's `~/.ssh/config` (host
aliases, key paths, control-master settings) for free — same behavior as
running `ssh kakao-login-1 ...` from a terminal.
"""

import asyncio
import shlex
from dataclasses import dataclass


@dataclass
class SSHResult:
    returncode: int
    stdout: str
    stderr: str


# Some clusters (skt) don't have slurm tools on the default non-login-shell PATH.
# Prepend the common install location so sinfo/squeue/sacct/scancel/sbatch all work.
_SLURM_PATH_PREFIX = "export PATH=/opt/slurm/bin:$PATH; "


async def ssh_run(host: str, cmd: str, timeout: float = 60.0) -> SSHResult:
    """Run `cmd` on `host` over ssh and return its output."""
    proc = await asyncio.create_subprocess_exec(
        "ssh",
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=10",
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


async def rsync_to(host: str, local_path: str, remote_path: str,
                   *, delete: bool = False, timeout: float = 120.0) -> SSHResult:
    """rsync local_path to host:remote_path. Creates parent dirs."""
    args = ["rsync", "-az"]
    if delete:
        args.append("--delete")
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


async def ssh_tail_lines(host: str, remote_pattern: str):
    """Async generator yielding `tail -F` lines from a remote file glob.

    `remote_pattern` may contain globs (`*`); we explicitly avoid shlex.quote
    so the remote login shell expands them. The path is constructed from
    cluster config + job id, not user input, so injection isn't a risk.

    Wraps in a small bash retry loop: if the file doesn't exist yet (PENDING
    job, eval mid-startup) the loop sleeps and retries until it does, then
    `tail -F` follows-by-name forever.
    """
    cmd = (
        f'while ! ls {remote_pattern} >/dev/null 2>&1; do sleep 2; done; '
        f'exec tail -n 200 -F {remote_pattern}'
    )
    proc = await asyncio.create_subprocess_exec(
        "ssh", "-o", "BatchMode=yes", host, cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    assert proc.stdout is not None
    try:
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            yield line.decode(errors="replace").rstrip("\n")
    finally:
        if proc.returncode is None:
            proc.kill()
            await proc.wait()
