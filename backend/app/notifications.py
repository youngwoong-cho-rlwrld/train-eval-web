"""Slack notifications for job status changes.

The backend is otherwise request-driven; this module adds the one persistent
background task (started from the FastAPI lifespan): a poller that lists jobs
across clusters, diffs each job's state against the last-seen value, and posts
to a Slack incoming webhook on notable transitions (running / completed /
failed / cancelled). Submit-time "submitted" pings are fired inline from the
submit route via note_submitted().

Delivery uses stdlib urllib (no extra deps); the blocking POST runs in a
worker thread so it never stalls the event loop. Last-seen state is persisted
to ~/.train-eval-web/notify_state.json, but each process start still seeds a
fresh baseline before posting transitions. Jobs first seen in a terminal state
are treated as baseline, not as fresh transitions, so enabling Slack does not
replay old history.
"""
from __future__ import annotations

import asyncio
import fcntl
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

from . import clusters, jobs, notifications_config
from .jobs import short_state

_STATE_FILE = Path.home() / ".train-eval-web" / "notify_state.json"
_POLL_INTERVAL = float(os.environ.get("TRAIN_EVAL_WEB_NOTIFY_INTERVAL", "45"))
# Optional public base URL of the frontend, e.g. http://100.80.190.34:3000.
# When set, messages link to the job detail page.
_BASE_URL = (os.environ.get("TRAIN_EVAL_WEB_BASE_URL") or "").rstrip("/")

_EMOJI = {
    "submitted": ":rocket:",
    "running": ":runner:",
    "completed": ":white_check_mark:",
    "failed": ":x:",
    "cancelled": ":no_entry_sign:",
}
_LABEL = {
    "submitted": "Submitted",
    "running": "Running",
    "completed": "Completed",
    "failed": "Failed",
    "cancelled": "Cancelled",
}


def _event_for_state(state: str) -> str | None:
    """Map a raw slurm/mlxp state to a notification event key (or None for
    non-notable states like PENDING/COMPLETING/CONFIGURING/SUSPENDED)."""
    u = short_state(state or "").upper()
    if u == "RUNNING":
        return "running"
    if u.startswith("COMPLET") and u != "COMPLETING":  # COMPLETED
        return "completed"
    if u.startswith("CANCEL"):
        return "cancelled"
    if u.startswith((
        "FAIL", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL",
        "PREEMPT", "BOOT_FAIL", "DEADLINE", "ERROR",
    )):
        return "failed"
    return None


def _send_slack_blocking(url: str, text: str) -> None:
    body = json.dumps({"text": text}).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    urllib.request.urlopen(req, timeout=10).read()


async def _post(text: str) -> None:
    url = notifications_config.webhook_url()
    if not url:
        return
    try:
        await asyncio.to_thread(_send_slack_blocking, url, text)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        print(f"[notifications] slack post failed: {exc}")


def _job_line(cluster, job_id, job_name, phase, variant, event, state) -> str:
    label = _LABEL[event]
    if event == "failed" and state:
        label = f"Failed ({short_state(state)})"
    bits = [f"{_EMOJI.get(event, '')} *{label}*".strip(), f"`{cluster}/{job_id}`"]
    name = variant or job_name
    if name:
        bits.append(str(name))
    if phase:
        bits.append(f"({phase})")
    line = " ".join(bits)
    if _BASE_URL:
        line += f"\n<{_BASE_URL}/jobs/{cluster}/{job_id}|open>"
    return line


async def note_submitted(cluster, job_id, job_name, phase=None, variant=None) -> None:
    """Submit-time 'submitted' ping, called inline from the submit route.
    Never raises — a notification failure must not break job submission."""
    try:
        s = notifications_config.get_settings()
        if not (s.enabled and s.configured and s.notify_submitted):
            return
        await _post(_job_line(cluster, job_id, job_name, phase, variant, "submitted", ""))
    except Exception as exc:  # noqa: BLE001 - notifications are best-effort
        print(f"[notifications] note_submitted failed: {exc}")


async def send_test() -> bool:
    """Post a test message. Returns False if no webhook or the post failed."""
    url = notifications_config.webhook_url()
    if not url:
        return False
    try:
        await asyncio.to_thread(
            _send_slack_blocking, url, ":bell: train-eval-web test notification"
        )
        return True
    except (urllib.error.URLError, OSError, ValueError):
        return False


class _Monitor:
    """Polls job states and posts on transitions. State entry per job id:
    {"cluster": str, "event": str|None}."""

    def __init__(self) -> None:
        self.state: dict[str, dict] = {}
        self.primed = False
        self._loaded = False

    def _load_state(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        try:
            data = json.loads(_STATE_FILE.read_text())
        except (OSError, json.JSONDecodeError):
            return
        if isinstance(data, dict):
            self.state = data
            # Do not notify on the first tick after process start. The jobs
            # list intentionally includes recent terminal jobs, so a persisted
            # diff would replay historical completions when Slack is enabled.
            self.primed = False

    def _persist(self) -> None:
        try:
            _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            _STATE_FILE.write_text(json.dumps(self.state))
        except OSError:
            pass

    async def _collect(self) -> dict[str, dict]:
        """Current jobs keyed by id. Fetches per cluster and, on a cluster
        error, preserves that cluster's prior entries so a transient SSH
        failure can't drop-then-readd a job and cause a spurious re-notify."""
        current: dict[str, dict] = {}
        for c in clusters.list_clusters():
            try:
                js = await jobs.list_jobs([c], hours=24)
            except Exception:
                for jid, ent in self.state.items():
                    if ent.get("cluster") == c:
                        current[jid] = dict(ent)
                continue
            for j in js:
                # Slurm job ids are only unique within one cluster. Kakao and
                # SKT can both have (for example) job 153064; keying by the bare
                # id silently overwrote one transition and lost its Slack ping.
                current[f"{c}/{j.job_id}"] = {
                    "cluster": c,
                    "event": _event_for_state(j.state),
                    "_job": j,  # transient; stripped before persist
                }
        return current

    @staticmethod
    def _persistable(current: dict[str, dict]) -> dict[str, dict]:
        return {
            jid: {"cluster": e["cluster"], "event": e["event"]}
            for jid, e in current.items()
        }

    async def _tick(self) -> None:
        s = notifications_config.get_settings()
        if not (s.enabled and s.configured):
            self.primed = False  # re-seed cleanly when re-enabled
            return
        current = await self._collect()
        if not self.primed:
            self.state = self._persistable(current)
            self.primed = True
            self._persist()
            return
        for state_key, e in current.items():
            previous = self.state.get(state_key)
            if previous is None:
                continue
            new = e["event"]
            old = previous.get("event")
            if new and new != old and notifications_config.event_enabled(new):
                j = e.get("_job")
                await _post(_job_line(
                    e["cluster"], getattr(j, "job_id", "") if j else "",
                    getattr(j, "job_name", "") if j else "",
                    getattr(j, "phase", None) if j else None,
                    getattr(j, "variant", None) if j else None,
                    new,
                    getattr(j, "state", "") if j else "",
                ))
        self.state = self._persistable(current)
        self._persist()

    async def run(self) -> None:
        self._load_state()
        while True:
            try:
                await self._tick()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - keep the loop alive
                print(f"[notifications] monitor tick failed: {exc}")
            await asyncio.sleep(_POLL_INTERVAL)


_monitor = _Monitor()

_LOCK_FILE = _STATE_FILE.parent / "notify.lock"


async def run_monitor() -> None:
    """Run the poller, but only in ONE backend process per machine.

    Two backends sharing a webhook (a stray second uvicorn, a dev copy on
    another port) each poll and diff independently, so every transition posts
    twice. An advisory flock elects a single sender; the others stand by and
    take over if the holder exits (the lock dies with its process).
    """
    _LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = _LOCK_FILE.open("w")  # held for process lifetime
    standby_logged = False
    while True:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except OSError:
            if not standby_logged:
                standby_logged = True
                print(
                    "[notifications] another train-eval-web backend holds "
                    f"{_LOCK_FILE}; notification monitor on standby in this process"
                )
            await asyncio.sleep(_POLL_INTERVAL)
    if standby_logged:
        print("[notifications] lock acquired; notification monitor active")
    await _monitor.run()
