"""Persisted Slack notification settings.

Stored in ~/.train-eval-web/notifications.json so the backend job monitor and
the submit route agree on where/whether to post job-status updates. The
TRAIN_EVAL_WEB_SLACK_WEBHOOK_URL env var wins over the saved URL. The raw
webhook URL is a secret: it is never returned to the frontend (only a
`configured` boolean is).
"""
from __future__ import annotations


import json
import os
from pathlib import Path

from pydantic import BaseModel

_SETTINGS_DIR = Path.home() / ".train-eval-web"
_SETTINGS_FILE = _SETTINGS_DIR / "notifications.json"


class NotificationSettings(BaseModel):
    """Frontend-facing view. `configured` reports whether a webhook is saved;
    the URL itself is intentionally omitted."""
    enabled: bool = False
    configured: bool = False
    notify_submitted: bool = True
    notify_running: bool = False
    notify_completed: bool = True
    notify_failed: bool = True
    notify_cancelled: bool = True


class NotificationSettingsUpdate(BaseModel):
    enabled: bool = False
    # Omitted / empty preserves the currently-saved webhook URL.
    slack_webhook_url: str | None = None
    notify_submitted: bool = True
    notify_running: bool = False
    notify_completed: bool = True
    notify_failed: bool = True
    notify_cancelled: bool = True


def _load() -> dict:
    if not _SETTINGS_FILE.is_file():
        return {}
    try:
        return json.loads(_SETTINGS_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _save(data: dict) -> None:
    _SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
    _SETTINGS_FILE.write_text(json.dumps(data, indent=2))
    try:
        _SETTINGS_FILE.chmod(0o600)  # contains a secret webhook URL
    except OSError:
        pass


def webhook_url() -> str:
    """Env var > saved file."""
    env = os.environ.get("TRAIN_EVAL_WEB_SLACK_WEBHOOK_URL")
    if env:
        return env.strip()
    return (_load().get("slack_webhook_url") or "").strip()


def get_settings() -> NotificationSettings:
    data = _load()
    return NotificationSettings(
        enabled=bool(data.get("enabled", False)),
        configured=bool(webhook_url()),
        notify_submitted=bool(data.get("notify_submitted", True)),
        notify_running=bool(data.get("notify_running", False)),
        notify_completed=bool(data.get("notify_completed", True)),
        notify_failed=bool(data.get("notify_failed", True)),
        notify_cancelled=bool(data.get("notify_cancelled", True)),
    )


def save_settings(req: NotificationSettingsUpdate) -> NotificationSettings:
    data = _load()
    data["enabled"] = bool(req.enabled)
    if req.slack_webhook_url and req.slack_webhook_url.strip():
        data["slack_webhook_url"] = req.slack_webhook_url.strip()
    data["notify_submitted"] = bool(req.notify_submitted)
    data["notify_running"] = bool(req.notify_running)
    data["notify_completed"] = bool(req.notify_completed)
    data["notify_failed"] = bool(req.notify_failed)
    data["notify_cancelled"] = bool(req.notify_cancelled)
    _save(data)
    return get_settings()


def event_enabled(event: str) -> bool:
    s = get_settings()
    return {
        "submitted": s.notify_submitted,
        "running": s.notify_running,
        "completed": s.notify_completed,
        "failed": s.notify_failed,
        "cancelled": s.notify_cancelled,
    }.get(event, False)
