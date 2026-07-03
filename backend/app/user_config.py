"""Persisted user identity settings.

We store the username in ~/.train-eval-web/user.json so default job names
carry a per-user prefix (e.g. youngwoong_train_...). Empty means no prefix.
The TRAIN_EVAL_WEB_USERNAME env var wins over the saved value.
"""
from __future__ import annotations


import json
import os
import re
from pathlib import Path

from pydantic import BaseModel

_SETTINGS_DIR = Path.home() / ".train-eval-web"
_SETTINGS_FILE = _SETTINGS_DIR / "user.json"

# The username lands in slurm job names, wandb run ids, and staging paths,
# so keep it to scheduler/path-safe characters. Phase keywords are reserved:
# job_identity.parse_phase_and_variant keys on the first train/eval/resume
# token, so a username containing one would misparse every job it prefixes.
_USERNAME_RE = re.compile(r"[A-Za-z0-9._-]+")
_RESERVED_TOKENS = {"train", "eval", "resume"}


class UserSettings(BaseModel):
    username: str = ""


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


def validate_username(name: str) -> str:
    name = name.strip()
    if not name:
        return ""
    if not _USERNAME_RE.fullmatch(name):
        raise ValueError(
            "username may only contain letters, numbers, dot, underscore, or hyphen"
        )
    if _RESERVED_TOKENS & set(name.split("_")):
        raise ValueError("username must not contain 'train', 'eval', or 'resume'")
    return name


def get_username() -> str:
    """Env var > saved file. Empty string means no job-name prefix."""
    env = os.environ.get("TRAIN_EVAL_WEB_USERNAME")
    if env:
        return env.strip()
    return (_load().get("username") or "").strip()


def get_settings() -> UserSettings:
    return UserSettings(username=get_username())


def save_settings(req: UserSettings) -> UserSettings:
    data = _load()
    data["username"] = validate_username(req.username)
    _save(data)
    return get_settings()
