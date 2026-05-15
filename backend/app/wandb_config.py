"""Persisted wandb settings.

We store the wandb project name in ~/.train-eval-web/wandb.json so the
backend, the body script renderer, and the URL builder all agree on
where the runs live. The TRAIN_EVAL_WEB_WANDB_PROJECT env var still
wins over the saved value when present.
"""

import json
import os
from pathlib import Path

DEFAULT_PROJECT = "finetune-gr00t-n1d6"
_SETTINGS_DIR = Path.home() / ".train-eval-web"
_SETTINGS_FILE = _SETTINGS_DIR / "wandb.json"


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


def get_project() -> str:
    """Env var > saved file > default."""
    env = os.environ.get("TRAIN_EVAL_WEB_WANDB_PROJECT")
    if env:
        return env
    return _load().get("project") or DEFAULT_PROJECT


def set_project(name: str) -> str:
    data = _load()
    data["project"] = name.strip()
    _save(data)
    return data["project"]
