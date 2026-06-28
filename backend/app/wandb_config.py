"""Persisted wandb settings.

We store the wandb project name in ~/.train-eval-web/wandb.json so the
backend, the body script renderer, and the URL builder all agree on
where the runs live. The TRAIN_EVAL_WEB_WANDB_PROJECT env var still
wins over the saved value when present.
"""
from __future__ import annotations


import json
import os
from pathlib import Path

DEFAULT_PROJECT = "my project"
_SETTINGS_DIR = Path.home() / ".train-eval-web"
_SETTINGS_FILE = _SETTINGS_DIR / "wandb.json"

# Wandb identity overrides:
#   - entity: wandb.Api().default_entity after `wandb login` on this laptop is
#     used by default; this env var forces a specific entity.
#   - workspace: the browser workspace selector for the entity.
WANDB_ENTITY_OVERRIDE = os.environ.get("TRAIN_EVAL_WEB_WANDB_ENTITY")
WANDB_WORKSPACE_OVERRIDE = os.environ.get("TRAIN_EVAL_WEB_WANDB_WORKSPACE")


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


def set_project(name: str) -> None:
    data = _load()
    data["project"] = name.strip()
    _save(data)
