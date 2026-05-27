"""User-editable cluster environment settings.

Repo cluster env files are templates. The effective, user-specific env text is
saved outside git under ~/.train-eval-web/clusters/<cluster>.env.
"""

from __future__ import annotations

import shlex
from pathlib import Path

from pydantic import BaseModel, Field

from .paths import CLUSTERS_DIR


_SETTINGS_DIR = Path.home() / ".train-eval-web" / "clusters"
_BUILTIN_CLUSTER_ORDER = ("kakao", "skt", "mlxp")


class ClusterEnvSettings(BaseModel):
    name: str
    env_text: str
    template_text: str
    configured: bool
    source: str
    path: str | None = None


class ClusterEnvSettingsUpdate(BaseModel):
    env_text: str = Field(default="")


def list_cluster_names() -> list[str]:
    names = set(_BUILTIN_CLUSTER_ORDER)
    names.update(p.stem for p in CLUSTERS_DIR.glob("*.env") if p.is_file())
    return sorted(names, key=lambda n: (_order(n), n))


def list_settings() -> list[ClusterEnvSettings]:
    return [get_settings(name) for name in list_cluster_names()]


def get_settings(name: str) -> ClusterEnvSettings:
    _validate_name(name)
    saved = _saved_path(name)
    template = _template_path(name)
    if saved.is_file():
        text = saved.read_text()
        source = "saved"
        path = str(saved)
    else:
        text = template.read_text() if template.is_file() else ""
        source = "template"
        path = str(template) if template.is_file() else None
    template_text = template.read_text() if template.is_file() else ""
    return ClusterEnvSettings(
        name=name,
        env_text=text,
        template_text=template_text,
        configured=_is_configured(text),
        source=source,
        path=path,
    )


def save_settings(name: str, env_text: str) -> ClusterEnvSettings:
    _validate_name(name)
    _SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
    normalized = env_text.rstrip() + ("\n" if env_text.strip() else "")
    _saved_path(name).write_text(normalized)
    return get_settings(name)


def load_env_text(name: str) -> str:
    return get_settings(name).env_text


def parse_env_text(text: str) -> dict[str, str]:
    """Best-effort parser for simple export/KEY=value env files.

    Slurm runtime still uses bash sourcing for exact semantics. This parser is
    only for sync settings like MLXP config fields.
    """
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        try:
            parts = shlex.split(line, posix=True)
        except ValueError:
            continue
        if not parts:
            continue
        assignment = parts[0]
        if "=" not in assignment:
            continue
        key, value = assignment.split("=", 1)
        key = key.strip()
        if key:
            out[key] = value
    return out


def _validate_name(name: str) -> None:
    if name not in list_cluster_names():
        raise FileNotFoundError(f"unknown cluster {name}")


def _saved_path(name: str) -> Path:
    return _SETTINGS_DIR / f"{name}.env"


def _template_path(name: str) -> Path:
    return CLUSTERS_DIR / f"{name}.env"


def _is_configured(text: str) -> bool:
    values = parse_env_text(text)
    return any(v.strip() for k, v in values.items() if k not in {"CLUSTER"})


def _order(name: str) -> int:
    try:
        return _BUILTIN_CLUSTER_ORDER.index(name)
    except ValueError:
        return len(_BUILTIN_CLUSTER_ORDER)
