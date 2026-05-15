"""Variant config parsing.

Each `configs/experiments/<name>/config.sh` is sourced as a bash file. We
capture scalars + arrays (DATASETS, TASKS, TRAIN_EXTRA_ARGS, EVAL_SETS) and
return a typed Pydantic object.

Parsing is done in a local bash subprocess — no SSH needed since configs live
in the local repo.
"""

import asyncio
import re
from typing import Any

from pydantic import BaseModel

from .clusters import _BASH
from .paths import EXPERIMENTS_DIR


class Variant(BaseModel):
    name: str
    raw: str            # full file contents
    vars: dict[str, str]      # scalar variables (MAX_STEPS, MODEL_VERSION, ...)
    arrays: dict[str, list[str]]  # arrays (DATASETS, TASKS, EVAL_SETS, TRAIN_EXTRA_ARGS)


def list_variants() -> list[str]:
    """Variants directly under configs/experiments/.

    Skip names starting with `_` — those are templates / scratch (notably
    `_sample/`, which is the on-repo reference variant for new users).
    """
    return sorted(
        p.name
        for p in EXPERIMENTS_DIR.iterdir()
        if p.is_dir()
        and not p.name.startswith("_")
        and (p / "config.sh").is_file()
    )


async def load_variant(name: str) -> Variant:
    cfg_path = EXPERIMENTS_DIR / name / "config.sh"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Variant config not found: {cfg_path}")
    raw = cfg_path.read_text()
    vars, arrays = await _parse_bash(raw)
    return Variant(name=name, raw=raw, vars=vars, arrays=arrays)


_SCALAR_RE = re.compile(r'^declare -[a-zA-Z\-]+ ([A-Za-z_][A-Za-z0-9_]*)="(.*)"$')
# Array entries can be quoted or unquoted: `[0]="..." [1]="..."` or `[0]=foo`.
_ARRAY_LINE_RE = re.compile(r'^declare -[a-zA-Z\-]+ ([A-Za-z_][A-Za-z0-9_]*)=\((.*)\)$')
_ARRAY_ITEM_RE = re.compile(r'\[\d+\]=(?:"((?:[^"\\]|\\.)*)"|(\S+))')


async def _parse_bash(script_text: str) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Source a bash snippet and return (scalars, arrays)."""
    cmd = f"set -a\n{script_text}\nset +a\ndeclare -p"
    proc = await asyncio.create_subprocess_exec(
        _BASH, "-c", cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"bash failed: {stderr.decode()}")

    scalars: dict[str, str] = {}
    arrays: dict[str, list[str]] = {}
    for line in stdout.decode().splitlines():
        if m := _ARRAY_LINE_RE.match(line):
            name = m.group(1)
            items_str = m.group(2)
            items = []
            for im in _ARRAY_ITEM_RE.finditer(items_str):
                quoted, raw = im.group(1), im.group(2)
                value = quoted if quoted is not None else raw
                items.append(_bash_unescape(value))
            arrays[name] = items
        elif m := _SCALAR_RE.match(line):
            scalars[m.group(1)] = _bash_unescape(m.group(2))
    return scalars, arrays


def _bash_unescape(s: str) -> str:
    return s.replace(r"\"", '"').replace(r"\\", "\\").replace(r"\$", "$")
