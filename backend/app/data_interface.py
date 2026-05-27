"""Load GR00T modality config files for submit/job review."""

from __future__ import annotations

import ast
from pathlib import Path

from pydantic import BaseModel

from .paths import EXPERIMENTS_DIR
from .variants import Variant, load_variant


class DataInterfaceSummary(BaseModel):
    variant: str
    source: str | None = None
    path: str | None = None
    text: str | None = None
    config_name: str | None = None
    embodiment_tag: str | None = None
    action_horizon: int | None = None
    error: str | None = None


async def load_data_interface(variant_name: str) -> DataInterfaceSummary:
    variant = await load_variant(variant_name)
    return load_data_interface_for_variant(variant)


def load_data_interface_for_variant(variant: Variant) -> DataInterfaceSummary:
    rel = (variant.vars.get("TRAIN_MODALITY_CONFIG") or "").strip()
    if not rel:
        return DataInterfaceSummary(
            variant=variant.name,
            error="TRAIN_MODALITY_CONFIG is not set for this experiment",
        )
    if Path(rel).is_absolute() or ".." in Path(rel).parts:
        return DataInterfaceSummary(
            variant=variant.name,
            source=rel,
            error="TRAIN_MODALITY_CONFIG must be a relative file inside the experiment directory",
        )

    path = EXPERIMENTS_DIR / variant.name / rel
    shown_path = f"configs/experiments/{variant.name}/{rel}"
    if not path.is_file():
        return DataInterfaceSummary(
            variant=variant.name,
            source=rel,
            path=shown_path,
            error=f"modality.py not found: {shown_path}",
        )

    text = path.read_text()
    base = {
        "variant": variant.name,
        "source": rel,
        "path": shown_path,
        "text": text,
    }
    try:
        tree = ast.parse(text)
        config_name, embodiment_tag = _registered_metadata(tree)
        action_horizon = _action_horizon(tree, config_name)
    except SyntaxError as e:
        return DataInterfaceSummary(**base, error=f"could not parse Python file: {e.msg}")
    return DataInterfaceSummary(
        **base,
        config_name=config_name,
        embodiment_tag=embodiment_tag,
        action_horizon=action_horizon,
    )


def rewrite_action_horizon(text: str, action_horizon: int) -> str:
    """Return modality config text with action delta_indices set to range(N)."""
    if action_horizon <= 0:
        raise ValueError(f"action horizon must be positive, got {action_horizon}")
    try:
        tree = ast.parse(text)
    except SyntaxError as e:
        raise ValueError(f"could not parse modality config: {e.msg}") from e

    config_name, _ = _registered_metadata(tree)
    dicts = _top_level_dicts(tree)
    for config in _candidate_config_dicts(dicts, config_name):
        action = _dict_value(config, "action")
        if action is None:
            continue
        delta_indices = _call_keyword(action, "delta_indices")
        if delta_indices is None:
            continue
        start, end = _node_span(text, delta_indices)
        return text[:start] + f"list(range({action_horizon}))" + text[end:]
    raise ValueError("modality config action.delta_indices was not found")


def _registered_metadata(tree: ast.Module) -> tuple[str | None, str | None]:
    dict_names = list(_top_level_dicts(tree))
    registered_name: str | None = None
    embodiment_tag: str | None = None

    for node in tree.body:
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            continue
        call = node.value
        if _call_name(call.func) != "register_modality_config":
            continue
        if call.args and isinstance(call.args[0], ast.Name):
            registered_name = call.args[0].id
        for kw in call.keywords:
            if kw.arg == "embodiment_tag":
                embodiment_tag = _simple_expr(kw.value)

    config_name = registered_name or (dict_names[0] if len(dict_names) == 1 else None)
    return config_name, embodiment_tag


def _action_horizon(tree: ast.Module, config_name: str | None) -> int | None:
    dicts = _top_level_dicts(tree)
    candidates = _candidate_config_dicts(dicts, config_name)
    for config in candidates:
        value = _dict_value(config, "action")
        if value is None:
            continue
        delta_indices = _call_keyword(value, "delta_indices")
        if delta_indices is None:
            continue
        length = _sequence_len(delta_indices)
        if length is not None:
            return length
    return None


def _top_level_dicts(tree: ast.Module) -> dict[str, ast.Dict]:
    dicts: dict[str, ast.Dict] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Dict):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                dicts[target.id] = node.value
    return dicts


def _candidate_config_dicts(
    dicts: dict[str, ast.Dict],
    config_name: str | None,
) -> list[ast.Dict]:
    candidates: list[ast.Dict] = []
    if config_name and config_name in dicts:
        candidates.append(dicts[config_name])
    candidates.extend(value for key, value in dicts.items() if key != config_name)
    return candidates


def _dict_value(node: ast.Dict, key: str) -> ast.AST | None:
    for k, v in zip(node.keys, node.values):
        if isinstance(k, ast.Constant) and k.value == key:
            return v
    return None


def _call_keyword(node: ast.AST, key: str) -> ast.AST | None:
    if not isinstance(node, ast.Call):
        return None
    for kw in node.keywords:
        if kw.arg == key:
            return kw.value
    return None


def _sequence_len(node: ast.AST) -> int | None:
    if isinstance(node, (ast.List, ast.Tuple)):
        return len(node.elts)
    if isinstance(node, ast.Call) and _call_name(node.func) == "list":
        if len(node.args) == 1 and isinstance(node.args[0], ast.Call):
            return _range_len(node.args[0])
    if isinstance(node, ast.Call) and _call_name(node.func) == "range":
        return _range_len(node)
    return None


def _range_len(node: ast.Call) -> int | None:
    if _call_name(node.func) != "range":
        return None
    values: list[int] = []
    for arg in node.args:
        if not isinstance(arg, ast.Constant) or not isinstance(arg.value, int):
            return None
        values.append(arg.value)
    if len(values) == 1:
        start, stop, step = 0, values[0], 1
    elif len(values) == 2:
        start, stop = values
        step = 1
    elif len(values) == 3:
        start, stop, step = values
    else:
        return None
    if step == 0:
        return None
    return len(range(start, stop, step))


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _simple_expr(node: ast.AST) -> str | None:
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Constant):
        return str(node.value)
    try:
        return ast.unparse(node)
    except Exception:
        return None


def _node_span(text: str, node: ast.AST) -> tuple[int, int]:
    if (
        getattr(node, "lineno", None) is None
        or getattr(node, "col_offset", None) is None
        or getattr(node, "end_lineno", None) is None
        or getattr(node, "end_col_offset", None) is None
    ):
        raise ValueError("could not locate action.delta_indices source span")
    offsets: list[int] = []
    total = 0
    for line in text.splitlines(keepends=True):
        offsets.append(total)
        total += len(line)
    start = offsets[node.lineno - 1] + node.col_offset
    end = offsets[node.end_lineno - 1] + node.end_col_offset
    return start, end
