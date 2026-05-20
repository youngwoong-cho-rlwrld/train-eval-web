"""Parse GR00T N1.6 modality config files into UI-friendly summaries."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .paths import EXPERIMENTS_DIR
from .variants import Variant, load_variant


class ActionConfigSummary(BaseModel):
    index: int
    key: str | None = None
    rep: str | None = None
    type: str | None = None
    format: str | None = None


class DataInterfaceModality(BaseModel):
    name: str
    delta_indices: list[int] | None = None
    delta_indices_expr: str | None = None
    horizon: int | None = None
    modality_keys: list[str] = Field(default_factory=list)
    action_configs: list[ActionConfigSummary] = Field(default_factory=list)


class DataInterfaceSummary(BaseModel):
    variant: str
    source: str | None = None
    path: str | None = None
    text: str | None = None
    config_name: str | None = None
    embodiment_tag: str | None = None
    modalities: list[DataInterfaceModality] = Field(default_factory=list)
    error: str | None = None


async def load_data_interface(variant_name: str) -> DataInterfaceSummary:
    variant = await load_variant(variant_name)
    return parse_data_interface_for_variant(variant)


def parse_data_interface_for_variant(variant: Variant) -> DataInterfaceSummary:
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
            error=f"data interface file not found: {shown_path}",
        )

    text = path.read_text()

    try:
        tree = ast.parse(text)
    except SyntaxError as e:
        return DataInterfaceSummary(
            variant=variant.name,
            source=rel,
            path=shown_path,
            text=text,
            error=f"could not parse Python file: {e.msg}",
        )

    try:
        summary = _parse_tree(tree)
    except ValueError as e:
        return DataInterfaceSummary(
            variant=variant.name,
            source=rel,
            path=shown_path,
            error=str(e),
        )
    return summary.model_copy(update={
        "variant": variant.name,
        "source": rel,
        "path": shown_path,
        "text": text,
    })


def _parse_tree(tree: ast.Module) -> DataInterfaceSummary:
    dict_assignments: dict[str, ast.Dict] = {}
    selected_name: str | None = None
    embodiment_tag: str | None = None

    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    dict_assignments[target.id] = node.value
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            if _call_name(call.func) != "register_modality_config":
                continue
            if call.args:
                selected_name = _name_ref(call.args[0])
            for kw in call.keywords:
                if kw.arg == "embodiment_tag":
                    embodiment_tag = _enum_or_name(kw.value)

    if selected_name and selected_name in dict_assignments:
        config_name = selected_name
        config_dict = dict_assignments[selected_name]
    elif len(dict_assignments) == 1:
        config_name, config_dict = next(iter(dict_assignments.items()))
    else:
        raise ValueError("could not find a registered modality config dictionary")

    modalities: list[DataInterfaceModality] = []
    for key_node, value_node in zip(config_dict.keys, config_dict.values):
        name = _literal_str(key_node)
        if not name or not isinstance(value_node, ast.Call):
            continue
        if _call_name(value_node.func) != "ModalityConfig":
            continue
        modalities.append(_parse_modality(name, value_node))

    return DataInterfaceSummary(
        variant="",
        config_name=config_name,
        embodiment_tag=embodiment_tag,
        modalities=modalities,
    )


def _parse_modality(name: str, call: ast.Call) -> DataInterfaceModality:
    kwargs = {kw.arg: kw.value for kw in call.keywords if kw.arg}
    delta_node = kwargs.get("delta_indices")
    delta_indices = _int_list(delta_node)
    action_configs = _action_configs(kwargs.get("action_configs"))
    keys = _str_list(kwargs.get("modality_keys"))
    return DataInterfaceModality(
        name=name,
        delta_indices=delta_indices,
        delta_indices_expr=_expr(delta_node),
        horizon=len(delta_indices) if delta_indices is not None else None,
        modality_keys=keys,
        action_configs=[
            cfg.model_copy(update={"key": keys[cfg.index] if cfg.index < len(keys) else None})
            for cfg in action_configs
        ],
    )


def _action_configs(node: ast.AST | None) -> list[ActionConfigSummary]:
    if not isinstance(node, ast.List):
        return []
    out: list[ActionConfigSummary] = []
    for i, item in enumerate(node.elts):
        if not isinstance(item, ast.Call) or _call_name(item.func) != "ActionConfig":
            continue
        kwargs = {kw.arg: _enum_or_name(kw.value) for kw in item.keywords if kw.arg}
        out.append(ActionConfigSummary(
            index=i,
            rep=kwargs.get("rep"),
            type=kwargs.get("type"),
            format=kwargs.get("format"),
        ))
    return out


def _int_list(node: ast.AST | None) -> list[int] | None:
    if isinstance(node, ast.List):
        values: list[int] = []
        for item in node.elts:
            value = _literal_int(item)
            if value is None:
                return None
            values.append(value)
        return values
    if (
        isinstance(node, ast.Call)
        and _call_name(node.func) == "list"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Call)
        and _call_name(node.args[0].func) == "range"
    ):
        args = [_literal_int(arg) for arg in node.args[0].args]
        if any(arg is None for arg in args):
            return None
        ints = [arg for arg in args if arg is not None]
        if len(ints) == 1:
            return list(range(ints[0]))
        if len(ints) == 2:
            return list(range(ints[0], ints[1]))
        if len(ints) == 3:
            return list(range(ints[0], ints[1], ints[2]))
    return None


def _str_list(node: ast.AST | None) -> list[str]:
    if not isinstance(node, ast.List):
        return []
    return [value for item in node.elts if (value := _literal_str(item)) is not None]


def _literal_str(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _literal_int(node: ast.AST | None) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    return None


def _name_ref(node: ast.AST) -> str | None:
    return node.id if isinstance(node, ast.Name) else None


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _enum_or_name(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Constant):
        return str(node.value)
    return _expr(node)


def _expr(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    try:
        return ast.unparse(node)
    except Exception:
        return None
