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
        config_name, embodiment_tag = _registered_metadata(ast.parse(text))
    except SyntaxError as e:
        return DataInterfaceSummary(**base, error=f"could not parse Python file: {e.msg}")
    return DataInterfaceSummary(
        **base,
        config_name=config_name,
        embodiment_tag=embodiment_tag,
    )


def _registered_metadata(tree: ast.Module) -> tuple[str | None, str | None]:
    dict_names: list[str] = []
    registered_name: str | None = None
    embodiment_tag: str | None = None

    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
            dict_names.extend(target.id for target in node.targets if isinstance(target, ast.Name))
            continue
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
