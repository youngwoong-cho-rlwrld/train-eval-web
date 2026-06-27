"""Variant config parsing.

Each `configs/experiments/<name>/config.sh` is sourced as a bash file. We
capture scalars + arrays (DATASETS, TASKS, TRAIN_EXTRA_ARGS, EVAL_SETS) and
return a typed Pydantic object.

Parsing is done in a local bash subprocess — no SSH needed since configs live
in the local repo.
"""

import asyncio
import re
import shutil
from collections.abc import Callable, Set
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml
from pydantic import BaseModel

from .clusters import _BASH
from .paths import EXPERIMENTS_DIR
from .training_models import resolve_training_model


class Variant(BaseModel):
    name: str
    raw: str            # full file contents
    vars: dict[str, str]      # scalar variables (MAX_STEPS, MODEL_VERSION, ...)
    arrays: dict[str, list[str]]  # arrays (DATASETS, TASKS, EVAL_SETS, TRAIN_EXTRA_ARGS)


class VariantFile(BaseModel):
    kind: str
    label: str
    title: str
    path: str
    content: str
    exists: bool
    purpose: str


class VariantFileVersion(BaseModel):
    created_at: str
    path: str
    files: list[str]


class VariantFiles(BaseModel):
    variant: str
    model_family: str
    config: VariantFile
    second_file: VariantFile
    versions: list[VariantFileVersion]


class SaveVariantFilesRequest(BaseModel):
    config_title: str = "config.sh"
    config_content: str
    second_title: str
    second_content: str


class SaveVariantFilesResponse(VariantFiles):
    saved_version_path: str | None = None


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
    return await parse_variant_text(name, raw)


async def load_variant_files(name: str) -> VariantFiles:
    exp_dir = _experiment_dir(name)
    cfg_path = exp_dir / "config.sh"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Variant config not found: {cfg_path}")
    config_text = cfg_path.read_text()
    variant = await parse_variant_text(name, config_text)
    model_family = resolve_training_model(variant).family
    second_file = _active_second_file(exp_dir, variant, model_family)
    return VariantFiles(
        variant=name,
        model_family=model_family,
        config=VariantFile(
            kind="config_sh",
            label="config.sh",
            title="config.sh",
            path=_display_path(name, "config.sh"),
            content=config_text,
            exists=True,
            purpose="Defines the experiment: model family, datasets, training/eval defaults, and the second model-facing config file.",
        ),
        second_file=second_file,
        versions=_list_file_versions(exp_dir, name),
    )


async def save_variant_files(
    name: str,
    req: SaveVariantFilesRequest,
) -> SaveVariantFilesResponse:
    exp_dir = _experiment_dir(name)
    cfg_path = exp_dir / "config.sh"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Variant config not found: {cfg_path}")
    if req.config_title.strip() != "config.sh":
        raise ValueError("config title must remain config.sh")

    current = await load_variant_files(name)
    requested_variant = await parse_variant_text(name, req.config_content)
    model_family = resolve_training_model(requested_variant).family
    second_title = _validate_second_title(req.second_title, model_family)
    normalized_config = _set_second_file_ref(
        req.config_content,
        second_title,
        model_family,
    )
    await parse_variant_text(name, normalized_config)
    _validate_second_content(req.second_content, second_title, model_family)

    new_second_path = exp_dir / second_title
    old_second_path = exp_dir / current.second_file.title
    changed = (
        normalized_config != current.config.content
        or second_title != current.second_file.title
        or req.second_content != current.second_file.content
    )
    saved_version_path = _snapshot_variant_files(
        exp_dir,
        [cfg_path, old_second_path],
    ) if changed else None

    cfg_path.write_text(normalized_config)
    new_second_path.write_text(req.second_content)
    if old_second_path != new_second_path and old_second_path.is_file():
        old_second_path.unlink()

    saved = await load_variant_files(name)
    return SaveVariantFilesResponse(
        **saved.model_dump(),
        saved_version_path=_relative_display_path(saved_version_path) if saved_version_path else None,
    )


async def restore_variant_file_version(
    name: str,
    version: str,
) -> SaveVariantFilesResponse:
    exp_dir = _experiment_dir(name)
    if not re.fullmatch(r"\d{8}_\d{6}(?:_\d+)?", version):
        raise ValueError(f"invalid version id: {version!r}")
    version_dir = exp_dir / ".versions" / version
    if not version_dir.is_dir():
        raise FileNotFoundError(f"variant file version not found: {version}")
    restore_files = sorted(p for p in version_dir.iterdir() if p.is_file())
    if not restore_files:
        raise ValueError(f"variant file version is empty: {version}")
    if not any(p.name == "config.sh" for p in restore_files):
        raise ValueError(f"variant file version {version} does not include config.sh")

    current = await load_variant_files(name)
    current_paths = [
        exp_dir / "config.sh",
        exp_dir / current.second_file.title,
    ]
    saved_version_path = _snapshot_variant_files(exp_dir, current_paths)

    for src in restore_files:
        shutil.copy2(src, exp_dir / src.name)

    saved = await load_variant_files(name)
    return SaveVariantFilesResponse(
        **saved.model_dump(),
        saved_version_path=_relative_display_path(saved_version_path) if saved_version_path else None,
    )


async def parse_variant_text(name: str, raw: str) -> Variant:
    vars, arrays = await _parse_bash(raw)
    return Variant(name=name, raw=raw, vars=vars, arrays=arrays)


def _experiment_dir(name: str) -> Path:
    if "/" in name or "\\" in name or ".." in name:
        raise ValueError(f"invalid experiment name: {name!r}")
    exp_dir = EXPERIMENTS_DIR / name
    if not exp_dir.is_dir():
        raise FileNotFoundError(f"Variant not found: {exp_dir}")
    return exp_dir


def _display_path(variant: str, rel: str) -> str:
    return f"configs/experiments/{variant}/{rel}"


def _relative_display_path(path: Path) -> str:
    try:
        return str(path.relative_to(EXPERIMENTS_DIR.parent.parent))
    except ValueError:
        return str(path)


def _active_second_file(exp_dir: Path, variant: Variant, model_family: str) -> VariantFile:
    title = _active_second_title(exp_dir, variant, model_family)
    path = exp_dir / title
    return VariantFile(
        kind=_second_kind(model_family),
        label=_second_label(model_family),
        title=title,
        path=_display_path(variant.name, title),
        content=path.read_text() if path.is_file() else _default_second_content(variant, model_family),
        exists=path.is_file(),
        purpose=_second_purpose(model_family),
    )


def _active_second_title(exp_dir: Path, variant: Variant, model_family: str) -> str:
    spec = _spec_for(model_family)
    rel = (variant.vars.get(spec.config_var) or "").strip()
    if rel and _is_safe_relative_file(rel, spec.suffixes):
        return rel
    if model_family == "n1.5":
        for candidate in ("data_config.yaml", "data_config.yml"):
            if (exp_dir / candidate).is_file():
                return candidate
        return spec.default_name
    py_files = sorted(p.name for p in exp_dir.glob("*.py") if p.is_file())
    return py_files[0] if len(py_files) == 1 else spec.default_name


def _is_safe_relative_file(rel: str, suffixes: Set[str]) -> bool:
    path = Path(rel)
    return (
        bool(rel)
        and not path.is_absolute()
        and path.name == rel
        and not rel.startswith(".")
        and ".." not in path.parts
        and path.suffix in suffixes
    )


def _second_kind(model_family: str) -> str:
    return _spec_for(model_family).kind


def _second_label(model_family: str) -> str:
    return _spec_for(model_family).label


def _second_purpose(model_family: str) -> str:
    return _spec_for(model_family).purpose


def _default_second_content(variant: Variant, model_family: str) -> str:
    if model_family == "n1.5":
        return _generated_data_config_yaml(variant)
    return ""


def _generated_data_config_yaml(variant: Variant) -> str:
    rows: list[str] = ["train:", "  datasets:"]
    if variant.arrays.get("DATASETS"):
        entries = variant.arrays["DATASETS"]
    elif variant.vars.get("DATASET_NAME"):
        cfg = variant.vars.get("DATA_CONFIG", "allex_thetwo_ck40_egostereo")
        entries = [f"{variant.vars['DATASET_NAME']}|{cfg}|1.0"]
    else:
        entries = []
    for entry in entries:
        parts = entry.split("|", 2)
        if len(parts) == 3:
            name, cfg, weight = parts
        else:
            name = entry
            cfg = variant.vars.get("DATA_CONFIG", "allex_thetwo_ck40_egostereo")
            weight = "1.0"
        rows.extend([
            f"    - path: $DATA_DIR/{name}",
            "      embodiment_tag: new_embodiment",
            f"      data_config: {cfg}",
            f"      weight: {weight}",
        ])
    return "\n".join(rows) + "\n"


def _validate_second_title(title: str, model_family: str) -> str:
    clean = title.strip()
    spec = _spec_for(model_family)
    if not _is_safe_relative_file(clean, spec.suffixes):
        suffix = ".yaml/.yml" if model_family == "n1.5" else ".py"
        raise ValueError(f"second file title must be a single {suffix} filename")
    if clean == "config.sh":
        raise ValueError("second file title cannot be config.sh")
    return clean


def _validate_second_content(text: str, title: str, model_family: str) -> None:
    _spec_for(model_family).validator(text, title)


def _validate_n15_data_config_yaml(text: str, title: str) -> None:
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise ValueError(f"{title}: YAML syntax error: {e}")
    datasets = data.get("train", {}).get("datasets") if isinstance(data, dict) else None
    if not isinstance(datasets, list) or not datasets:
        raise ValueError(f"{title}: expected non-empty train.datasets list")
    for idx, dataset in enumerate(datasets):
        if not isinstance(dataset, dict):
            raise ValueError(f"{title}: train.datasets[{idx}] must be an object")
        missing = {"path", "embodiment_tag", "data_config"} - set(dataset)
        if missing:
            raise ValueError(
                f"{title}: train.datasets[{idx}] missing {', '.join(sorted(missing))}"
            )


def _validate_python(text: str, title: str) -> None:
    try:
        compile(text, title, "exec")
    except SyntaxError as e:
        raise ValueError(f"{title}: Python syntax error at line {e.lineno}: {e.msg}")


@dataclass(frozen=True)
class SecondFileSpec:
    config_var: str
    suffixes: frozenset[str]
    default_name: str
    kind: str
    label: str
    purpose: str
    validator: Callable[[str, str], None]


# Keyed by model family. The n1.6 spec is the default for any non-n1.5 family,
# preserving the original `else` semantics (physixel/dexjoco-* resolve here).
_SECOND_FILE_SPECS: dict[str, SecondFileSpec] = {
    "n1.5": SecondFileSpec(
        config_var="TRAIN_DATA_CONFIG",
        suffixes=frozenset({".yaml", ".yml"}),
        default_name="data_config.yaml",
        kind="data_config_yaml",
        label="data_config.yaml",
        purpose="YAML passed to gr00t_finetune.py as --data-config. It lists datasets and DATA_CONFIG_MAP keys such as allex_thetwo_ck40_egostereo.",
        validator=_validate_n15_data_config_yaml,
    ),
    "n1.6": SecondFileSpec(
        config_var="TRAIN_MODALITY_CONFIG",
        suffixes=frozenset({".py"}),
        default_name="modality.py",
        kind="modality_py",
        label="modality.py",
        purpose="Python modality config passed to GR00T N1.6/Physixel as --modality-config-path for train and --modality-config for eval.",
        validator=_validate_python,
    ),
}


def _spec_for(model_family: str) -> SecondFileSpec:
    return _SECOND_FILE_SPECS["n1.5"] if model_family == "n1.5" else _SECOND_FILE_SPECS["n1.6"]


def _set_second_file_ref(config_text: str, second_title: str, model_family: str) -> str:
    if model_family == "n1.5":
        return _set_scalar(config_text, "TRAIN_DATA_CONFIG", second_title)
    return _set_scalar(config_text, "TRAIN_MODALITY_CONFIG", second_title)


def _set_scalar(config_text: str, key: str, value: str) -> str:
    line = f"{key}={value}"
    pattern = re.compile(rf"(?m)^(?:export\s+)?{re.escape(key)}=.*$")
    if pattern.search(config_text):
        return pattern.sub(line, config_text)
    suffix = "" if config_text.endswith("\n") else "\n"
    return f"{config_text}{suffix}\n{line}\n"


def _snapshot_variant_files(exp_dir: Path, paths: list[Path]) -> Path | None:
    existing = [p for p in dict.fromkeys(paths) if p and p.is_file()]
    if not existing:
        return None
    root = exp_dir / ".versions"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = root / stamp
    counter = 1
    while dest.exists():
        counter += 1
        dest = root / f"{stamp}_{counter}"
    dest.mkdir(parents=True, exist_ok=False)
    for path in existing:
        shutil.copy2(path, dest / path.name)
    return dest


def _list_file_versions(exp_dir, variant: str) -> list[VariantFileVersion]:
    root = exp_dir / ".versions"
    if not root.is_dir():
        return []
    out: list[VariantFileVersion] = []
    for version in sorted((p for p in root.iterdir() if p.is_dir()), reverse=True):
        files = sorted(p.name for p in version.iterdir() if p.is_file())
        if not files:
            continue
        out.append(VariantFileVersion(
            created_at=version.name,
            path=_display_path(variant, f".versions/{version.name}"),
            files=files,
        ))
    return out


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
