#!/usr/bin/env python3
"""Compare N1.6 and PhysiXel AH40 training-time behavior.

Run this script from a checked-out GR00T/PhysiXel worktree with that
worktree's virtual environment. It writes JSON artifacts that can be compared
across the N1.6 and PhysiXel worktrees.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import random
import subprocess
import sys
import traceback
from collections.abc import Mapping
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    probe = sub.add_parser("probe")
    probe.add_argument("--label", required=True)
    probe.add_argument("--expected-commit", required=True)
    probe.add_argument("--out-dir", required=True)
    probe.add_argument("--base-model", default="nvidia/GR00T-N1.6-3B")
    probe.add_argument("--modality-config", required=True)
    probe.add_argument("--dataset-path", action="append", required=True)
    probe.add_argument("--global-batch-size", type=int, default=128)
    probe.add_argument("--num-gpus", type=int, default=2)
    probe.add_argument("--batch-size", type=int, default=1)
    probe.add_argument("--seed", type=int, default=42)
    probe.add_argument("--forward-seed", type=int, default=20260526)
    probe.add_argument("--run-forward", action="store_true")
    probe.add_argument("--run-optimizer-step", action="store_true")
    probe.add_argument("--include-eval-batch", action="store_true")
    probe.add_argument(
        "--parameter-scope",
        choices=("full", "selected", "none"),
        default="full",
        help="How many parameters to hash before the batch/forward probes.",
    )
    probe.add_argument("--state-part-mode", default=None)
    probe.add_argument("--state-part-token-count", type=int, default=None)
    probe.add_argument("--state-part-seed", type=int, default=None)

    cmp_parser = sub.add_parser("compare")
    cmp_parser.add_argument("--left", required=True)
    cmp_parser.add_argument("--right", required=True)
    cmp_parser.add_argument("--out", required=True)

    ckpt_cmp = sub.add_parser("compare-checkpoints")
    ckpt_cmp.add_argument("--left-run-dir", required=True)
    ckpt_cmp.add_argument("--right-run-dir", required=True)
    ckpt_cmp.add_argument("--out", required=True)
    ckpt_cmp.add_argument("--steps", default=None, help="Comma-separated checkpoint steps.")
    ckpt_cmp.add_argument(
        "--parameter-scope",
        choices=("full", "selected"),
        default="selected",
    )

    args = parser.parse_args()
    if args.cmd == "probe":
        return run_probe(args)
    if args.cmd == "compare":
        return run_compare(args)
    if args.cmd == "compare-checkpoints":
        return run_compare_checkpoints(args)
    raise AssertionError(args.cmd)


def run_probe(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"label": args.label, "status": "running"}
    try:
        _set_reproducible_seeds(args.seed)
        report["environment"] = environment_report(args.expected_commit)

        load_modality_config(Path(args.modality_config))
        config = build_config(args)

        from gr00t.experiment.experiment import setup_logging
        from gr00t.model.registry import MODEL_REGISTRY
        from transformers import set_seed

        setup_logging()
        set_seed(config.data.seed)

        save_cfg_dir = out_dir / "experiment_cfg"
        save_cfg_dir.mkdir(parents=True, exist_ok=True)
        pipeline_cls = MODEL_REGISTRY.get(type(config.model))
        pipeline = pipeline_cls(config, save_cfg_dir)

        pipeline.setup()
        model = pipeline.return_model()
        train_dataset, _ = pipeline.return_dataset()
        data_collator = pipeline.return_collator()

        report["model_config"] = simple_model_config(model.config)
        report["parameters"] = parameter_manifest(model, args.parameter_scope)
        report["trainable_parameter_summary"] = trainable_parameter_summary(model)

        batch = first_batch(train_dataset, data_collator, args.batch_size)
        report["batch"] = summarize_tree(batch)
        (out_dir / "batch_summary.json").write_text(
            json.dumps(report["batch"], indent=2, sort_keys=True) + "\n"
        )

        if args.include_eval_batch:
            processor = pipeline.return_processor()
            processor.eval()
            eval_batch = first_batch(train_dataset, data_collator, args.batch_size)
            processor.train()
            report["eval_batch"] = summarize_tree(eval_batch)
            (out_dir / "eval_batch_summary.json").write_text(
                json.dumps(report["eval_batch"], indent=2, sort_keys=True) + "\n"
            )

        if args.run_forward:
            report["forward"] = run_forward_probe(model, batch, args.forward_seed)
            (out_dir / "forward_summary.json").write_text(
                json.dumps(report["forward"], indent=2, sort_keys=True) + "\n"
            )

        if args.run_optimizer_step:
            report["optimizer_step"] = run_optimizer_probe(model, batch, args.forward_seed)
            (out_dir / "optimizer_step_summary.json").write_text(
                json.dumps(report["optimizer_step"], indent=2, sort_keys=True) + "\n"
            )

        report["status"] = "ok"
    except Exception as exc:  # noqa: BLE001 - diagnostic artifact needs full failure context.
        report["status"] = "error"
        report["error"] = repr(exc)
        report["traceback"] = traceback.format_exc()
        (out_dir / "probe_error.txt").write_text(report["traceback"])
    finally:
        (out_dir / "probe_summary.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
    return 0 if report["status"] == "ok" else 1


def _set_reproducible_seeds(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def environment_report(expected_commit: str) -> dict[str, Any]:
    import gr00t
    import gr00t.model.gr00t_n1d6.processing_gr00t_n1d6 as processing

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    return {
        "cwd": str(Path.cwd()),
        "expected_commit": expected_commit,
        "actual_commit": commit,
        "commit_matches": commit == expected_commit,
        "python": sys.executable,
        "python_version": sys.version,
        "gr00t_file": getattr(gr00t, "__file__", None),
        "processor_file": getattr(processing, "__file__", None),
        "pythonpath_head": sys.path[:8],
    }


def load_modality_config(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    sys.path.insert(0, str(path.parent))
    importlib.import_module(path.stem)


def build_config(args: argparse.Namespace):
    from gr00t.configs.base_config import get_default_config
    from gr00t.data.embodiment_tags import EmbodimentTag

    config = get_default_config().load_dict(
        {
            "data": {
                "download_cache": False,
                "datasets": [
                    {
                        "dataset_paths": args.dataset_path,
                        "mix_ratio": 1.0,
                        "embodiment_tag": EmbodimentTag.NEW_EMBODIMENT.value,
                    }
                ],
            }
        }
    )
    config.load_config_path = None

    config.model.tune_llm = False
    config.model.tune_visual = False
    config.model.tune_projector = True
    config.model.tune_diffusion_model = True
    config.model.state_dropout_prob = 0.0
    config.model.load_bf16 = False
    config.model.reproject_vision = False
    config.model.eagle_collator = True
    config.model.model_name = "nvidia/Eagle-Block2A-2B-v2"
    config.model.backbone_trainable_params_fp32 = True
    config.model.use_relative_action = True
    config.model.random_diffusion = False
    if args.state_part_mode is not None and hasattr(config.model, "state_part_mode"):
        config.model.state_part_mode = args.state_part_mode
    if args.state_part_token_count is not None and hasattr(config.model, "state_part_token_count"):
        config.model.state_part_token_count = args.state_part_token_count
    if args.state_part_seed is not None and hasattr(config.model, "state_part_seed"):
        config.model.state_part_seed = args.state_part_seed

    config.training.start_from_checkpoint = args.base_model
    config.training.optim = "adamw_torch"
    config.training.global_batch_size = args.global_batch_size
    config.training.dataloader_num_workers = 0
    config.training.learning_rate = 1e-4
    config.training.gradient_accumulation_steps = 1
    config.training.output_dir = str(Path(args.out_dir) / "training_output")
    config.training.save_steps = 30000
    config.training.save_total_limit = 5
    config.training.num_gpus = args.num_gpus
    config.training.use_wandb = False
    config.training.max_steps = 1
    config.training.weight_decay = 1e-5
    config.training.warmup_ratio = 0.05
    config.training.wandb_project = "physixel"
    config.training.experiment_name = f"diagnostic_{args.label}"

    config.data.shard_size = 2**10
    config.data.episode_sampling_rate = 0.1
    config.data.num_shards_per_epoch = 128
    config.data.allow_padding = True
    config.data.override_pretraining_statistics = False
    config.data.seed = args.seed
    return config


def simple_model_config(config: Any) -> dict[str, Any]:
    keys = [
        "action_horizon",
        "max_state_dim",
        "max_action_dim",
        "state_part_token_count",
        "state_part_seed",
        "state_part_mode",
        "state_part_permutation",
        "state_part_groups",
        "random_diffusion",
        "state_dropout_prob",
        "state_additive_noise_scale",
        "use_relative_action",
    ]
    return {k: getattr(config, k, None) for k in keys}


def first_batch(train_dataset: Any, data_collator: Any, batch_size: int) -> dict[str, Any]:
    import torch

    loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=data_collator,
        num_workers=0,
        pin_memory=False,
    )
    return next(iter(loader))


def parameter_manifest(model: Any, scope: str) -> dict[str, Any]:
    manifest = {}
    if scope == "none":
        return manifest
    for name, param in model.named_parameters():
        if scope == "selected" and not selected_name(name):
            continue
        if not param.requires_grad and not name.startswith("action_head"):
            continue
        tensor = param.detach().cpu()
        manifest[name] = tensor_summary(tensor, include_hash=True)
        manifest[name]["requires_grad"] = bool(param.requires_grad)
    return manifest


def trainable_parameter_summary(model: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, param in model.named_parameters():
        group = parameter_group(name)
        item = out.setdefault(group, {"tensors": 0, "elements": 0, "trainable_elements": 0})
        item["tensors"] += 1
        item["elements"] += int(param.numel())
        if param.requires_grad:
            item["trainable_elements"] += int(param.numel())
    return out


def parameter_group(name: str) -> str:
    parts = name.split(".")
    if parts[0] == "action_head" and len(parts) > 1:
        return ".".join(parts[:2])
    return parts[0]


def run_forward_probe(model: Any, batch: dict[str, Any], seed: int) -> dict[str, Any]:
    import torch

    _set_reproducible_seeds(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    batch = move_to_device(batch, device)
    before_rng = rng_state_summary()
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        outputs = model(**dict(batch))
    return {
        "device": str(device),
        "rng_before": before_rng,
        "rng_after": rng_state_summary(),
        "outputs": summarize_tree(outputs),
    }


def run_optimizer_probe(model: Any, batch: dict[str, Any], seed: int) -> dict[str, Any]:
    import torch

    _set_reproducible_seeds(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    batch = move_to_device(batch, device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4,
        weight_decay=1e-5,
    )
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        outputs = model(**dict(batch))
        loss = model_output_loss(outputs)
    loss.backward()
    grad_summary = selected_grad_summary(model)
    optimizer.step()
    post_summary = selected_param_summary(model)
    return {
        "device": str(device),
        "loss": float(loss.detach().cpu().item()),
        "outputs": summarize_tree(outputs),
        "gradients": grad_summary,
        "post_step_parameters": post_summary,
    }


def selected_grad_summary(model: Any) -> dict[str, Any]:
    out = {}
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None and selected_name(name):
            out[name] = tensor_summary(param.grad.detach().cpu(), include_hash=False)
    return out


def selected_param_summary(model: Any) -> dict[str, Any]:
    out = {}
    for name, param in model.named_parameters():
        if param.requires_grad and selected_name(name):
            out[name] = tensor_summary(param.detach().cpu(), include_hash=True)
    return out


def selected_name(name: str) -> bool:
    return (
        name.startswith("action_head.state_encoder")
        or name.startswith("action_head.action_encoder")
        or name.startswith("action_head.model.transformer_blocks.0")
        or name.startswith("action_head.action_decoder")
    )


def move_to_device(value: Any, device: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, Mapping):
        moved = {k: move_to_device(v, device) for k, v in value.items()}
        if type(value).__name__ == "BatchFeature":
            return type(value)(data=moved)
        return moved
    if isinstance(value, list):
        return [move_to_device(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(v, device) for v in value)
    return value


def model_output_loss(outputs: Any) -> Any:
    if isinstance(outputs, Mapping):
        return outputs["loss"]
    return outputs.loss


def rng_state_summary() -> dict[str, Any]:
    import torch

    out = {"torch_cpu": sha256_bytes(torch.get_rng_state().cpu().numpy().tobytes())}
    if torch.cuda.is_available():
        out["torch_cuda0"] = sha256_bytes(torch.cuda.get_rng_state(0).cpu().numpy().tobytes())
    return out


def summarize_tree(value: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return tensor_summary(value.detach().cpu(), include_hash=True)
    if isinstance(value, Mapping):
        return {str(k): summarize_tree(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [summarize_tree(v) for v in value]
    return {"type": type(value).__name__, "repr": repr(value)[:500], "hash": sha256_bytes(repr(value).encode())}


def tensor_summary(tensor: Any, *, include_hash: bool) -> dict[str, Any]:
    import torch

    out: dict[str, Any] = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "numel": int(tensor.numel()),
    }
    if tensor.numel() == 0:
        out.update({"min": None, "max": None, "mean": None, "norm": 0.0})
    elif torch.is_floating_point(tensor):
        f = tensor.float()
        out.update(
            {
                "min": float(f.min().item()),
                "max": float(f.max().item()),
                "mean": float(f.mean().item()),
                "norm": float(torch.linalg.vector_norm(f).item()),
            }
        )
    else:
        out.update(
            {
                "min": scalar_value(tensor.min()),
                "max": scalar_value(tensor.max()),
                "mean": float(tensor.float().mean().item()),
                "norm": float(torch.linalg.vector_norm(tensor.float()).item()),
            }
        )
    if include_hash:
        out["sha256"] = tensor_hash(tensor)
    return out


def scalar_value(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value


def tensor_hash(tensor: Any) -> str:
    import torch

    tensor = tensor.contiguous()
    if tensor.dtype == torch.bfloat16:
        arr = tensor.view(torch.uint16).numpy()
    else:
        arr = tensor.numpy()
    return sha256_bytes(arr.tobytes())


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run_compare(args: argparse.Namespace) -> int:
    left = json.loads((Path(args.left) / "probe_summary.json").read_text())
    right = json.loads((Path(args.right) / "probe_summary.json").read_text())
    result = compare_reports(left, right)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


def compare_reports(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    result = {
        "left_label": left.get("label"),
        "right_label": right.get("label"),
        "summary": {},
        "differences": {},
    }
    result["summary"]["status"] = [left.get("status"), right.get("status")]
    result["summary"]["commit_matches"] = [
        left.get("environment", {}).get("commit_matches"),
        right.get("environment", {}).get("commit_matches"),
    ]
    result["differences"]["model_config"] = dict_diff(
        left.get("model_config", {}), right.get("model_config", {})
    )
    result["differences"]["batch"] = first_tree_differences(left.get("batch", {}), right.get("batch", {}))
    result["differences"]["eval_batch"] = first_tree_differences(
        left.get("eval_batch", {}), right.get("eval_batch", {})
    )
    result["differences"]["parameters"] = compare_tensor_manifests(
        left.get("parameters", {}), right.get("parameters", {})
    )
    result["differences"]["forward"] = first_tree_differences(
        left.get("forward", {}), right.get("forward", {})
    )
    result["differences"]["optimizer_step"] = first_tree_differences(
        left.get("optimizer_step", {}), right.get("optimizer_step", {})
    )
    result["summary"]["first_different_stage"] = first_different_stage(result["differences"])
    return result


def dict_diff(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    diff = {}
    for key in sorted(set(left) | set(right)):
        if left.get(key) != right.get(key):
            diff[key] = {"left": left.get(key), "right": right.get(key)}
    return diff


def first_tree_differences(left: Any, right: Any, path: str = "", limit: int = 40) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    if type(left) is not type(right):
        return [{"path": path, "left_type": type(left).__name__, "right_type": type(right).__name__}]
    if isinstance(left, dict):
        if "sha256" in left or "shape" in left:
            if left != right:
                return [{"path": path, "left": compact_summary(left), "right": compact_summary(right)}]
            return []
        out = []
        for key in sorted(set(left) | set(right)):
            out.extend(first_tree_differences(left.get(key), right.get(key), f"{path}.{key}" if path else str(key), limit - len(out)))
            if len(out) >= limit:
                break
        return out
    if isinstance(left, list):
        out = []
        for idx in range(max(len(left), len(right))):
            lval = left[idx] if idx < len(left) else None
            rval = right[idx] if idx < len(right) else None
            out.extend(first_tree_differences(lval, rval, f"{path}[{idx}]", limit - len(out)))
            if len(out) >= limit:
                break
        return out
    if left != right:
        return [{"path": path, "left": left, "right": right}]
    return []


def compare_tensor_manifests(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    shared = sorted(set(left) & set(right))
    missing_left = sorted(set(right) - set(left))
    missing_right = sorted(set(left) - set(right))
    hash_diff = [name for name in shared if left[name].get("sha256") != right[name].get("sha256")]
    shape_diff = [name for name in shared if left[name].get("shape") != right[name].get("shape")]
    return {
        "left_count": len(left),
        "right_count": len(right),
        "missing_left": missing_left[:40],
        "missing_right": missing_right[:40],
        "shape_diff": shape_diff[:40],
        "hash_diff_count": len(hash_diff),
        "hash_diff_examples": hash_diff[:40],
    }


def compact_summary(value: dict[str, Any]) -> dict[str, Any]:
    keys = ["shape", "dtype", "numel", "sha256", "min", "max", "mean", "norm", "repr", "hash"]
    return {k: value.get(k) for k in keys if k in value}


def first_different_stage(differences: dict[str, Any]) -> str | None:
    order = ["model_config", "batch", "eval_batch", "parameters", "forward", "optimizer_step"]
    for key in order:
        value = differences.get(key)
        if isinstance(value, dict) and any(value.values()):
            return key
        if isinstance(value, list) and value:
            return key
    return None


def run_compare_checkpoints(args: argparse.Namespace) -> int:
    left_dir = Path(args.left_run_dir)
    right_dir = Path(args.right_run_dir)
    steps = parse_steps(args.steps, left_dir, right_dir)
    report: dict[str, Any] = {
        "left_run_dir": str(left_dir),
        "right_run_dir": str(right_dir),
        "steps": steps,
        "parameter_scope": args.parameter_scope,
        "step_reports": {},
        "first_different_step": None,
    }
    for step in steps:
        left_ckpt = left_dir / f"checkpoint-{step}"
        right_ckpt = right_dir / f"checkpoint-{step}"
        left_manifest = checkpoint_manifest(left_ckpt, args.parameter_scope)
        right_manifest = checkpoint_manifest(right_ckpt, args.parameter_scope)
        diff = compare_tensor_manifests(left_manifest, right_manifest)
        left_state = trainer_state_summary(left_ckpt)
        right_state = trainer_state_summary(right_ckpt)
        step_report = {
            "left_checkpoint": str(left_ckpt),
            "right_checkpoint": str(right_ckpt),
            "parameters": diff,
            "trainer_state": {
                "left": left_state,
                "right": right_state,
                "diff": dict_diff(left_state, right_state),
            },
        }
        if diff["hash_diff_count"] and report["first_different_step"] is None:
            report["first_different_step"] = step
        report["step_reports"][str(step)] = step_report
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: report[k] for k in ["steps", "first_different_step"]}, indent=2))
    return 0


def parse_steps(raw_steps: str | None, left_dir: Path, right_dir: Path) -> list[int]:
    if raw_steps:
        return [int(item) for item in raw_steps.split(",") if item.strip()]
    left_steps = discover_checkpoint_steps(left_dir)
    right_steps = discover_checkpoint_steps(right_dir)
    return sorted(set(left_steps) & set(right_steps))


def discover_checkpoint_steps(run_dir: Path) -> list[int]:
    out = []
    for path in run_dir.glob("checkpoint-*"):
        suffix = path.name.removeprefix("checkpoint-")
        if suffix.isdigit():
            out.append(int(suffix))
    return sorted(out)


def checkpoint_manifest(checkpoint_dir: Path, scope: str) -> dict[str, Any]:
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(checkpoint_dir)
    manifest: dict[str, Any] = {}
    for tensor_file in checkpoint_tensor_files(checkpoint_dir):
        for name, tensor in load_tensor_file(tensor_file).items():
            clean_name = name.removeprefix("module.")
            if scope == "selected" and not selected_name(clean_name):
                continue
            if scope == "full" and not (clean_name.startswith("action_head") or selected_name(clean_name)):
                continue
            manifest[clean_name] = tensor_summary(tensor.detach().cpu(), include_hash=True)
    return manifest


def checkpoint_tensor_files(checkpoint_dir: Path) -> list[Path]:
    patterns = [
        "model.safetensors",
        "pytorch_model.bin",
        "*.safetensors",
        "pytorch_model-*.bin",
    ]
    files: list[Path] = []
    for pattern in patterns:
        for path in checkpoint_dir.glob(pattern):
            if path.name.startswith("optimizer") or path.name.startswith("scheduler"):
                continue
            if path not in files:
                files.append(path)
    if not files:
        raise FileNotFoundError(f"No model tensor files found in {checkpoint_dir}")
    return sorted(files)


def load_tensor_file(path: Path) -> dict[str, Any]:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path), device="cpu")
    import torch

    data = torch.load(path, map_location="cpu")
    if isinstance(data, Mapping) and "state_dict" in data:
        data = data["state_dict"]
    return dict(data)


def trainer_state_summary(checkpoint_dir: Path) -> dict[str, Any]:
    state_path = checkpoint_dir / "trainer_state.json"
    if not state_path.is_file():
        return {}
    state = json.loads(state_path.read_text())
    history = state.get("log_history") or []
    return {
        "global_step": state.get("global_step"),
        "epoch": state.get("epoch"),
        "best_metric": state.get("best_metric"),
        "last_log": history[-1] if history else None,
    }


if __name__ == "__main__":
    raise SystemExit(main())
