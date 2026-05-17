#!/usr/bin/env python3
"""Run the ALLEX Isaac server with extensions needed by headless eval.

Some IsaacLab assets create ``PreviewSurfaceCfg`` materials during scene
construction. In the headless Isaac Sim app used by eval, the USD material
commands are not always registered unless ``omni.usd.commands`` is imported
after the Kit app starts. This wrapper mirrors ``server_v2.py``'s entrypoint
and performs that import before task modules are loaded.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import asdict
from pathlib import Path

import tyro


def _load_server_module(server_path: Path):
    spec = importlib.util.spec_from_file_location("allex_server_v2", server_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import server module: {server_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: isaac_server_runner.py <server_v2.py> [server args...]")

    server_path = Path(sys.argv[1]).resolve()
    server_args = sys.argv[2:]
    isaac_root = server_path.parents[2]
    os.chdir(isaac_root)
    sys.path.insert(0, str(isaac_root))

    server = _load_server_module(server_path)
    cfg = tyro.cli(server.AllexEnvServerConfig, args=server_args)

    if cfg.task_name is not None:
        task_name_value = cfg.task_name.strip()
        if task_name_value:
            try:
                server._update_task_config_from_task_name(task_name_value)
                print(f"[INFO] Applied task preset: {task_name_value}")
            except Exception as exc:
                raise RuntimeError(f"Failed to apply --task_name preset: {exc}") from exc

    server._set_eval_env_flag(cfg.eval_set)
    server._set_force_progress_in_train(cfg.force_progress_in_train)

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(asdict(cfg.app_launcher))
    simulation_app = app_launcher.app

    import omni.kit.commands
    import omni.usd.commands as usd_commands

    omni.kit.commands.register_all_commands_in_module(usd_commands)

    import isaaclab_tasks  # noqa: F401
    import allex_sim.tasks.uni_pick_place  # noqa: F401
    from isaaclab.utils.seed import configure_seed
    from isaaclab_tasks.utils import parse_env_cfg

    server.configure_seed = configure_seed
    server.parse_env_cfg = parse_env_cfg
    server.main(cfg, simulation_app)


if __name__ == "__main__":
    main()
