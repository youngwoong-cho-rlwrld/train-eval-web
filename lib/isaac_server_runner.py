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


def _patch_per_env_seed_reset(server) -> None:
    """Expand scalar reset seeds into deterministic per-env seeds.

    The upstream server API accepts one scalar seed per vectorized reset. When
    num_envs > 1, that makes the eval logs and some reset paths treat the whole
    vectorized batch as a single seeded episode. We keep the client API scalar,
    but reset each sub-env with a unique seed derived from that scalar.
    """

    original_reset = server.AllexEnvServer.reset

    def reset_with_per_env_seeds(self, seed=None):
        num_envs = int(getattr(self.config, "num_envs", 1) or 1)
        if seed is None or num_envs <= 1:
            return original_reset(self, seed)
        try:
            base_seed = int(seed)
        except (TypeError, ValueError):
            return original_reset(self, seed)

        try:
            try:
                if hasattr(self.env, "sim") and hasattr(self.env.sim, "reset"):
                    self.env.sim.reset()
            except Exception as exc:
                self.logger.warning(f"env.sim.reset() failed (continuing): {exc}")

            env_seeds = [base_seed * num_envs + env_idx for env_idx in range(num_envs)]
            self.logger.info(
                f"Resetting vectorized environment with per-env seeds from base seed {base_seed}: {env_seeds}"
            )

            obs = None
            info = {}
            for env_idx, env_seed in enumerate(env_seeds):
                env_ids = server.torch.tensor([env_idx], dtype=server.torch.int64, device=self.env.device)
                server.configure_seed(env_seed)
                obs, info = self.env.reset(seed=env_seed, env_ids=env_ids)
            if obs is None:
                obs, info = self.env.reset(seed=base_seed)

            if isinstance(info, dict):
                info["per_env_seeds"] = env_seeds

            if self.config.reset_stabilization_steps > 0:
                if self.config.stabilization_mode == server.StabilizationMode.POLICY_ACTION:
                    self._pending_stabilization = (obs, info)
                    obs_serialized = self._transform_observation(obs)
                    self._episode_step_count = 0
                    self._reset_count += 1
                    return {
                        "status": "ok",
                        "observation": obs_serialized,
                        "info": info,
                        "needs_stabilization": True,
                    }
                obs, info = self._stabilize_environment(obs, info)

            obs_serialized = self._transform_observation(obs)
            self._episode_step_count = 0
            self._reset_count += 1
            return {
                "status": "ok",
                "observation": obs_serialized,
                "info": info,
            }
        except TypeError as exc:
            if "env_ids" not in str(exc):
                self.logger.error(f"Error resetting environment with per-env seeds: {exc}", exc_info=True)
                return {"status": "error", "message": f"Error resetting environment: {exc}"}
            self.logger.warning(
                f"Environment reset does not accept env_ids; falling back to scalar seed {base_seed}"
            )
            return original_reset(self, base_seed)
        except Exception as exc:
            self.logger.error(f"Error resetting environment with per-env seeds: {exc}", exc_info=True)
            return {"status": "error", "message": f"Error resetting environment: {exc}"}

    server.AllexEnvServer.reset = reset_with_per_env_seeds


def _patch_headless_viewport_camera() -> None:
    """Ignore IsaacLab viewport camera updates when headless has no viewport prim."""

    from isaaclab.envs.ui.viewport_camera_controller import ViewportCameraController

    original_update = ViewportCameraController.update_view_location

    def update_view_location_headless_safe(self, eye=None, lookat=None):
        try:
            return original_update(self, eye=eye, lookat=lookat)
        except RuntimeError as exc:
            if "invalid null prim" not in str(exc).lower():
                raise
            return None

    ViewportCameraController.update_view_location = update_view_location_headless_safe


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

    _patch_headless_viewport_camera()

    import isaaclab_tasks  # noqa: F401
    import allex_sim.tasks.uni_pick_place  # noqa: F401
    from isaaclab.utils.seed import configure_seed
    from isaaclab_tasks.utils import parse_env_cfg

    server.configure_seed = configure_seed
    server.parse_env_cfg = parse_env_cfg
    _patch_per_env_seed_reset(server)
    server.main(cfg, simulation_app)


if __name__ == "__main__":
    main()
