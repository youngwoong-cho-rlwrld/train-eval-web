"""Websocket policy server exposing a fine-tuned GAM (Geometric Action Model)
DexJoCo policy over the openpi-client protocol the DexJoCo eval client speaks.

Byte-compatible with lib/dexjoco/gr00t_dexjoco_server.py:
  obs in  (single-arm): {"base", "wrist": uint8[H,W,3], "state": float[23], "prompt": str}
  obs in  (dual-arm):   {"base", "wrist_left", "wrist_right": uint8[H,W,3], "state": float[46], "prompt": str}
  obs out : {"actions": float32[horizon, D]}  D=22 single-arm, 44 dual-arm

GAM is single-view: only "base" (-> the model's single "front"/"ego" training view)
is consumed; wrist frames are ignored. State and actions are raw joint-space
vectors (no EEF math).

Multi-embodiment mapping (a 44-D multitask model served for a single-arm task,
--embodiment-tag dexjoco_single_arm):
  * incoming state is zero-padded 23 -> model proprio_dim (46), native dims in [0:23];
  * outgoing actions are sliced 44 -> 22 (the [0:22] right-arm block).
This mirrors the dataset's single_to_dual padding (see dexjoco/DEXJOCO_INTEGRATION.md).

The model is built with the GAM repo's ``load_stage1_policy`` (run with
PYTHONPATH=$GAM_DIR/src). Each request is stateless: the rollout closure's
episode history is reset and one model step (== chunk_size env actions) is
returned, matching the GR00T server's per-call action chunk.

The websocket/msgpack/healthz/worker-thread protocol is copied verbatim from the
GR00T server so the harness client and eval_body_dexjoco.sh need no changes.
"""
import argparse
import asyncio
import http
import logging
import os
import traceback

import numpy as np
import torch
import websockets
import websockets.asyncio.server as _server
import websockets.frames
from omegaconf import OmegaConf

import msgpack_numpy  # copied next to this file from openpi_client

from eval_libero_unified import load_stage1_policy  # GAM repo src on PYTHONPATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gam_dexjoco_server")

DEFAULT_PROMPT = "Grasp the watering can and apply water to the plant."

# Output action dim the DexJoCo client expects per embodiment tag.
_TAG_ACTION_DIM = {
    "dexjoco_single_arm": 22,
    "dexjoco_dual_arm": 44,
}


def _resolve_checkpoint(checkpoint_path: str) -> tuple[str, str]:
    """Return (ckpt_file, config_yaml) from a submission dir or a direct .pt.

    A submission passes its checkpoint dir; the training wrapper writes
    ``checkpoint-final.pt`` + ``config.yaml`` there (see dexjoco/train_dexjoco.sh).
    A direct ``.pt`` path is also accepted (config.yaml is read from its parent).
    """
    if os.path.isdir(checkpoint_path):
        ckpt_file = os.path.join(checkpoint_path, "checkpoint-final.pt")
        config_dir = checkpoint_path
    else:
        ckpt_file = checkpoint_path
        config_dir = os.path.dirname(checkpoint_path)
    if not os.path.exists(ckpt_file):
        raise FileNotFoundError(f"GAM checkpoint not found: {ckpt_file}")
    config_yaml = os.path.join(config_dir, "config.yaml")
    if not os.path.exists(config_yaml):
        raise FileNotFoundError(f"GAM config.yaml not found next to checkpoint: {config_yaml}")
    return ckpt_file, config_yaml


class GamDexJoCoPolicy:
    def __init__(self, checkpoint_path: str, embodiment_tag: str, default_prompt: str):
        if embodiment_tag not in _TAG_ACTION_DIM:
            raise ValueError(
                f"Unknown --embodiment-tag {embodiment_tag!r}; expected one of {sorted(_TAG_ACTION_DIM)}."
            )
        self.default_prompt = default_prompt
        self.d_out = _TAG_ACTION_DIM[embodiment_tag]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ckpt_file, config_yaml = _resolve_checkpoint(checkpoint_path)
        cfg = OmegaConf.load(config_yaml)
        ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)

        self.model_n_dims = int(cfg.action_head.n_dims)
        self.proprio_dim = int(cfg.proprioception.proprio_dim)
        if self.d_out > self.model_n_dims:
            raise ValueError(
                f"embodiment-tag wants {self.d_out}-D actions but the model outputs "
                f"{self.model_n_dims}-D (checkpoint/config mismatch)."
            )

        # EMA weights when the checkpoint carries them (train_dexjoco.sh runs
        # with training.ema.enabled); plain weights otherwise instead of the
        # hard failure _require_stage1_ema_state would raise.
        use_ema = any(
            ckpt.get(key) is not None
            for key in (
                "student_da3_ema",
                "action_head_ema",
                "future_predictor_ema",
                "text_conditioner_proj_ema",
            )
        )
        # raw_task_text: DexJoCo trains on the LeRobot tasks.jsonl strings as-is
        # (no LIBERO lowercase/punctuation normalization).
        # rollout_decode_horizon=1: with active_action_horizon=1 the AR decode
        # is exactly the one executed model step; the default "full" would
        # decode the whole native train horizon per request only to discard it.
        self.policy, self.info = load_stage1_policy(
            cfg,
            ckpt,
            ckpt_file,
            device,
            stats_key=None,
            action_stats_json=None,
            decode_visuals=False,
            rollout_decode_horizon=1,
            text_prompt_normalization="raw_task_text",
            use_ema=use_ema,
        )
        # One model step per call == chunk_size env actions == the GR00T-style
        # action chunk; the harness executes the whole chunk open-loop.
        self.policy.active_action_horizon = 1
        logger.info(
            "GAM policy loaded from %s (model_n_dims=%d proprio_dim=%d d_out=%d chunk_size=%s stats_key=%s)",
            ckpt_file, self.model_n_dims, self.proprio_dim, self.d_out,
            self.info.get("chunk_size"), self.info.get("action_stats_key"),
        )

    def _prep_state(self, state) -> np.ndarray:
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        if state.shape[0] > self.proprio_dim:
            raise ValueError(
                f"Incoming state dim {state.shape[0]} exceeds model proprio_dim {self.proprio_dim}."
            )
        if state.shape[0] < self.proprio_dim:  # single-arm obs into a dual model
            padded = np.zeros(self.proprio_dim, dtype=np.float32)
            padded[: state.shape[0]] = state
            state = padded
        return state

    def infer(self, obs: dict) -> dict:
        prompt = obs.get("prompt", self.default_prompt)
        if isinstance(prompt, bytes):
            prompt = prompt.decode()

        # Single view: the client's "base" frame is the model's front/ego view.
        closure_obs = {
            "front": np.asarray(obs["base"], dtype=np.uint8),
            "state": self._prep_state(obs["state"]),
        }
        # Stateless: clear rollout history so every call is a fresh H_eff=1 step.
        self.policy.reset_episode()
        with torch.no_grad():
            act = self.policy(closure_obs, prompt)
        act = np.asarray(act.reshape(-1, self.model_n_dims).float().cpu().numpy(), dtype=np.float32)
        horizon = int(self.info.get("chunk_size") or act.shape[0])
        act = act[:horizon, : self.d_out]  # one chunk; slice dual layout -> single-arm block when needed
        return {"actions": np.ascontiguousarray(act, dtype=np.float32)}


class WebsocketPolicyServer:
    def __init__(self, policy, host: str, port: int, metadata: dict | None = None):
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}

    def serve_forever(self):
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            ping_interval=None,
            ping_timeout=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, ws):
        logger.info("Connection from %s opened", ws.remote_address)
        packer = msgpack_numpy.Packer()
        await ws.send(packer.pack(self._metadata))
        while True:
            try:
                obs = msgpack_numpy.unpackb(await ws.recv())
                # Worker thread, NOT inline: the first get_action can spend
                # minutes in torch.compile / DA3 warmup, and blocking the event
                # loop there leaves the client's keepalive unanswered.
                action = await asyncio.to_thread(self._policy.infer, obs)
                await ws.send(packer.pack(action))
            except websockets.ConnectionClosed:
                logger.info("Connection from %s closed", ws.remote_address)
                break
            except Exception:
                await ws.send(traceback.format_exc())
                await ws.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(connection, request):
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-path", required=True,
                    help="Submission checkpoint dir (holds checkpoint-final.pt + config.yaml) or a direct .pt path.")
    ap.add_argument("--embodiment-tag", required=True, choices=sorted(_TAG_ACTION_DIM))
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    args = ap.parse_args()

    policy = GamDexJoCoPolicy(args.checkpoint_path, args.embodiment_tag, args.prompt)
    server = WebsocketPolicyServer(policy, args.host, args.port)
    logger.info("serving GAM DexJoCo policy on %s:%d", args.host, args.port)
    server.serve_forever()
