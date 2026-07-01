"""Websocket policy server exposing a fine-tuned GR00T N1.6 / PhysiXel policy
over the openpi-client protocol that the DexJoCo eval client speaks.

obs in  (single-arm): {"base", "wrist": uint8[H,W,3], "state": float[23], "prompt": str}
obs in  (dual-arm):   {"base", "wrist_left", "wrist_right": uint8[H,W,3], "state": float[46], "prompt": str}
obs out : {"actions": float32[horizon, D]}  D=22 single-arm, 44 dual-arm
          (single: [xyz3, rotvec3, hand16]; dual: [r_xyz3, r_rotvec3, r_hand16, l_xyz3, l_rotvec3, l_hand16])

The protocol mirrors openpi.serving.websocket_policy_server.WebsocketPolicyServer
(metadata-on-connect, recv->infer->send, /healthz, keepalive ping disabled).
"""
import argparse
import asyncio
import http
import logging
import traceback

import numpy as np
import websockets
import websockets.asyncio.server as _server
import websockets.frames

import msgpack_numpy  # copied next to this file from openpi_client

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gr00t_dexjoco_server")

DEFAULT_PROMPT = "Grasp the watering can and apply water to the plant."


class Gr00tDexJoCoPolicy:
    def __init__(self, model_path: str, embodiment_tag: str, default_prompt: str, image_size: int | None = None):
        self.default_prompt = default_prompt
        self.image_size = image_size
        self.policy = Gr00tPolicy(
            embodiment_tag=EmbodimentTag(embodiment_tag),
            model_path=model_path,
            device="cuda",
            strict=True,
        )
        logger.info("Gr00tPolicy loaded from %s (action_horizon=%s)",
                    model_path, getattr(self.policy.model.config, "action_horizon", "?"))

    def _prep_img(self, img) -> np.ndarray:
        img = np.asarray(img, dtype=np.uint8)
        if self.image_size:
            # Upscale to match training resolution for models whose processor does
            # not upscale the client's 224x224 (e.g. PhysiXel's area-budget resize,
            # which only downscales -> train=256/eval=224 mismatch). N1.6's processor
            # already upscales via shortest_image_edge, so it leaves image_size unset.
            import cv2
            img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        return img[None, None]  # (B=1, T=1, H, W, 3)

    def infer(self, obs: dict) -> dict:
        prompt = obs.get("prompt", self.default_prompt)
        if isinstance(prompt, bytes):
            prompt = prompt.decode()

        # Camera-agnostic: the client always sends "base" (-> "front") plus either
        # "wrist" (single-arm) or "wrist_left"+"wrist_right" (dual-arm). The video
        # keys must match the checkpoint's baked modality config; state is passed raw
        # (23 single-arm / 46 dual-arm) and action is returned as-is (22 / 44).
        video = {"front": self._prep_img(obs["base"])}
        for k in ("wrist", "wrist_left", "wrist_right"):
            if k in obs:
                video[k] = self._prep_img(obs[k])
        state = np.asarray(obs["state"], dtype=np.float32).reshape(-1)

        observation = {
            "video": video,
            "state": {"state": state[None, None].astype(np.float32)},  # (1, 1, state_dim)
            "language": {"annotation.human.task_description": [[prompt]]},  # (B)(T)
        }
        action_chunk, _ = self.policy.get_action(observation)
        act = np.asarray(action_chunk["action"], dtype=np.float32)
        if act.ndim == 3:      # (B, horizon, D) -> (horizon, D)
            act = act[0]
        return {"actions": act.astype(np.float32)}


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
                action = self._policy.infer(obs)
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
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--embodiment_tag", default="new_embodiment")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--image_size", type=int, default=None,
                    help="If set, resize incoming camera images to NxN before inference "
                         "(for models whose processor does not upscale, e.g. PhysiXel area-budget).")
    args = ap.parse_args()

    policy = Gr00tDexJoCoPolicy(args.model_path, args.embodiment_tag, args.prompt, image_size=args.image_size)
    server = WebsocketPolicyServer(policy, args.host, args.port)
    logger.info("serving GR00T policy on %s:%d", args.host, args.port)
    server.serve_forever()
