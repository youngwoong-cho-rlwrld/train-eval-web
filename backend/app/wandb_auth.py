"""Wandb credentials endpoints.

Lets the user paste their wandb API key once via the UI. The key is
persisted to `~/.netrc` (wandb's standard location) by calling
`wandb.login(key=...)`. After that, the backend's wandb.Api() picks it
up on next import.
"""

import asyncio
from pydantic import BaseModel


class WandbStatus(BaseModel):
    logged_in: bool
    entity: str | None = None
    project: str
    error: str | None = None


class LoginRequest(BaseModel):
    key: str


async def get_status() -> WandbStatus:
    """Probe wandb to see whether the local netrc/key works."""
    from .details import WANDB_PROJECT, WANDB_ENTITY_OVERRIDE

    def _probe() -> tuple[str | None, str | None]:
        try:
            import wandb
            api = wandb.Api(timeout=5)
            entity = WANDB_ENTITY_OVERRIDE or api.default_entity
            return entity, None
        except Exception as e:
            return None, str(e)

    entity, err = await asyncio.to_thread(_probe)
    return WandbStatus(
        logged_in=entity is not None,
        entity=entity,
        project=WANDB_PROJECT,
        error=err,
    )


async def login(key: str) -> WandbStatus:
    """Persist the API key via wandb.login (writes ~/.netrc)."""
    from . import details

    def _do() -> tuple[str | None, str | None]:
        try:
            import wandb
            ok = wandb.login(key=key.strip(), relogin=True, force=True, verify=True)
            if not ok:
                return None, "wandb.login returned false"
            api = wandb.Api(timeout=5)
            return api.default_entity, None
        except Exception as e:
            return None, str(e)

    entity, err = await asyncio.to_thread(_do)

    # Clear the cached entity in details.py so the next request re-resolves it.
    details._wandb_entity_cache = None
    return WandbStatus(
        logged_in=entity is not None,
        entity=entity,
        project=details.WANDB_PROJECT,
        error=err,
    )
