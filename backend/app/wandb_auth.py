"""Wandb credentials and project settings endpoints.

The API key is persisted to `~/.netrc` via `wandb.login(key=...)`. The
project name is persisted to `~/.train-eval-web/wandb.json` (see
wandb_config) and is used both for the backend's run lookup and as the
default project passed to training jobs.
"""
from __future__ import annotations


import asyncio
from pydantic import BaseModel

from .wandb_config import WANDB_ENTITY_OVERRIDE, get_project, set_project


class WandbStatus(BaseModel):
    logged_in: bool
    entity: str | None = None
    project: str
    error: str | None = None


class LoginRequest(BaseModel):
    key: str


class ProjectRequest(BaseModel):
    project: str


async def get_status() -> WandbStatus:
    """Probe wandb to see whether the local netrc/key works."""

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
        project=get_project(),
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

    # Clear cached wandb identity data so the next request re-resolves it.
    details._wandb_entity_cache = None
    details._wandb_workspace_cache.clear()
    return WandbStatus(
        logged_in=entity is not None,
        entity=entity,
        project=get_project(),
        error=err,
    )


async def set_project_endpoint(project: str) -> WandbStatus:
    from . import details

    set_project(project)
    details._wandb_workspace_cache.clear()
    return await get_status()
