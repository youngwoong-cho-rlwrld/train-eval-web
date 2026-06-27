"""Eval-harness abstraction.

An eval job runs in one of two environments:
  - ``isaac``: Isaac Sim, driven by ``scripts/eval_allex.py`` against a GR00T
    policy server (``lib/eval_body.sh``).
  - ``dexjoco``: the DexJoCo MuJoCo benchmark, driven by ``dexjoco-openpi-eval``
    against either a GR00T or an openpi/pi0.5 policy server
    (``lib/eval_body_dexjoco.sh``).

Each harness owns the behavior that used to be scattered as
``if EVAL_HARNESS == "dexjoco"`` string checks: the entrypoint flag list shown
in the submit preview and the submit-time required-field validation. Eval
completion + progress probing hang off the same abstraction (see
``eval_completion.py`` / ``details.py``).

``harness_for(variant)`` selects the harness from the ``EVAL_HARNESS`` config
var, defaulting to ``isaac``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .variants import Variant


class EvalHarness(ABC):
    """Strategy for one eval environment."""

    name: str

    @abstractmethod
    def eval_flags(self, variant: "Variant") -> list[tuple[str, str]]:
        """The ``(flag, value)`` list the eval entrypoint receives.

        Mirrors the harness's ``lib/eval_body*.sh`` client invocation so the
        submit preview shows the flags the body would actually emit.
        """

    def validate_submit(self, req) -> None:
        """Raise ``ValueError`` if a harness-required submit field is missing.

        Default: the harness has no extra required field.
        """
        return None


class IsaacHarness(EvalHarness):
    name = "isaac"

    def eval_flags(self, variant: "Variant") -> list[tuple[str, str]]:
        v = variant
        return [
            ("--task-name", v.vars.get("TASK_NAME", "")),
            ("--instruction", v.vars.get("INSTRUCTION", "")),
            ("--n-episodes", v.vars.get("N_EPISODES", "")),
            ("--n-runs", v.vars.get("N_RUNS", "")),
            ("EVAL_NUM_ENVS_PER_GPU", "1"),
            ("--execution-horizon", v.vars.get("EXECUTION_HORIZON", "")),
            ("--max-episode-steps", v.vars.get("MAX_EPISODE_STEPS", "")),
            ("(eval_sets)", " ".join(v.arrays.get("EVAL_SETS") or [])),
        ]


class DexjocoHarness(EvalHarness):
    name = "dexjoco"

    def eval_flags(self, variant: "Variant") -> list[tuple[str, str]]:
        v = variant
        return [
            ("--task", v.vars.get("DEXJOCO_TASK", "")),
            ("--server", v.vars.get("DEXJOCO_SERVER_TYPE", "groot")),
            ("(families)", " ".join(v.arrays.get("EVAL_SETS") or [])),
            ("--episodes", v.vars.get("N_EPISODES", "")),
            ("--n-runs", v.vars.get("N_RUNS", "")),
            ("--seed", v.vars.get("EVAL_BASE_SEED", "")),
            ("--checkpoint", "<eval-checkpoint>"),
        ]

    def validate_submit(self, req) -> None:
        task = getattr(req, "dexjoco_task", None)
        if not (task and task.strip()):
            raise ValueError("dexjoco_task is required for DexJoCo evals")


DEFAULT_HARNESS = "isaac"
_HARNESSES: dict[str, EvalHarness] = {
    harness.name: harness for harness in (IsaacHarness(), DexjocoHarness())
}


def harness_for(variant: "Variant") -> EvalHarness:
    """Select the eval harness for ``variant`` from its ``EVAL_HARNESS`` var."""
    name = (variant.vars.get("EVAL_HARNESS") or DEFAULT_HARNESS).strip().lower()
    return _HARNESSES.get(name, _HARNESSES[DEFAULT_HARNESS])
