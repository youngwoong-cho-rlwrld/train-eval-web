from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


# DexJoCo single-arm (Franka 7-DoF EEF + Allegro 16-DoF hand) for the water_plant task.
# state[23] = proprio (eef pose quat + hand); action[22] = [xyz(3), rotvec(3), hand(16)].
# Action is ABSOLUTE rotvec EEF executed directly by the env; we regress it raw
# (NON_EEF / ABSOLUTE / DEFAULT) to match the pi0.5 baseline and the eval contract,
# rather than GR00T's default state-relative chunks (action[22] and state[23] are not
# the same space, so RELATIVE is invalid here).
dexjoco_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["front", "wrist"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["state"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(16)),
        modality_keys=["action"],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(dexjoco_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
