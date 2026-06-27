from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


# DexJoCo dual-arm (bimanual) tasks: two Franka EEFs + two Allegro hands.
# state[46] = [r_arm_pose(7), l_arm_pose(7), r_hand(16), l_hand(16)] (quat poses).
# action[44] = [r_xyz(3), r_rotvec(3), r_hand(16), l_xyz(3), l_rotvec(3), l_hand(16)].
# Cameras: ego + wrist_left + wrist_right. Action is regressed raw (NON_EEF/ABSOLUTE/
# DEFAULT) to match the pi0.5 baseline + eval contract, identical to the single-arm
# config except for the three video keys. state/action dims live in the dataset's
# meta/modality.json (state 0:46, action 0:44).
dexjoco_dual_arm_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["front", "wrist_left", "wrist_right"],
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

register_modality_config(dexjoco_dual_arm_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
