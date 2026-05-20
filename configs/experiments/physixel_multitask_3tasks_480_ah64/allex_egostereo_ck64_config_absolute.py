# Modality config for ALLEX robot with egostereo cameras and 64-step action horizon.
# Compatible with GR00T N1.6. Uses ABSOLUTE action representation.

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


allex_egostereo_ck64_config_absolute = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["camera_ego_left", "camera_ego_right"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "right_arm_joints",
            "left_arm_joints",
            "right_hand_joints",
            "left_hand_joints",
            "neck_joints",
            "waist_joints",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(64)),
        modality_keys=[
            "right_arm_joints",
            "left_arm_joints",
            "right_hand_joints",
            "left_hand_joints",
            "neck_joints",
            "waist_joints",
        ],
        action_configs=[
            ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
            ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
            ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
            ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
            ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
            ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(allex_egostereo_ck64_config_absolute, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
