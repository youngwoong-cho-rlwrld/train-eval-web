# ============================================================================
# Modality config template (GR00T n1.6 / PhysiXel / DexJoCo).
#
# TRAIN_MODALITY_CONFIG in config.sh points at this file. Training and Isaac
# eval both import it. It declares which dataset keys feed the video / state /
# action / language modalities and how the action is represented.
#
# This is EMBODIMENT- AND DATASET-SPECIFIC — the keys and dimensions below match
# one concrete robot (a Franka 7-DoF EEF arm + Allegro 16-DoF hand). Edit
# `modality_keys`, `delta_indices`, and the `ActionConfig` to match YOUR dataset
# before training. If unsure, copy the modality config that ships with your
# model repo / dataset and adapt it here.
#
# n1.5 variants do NOT use this file (they use DATA_CONFIG instead) — you can
# delete it for an n1.5 variant.
# ============================================================================
from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


# Example: single-arm (Franka 7-DoF EEF + Allegro 16-DoF hand).
#   state[..]  = proprioception (eef pose + hand joints)
#   action[..] = [xyz(3), rotvec(3), hand(16)], absolute EEF executed by the env.
# `delta_indices=list(range(N))` predicts N future action steps — keep N in sync
# with TRAIN_ACTION_HORIZON in config.sh when ACTION_HORIZON_MODE=modality.
modality_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["front", "wrist"],          # EDIT: your camera keys
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["state"],                   # EDIT: your proprio key(s)
    ),
    "action": ModalityConfig(
        delta_indices=list(range(16)),             # EDIT: == TRAIN_ACTION_HORIZON
        modality_keys=["action"],                  # EDIT: your action key(s)
        action_configs=[
            # ABSOLUTE / NON_EEF / DEFAULT = regress the raw action vector as-is.
            # Use this when the action space is executed directly by the env and
            # is NOT the same space as `state` (so RELATIVE would be invalid).
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

# The embodiment tag must match what training/eval pass (NEW_EMBODIMENT is the
# default for freshly fine-tuned checkpoints).
register_modality_config(modality_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
