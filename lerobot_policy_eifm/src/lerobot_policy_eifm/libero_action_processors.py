"""
LIBERO-specific action format conversion processor step.

Registered as 'libero_action_to_ee6d' in the ProcessorStepRegistry so it can
be referenced from policy_preprocessor.json.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.processor import ProcessorStep, ProcessorStepRegistry
from lerobot.processor.core import EnvTransition, TransitionKey
from lerobot.utils.constants import ACTION

from .utils import axis_angle_to_rot6d


@dataclass
@ProcessorStepRegistry.register(name="libero_action_to_ee6d")
class LiberoActionToEE6DProcessorStep(ProcessorStep):
    """Convert 7D LIBERO actions to 20D ee6d format required by X-VLA / EI-FM.

    LIBERO dataset actions are 7D: [delta_pos(3), delta_rot_axis_angle(3), gripper±1(1)]
    X-VLA EE6DActionSpace expects 20D: [pos(3), rot6d(6), g01(1)] × 2 (dual-head)

    Conversion:
    - rot6d  = axis_angle_to_rot6d(delta_rot_aa)   [3D axis-angle → 6D rotation]
    - g01    = (gripper > 0).float()                [{-1,+1} → {0,1} for BCE loss]
    - output = [pos, rot6d, g01] tiled twice        [10D × 2 = 20D]

    Handles action shapes: (B, 7) and (B, T, 7).
    No-ops when action is None, already 20D, or not a Tensor.
    """

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy()
        action = new_transition.get(TransitionKey.ACTION)

        if action is None or not isinstance(action, torch.Tensor):
            return new_transition

        # Skip if already converted
        if action.shape[-1] == 20:
            return new_transition

        # Only handle 7D LIBERO actions
        if action.shape[-1] != 7:
            return new_transition

        device = action.device
        dtype = action.dtype
        orig_shape = action.shape  # (B, 7) or (B, T, 7)

        # Flatten to (N, 7) for vectorised processing
        action_np = action.float().cpu().numpy().reshape(-1, 7)

        pos = action_np[:, :3]       # (N, 3) delta position
        aa = action_np[:, 3:6]       # (N, 3) delta rotation as axis-angle
        gripper = action_np[:, 6:7]  # (N, 1) gripper open/close ±1

        rot6d = axis_angle_to_rot6d(aa)                          # (N, 6)
        gripper01 = (gripper > 0).astype(np.float32)             # (N, 1) → {0, 1}

        chunk10 = np.concatenate([pos, rot6d, gripper01], axis=-1)  # (N, 10)
        action_20d = np.concatenate([chunk10, chunk10], axis=-1)     # (N, 20) dual-head

        # Restore original leading dims
        new_shape = orig_shape[:-1] + (20,)
        action_20d = action_20d.reshape(new_shape)

        new_transition[TransitionKey.ACTION] = torch.from_numpy(action_20d).to(device=device, dtype=dtype)
        return new_transition

    def transform_features(self, features):
        """Update action metadata so saved processor configs reflect the converted 20D layout."""
        new_features = {ft: feats.copy() for ft, feats in features.items()}
        action_features = new_features.get(PipelineFeatureType.ACTION)
        if action_features is None or ACTION not in action_features:
            return new_features

        action_feature = action_features[ACTION]
        action_features[ACTION] = PolicyFeature(
            type=action_feature.type,
            shape=(20,),
        )
        return new_features

    def get_config(self) -> dict[str, Any]:
        return {}
