# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Policy transforms for the BDS VR_H5D robot (left arm + left vacuum).

Ported from the openpi fork's ``openpi/policies/bds_policy.py`` so that the
state/action slicing matches the ``pi05_bds_vfe_sim_pick_lora_0710`` config.
"""

import dataclasses
from collections.abc import Sequence

import einops
import numpy as np
from openpi import transforms
from openpi.models import model as _model

# observation.state is 96-dim (pos/vel/eff for 32 joints). Both arms live at
# 14..27 and the vacuums at 30, 31, in the same order as the 16-dim action.
STATE_INDICES = (*range(14, 28), 30, 31)
ACTION_DIM = 16
FULL_STATE_DIM = 96

# Left arm only: 7 left arm joints + left vacuum.
LEFT_STATE_INDICES = (*range(14, 21), 30)  # in the 96-dim observation.state
LEFT_ACTION_INDICES = (*range(7), 14)  # in the 16-dim action


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class BDSInputs(transforms.DataTransformFn):
    """Convert BDS dataset / inference inputs to the model input format."""

    model_type: _model.ModelType
    # Indices into the 96-dim observation.state used as the model state.
    state_indices: Sequence[int] = STATE_INDICES
    # Indices into the 16-dim dataset action predicted by the model. None = all 16.
    action_indices: Sequence[int] | None = None

    def __call__(self, data: dict) -> dict:
        head_image = _parse_image(data["observation.images.color.head"])
        outside_image = _parse_image(data["observation.images.color.outside"])

        state = np.asarray(data["observation.state"])
        # Accept both the full 96-dim dataset state and an already-sliced state.
        if state.shape[-1] == FULL_STATE_DIM:
            state = state[..., list(self.state_indices)]
        elif state.shape[-1] != len(self.state_indices):
            raise ValueError(
                f"Expected observation.state of dim {FULL_STATE_DIM} or "
                f"{len(self.state_indices)}, got {state.shape}"
            )

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": head_image,
                "left_wrist_0_rgb": outside_image,
                "right_wrist_0_rgb": np.zeros_like(head_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_
                if self.model_type == _model.ModelType.PI0_FAST
                else np.False_,
            },
        }

        if "actions" in data:
            actions = np.asarray(data["actions"])
            inputs["actions"] = (
                actions
                if self.action_indices is None
                else actions[..., list(self.action_indices)]
            )

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class BDSOutputs(transforms.DataTransformFn):
    """Convert model outputs back to the BDS action layout (inference only)."""

    # Number of action dims the model predicts (16, or 8 for left arm only).
    action_dim: int = ACTION_DIM

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., : self.action_dim])}
