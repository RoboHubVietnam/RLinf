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

import dataclasses
import pathlib
from collections.abc import Sequence

import openpi.models.model as _model
import openpi.transforms as _transforms
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory
from typing_extensions import override

from rlinf.models.embodiment.openpi.policies import bds_policy


@dataclasses.dataclass(frozen=True)
class LeRobotBDSDataConfig(DataConfigFactory):
    """Data config for the BDS VR_H5D robot (LeRobot v2.1, 16-dim action).

    Defaults select the left arm + left vacuum (8 dims), matching the openpi
    ``pi05_bds_vfe_sim_pick_lora_0710`` modality.
    """

    extra_delta_transform: bool = True
    # Indices into the 96-dim observation.state used as the model state.
    state_indices: Sequence[int] = bds_policy.LEFT_STATE_INDICES
    # Indices into the 16-dim dataset action predicted by the model. Must be in
    # the same joint order as ``state_indices`` (delta actions use the state).
    action_indices: Sequence[int] = bds_policy.LEFT_ACTION_INDICES

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation.images.color.head": "observation.images.color.head",
                        "observation.images.color.outside": "observation.images.color.outside",
                        "observation.state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[
                bds_policy.BDSInputs(
                    model_type=model_config.model_type,
                    state_indices=self.state_indices,
                    action_indices=self.action_indices,
                )
            ],
            outputs=[bds_policy.BDSOutputs(action_dim=len(self.action_indices))],
        )

        # Dataset actions are absolute joint positions: train on deltas for the
        # arm joints and keep the vacuum commands (dataset dims 14, 15) absolute.
        if self.extra_delta_transform:
            delta_action_mask = tuple(i not in (14, 15) for i in self.action_indices)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )
