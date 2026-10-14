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

"""Helpers for building openpi LeRobot datasets with extra RLinf options."""

import contextlib
from collections.abc import Iterator


@contextlib.contextmanager
def action_stride_patch(stride: int) -> Iterator[None]:
    """Make openpi's ``create_data_loader`` space action-chunk frames by ``stride``.

    The openpi build used by RLinf always samples ``action_horizon`` consecutive
    frames (``t / fps``). Some datasets are trained with spaced actions (e.g. the
    BDS config uses frames ``t, t+2, ..., t+2*(H-1)``). Inside this context the
    dataset is built with ``t * stride / fps`` instead; ``stride == 1`` is a no-op.

    Args:
        stride: Frame spacing between consecutive actions in a chunk.

    Yields:
        None. The patch is reverted on exit.
    """
    if stride < 1:
        raise ValueError(f"action_stride must be >= 1, got {stride}")
    if stride == 1:
        yield
        return

    import openpi.training.data_loader as openpi_data_loader
    import openpi.transforms as openpi_transforms
    from lerobot.common.datasets import lerobot_dataset

    from rlinf.data.lerobot_paths import resolve_lerobot_dataset_root

    original = openpi_data_loader.create_torch_dataset

    def create_torch_dataset(data_config, action_horizon, model_config):
        repo_id = data_config.repo_id
        if repo_id is None:
            raise ValueError("Repo ID is not set. Cannot create dataset.")
        if repo_id == "fake":
            return original(data_config, action_horizon, model_config)

        root = resolve_lerobot_dataset_root(repo_id)
        meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id, root=root)
        dataset = lerobot_dataset.LeRobotDataset(
            repo_id,
            root=root,
            delta_timestamps={
                key: [t * stride / meta.fps for t in range(action_horizon)]
                for key in data_config.action_sequence_keys
            },
        )
        if data_config.prompt_from_task:
            dataset = openpi_data_loader.TransformedDataset(
                dataset, [openpi_transforms.PromptFromLeRobotTask(meta.tasks)]
            )
        return dataset

    openpi_data_loader.create_torch_dataset = create_torch_dataset
    try:
        yield
    finally:
        openpi_data_loader.create_torch_dataset = original
