# Copyright 2025 The RLinf Authors.
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

"""GR00T data pipeline for RECAP stage-4 CFG training (FSDPCfgWorker).

GR00T counterpart of the OpenPI CFG data path in ``fsdp_cfg_worker``. Wraps
GR00T's ``LeRobotSingleDataset`` to attach the per-step advantage label (from
``meta/advantages_{tag}.parquet``) and collates batches with GR00T's own
``collate`` into ``(observation, actions, advantage)`` — the 3-tuple
``FSDPCfgWorker.run_training`` consumes.
"""

from pathlib import Path
from typing import Any

import numpy as np
import torch

from rlinf.data.datasets.recap.common import BaseDataLoaderImpl


def load_advantages_lookup(
    data_path: str, advantage_tag: str | None = None
) -> dict[tuple[int, int], bool]:
    """Load (episode_index, frame_index) -> advantage(bool) from the parquet."""
    import pandas as pd

    if advantage_tag:
        meta_path = Path(data_path) / "meta" / f"advantages_{advantage_tag}.parquet"
    else:
        meta_path = Path(data_path) / "meta" / "advantages.parquet"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Advantage file not found: {meta_path}. Run compute_advantages.py first."
        )
    adv_df = pd.read_parquet(meta_path)
    return dict(
        zip(
            zip(
                adv_df["episode_index"].values.astype(int).tolist(),
                adv_df["frame_index"].values.astype(int).tolist(),
            ),
            adv_df["advantage"].values.astype(bool).tolist(),
        )
    )


class AdvantagePreservingGr00tDataset(torch.utils.data.Dataset):
    """Wrap a GR00T ``LeRobotSingleDataset`` to add a per-sample ``advantage`` bool.

    GR00T's transform strips dataset bookkeeping, so the advantage is looked up
    from the underlying dataset's ``all_steps[idx] -> (episode_index,
    frame_index)`` mapping and injected as an extra sample key (a 0-d numpy bool
    that the GR00T collate stacks into a ``(B,)`` tensor).
    """

    def __init__(
        self,
        base_dataset: Any,
        advantages_lookup: dict[tuple[int, int], bool],
        default_advantage: bool = False,
    ):
        self.base_dataset = base_dataset
        self.advantages_lookup = advantages_lookup
        self.default_advantage = default_advantage

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> dict:
        sample = self.base_dataset[idx]
        episode_index, frame_index = self.base_dataset.all_steps[idx]
        advantage = self.advantages_lookup.get(
            (int(episode_index), int(frame_index)), self.default_advantage
        )
        sample["advantage"] = np.asarray(bool(advantage))
        return sample


class Gr00tCfgCollator:
    """Collate GR00T CFG samples into ``(observation, actions, advantage)``.

    Pops the ``advantage`` field (stacked separately into a ``(B,)`` bool
    tensor), then defers to GR00T's ``collate`` for everything else (eagle
    content -> ``eagle_*``; state/action/masks stacked). Returns the full batch
    dict as ``observation`` (it already carries ``action``/``action_mask``),
    plus ``actions`` and ``advantage`` for the worker contract.
    """

    def __init__(self, eagle_path: str | None = None):
        from gr00t.model.transforms import DefaultDataCollator

        self._collator = (
            DefaultDataCollator() if eagle_path is None else DefaultDataCollator(eagle_path)
        )

    def __call__(self, features: list[dict]):
        advantage = torch.from_numpy(
            np.stack([np.asarray(bool(f.pop("advantage"))) for f in features])
        )
        batch = self._collator(features)
        return batch, batch["action"], advantage


class Gr00tCfgDataLoaderImpl(BaseDataLoaderImpl):
    """DataLoaderImpl for the GR00T CFG path (collator already yields 3-tuples)."""

    def __iter__(self):
        yield from self._data_loader


def build_gr00t_cfg_dataloader(cfg, world_size: int, rank: int):
    """Build the GR00T CFG (observation, actions, advantage) dataloader.

    Used by FSDPCfgWorker when ``actor.model.model_type == 'gr00t_cfg'``.
    """
    from rlinf.utils.patcher import Patcher

    Patcher.clear()
    Patcher.add_patch(
        "gr00t.data.embodiment_tags.EmbodimentTag",
        "rlinf.models.embodiment.gr00t.embodiment_tags.EmbodimentTag",
    )
    Patcher.add_patch(
        "gr00t.data.embodiment_tags.EMBODIMENT_TAG_MAPPING",
        "rlinf.models.embodiment.gr00t.embodiment_tags.EMBODIMENT_TAG_MAPPING",
    )
    Patcher.apply()

    from gr00t.data.dataset import LeRobotSingleDataset
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.experiment.data_config import load_data_config

    data_cfg = cfg.get("data", {})
    model_cfg = cfg.actor.model
    advantage_tag = data_cfg.get("advantage_tag", None)
    datasets_config = data_cfg.get("train_data_paths", [])
    if not datasets_config:
        raise ValueError(
            "data.train_data_paths must contain at least one entry with 'dataset_path'."
        )

    data_config_class = model_cfg.get(
        "data_config_class",
        "rlinf.models.embodiment.gr00t.gr00t_n1d5.modality_config:LiberoFrankaDataConfig",
    )
    embodiment_tag = model_cfg.get("embodiment_tag", "libero_franka")
    video_backend = data_cfg.get("video_backend", "torchcodec")
    dc = load_data_config(data_config_class)

    datasets = []
    for ds_entry in datasets_config:
        ds_path = ds_entry["dataset_path"]
        base = LeRobotSingleDataset(
            dataset_path=ds_path,
            modality_configs=dc.modality_config(),
            transforms=dc.transform(),
            embodiment_tag=EmbodimentTag(embodiment_tag),
            video_backend=video_backend,
        )
        advantages_lookup = load_advantages_lookup(ds_path, advantage_tag)
        datasets.append(AdvantagePreservingGr00tDataset(base, advantages_lookup))

    dataset = (
        datasets[0]
        if len(datasets) == 1
        else torch.utils.data.ConcatDataset(datasets)
    )

    micro_batch_size = cfg.actor.micro_batch_size
    num_workers = int(data_cfg.get("num_workers", 8))
    sampler = None
    if torch.distributed.is_initialized():
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
        )
    collator = Gr00tCfgCollator(eagle_path=model_cfg.get("eagle_path", None))
    torch_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=micro_batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collator,
        prefetch_factor=4 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )
    data_config = {"model_type": "gr00t_cfg"}
    return Gr00tCfgDataLoaderImpl(data_config, torch_loader), data_config
