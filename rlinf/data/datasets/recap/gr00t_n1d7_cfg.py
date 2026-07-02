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

"""GR00T N1.7 data pipeline for RECAP stage-4 CFG training (FSDPCfgWorker).

GR00T N1.7 counterpart of ``gr00t_cfg.py`` (N1.5). N1.7 uses a different data
path: per-frame samples carry ``vlm_content`` (text+images) + normalized state +
normalized action, and ``Gr00tN1d7DataCollator`` (the processor's own collator)
batches them into ``{"inputs": {input_ids, pixel_values, state, action, ...}}``.

This module builds a per-frame ``torch.utils.data.Dataset`` over the LeRobot
rollout dataset using the official ``LeRobotEpisodeLoader`` + ``extract_step_data``
+ ``Gr00tN1d7Processor`` (so state/action normalization matches the checkpoint),
attaches the per-step advantage label from ``meta/advantages_{tag}.parquet``, and
collates into the ``(observation, actions, advantage)`` 3-tuple
``FSDPCfgWorker.run_training`` consumes.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rlinf.data.datasets.recap.common import BaseDataLoaderImpl
from rlinf.data.datasets.recap.gr00t_cfg import load_advantages_lookup


def _load_n1d7_processor(model_path: str, backbone_model_path: str | None):
    """Instantiate a Gr00tN1d7Processor from the checkpoint dir (offline).

    Mirrors ``GR00T_N1_7_ForRLActionPrediction._load_processor_from_dir`` but
    points the underlying Qwen3-VL processor at the local Cosmos backbone so no
    network access is required inside dataloader workers.
    """
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor

    processor_dir = Path(model_path)
    with open(processor_dir / "processor_config.json") as f:
        processor_cfg = json.load(f)["processor_kwargs"]
    with open(processor_dir / "statistics.json") as f:
        processor_cfg["statistics"] = json.load(f)
    with open(processor_dir / "embodiment_id.json") as f:
        processor_cfg["embodiment_id_mapping"] = json.load(f)
    if backbone_model_path is not None:
        # Load the Qwen3-VL tokenizer/processor from the local Cosmos dir.
        processor_cfg["model_name"] = str(backbone_model_path)
        processor_cfg.setdefault("transformers_loading_kwargs", {})
        processor_cfg["transformers_loading_kwargs"]["local_files_only"] = True
    return Gr00tN1d7Processor(**processor_cfg)


class N1d7AdvantageDataset(torch.utils.data.Dataset):
    """Per-frame GR00T N1.7 dataset with a per-sample ``advantage`` bool.

    Each item is the processor output for one ``(episode, step)`` (the same
    per-sample dict the official sharded dataset produces), plus an ``advantage``
    key looked up from the advantages parquet.
    """

    def __init__(
        self,
        dataset_path: str,
        processor: Any,
        embodiment_tag: Any,
        advantages_lookup: dict[tuple[int, int], bool],
        video_backend: str = "torchcodec",
        allow_padding: bool = True,
        default_advantage: bool = False,
        conditioning: str = "film",
        advantage_dropout_prob: float = 0.3,
        positive_only_conditional: bool = False,
    ):
        from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader

        self.processor = processor
        self.embodiment_tag = embodiment_tag
        self.advantages_lookup = advantages_lookup
        self.allow_padding = allow_padding
        self.default_advantage = default_advantage
        # For "text" conditioning the advantage is injected into the task prompt
        # here (the paper's method); routing/dropout happens per-sample below. For
        # "film"/"token" the head consumes the advantage; the text is untouched.
        self.conditioning = conditioning
        self.advantage_dropout_prob = advantage_dropout_prob
        self.positive_only_conditional = positive_only_conditional
        # Per-embodiment modality config {video, state, action, language}.
        self.modality_configs = processor.modality_configs[embodiment_tag.value]

        self.loader = LeRobotEpisodeLoader(
            dataset_path=dataset_path,
            modality_configs=self.modality_configs,
            video_backend=video_backend,
        )

        action_delta = self.modality_configs["action"].delta_indices
        self.action_horizon = max(action_delta) - min(action_delta) + 1

        # Flat (episode_index, step_index) index over effective episode lengths.
        self.index: list[tuple[int, int]] = []
        for ep in range(len(self.loader.episode_lengths)):
            eff_len = max(0, self.loader.get_episode_length(ep) - self.action_horizon + 1)
            for step in range(eff_len):
                self.index.append((ep, step))

        self._cache_ep: int | None = None
        self._cache_data = None

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict:
        import random

        from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
        from gr00t.data.types import MessageType

        ep, step = self.index[i]
        if self._cache_ep != ep:
            self._cache_data = self.loader[ep]
            self._cache_ep = ep
        vla_step_data = extract_step_data(
            self._cache_data,
            step,
            self.modality_configs,
            self.embodiment_tag,
            self.allow_padding,
        )
        advantage = bool(
            self.advantages_lookup.get((int(ep), int(step)), self.default_advantage)
        )

        # Text conditioning (RECAP-faithful): inject the advantage indicator into
        # the task prompt, with per-sample CFG dropout. "film"/"token" leave the
        # text untouched (the head applies the conditioning instead).
        if self.conditioning == "text":
            if random.random() < self.advantage_dropout_prob:
                phrase = ""  # unconditional (CFG dropout)
            elif advantage:
                phrase = " Advantage: positive"
            elif self.positive_only_conditional:
                phrase = ""  # treat negatives as unconditional
            else:
                phrase = " Advantage: negative"
            if phrase and vla_step_data.text:
                vla_step_data.text = f"{vla_step_data.text}{phrase}"

        messages = [{"type": MessageType.EPISODE_STEP.value, "content": vla_step_data}]
        sample = self.processor(messages)
        sample["advantage"] = np.asarray(advantage)
        return sample


class N1d7CfgCollator:
    """Collate N1.7 CFG samples into ``(observation, actions, advantage)``.

    Pops ``advantage`` (stacked into a ``(B,)`` bool tensor), then defers to the
    processor's own ``Gr00tN1d7DataCollator`` for everything else (vlm_content ->
    input_ids/pixel_values; state/action/masks stacked). The collator returns
    ``BatchFeature({"inputs": {...}})``; we hand back the inner ``inputs`` dict as
    ``observation`` (it already carries ``action``/``action_mask``).
    """

    def __init__(self, processor: Any):
        self._collator = processor._collator

    def __call__(self, features: list[dict]):
        advantage = torch.from_numpy(
            np.stack([np.asarray(bool(f.pop("advantage"))) for f in features])
        )
        batch = self._collator(features)
        inputs = batch["inputs"]
        return inputs, inputs["action"], advantage


class Gr00tN1d7CfgDataLoaderImpl(BaseDataLoaderImpl):
    """DataLoaderImpl for the GR00T N1.7 CFG path (3-tuple collator)."""

    def __iter__(self):
        yield from self._data_loader


def build_gr00t_n1d7_cfg_dataloader(cfg, world_size: int, rank: int):
    """Build the GR00T N1.7 CFG (observation, actions, advantage) dataloader.

    Used by FSDPCfgWorker when ``actor.model.model_type == 'gr00t_n1d7_cfg'``.
    """
    from rlinf.utils.patcher import Patcher

    Patcher.clear()
    Patcher.add_patch(
        "gr00t.data.embodiment_tags.EmbodimentTag",
        "rlinf.models.embodiment.gr00t.embodiment_tags.EmbodimentTag",
    )
    Patcher.apply()

    from gr00t.data.embodiment_tags import EmbodimentTag

    data_cfg = cfg.get("data", {})
    model_cfg = cfg.actor.model
    advantage_tag = data_cfg.get("advantage_tag", None)
    datasets_config = data_cfg.get("train_data_paths", [])
    if not datasets_config:
        raise ValueError(
            "data.train_data_paths must contain at least one entry with 'dataset_path'."
        )

    embodiment_tag = EmbodimentTag(model_cfg.get("embodiment_tag", "libero_sim"))
    video_backend = data_cfg.get("video_backend", "torchcodec")
    conditioning = model_cfg.get("conditioning", "film")
    advantage_dropout_prob = model_cfg.get("advantage_cfg_dropout_prob", 0.3)
    positive_only = model_cfg.get("positive_only_conditional", False)
    processor = _load_n1d7_processor(
        model_cfg.model_path, model_cfg.get("backbone_model_path", None)
    )

    datasets = []
    for ds_entry in datasets_config:
        ds_path = ds_entry["dataset_path"]
        advantages_lookup = load_advantages_lookup(ds_path, advantage_tag)
        datasets.append(
            N1d7AdvantageDataset(
                dataset_path=ds_path,
                processor=processor,
                embodiment_tag=embodiment_tag,
                advantages_lookup=advantages_lookup,
                video_backend=video_backend,
                conditioning=conditioning,
                advantage_dropout_prob=advantage_dropout_prob,
                positive_only_conditional=positive_only,
            )
        )

    dataset = (
        datasets[0] if len(datasets) == 1 else torch.utils.data.ConcatDataset(datasets)
    )

    micro_batch_size = cfg.actor.micro_batch_size
    # Heavy per-item video decode + tokenization; keep workers modest (shm) and
    # default to 0 unless explicitly raised.
    num_workers = int(data_cfg.get("num_workers", 0))
    sampler = None
    if torch.distributed.is_initialized():
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
        )
    collator = N1d7CfgCollator(processor)
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
    data_config = {"model_type": "gr00t_n1d7_cfg"}
    return Gr00tN1d7CfgDataLoaderImpl(data_config, torch_loader), data_config
