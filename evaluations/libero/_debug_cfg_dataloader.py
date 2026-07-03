"""Validate the stage-4 N1.7 CFG dataloader against the real LeRobot rollout
dataset (gr00t LeRobotEpisodeLoader + extract_step_data + Gr00tN1d7Processor +
Gr00tN1d7DataCollator). Uses a dummy advantage lookup. Exercises the riskiest
path (incl. whether stats.json from compute_returns is gr00t-compatible)."""

import numpy as np
import torch

from rlinf.utils.patcher import Patcher

Patcher.clear()
Patcher.add_patch(
    "gr00t.data.embodiment_tags.EmbodimentTag",
    "rlinf.models.embodiment.gr00t.embodiment_tags.EmbodimentTag",
)
Patcher.apply()

from gr00t.data.embodiment_tags import EmbodimentTag  # noqa: E402

from rlinf.data.datasets.recap.gr00t_n1d7_cfg import (  # noqa: E402
    N1d7AdvantageDataset,
    N1d7CfgCollator,
    _load_n1d7_processor,
)

DS = "/data/datasets/n1d7_libero_rollouts"
proc = _load_n1d7_processor(
    "/data/checkpoints/GR00T-N1.7-LIBERO/libero_10",
    "/data/checkpoints/Cosmos-Reason2-2B",
)
print("processor loaded; embodiments:", list(proc.modality_configs.keys())[-3:])

ds = N1d7AdvantageDataset(
    dataset_path=DS,
    processor=proc,
    embodiment_tag=EmbodimentTag("libero_sim"),
    advantages_lookup={},  # dummy -> all default_advantage False
    video_backend="torchcodec",
)
print("dataset len (frames):", len(ds))

s0 = ds[0]
print("sample keys:", sorted(s0.keys()))
for k, v in s0.items():
    if isinstance(v, np.ndarray):
        print(f"  {k}: np{v.shape} {v.dtype}")
    elif torch.is_tensor(v):
        print(f"  {k}: torch{tuple(v.shape)} {v.dtype}")
    elif isinstance(v, dict):
        print(f"  {k}: dict keys={list(v.keys())}")
    else:
        print(f"  {k}: {type(v).__name__}")

collator = N1d7CfgCollator(proc)
batch = collator([ds[0], ds[1], ds[2], ds[3]])
observation, actions, advantage = batch
print("=== collated ===")
print("observation['inputs'] keys:", sorted(observation.keys()))
for k, v in observation.items():
    if torch.is_tensor(v):
        print(f"  obs.{k}: {tuple(v.shape)} {v.dtype}")
print("actions:", tuple(actions.shape), actions.dtype)
print("advantage:", tuple(advantage.shape), advantage.dtype, advantage.tolist())
print("OK: stage-4 dataloader produces a valid (observation, actions, advantage) batch")
