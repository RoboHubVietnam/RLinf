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

"""Advantage-conditioned (CFG) GR00T N1.5 variant for RECAP stage 4.

This package adds classifier-free-guidance (CFG / CFGRL) flow-matching training
on top of the official GR00T N1.5 ``FlowmatchingActionHead``. It is the GR00T
counterpart of ``rlinf/models/embodiment/openpi_cfg`` and is trained via
``FSDPCfgWorker`` with per-step advantage labels produced by RECAP stages 1-3.
"""

import torch
from omegaconf import DictConfig

# NOTE: do NOT import cfg_action_head / gr00t_cfg_model (or any gr00t.* module) at
# module top level. Importing this package happens before get_model() runs the
# EmbodimentTag Patcher; a premature `gr00t.data.schema` import would freeze the
# pydantic DatasetMetadata validator with the un-patched enum and reject custom
# embodiment tags (e.g. libero_franka). All gr00t imports stay lazy inside
# get_model(), mirroring gr00t_n1d5/__init__.py.

__all__ = ["get_model"]


def get_model(cfg: DictConfig, torch_dtype=torch.bfloat16):
    """Build a GR00T N1.5 CFG model from a config (mirrors gr00t_n1d5.get_model)."""
    from pathlib import Path

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

    from gr00t.experiment.data_config import load_data_config

    from rlinf.models.embodiment.gr00t.gr00t_n1d5_cfg.gr00t_cfg_model import (
        GR00T_N1_5_ForCFG,
    )
    from rlinf.models.embodiment.gr00t.utils import replace_dropout_with_identity

    if cfg.embodiment_tag in ("libero_franka", "isaaclab_franka"):
        data_config = load_data_config(
            "rlinf.models.embodiment.gr00t.gr00t_n1d5.modality_config:LiberoFrankaDataConfig"
        )
    elif cfg.embodiment_tag == "maniskill_widowx":
        data_config = load_data_config(
            "rlinf.models.embodiment.gr00t.gr00t_n1d5.modality_config:ManiskillWidowXDataConfig"
        )
    else:
        raise ValueError(f"Invalid embodiment tag: {cfg.embodiment_tag}")
    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()

    model_path = Path(cfg.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    model = GR00T_N1_5_ForCFG.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        embodiment_tag=cfg.embodiment_tag,
        modality_config=modality_config,
        modality_transform=modality_transform,
        denoising_steps=cfg.get("denoising_steps", None),
        output_action_chunks=cfg.get("num_action_chunks", 1),
        obs_converter_type=cfg.get("obs_converter_type", "libero"),
        tune_visual=cfg.get("tune_visual", False),
        tune_llm=cfg.get("tune_llm", False),
        advantage_cfg_dropout_prob=cfg.get("advantage_cfg_dropout_prob", 0.3),
        cfg_guidance_weight=cfg.get("cfg_guidance_weight", 1.0),
        positive_only_conditional=cfg.get("positive_only_conditional", False),
    )
    model.to(torch_dtype)

    # The advantage_embedding is a CFG-only parameter absent from the GR00T SFT
    # checkpoint. HuggingFace ``from_pretrained`` materializes such missing keys
    # from uninitialized memory (the module's ``nn.init`` in ``__init__`` is
    # bypassed under the low_cpu_mem_usage / meta-device load path), so the weight
    # arrives as garbage — sometimes finite, sometimes NaN / ~3.4e38. That NaN
    # then poisons ``pred`` -> loss -> grad on the very first step (lr=0). Re-init
    # it deterministically after load so CFG training starts from a clean token.
    import torch.nn as nn

    adv_emb = model.action_head.advantage_embedding.embedding
    if adv_emb.weight.is_meta:
        adv_emb.to_empty(device="cpu")
    with torch.no_grad():
        nn.init.normal_(adv_emb.weight, mean=0.0, std=0.02)
    if not torch.isfinite(adv_emb.weight).all():
        raise RuntimeError("advantage_embedding re-init produced non-finite values")

    if cfg.get("disable_dropout", False):
        replace_dropout_with_identity(model)

    return model
