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

"""Advantage-conditioned (CFG) GR00T N1.7 variant for RECAP stage 4.

Adds classifier-free-guidance (CFGRL) flow-matching training on top of the
official GR00T N1.7 ``Gr00tN1d7ActionHead`` (Cosmos-Reason2-2B backbone). The
N1.7 counterpart of ``gr00t_n1d5_cfg``; trained via ``FSDPCfgWorker`` with
per-step advantage labels produced by RECAP stages 1-3.

All ``gr00t.*`` imports stay lazy inside ``get_model`` so the EmbodimentTag
patch is applied before the pydantic schema validator freezes the enum (mirrors
gr00t_n1d7/__init__.py).
"""

import torch
from omegaconf import DictConfig, OmegaConf

from rlinf.utils.logging import get_logger

__all__ = ["get_model"]


def get_model(cfg: DictConfig, torch_dtype=torch.bfloat16):
    """Build a GR00T N1.7 CFG model from a config (mirrors gr00t_n1d7.get_model)."""
    from pathlib import Path

    from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7
    from transformers import AutoConfig, AutoModel

    AutoConfig.register("Gr00tN1d7", Gr00tN1d7Config)
    AutoModel.register(Gr00tN1d7Config, Gr00tN1d7)

    logger = get_logger()
    logger.info(
        "Successfully registered custom architecture Gr00tN1d7, authentication passed!"
    )

    from rlinf.utils.patcher import Patcher

    Patcher.clear()
    Patcher.add_patch(
        "gr00t.data.embodiment_tags.EmbodimentTag",
        "rlinf.models.embodiment.gr00t.embodiment_tags.EmbodimentTag",
    )
    Patcher.apply()

    from gr00t.data.embodiment_tags import EmbodimentTag

    from rlinf.models.embodiment.gr00t.gr00t_n1d7_cfg.gr00t_cfg_model import (
        GR00T_N1_7_ForCFG,
    )
    from rlinf.models.embodiment.gr00t.utils import replace_dropout_with_identity

    embodiment_tag_by_cfg = {
        "libero_sim": EmbodimentTag.LIBERO_SIM,
        "libero_panda": EmbodimentTag.LIBERO_PANDA,
        "libero_franka": EmbodimentTag.LIBERO_FRANKA,
        "isaaclab_franka": EmbodimentTag.ISAACLAB_FRANKA,
        "maniskill_widowx": EmbodimentTag.MANISKILL_WIDOWX,
        "robocasa_panda_omron": EmbodimentTag.ROBOCASA_PANDA_OMRON,
        "gr1": EmbodimentTag.GR1,
        "behavior_r1_pro": EmbodimentTag.BEHAVIOR_R1_PRO,
        "new_embodiment": EmbodimentTag.NEW_EMBODIMENT,
    }
    emb_tag = embodiment_tag_by_cfg.get(cfg.embodiment_tag)
    if emb_tag is None:
        raise ValueError(
            f"Invalid or unsupported embodiment tag: {cfg.embodiment_tag}. "
            f"Supported tags are: {list(embodiment_tag_by_cfg.keys())}."
        )

    model_path = Path(cfg.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    config = Gr00tN1d7Config.from_pretrained(str(model_path))
    _action_dim = cfg.get("action_dim")
    if _action_dim is not None:
        config.action_dim = _action_dim

    backbone_model_path = OmegaConf.select(cfg, "backbone_model_path", default=None)

    model = GR00T_N1_7_ForCFG.from_pretrained(
        config=config,
        local_model_path=str(model_path),
        pretrained_model_name_or_path=str(model_path),
        backbone_model_path=backbone_model_path,
        torch_dtype=torch_dtype,
        embodiment_tag=emb_tag,
        denoising_steps=cfg.denoising_steps,
        output_action_chunks=cfg.num_action_chunks,
        obs_converter_type=cfg.obs_converter_type,
        rl_head_config=cfg.rl_head_config,
        advantage_cfg_dropout_prob=cfg.get("advantage_cfg_dropout_prob", 0.3),
        cfg_guidance_weight=cfg.get("cfg_guidance_weight", 1.0),
        positive_only_conditional=cfg.get("positive_only_conditional", False),
        conditioning=cfg.get("conditioning", "film"),
    )

    model.to(torch_dtype)

    # The advantage_embedding is a CFG-only parameter absent from the GR00T SFT
    # checkpoint. HuggingFace ``from_pretrained`` materializes such missing keys
    # from uninitialized memory (the module's ``nn.init`` in ``__init__`` is
    # bypassed under the low_cpu_mem_usage / meta-device load path), so the weight
    # arrives as garbage — sometimes finite, sometimes NaN / ~3.4e38. That NaN
    # then poisons pred -> loss -> grad on the very first step. Re-init it
    # deterministically after load so CFG training starts from a clean token.
    # (Same root cause / fix as the N1.5 CFG model.)
    # Zero-init (adaLN-zero): conditioning starts as an exact no-op so the
    # unconditional (NULL) pass equals the loaded SFT policy at step 0 and the
    # POS-NULL guidance direction is learned rather than random.
    import torch.nn as nn

    adv_emb = model.action_head.advantage_embedding.embedding
    if adv_emb.weight.is_meta:
        adv_emb.to_empty(device="cpu")
    with torch.no_grad():
        nn.init.zeros_(adv_emb.weight)
    if not torch.isfinite(adv_emb.weight).all():
        raise RuntimeError("advantage_embedding re-init produced non-finite values")

    # Prompt-tuning-style CFG: train ONLY the advantage embedding, freezing the
    # pretrained action head (backbone is always frozen). The base flow field
    # is untouched by construction — the unconditional (NULL) policy stays
    # exactly the loaded SFT policy — and the conditioning is learned as a pure
    # temb-space bias consumed by the existing adaLN layers. Needs a much
    # larger lr than a head fine-tune (from-scratch param); pair with
    # optim.param_group_overrides or a high optim.lr.
    if cfg.get("train_advantage_embedding_only", False):
        frozen = 0
        for name, param in model.named_parameters():
            if param.requires_grad and "advantage_embedding" not in name:
                param.requires_grad = False
                frozen += 1
        logger.info(
            f"train_advantage_embedding_only: froze {frozen} params; only the "
            "advantage embedding trains"
        )

    if cfg.rl_head_config.get("disable_dropout", False):
        replace_dropout_with_identity(model)

    return model
