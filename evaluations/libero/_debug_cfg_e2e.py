"""End-to-end real-feature isolation: build a dummy LIBERO obs, run the CFG
model's real obs pipeline + Cosmos backbone, then compare the inherited base
get_action vs get_action_cfg (with and without the advantage token) on the SAME
real backbone output. Pinpoints whether the 0% is the token or the invocation."""

import numpy as np
import torch
from omegaconf import OmegaConf
from transformers.feature_extraction_utils import BatchFeature

from rlinf.models.embodiment.gr00t.gr00t_n1d7_cfg import get_model

cfg = OmegaConf.create(
    {
        "model_type": "gr00t_n1d7_cfg",
        "model_path": "/data/checkpoints/GR00T-N1.7-LIBERO/libero_10",
        "backbone_model_path": "/data/checkpoints/Cosmos-Reason2-2B",
        "embodiment_tag": "libero_sim",
        "num_action_chunks": 16,
        "denoising_steps": 4,
        "obs_converter_type": "libero",
        "cfg_guidance_weight": 1.0,
        "rl_head_config": {"add_value_head": False, "disable_dropout": True, "padding_value": 570},
    }
)

model = get_model(cfg, torch.bfloat16).to("cuda")
model.eval()
head = model.action_head

B = 2
torch.manual_seed(0)
np.random.seed(0)
env_obs = {
    "main_images": torch.randint(0, 255, (B, 256, 256, 3), dtype=torch.uint8),
    "wrist_images": torch.randint(0, 255, (B, 256, 256, 3), dtype=torch.uint8),
    "states": torch.randn(B, 8),
    "task_descriptions": ["pick up the object"] * B,
}

# Run the model's real obs pipeline to get a real backbone_output + action_input.
observations, obs_copy, is_batch = model._prepare_rollout_observation(env_obs)
normalized_input = model.apply_transforms(obs_copy)
normalized_input = model._cast_float_tensors_to_compute_dtype(normalized_input, model.compute_dtype)
from rlinf.models.embodiment.gr00t.gr00t_n1d7.gr00t_action_model import (
    _canonicalize_gr00t_text_forward_inputs,
)

normalized_input = _canonicalize_gr00t_text_forward_inputs(
    normalized_input, getattr(model, "padding_value", 0)
)

with torch.inference_mode(), torch.autocast("cuda", dtype=model.compute_dtype):
    backbone_inputs, action_inputs = model.prepare_input(normalized_input)
    backbone_outputs = model.backbone(backbone_inputs)
    bo = backbone_outputs
    print("backbone_features:", tuple(bo["backbone_features"].shape),
          "has image_mask:", "image_mask" in bo, "has attn:", "backbone_attention_mask" in bo)

    def stats(name, t):
        t = t.float()
        print(f"{name}: finite={torch.isfinite(t).all().item()} mean={t.mean():.4f} "
              f"std={t.std():.4f} min={t.min():.3f} max={t.max():.3f}")

    import os

    # Same seed before each call so the initial denoising noise is identical and
    # only the code path differs.
    # 1) base get_action (what the SFT model uses -> 41%)
    torch.manual_seed(123)
    out_base = head.get_action(BatchFeature(dict(bo)), BatchFeature(dict(action_inputs)))
    stats("base get_action    ", out_base["action_pred"])
    # 2) get_action_cfg WITH token
    os.environ["CFG_DISABLE_ADV_TOKEN"] = "0"
    torch.manual_seed(123)
    out_cfg = head.get_action_cfg(BatchFeature(dict(bo)), BatchFeature(dict(action_inputs)))
    stats("get_action_cfg+tok ", out_cfg["action_pred"])
    # 3) get_action_cfg WITHOUT token
    os.environ["CFG_DISABLE_ADV_TOKEN"] = "1"
    torch.manual_seed(123)
    out_cfg2 = head.get_action_cfg(BatchFeature(dict(bo)), BatchFeature(dict(action_inputs)))
    stats("get_action_cfg-tok ", out_cfg2["action_pred"])

    d_tok = (out_base["action_pred"].float() - out_cfg["action_pred"].float()).abs()
    d_notok = (out_base["action_pred"].float() - out_cfg2["action_pred"].float()).abs()
    print(f"|base - cfg+tok|: mean={d_tok.mean():.4f} max={d_tok.max():.4f}")
    print(f"|base - cfg-tok|: mean={d_notok.mean():.4f} max={d_notok.max():.4f}")
