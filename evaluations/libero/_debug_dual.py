"""Compare base get_action between the SFT model (gr00t_n1d7) and the CFG model
(gr00t_n1d7_cfg) on IDENTICAL backbone features + same seed. If they differ, the
CFG model's action-head weights are wrong despite the key-match (the real bug)."""

import numpy as np
import torch
from omegaconf import OmegaConf
from transformers.feature_extraction_utils import BatchFeature

from rlinf.models.embodiment.gr00t.gr00t_n1d7.gr00t_action_model import (
    _canonicalize_gr00t_text_forward_inputs,
)

BASE = {
    "model_path": "/data/checkpoints/GR00T-N1.7-LIBERO/libero_10",
    "backbone_model_path": "/data/checkpoints/Cosmos-Reason2-2B",
    "embodiment_tag": "libero_sim", "num_action_chunks": 16, "denoising_steps": 4,
    "obs_converter_type": "libero", "add_value_head": False,
    "rl_head_config": {"add_value_head": False, "disable_dropout": True, "padding_value": 570,
                       "joint_logprob": False, "noise_method": "flow_sde", "ignore_last": False,
                       "safe_get_logprob": False, "noise_anneal": False, "noise_params": [0.7,0.3,400],
                       "noise_level": 0.5, "action_noise_scale": 0.1, "chunk_critic_input": False,
                       "detach_critic_input": True, "use_vlm_value": False, "value_vlm_mode": "mean_token"},
}

B = 2
env_obs = {
    "main_images": torch.randint(0, 255, (B, 256, 256, 3), dtype=torch.uint8),
    "wrist_images": torch.randint(0, 255, (B, 256, 256, 3), dtype=torch.uint8),
    "states": torch.randn(B, 8), "task_descriptions": ["pick up the object"] * B,
}


def build_bo_and_action(model_type):
    from rlinf.models import get_model as registry_get_model
    cfg = OmegaConf.create({**BASE, "model_type": model_type, "precision": "bf16",
                            "is_lora": False, "load_to_device": False,
                            "cfg_guidance_weight": 1.0, "advantage_cfg_dropout_prob": 0.3,
                            "positive_only_conditional": False})
    model = registry_get_model(cfg).to("cuda")
    model.eval()
    obs, obs_copy, is_batch = model._prepare_rollout_observation(env_obs)
    ni = model.apply_transforms(obs_copy)
    ni = model._cast_float_tensors_to_compute_dtype(ni, model.compute_dtype)
    ni = _canonicalize_gr00t_text_forward_inputs(ni, getattr(model, "padding_value", 0))
    with torch.inference_mode(), torch.autocast("cuda", dtype=model.compute_dtype):
        bi, ai = model.prepare_input(ni)
        bo = model.backbone(bi)
        torch.manual_seed(7)
        act = model.action_head.get_action(BatchFeature(dict(bo)), BatchFeature(dict(ai)))["action_pred"].clone().float().cpu()
    del model
    torch.cuda.empty_cache()
    return act


print(">>> loading SFT model (gr00t_n1d7)")
sft_act = build_bo_and_action("gr00t_n1d7")
print(">>> loading CFG model (gr00t_n1d7_cfg)")
cfg_act = build_bo_and_action("gr00t_n1d7_cfg")

print(f"SFT action: mean={sft_act.mean():.4f} std={sft_act.std():.4f}")
print(f"CFG action: mean={cfg_act.mean():.4f} std={cfg_act.std():.4f}")
print(f"base get_action |SFT - CFG|: mean={(sft_act-cfg_act).abs().mean():.5f} max={(sft_act-cfg_act).abs().max():.5f}")
