"""Diagnose the FiLM CFG head: on REAL backbone features, compare base get_action
vs get_action_cfg (same seed), and measure temb vs advantage-embedding magnitude
+ confirm the timestep hook fires."""

import numpy as np
import torch
from omegaconf import OmegaConf
from transformers.feature_extraction_utils import BatchFeature

from rlinf.models.embodiment.gr00t.gr00t_n1d7_cfg import get_model
from rlinf.models.embodiment.gr00t.gr00t_n1d7.gr00t_action_model import (
    _canonicalize_gr00t_text_forward_inputs,
)

cfg = OmegaConf.create({
    "model_type": "gr00t_n1d7_cfg", "model_path": "/data/checkpoints/GR00T-N1.7-LIBERO/libero_10",
    "backbone_model_path": "/data/checkpoints/Cosmos-Reason2-2B", "embodiment_tag": "libero_sim",
    "num_action_chunks": 16, "denoising_steps": 4, "obs_converter_type": "libero",
    "cfg_guidance_weight": 1.0, "advantage_cfg_dropout_prob": 0.3, "positive_only_conditional": False,
    "rl_head_config": {"add_value_head": False, "disable_dropout": True, "padding_value": 570},
})
model = get_model(cfg, torch.bfloat16).to("cuda")
model.eval()
head = model.action_head
print("inner_dim:", head.model.inner_dim, "adv_emb shape:", tuple(head.advantage_embedding.embedding.weight.shape))
print("adv_emb weight norm per-row:", head.advantage_embedding.embedding.weight.norm(dim=1).tolist())

# Hook fire check
fired = {"n": 0, "temb_norm": None, "adv_norm": None}
orig = head._timestep_adv_hook
def wrapped(module, inputs, temb):
    out = orig(module, inputs, temb)
    fired["n"] += 1
    fired["temb_norm"] = float(temb.float().norm(dim=-1).mean())
    fired["adv_norm"] = float((out - temb).float().norm(dim=-1).mean()) if head._adv_labels is not None else 0.0
    return out
head._timestep_adv_hook = wrapped
head.model.timestep_encoder._forward_hooks.clear()
head.model.timestep_encoder.register_forward_hook(wrapped)

B = 2
env_obs = {
    "main_images": torch.randint(0, 255, (B, 256, 256, 3), dtype=torch.uint8),
    "wrist_images": torch.randint(0, 255, (B, 256, 256, 3), dtype=torch.uint8),
    "states": torch.randn(B, 8), "task_descriptions": ["pick up the object"] * B,
}
obs, obs_copy, is_batch = model._prepare_rollout_observation(env_obs)
ni = model.apply_transforms(obs_copy)
ni = model._cast_float_tensors_to_compute_dtype(ni, model.compute_dtype)
ni = _canonicalize_gr00t_text_forward_inputs(ni, getattr(model, "padding_value", 0))
with torch.inference_mode(), torch.autocast("cuda", dtype=model.compute_dtype):
    bi, ai = model.prepare_input(ni)
    bo = model.backbone(bi)

    def stats(name, t):
        t = t.float()
        print(f"{name}: finite={torch.isfinite(t).all().item()} mean={t.mean():.4f} std={t.std():.4f} min={t.min():.3f} max={t.max():.3f}")

    torch.manual_seed(123)
    out_base = head.get_action(BatchFeature(dict(bo)), BatchFeature(dict(ai)))
    stats("base get_action  ", out_base["action_pred"])
    print("  hook fired during base get_action:", fired["n"], "(should be >0; adv_norm should be ~0 since _adv_labels None)")

    fired["n"] = 0
    torch.manual_seed(123)
    out_cfg = head.get_action_cfg(BatchFeature(dict(bo)), BatchFeature(dict(ai)))
    stats("get_action_cfg POS", out_cfg["action_pred"])
    print(f"  hook fired: {fired['n']}, temb_norm~{fired['temb_norm']:.3f}, adv_add_norm~{fired['adv_norm']:.3f}")
    d = (out_base["action_pred"].float() - out_cfg["action_pred"].float()).abs()
    print(f"|base - cfg(POS)|: mean={d.mean():.4f} max={d.max():.4f}")
