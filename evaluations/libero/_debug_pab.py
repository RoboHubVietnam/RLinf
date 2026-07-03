"""Compare the FULL predict_action_batch raw_action between the SFT model
(gr00t_n1d7) and the CFG model (gr00t_n1d7_cfg) on identical env_obs + seed.
Tests the entire eval path (obs -> head -> unnormalize -> action_convert)."""

import numpy as np
import torch

from rlinf.models import get_model as registry_get_model
from omegaconf import OmegaConf

BASE = {
    "model_path": "/data/checkpoints/GR00T-N1.7-LIBERO/libero_10",
    "backbone_model_path": "/data/checkpoints/Cosmos-Reason2-2B",
    "embodiment_tag": "libero_sim", "num_action_chunks": 16, "denoising_steps": 4,
    "obs_converter_type": "libero", "add_value_head": False, "precision": "bf16",
    "is_lora": False, "load_to_device": False,
    "rl_head_config": {"add_value_head": False, "disable_dropout": True, "padding_value": 570,
                       "joint_logprob": False, "noise_method": "flow_sde", "ignore_last": False,
                       "safe_get_logprob": False, "noise_anneal": False, "noise_params": [0.7,0.3,400],
                       "noise_level": 0.5, "action_noise_scale": 0.1, "chunk_critic_input": False,
                       "detach_critic_input": True, "use_vlm_value": False, "value_vlm_mode": "mean_token"},
    "cfg_guidance_weight": 1.0, "advantage_cfg_dropout_prob": 0.3, "positive_only_conditional": False,
}
B = 2
def make_obs():
    g = torch.Generator().manual_seed(0)
    return {
        "main_images": torch.randint(0, 255, (B, 256, 256, 3), generator=g, dtype=torch.uint8),
        "wrist_images": torch.randint(0, 255, (B, 256, 256, 3), generator=g, dtype=torch.uint8),
        "states": torch.randn(B, 8, generator=g), "task_descriptions": ["pick up the object"] * B,
    }

def run(model_type):
    model = registry_get_model(OmegaConf.create({**BASE, "model_type": model_type})).to("cuda")
    model.eval()
    torch.manual_seed(7)
    raw, result = model.predict_action_batch(make_obs(), mode="eval")
    raw = torch.as_tensor(raw).clone().float().cpu()
    del model; torch.cuda.empty_cache()
    return raw

print(">>> SFT predict_action_batch"); sft = run("gr00t_n1d7")
print(">>> CFG predict_action_batch"); cfg = run("gr00t_n1d7_cfg")
print(f"SFT raw_action: shape={tuple(sft.shape)} mean={sft.mean():.4f} std={sft.std():.4f} min={sft.min():.3f} max={sft.max():.3f}")
print(f"CFG raw_action: shape={tuple(cfg.shape)} mean={cfg.mean():.4f} std={cfg.std():.4f} min={cfg.min():.3f} max={cfg.max():.3f}")
print(f"|SFT - CFG| raw_action: mean={(sft-cfg).abs().mean():.5f} max={(sft-cfg).abs().max():.5f}")
