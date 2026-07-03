"""Isolate the get_action_cfg collapse: compare base get_action vs get_action_cfg
on identical synthetic inputs through the SAME loaded CFG model/head."""

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
        "advantage_cfg_dropout_prob": 0.3,
        "positive_only_conditional": False,
        "rl_head_config": {"add_value_head": False, "disable_dropout": True, "padding_value": 570},
    }
)

model = get_model(cfg, torch.bfloat16).to("cuda")
model.eval()
head = model.action_head
conf = head.config
print("state_history_length:", conf.state_history_length, "max_state_dim:", conf.max_state_dim)
print("action_horizon:", conf.action_horizon, "action_dim:", head.action_dim)

B, S = 2, 64
dt = torch.bfloat16
dev = "cuda"
vl = torch.randn(B, S, conf.backbone_embedding_dim, device=dev, dtype=dt)
attn = torch.ones(B, S, device=dev, dtype=torch.bool)
img = torch.zeros(B, S, device=dev, dtype=torch.bool)
img[:, : S // 2] = True  # first half are "image" tokens
bo = BatchFeature(
    data={"backbone_features": vl, "backbone_attention_mask": attn, "image_mask": img}
)
state = torch.randn(B, conf.state_history_length, conf.max_state_dim, device=dev, dtype=dt)
emb = torch.full((B,), 2, device=dev, dtype=torch.long)  # libero_sim
ai = BatchFeature(data={"state": state, "embodiment_id": emb})


def stats(name, t):
    t = t.float()
    print(
        f"{name}: shape={tuple(t.shape)} finite={torch.isfinite(t).all().item()} "
        f"mean={t.mean().item():.4f} std={t.std().item():.4f} "
        f"min={t.min().item():.3f} max={t.max().item():.3f}"
    )


with torch.no_grad():
    torch.manual_seed(0)
    out_base = head.get_action(BatchFeature(dict(bo)), BatchFeature(dict(ai)))
    stats("base get_action     ", out_base["action_pred"])
    torch.manual_seed(0)
    out_cfg = head.get_action_cfg(BatchFeature(dict(bo)), BatchFeature(dict(ai)))
    stats("cfg  get_action_cfg ", out_cfg["action_pred"])
    d = (out_base["action_pred"].float() - out_cfg["action_pred"].float()).abs()
    print(f"abs diff: mean={d.mean().item():.4f} max={d.max().item():.4f}")
