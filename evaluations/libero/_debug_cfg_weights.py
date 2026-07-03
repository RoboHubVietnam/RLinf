"""Check whether GR00T_N1_7_ForCFG actually loaded the SFT checkpoint weights
into the (swapped) CFG action head + backbone, vs leaving them at random init."""

import glob

import torch
from omegaconf import OmegaConf
from safetensors.torch import load_file

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

model = get_model(cfg, torch.bfloat16)
sd = model.state_dict()

# Load checkpoint shards.
ckpt = {}
for f in sorted(glob.glob("/data/checkpoints/GR00T-N1.7-LIBERO/libero_10/*.safetensors")):
    ckpt.update(load_file(f))

print("model keys:", len(sd), "ckpt keys:", len(ckpt))
# action_head base keys that should have loaded from checkpoint
sample_keys = [k for k in sd if k.startswith("action_head.") and "advantage" not in k]
print("action_head non-adv keys in model:", len(sample_keys))

matched = mism = missing = 0
shown = 0
for k in sample_keys:
    if k not in ckpt:
        missing += 1
        if shown < 8:
            print(f"  MISSING in ckpt: {k}")
            shown += 1
        continue
    a = sd[k].float()
    b = ckpt[k].float()
    if a.shape != b.shape:
        mism += 1
        continue
    if torch.allclose(a, b, atol=1e-3):
        matched += 1
    else:
        mism += 1
        if mism <= 5:
            print(f"  DIFF {k}: model_norm={a.norm():.3f} ckpt_norm={b.norm():.3f}")
print(f"action_head: matched={matched} mismatch={mism} missing_in_ckpt={missing}")

# Backbone spot check
bb_keys = [k for k in sd if k.startswith("backbone.")][:5]
for k in bb_keys:
    inckpt = k in ckpt
    close = (inckpt and sd[k].shape == ckpt[k].shape
             and torch.allclose(sd[k].float(), ckpt[k].float(), atol=1e-3))
    print(f"  backbone {k}: in_ckpt={inckpt} close={close}")

# What ckpt action_head keys are NOT in model (would indicate name mismatch)?
ck_ah = [k for k in ckpt if k.startswith("action_head.")]
not_in_model = [k for k in ck_ah if k not in sd]
print(f"ckpt action_head keys: {len(ck_ah)}, NOT in model: {len(not_in_model)}")
for k in not_in_model[:8]:
    print(f"  ckpt-only: {k}")
