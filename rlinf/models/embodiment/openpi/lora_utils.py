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

"""Helpers to turn RLinf FSDP openpi checkpoints into plain PI0Pytorch weights.

With ``actor.model.is_lora: True`` the saved ``full_weights.pt`` holds peft-wrapped
keys (``base_model.model.``, ``.base_layer.``, ``.lora_A/B.``). Loading it with
``strict=False`` into an unwrapped model silently drops every LoRA'd layer, so
merge it first.
"""

import os

import torch

_PEFT_PREFIX = "base_model.model."


def resolve_weights_path(checkpoint: str) -> str:
    """Accept a global_step_<N> dir, its actor/ dir, an exported dir holding
    ``model.safetensors``, or the weights file itself."""
    candidates = [
        checkpoint,
        os.path.join(checkpoint, "model_state_dict", "full_weights.pt"),
        os.path.join(checkpoint, "actor", "model_state_dict", "full_weights.pt"),
        os.path.join(checkpoint, "model.safetensors"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f"No full_weights.pt or model.safetensors found under {checkpoint}"
    )


def merge_lora_state_dict(
    state_dict: dict[str, torch.Tensor], lora_scaling: float = 1.0
) -> dict[str, torch.Tensor]:
    """Fold peft LoRA weights into the base Linear weights and strip peft prefixes.

    Args:
        state_dict: State dict saved by the FSDP actor.
        lora_scaling: ``lora_alpha / lora_rank``; RLinf sets ``lora_alpha = lora_rank``.

    Returns:
        State dict with plain ``PI0Pytorch`` key names. Returned unchanged (apart
        from prefixes) if it holds no LoRA weights.
    """
    merged = {}
    for key, value in state_dict.items():
        if ".lora_A." in key or ".lora_B." in key:
            continue
        if ".base_layer." in key:
            module, param = key.rsplit(".base_layer.", 1)
            if param == "weight":
                lora_a = state_dict[f"{module}.lora_A.default.weight"].float()
                lora_b = state_dict[f"{module}.lora_B.default.weight"].float()
                value = (value.float() + lora_scaling * lora_b @ lora_a).to(value.dtype)
            key = f"{module}.{param}"
        merged[key.replace(_PEFT_PREFIX, "")] = value
    return merged
