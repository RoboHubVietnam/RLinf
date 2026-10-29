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

"""Export an RLinf FSDP openpi checkpoint as a plain openpi PyTorch model dir.

Writes ``<output>/model.safetensors`` (LoRA merged) and copies the norm stats
(``<model_path>/<asset_id>/norm_stats.json``), so the result can be used as
``actor.model.model_path`` for a new run (e.g. RECAP CFG training from an SFT
checkpoint). Loading ``full_weights.pt`` of a LoRA run directly would silently
drop every LoRA'd layer.

Usage:
    python examples/recap/process/export_openpi_checkpoint.py \
        --checkpoint <run>/checkpoints/global_step_<N> \
        --model-path <base model dir used for training> \
        --asset-id bds/vfe_sim_pick_success_0710 \
        --output <output dir>
"""

import argparse
import os
import shutil

import safetensors.torch
import torch

from rlinf.models.embodiment.openpi.lora_utils import (
    merge_lora_state_dict,
    resolve_weights_path,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--asset-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lora-scaling", type=float, default=1.0)
    args = parser.parse_args()

    weights_path = resolve_weights_path(args.checkpoint)
    print(f"Loading {weights_path}")
    state_dict = torch.load(weights_path, map_location="cpu", mmap=True)
    num_lora = sum(".lora_A." in k for k in state_dict)
    state_dict = merge_lora_state_dict(state_dict, args.lora_scaling)
    print(f"Merged {num_lora} LoRA adapters -> {len(state_dict)} tensors")

    os.makedirs(args.output, exist_ok=True)
    safetensors.torch.save_file(
        {k: v.contiguous() for k, v in state_dict.items()},
        os.path.join(args.output, "model.safetensors"),
    )
    src = os.path.join(args.model_path, args.asset_id, "norm_stats.json")
    dst = os.path.join(args.output, args.asset_id, "norm_stats.json")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
