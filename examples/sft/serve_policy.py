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

"""Serve an RLinf-trained openpi (pi0 / pi0.5) checkpoint over websocket.

Drop-in replacement for openpi ``scripts/serve_policy.py``: same transforms,
same ``openpi_client`` protocol and metadata, so existing clients work as is.

The RLinf FSDP checkpoint (``actor/model_state_dict/full_weights.pt``) stores
peft-wrapped keys when trained with LoRA; they are merged into plain
``PI0Pytorch`` weights here. Norm stats are read from the base model dir
(``<model_path>/<asset_id>/norm_stats.json``), as during training.
"""

import argparse
import logging
import os
import socket

import openpi.policies.policy as _policy
import openpi.transforms as transforms
import torch
from openpi.models_pytorch import pi0_pytorch
from openpi.serving import websocket_policy_server
from openpi.training import checkpoints as _checkpoints

from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

_PEFT_PREFIX = "base_model.model."


def resolve_weights_path(checkpoint: str) -> str:
    """Accept a global_step_<N> dir, its actor/ dir, or the .pt file itself."""
    candidates = [
        checkpoint,
        os.path.join(checkpoint, "model_state_dict", "full_weights.pt"),
        os.path.join(checkpoint, "actor", "model_state_dict", "full_weights.pt"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"No full_weights.pt found under {checkpoint}")


def merge_lora_state_dict(
    state_dict: dict[str, torch.Tensor], lora_scaling: float
) -> dict[str, torch.Tensor]:
    """Fold peft LoRA weights into the base Linear weights and strip peft prefixes."""
    merged = {}
    for key, value in state_dict.items():
        if ".lora_A." in key or ".lora_B." in key:
            continue
        if ".base_layer." in key:
            module, param = key.rsplit(".base_layer.", 1)
            if param == "weight":
                lora_a = state_dict[f"{module}.lora_A.default.weight"].float()
                lora_b = state_dict[f"{module}.lora_B.default.weight"].float()
                value = (value.float() + lora_scaling * lora_b @ lora_a).to(
                    value.dtype
                )
            key = f"{module}.{param}"
        merged[key.replace(_PEFT_PREFIX, "")] = value
    return merged


def load_model(train_config, weights_path: str, lora_scaling: float, compile: bool):
    model = pi0_pytorch.PI0Pytorch(config=train_config.model)
    if not compile:
        # PI0Pytorch always wraps sample_actions in torch.compile(max-autotune); the
        # first request then blocks for minutes and clients hit keepalive timeouts.
        del model.sample_actions
    state_dict = torch.load(weights_path, map_location="cpu", mmap=True)
    num_lora = sum(".lora_A." in k for k in state_dict)
    if num_lora:
        logging.info("Merging %d LoRA adapters (scaling=%s)", num_lora, lora_scaling)
        state_dict = merge_lora_state_dict(state_dict, lora_scaling)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    # lm_head is tied to embed_tokens, so one of them is always reported missing.
    missing = [k for k in missing if "embed_tokens" not in k and "lm_head" not in k]
    if missing:
        raise RuntimeError(f"Missing keys when loading {weights_path}: {missing}")
    if unexpected:
        logging.warning("Ignoring unexpected keys: %s", unexpected)
    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    return model


def create_policy(args) -> _policy.Policy:
    train_config = get_openpi_config(args.config_name, model_path=args.model_path)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if data_config.asset_id is None:
        raise ValueError("Asset id is required to load norm stats.")
    norm_stats = _checkpoints.load_norm_stats(args.model_path, data_config.asset_id)

    weights_path = resolve_weights_path(args.checkpoint)
    logging.info("Loading weights from %s", weights_path)
    model = load_model(train_config, weights_path, args.lora_scaling, args.compile)

    sample_kwargs = {"num_steps": args.num_steps} if args.num_steps else None
    return _policy.Policy(
        model,
        transforms=[
            transforms.InjectDefaultPrompt(args.default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(
                norm_stats, use_quantiles=data_config.use_quantile_norm
            ),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(
                norm_stats, use_quantiles=data_config.use_quantile_norm
            ),
            *data_config.data_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
        is_pytorch=True,
        pytorch_device="cuda" if torch.cuda.is_available() else "cpu",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="RLinf checkpoint: global_step_<N> dir, its actor/ dir, or full_weights.pt",
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Base model dir used for training (holds <asset_id>/norm_stats.json)",
    )
    parser.add_argument("--config-name", required=True, help="openpi config name")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--default-prompt", default=None)
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Denoise steps (default: PI0Pytorch.sample_actions default, 10)",
    )
    parser.add_argument(
        "--lora-scaling",
        type=float,
        default=1.0,
        help="lora_alpha / lora_rank; RLinf sets lora_alpha = lora_rank",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile sample_actions (faster, but the first request takes minutes)",
    )
    parser.add_argument(
        "--record", action="store_true", help="Record requests to policy_records/"
    )
    args = parser.parse_args()

    policy = create_policy(args)
    policy_metadata = policy.metadata
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    logging.info(
        "Creating server (host: %s, ip: %s, port: %d)",
        hostname,
        socket.gethostbyname(hostname),
        args.port,
    )
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy, host="0.0.0.0", port=args.port, metadata=policy_metadata
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
