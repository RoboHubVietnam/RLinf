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

"""GR00T N1.5 model wrapper for RECAP advantage-conditioned (CFG) training.

Counterpart of ``GR00T_N1_5_ForRLActionPrediction`` but for the offline RECAP
stage-4 path: it holds a :class:`CFGFlowmatchingActionHead` and exposes the
``forward(data) -> (loss, metrics)`` contract consumed by ``FSDPCfgWorker`` plus
a CFG-guided ``predict_action_batch`` for LIBERO evaluation.
"""

import json
from pathlib import Path
from typing import Any, Literal, Union

import numpy as np
import torch
from gr00t.data.dataset import ModalityConfig
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.schema import DatasetMetadata
from gr00t.data.transform.base import ComposedModalityTransform
from gr00t.model.action_head.flow_matching_action_head import (
    FlowmatchingActionHeadConfig,
)
from gr00t.model.gr00t_n1 import GR00T_N1_5, GR00T_N1_5_Config
from transformers.feature_extraction_utils import BatchFeature

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.gr00t.gr00t_n1d5_cfg.cfg_action_head import (
    CFGFlowmatchingActionHead,
)
from rlinf.models.embodiment.gr00t.simulation_io import (
    ACTION_CONVERSION_N1D5,
    OBS_CONVERSION,
)
from rlinf.models.embodiment.gr00t.utils import (
    squeeze_dict_values,
    unsqueeze_dict_values,
)


class GR00T_N1_5_ForCFG(GR00T_N1_5, BasePolicy):
    """GR00T N1.5 with an advantage-conditioned (CFG) flow-matching head."""

    _no_split_modules = [
        "Eagle2_5_VLForConditionalGeneration",
        "CFGFlowmatchingActionHead",
        "TimestepEncoder",
        "TimestepEmbedding",
        "AdvantageEmbedding",
    ]

    def __init__(
        self,
        config: GR00T_N1_5_Config,
        local_model_path: str,
        embodiment_tag: Union[str, EmbodimentTag],
        modality_config: dict[str, ModalityConfig],
        modality_transform: ComposedModalityTransform,
        compute_dtype: torch.dtype = torch.bfloat16,
        denoising_steps: int | None = None,
        obs_converter_type: str = "libero",
        output_action_chunks: int = 1,
        advantage_cfg_dropout_prob: float = 0.3,
        cfg_guidance_weight: float = 1.0,
        positive_only_conditional: bool = False,
        padding_value: int = 570,
    ):
        super().__init__(config, local_model_path)

        self._modality_config = modality_config
        self._modality_transform = modality_transform
        self.model_path = Path(local_model_path)
        self.compute_dtype = compute_dtype
        self.output_action_chunks = output_action_chunks
        self.cfg_guidance_weight = cfg_guidance_weight
        # Eagle backbone inputs are padded to this length at inference (the base
        # GR00T_N1_5 RL action model does the same with rl_head_config.padding_value
        # = 570). Skipping this feeds the backbone variable-length token ids and
        # produces wrong VL features -> a near-random policy.
        self.padding_value = padding_value

        if isinstance(embodiment_tag, str):
            self.embodiment_tag = EmbodimentTag(embodiment_tag)
        else:
            self.embodiment_tag = embodiment_tag

        self.obs_convert_fn = OBS_CONVERSION[obs_converter_type]
        self.action_convert_fn = ACTION_CONVERSION_N1D5[obs_converter_type]
        self._load_metadata(self.model_path / "experiment_cfg")

        # Replace the base action head with the advantage-conditioned variant.
        action_head_cfg = FlowmatchingActionHeadConfig(**config.action_head_cfg)
        self.action_head = CFGFlowmatchingActionHead(
            action_head_cfg,
            advantage_cfg_dropout_prob=advantage_cfg_dropout_prob,
            cfg_guidance_weight=cfg_guidance_weight,
            positive_only_conditional=positive_only_conditional,
        )
        if denoising_steps is not None:
            self.action_head.num_inference_timesteps = denoising_steps

    def eval(self):
        self._modality_transform.eval()
        super().eval()

    # ------------------------------------------------------------------
    # Training: CFG loss (consumed by FSDPCfgWorker)
    # ------------------------------------------------------------------
    def forward(self, forward_type=ForwardType.DEFAULT, data: dict | None = None, **kwargs):
        if data is None:
            data = kwargs
        return self.default_forward(data)

    def default_forward(self, data: dict) -> tuple[torch.Tensor, dict]:
        """Run one CFG training step.

        Args:
            data: ``{"observation": <dict of GR00T model inputs>,
                     "actions": (B, H, action_dim),
                     "advantage": (B,) bool}``.

        Returns:
            (loss, metrics) — see :meth:`CFGFlowmatchingActionHead.forward_cfg`.
        """
        observation = data["observation"]
        actions = data["actions"]
        advantage = data["advantage"]

        model_inputs = dict(observation)
        model_inputs["action"] = actions
        if "action_mask" not in model_inputs:
            model_inputs["action_mask"] = torch.ones_like(actions)

        backbone_inputs, action_inputs = self.prepare_input(model_inputs)
        # The Eagle backbone is frozen for CFG (tune_visual/tune_llm=False); run it
        # under no_grad so autograd doesn't build (and later trip over in-place
        # views in) the backbone graph. The trainable action head still receives
        # gradients via its own vlln / vl_self_attention / DiT.
        #
        # NaN-robustness: when the model is loaded in fp32 (the action head needs
        # fp32 for a stable flow-matching backward — the DiT timestep_embedder
        # produces NaN grads in bf16), the frozen backbone's flash-attention still
        # requires bf16/fp16 inputs. Run the backbone under a bf16 autocast so
        # flash-attn gets bf16 activations regardless of the master dtype, then
        # cast the features back to the action head's dtype below. When the model
        # is bf16 this autocast is a no-op.
        backbone_trainable = any(p.requires_grad for p in self.backbone.parameters())
        if backbone_trainable:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                backbone_outputs = self.backbone(backbone_inputs)
        else:
            with torch.no_grad(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16
            ):
                backbone_outputs = self.backbone(backbone_inputs)
        # The Eagle backbone returns features as a view; under FSDP a later
        # in-place op on its base trips autograd's view-inplace guard inside the
        # action head's vlln. Clone to a fresh tensor to break the view chain, and
        # cast to the action head's parameter dtype so the trainable head computes
        # in its master precision (fp32) even though the backbone emitted bf16.
        head_dtype = next(self.action_head.parameters()).dtype
        if "backbone_features" in backbone_outputs:
            backbone_outputs["backbone_features"] = (
                backbone_outputs["backbone_features"].clone().to(head_dtype)
            )
        loss, metrics = self.action_head.forward_cfg(
            backbone_outputs, action_inputs, advantage
        )
        return loss, metrics

    # ------------------------------------------------------------------
    # Inference: CFG-guided action prediction for LIBERO eval
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_action_batch(
        self,
        env_obs,
        mode: Literal["train", "eval"] = "eval",
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        # Match the RL wrapper's inference/training dtype round-trip.
        env_obs["states"] = env_obs["states"].to(torch.bfloat16)
        env_obs["states"] = env_obs["states"].cpu().float()

        observations = self.obs_convert_fn(env_obs)
        obs_copy = observations.copy()

        is_batch = self._check_state_is_batched(obs_copy)
        if not is_batch:
            obs_copy = unsqueeze_dict_values(obs_copy)
        for k, v in obs_copy.items():
            if not isinstance(v, np.ndarray):
                obs_copy[k] = np.array(v)

        normalized_input = self.apply_transforms(obs_copy)
        for key in normalized_input:
            if normalized_input[key].dtype == torch.float32:
                normalized_input[key] = normalized_input[key].to(torch.bfloat16)

        # Pad eagle inputs to padding_value so the backbone sees a fixed-length
        # token sequence (mirrors GR00T_N1_5_ForRLActionPrediction.predict_action_batch).
        normalized_input["eagle_input_ids"] = torch.nn.functional.pad(
            normalized_input["eagle_input_ids"],
            pad=(0, self.padding_value - normalized_input["eagle_input_ids"].shape[-1]),
            mode="constant",
            value=0,
        )
        normalized_input["eagle_attention_mask"] = torch.nn.functional.pad(
            normalized_input["eagle_attention_mask"],
            pad=(0, self.padding_value - normalized_input["eagle_attention_mask"].shape[-1]),
            mode="constant",
            value=0,
        )

        backbone_inputs, action_inputs = self.prepare_input(normalized_input)
        backbone_outputs = self.backbone(backbone_inputs)
        action_head_outputs = self.action_head.get_action_cfg(
            backbone_outputs, action_inputs
        )
        normalized_action = action_head_outputs["action_pred"].float()

        unnormalized_action = self.unapply_transforms(
            {"action": normalized_action.cpu()}
        )
        if not is_batch:
            unnormalized_action = squeeze_dict_values(unnormalized_action)
        raw_action = self.action_convert_fn(
            unnormalized_action, chunk_size=self.output_action_chunks
        )
        return torch.from_numpy(raw_action), {"action_pred": normalized_action}

    # ------------------------------------------------------------------
    # Transform / metadata helpers (mirrors GR00T_N1_5_ForRLActionPrediction)
    # ------------------------------------------------------------------
    def _check_state_is_batched(self, obs: dict[str, Any]) -> bool:
        for k, v in obs.items():
            if "state" in k and len(v.shape) < 3:  # (B, Time, Dim)
                return False
        return True

    def apply_transforms(self, obs: dict[str, Any]) -> dict[str, Any]:
        return self._modality_transform(obs)

    def unapply_transforms(self, action: dict[str, Any]) -> dict[str, Any]:
        return self._modality_transform.unapply(action)

    def _load_metadata(self, exp_cfg_dir: Path):
        metadata_path = exp_cfg_dir / "metadata.json"
        with open(metadata_path, "r") as f:
            metadatas = json.load(f)
        metadata_dict = metadatas.get(self.embodiment_tag.value)
        if metadata_dict is None:
            raise ValueError(
                f"No metadata found for embodiment tag: {self.embodiment_tag.value}; "
                f"make sure metadata.json is present at {metadata_path}"
            )
        metadata = DatasetMetadata.model_validate(metadata_dict)
        self._modality_transform.set_metadata(metadata)
        self.metadata = metadata

        valid_action_dim = 0
        for v in metadata.modalities.action.values():
            valid_action_dim += v.shape[0]
        self.valid_action_dim = valid_action_dim
        self.image_nums = len(metadata.modalities.video.keys())
