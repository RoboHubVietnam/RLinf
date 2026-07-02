# Copyright 2026 The RLinf Authors.
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

"""GR00T N1.7 model wrapper for RECAP advantage-conditioned (CFG) training.

N1.7 counterpart of ``gr00t_n1d5_cfg.gr00t_cfg_model.GR00T_N1_5_ForCFG``. It
subclasses the maintained RL action model
:class:`~rlinf.models.embodiment.gr00t.gr00t_n1d7.gr00t_action_model.GR00T_N1_7_ForRLActionPrediction`
to reuse its Cosmos backbone loading, GR00T N1.7 processor, observation/action
conversion, transforms and metadata handling, and only swaps in the
advantage-conditioned :class:`CFGGr00tN1d7ActionHead`.

* Training (consumed by ``FSDPCfgWorker``): :meth:`default_forward` runs the
  backbone then the CFG flow-matching loss with per-step advantage labels from
  the RECAP advantages parquet, returning ``(loss, metrics)``.
* Inference (LIBERO eval): the inherited ``predict_action_batch`` is reused
  verbatim; only the deterministic eval denoiser
  (:meth:`_get_action_from_normalized_input`) is overridden to run the
  CFG-guided sampler ``CFGGr00tN1d7ActionHead.get_action_cfg``.
"""

from contextlib import nullcontext
from typing import Any, Literal, Optional

import torch

from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.gr00t.gr00t_n1d7.gr00t_action_model import (
    GR00T_N1_7_ForRLActionPrediction,
)
from rlinf.models.embodiment.gr00t.gr00t_n1d7_cfg.cfg_action_head import (
    CFGGr00tN1d7ActionHead,
)


class GR00T_N1_7_ForCFG(GR00T_N1_7_ForRLActionPrediction):
    """GR00T N1.7 with an advantage-conditioned (CFG) flow-matching head."""

    def __init__(
        self,
        config,
        rl_head_config: dict[str, Any],
        embodiment_tag,
        local_model_path: str,
        advantage_cfg_dropout_prob: float = 0.3,
        cfg_guidance_weight: float = 1.0,
        positive_only_conditional: bool = False,
        conditioning: str = "film",
        denoising_steps: Optional[int] = None,
        output_action_chunks: int = 1,
        **kwargs,
    ):
        # Build the full RL action model (Cosmos backbone, processor, transforms,
        # metadata, RL flow-matching head). The RL head is then replaced below; its
        # shared base modules (state/action encoder, action decoder, DiT, vlln,
        # vl_self_attention, position_embedding) carry the same parameter names, so
        # the SFT checkpoint loads into the CFG head unchanged. ``advantage_embedding``
        # is the only extra parameter and is re-initialized in ``get_model``.
        super().__init__(
            config,
            rl_head_config,
            embodiment_tag,
            local_model_path,
            denoising_steps=denoising_steps,
            output_action_chunks=output_action_chunks,
            **kwargs,
        )

        self.cfg_guidance_weight = cfg_guidance_weight
        self.action_head = CFGGr00tN1d7ActionHead(
            config,
            advantage_cfg_dropout_prob=advantage_cfg_dropout_prob,
            cfg_guidance_weight=cfg_guidance_weight,
            positive_only_conditional=positive_only_conditional,
            conditioning=conditioning,
        )
        if denoising_steps is not None and hasattr(
            self.action_head, "num_inference_timesteps"
        ):
            self.action_head.num_inference_timesteps = denoising_steps
        self.action_head.env_action_dim = self.action_dim
        self.action_head.valid_action_dim = self.valid_action_dim

    # ------------------------------------------------------------------
    # Inference obs prep: text conditioning injects the advantage into the prompt
    # ------------------------------------------------------------------
    def _prepare_rollout_observation(self, env_obs):
        """For ``text`` conditioning, append 'Advantage: positive' to the task
        prompt at inference (mirrors the dataloader's training-time injection).
        For film/token the prompt is untouched (the head conditions instead)."""
        if getattr(self.action_head, "conditioning", "film") == "text":
            env_obs = dict(env_obs)
            td = env_obs.get("task_descriptions")
            if isinstance(td, (list, tuple)):
                env_obs["task_descriptions"] = [f"{t} Advantage: positive" for t in td]
            elif td is not None:
                env_obs["task_descriptions"] = f"{td} Advantage: positive"
        return super()._prepare_rollout_observation(env_obs)

    # ------------------------------------------------------------------
    # Training: CFG loss (consumed by FSDPCfgWorker)
    # ------------------------------------------------------------------
    def forward(self, forward_type=ForwardType.DEFAULT, data: dict | None = None, **kwargs):
        if forward_type != ForwardType.DEFAULT:
            raise NotImplementedError(forward_type)
        if data is None:
            data = kwargs
        return self.default_forward(data)

    def default_forward(self, data: dict) -> tuple[torch.Tensor, dict]:
        """Run one CFG training step.

        Args:
            data: ``{"observation": <dict of normalized GR00T N1.7 model inputs>,
                     "actions": (B, H, action_dim),
                     "advantage": (B,) bool}``.

        Returns:
            (loss, metrics) — see :meth:`CFGGr00tN1d7ActionHead.forward_cfg`.
        """
        observation = data["observation"]
        actions = data["actions"]
        advantage = data["advantage"]

        # The CFG collator yields processor-format model inputs (input_ids,
        # pixel_values, state, action, embodiment_id, ...). Mirror the base
        # ``Gr00tN1d7.forward`` exactly: prepare_input -> backbone -> head. Do NOT
        # run ``_normalize_gr00t_forward_inputs`` (the actor recompute path; it
        # filters out the vlm/image fields). The collator output already carries a
        # normalized ``action`` chunk; fall back to the passed ``actions`` only if
        # absent.
        model_inputs = dict(observation)
        if "action" not in model_inputs:
            model_inputs["action"] = actions
        if "action_mask" not in model_inputs:
            model_inputs["action_mask"] = torch.ones_like(model_inputs["action"])

        backbone_inputs, action_inputs = self.prepare_input(model_inputs)

        # The Cosmos backbone is frozen for CFG; run it under no_grad when no
        # backbone parameter requires grad so autograd does not build (and later
        # trip over in-place views in) the backbone graph. The trainable action
        # head still receives gradients via its own vlln / vl_self_attention / DiT.
        backbone_trainable = any(p.requires_grad for p in self.backbone.parameters())
        if backbone_trainable:
            backbone_outputs = self.backbone(backbone_inputs)
        else:
            with torch.no_grad():
                backbone_outputs = self.backbone(backbone_inputs)

        loss, metrics = self.action_head.forward_cfg(
            backbone_outputs, action_inputs, advantage
        )
        return loss, metrics

    # ------------------------------------------------------------------
    # Inference: CFG-guided deterministic eval denoiser
    # ------------------------------------------------------------------
    def _get_action_from_normalized_input(
        self, normalized_input: dict[str, Any]
    ) -> torch.Tensor:
        """Override the SFT eval denoiser with the CFG-guided sampler.

        Reuses the parent's full rollout pipeline (obs convert -> transforms ->
        text canonicalization -> unnormalize); only the action-head call changes
        from ``get_action`` to ``get_action_cfg``. Mirrors the base
        ``Gr00tN1d7.get_action`` exactly (prepare_input -> backbone -> head): it
        must NOT run ``_normalize_gr00t_forward_inputs`` (that is the actor
        recompute path; it drops the eval processor's ``vlm_content``/image
        fields, starving the backbone and collapsing the policy).
        """
        device_type = getattr(self.device, "type", "cpu")
        autocast_context = (
            torch.autocast(device_type=device_type, dtype=self.compute_dtype)
            if device_type == "cuda"
            else nullcontext()
        )
        # Mirror the base ``_get_action_from_normalized_input`` exactly: prepare_input
        # MUST run inside inference_mode+autocast (the base path wraps the whole
        # ``self.get_action`` call). Running it outside autocast leaves image/input
        # tensors in fp32, diverging the backbone features and collapsing eval SR
        # (86% -> 34%) even though the head math is identical.
        with torch.inference_mode(), autocast_context:
            backbone_inputs, action_inputs = self.prepare_input(normalized_input)
            backbone_outputs = self.backbone(backbone_inputs)
            model_pred = self.action_head.get_action_cfg(backbone_outputs, action_inputs)

        return model_pred["action_pred"].float()

    @torch.no_grad()
    def predict_action_batch(
        self,
        env_obs,
        mode: Literal["train", "eval"] = "eval",
        **kwargs,
    ):
        # CFG is an offline (eval / advantage-conditioned) policy: there is no
        # RL exploration/log-prob bookkeeping path. Force the deterministic eval
        # branch regardless of caller mode so rollouts use get_action_cfg.
        return super().predict_action_batch(env_obs, mode="eval", **kwargs)
