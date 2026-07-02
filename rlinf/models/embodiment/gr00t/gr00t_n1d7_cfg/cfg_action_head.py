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

"""Advantage-conditioned flow-matching action head for GR00T N1.7 (RECAP stage 4).

GR00T N1.7 counterpart of ``gr00t_n1d5_cfg.cfg_action_head``. Implements
classifier-free-guidance (CFG / CFGRL) training on top of the official
``Gr00tN1d7ActionHead``. A learned advantage indicator (NULL / NEG / POS)
conditions the flow-matching velocity field on optimality.

Two interchangeable conditioning-injection sites are supported (``conditioning``):

* ``"token"`` — append the advantage embedding as one extra token in the DiT
  cross-attention context (``vl_embs``), extending ``backbone_attention_mask`` (+1)
  and ``image_mask`` (+0). This is the original RECAP/CFGRL port (same as N1.5).
* ``"film"`` — add the advantage embedding to the DiT timestep embedding ``temb``
  (the adaLN modulation signal), via a forward hook on ``timestep_encoder``; the
  cross-attention context is left unchanged. Canonical DiT class-conditioning.

Both are faithful CFG (the method only specifies *what* to condition on + dropout
training + guided inference; the injection site is an architectural choice). They
are run as an ablation. Embedding dim differs by site: ``token`` uses
``backbone_embedding_dim``, ``film`` uses the DiT ``inner_dim``.

N1.7 specifics: state carries a history dim flattened to ``(B, 1, -1)``; no future
tokens (DiT input is ``cat(state_features, action_features)``); the DiT returns a
tuple under ``return_all_hidden_states=True`` (training) and a single tensor at
inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead
from transformers.feature_extraction_utils import BatchFeature

from rlinf.models.embodiment.gr00t.gr00t_n1d7_cfg.cfg_utils import (
    compute_cfg_routing_masks,
)


class AdvantageEmbedding(nn.Module):
    """Advantage indicator embedding (NULL / NEG / POS).

    NULL_IDX (0): unconditional (CFG dropout / unconditional pass).
    NEG_IDX  (1): A(o, a) <= eps_l  -> "advantage negative".
    POS_IDX  (2): A(o, a)  > eps_l  -> "advantage positive".
    """

    NULL_IDX: int = 0
    NEG_IDX: int = 1
    POS_IDX: int = 2

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(3, embedding_dim)
        # adaLN-zero (DiT): zero-init so conditioning starts as an exact no-op —
        # the unconditional (NULL) pass equals the base SFT policy at step 0 and
        # the POS/NEG directions are learned rather than random. A nonzero init
        # perturbs the base flow field immediately (observed as w=1.0 regressing
        # below the SFT baseline) and makes early CFG guidance extrapolate along
        # a random direction.
        nn.init.zeros_(self.embedding.weight)

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        """labels: (B,) long in {NULL,NEG,POS} -> (B, embedding_dim)."""
        return self.embedding(labels)


class CFGGr00tN1d7ActionHead(Gr00tN1d7ActionHead):
    """GR00T N1.7 flow-matching head with advantage conditioning (token or FiLM)."""

    def __init__(
        self,
        config,
        advantage_cfg_dropout_prob: float = 0.3,
        cfg_guidance_weight: float = 1.0,
        positive_only_conditional: bool = False,
        conditioning: str = "film",
    ):
        super().__init__(config)
        if conditioning not in ("film", "token", "text"):
            raise ValueError(
                f"conditioning must be 'film', 'token', or 'text', got {conditioning}"
            )
        self.conditioning = conditioning
        self.advantage_cfg_dropout_prob = advantage_cfg_dropout_prob
        self.cfg_guidance_weight = cfg_guidance_weight
        self.positive_only_conditional = positive_only_conditional
        self._adv_labels: torch.Tensor | None = None

        if conditioning == "film":
            # Advantage embedding lives in the DiT timestep-embedding space.
            self.advantage_embedding = AdvantageEmbedding(self.model.inner_dim)
            self.model.timestep_encoder.register_forward_hook(self._timestep_adv_hook)
        elif conditioning == "token":
            # Advantage embedding lives in the cross-attention context space.
            self.advantage_embedding = AdvantageEmbedding(config.backbone_embedding_dim)
        else:  # text — advantage is injected into the prompt by the dataloader /
            # inference obs prep (RECAP-faithful); the head runs the BASE flow head.
            # Keep a small unused embedding so the get_model NaN re-init is uniform.
            self.advantage_embedding = AdvantageEmbedding(self.model.inner_dim)

    # ------------------------------------------------------------------
    # FiLM/adaLN conditioning hook (film mode only)
    # ------------------------------------------------------------------
    def _timestep_adv_hook(self, module, inputs, temb):
        """Add the advantage embedding to the DiT timestep embedding (adaLN cond)."""
        if self._adv_labels is None:
            return temb
        adv = self.advantage_embedding(self._adv_labels).to(temb.dtype)
        return temb + adv

    # ------------------------------------------------------------------
    # Token conditioning helper (token mode only)
    # ------------------------------------------------------------------
    def _append_advantage_token(self, vl_embs, vl_attn_mask, image_mask, labels):
        """Append one advantage token to the cross-attention context (B, S+1, D)."""
        bsz = vl_embs.shape[0]
        adv_token = self.advantage_embedding(labels).to(vl_embs.dtype).unsqueeze(1)
        vl_embs_aug = torch.cat([vl_embs, adv_token], dim=1)

        def _extend(mask, fill):
            if mask is None:
                return None
            extra = torch.full((bsz, 1), fill, dtype=mask.dtype, device=mask.device)
            return torch.cat([mask, extra], dim=1)

        return vl_embs_aug, _extend(vl_attn_mask, 1), _extend(image_mask, 0)

    def _condition(self, vl_embs, vl_attn_mask, image_mask, labels):
        """Apply the configured conditioning, returning the (vl, mask, img) to feed
        the DiT. For ``film`` it sets ``self._adv_labels`` (consumed by the hook) and
        returns the context unchanged; for ``token`` it appends the advantage token."""
        if self.conditioning == "film":
            self._adv_labels = labels
            return vl_embs, vl_attn_mask, image_mask
        if self.conditioning == "token":
            return self._append_advantage_token(vl_embs, vl_attn_mask, image_mask, labels)
        # text: conditioning is already in the prompt -> base head, no-op here.
        return vl_embs, vl_attn_mask, image_mask

    def process_backbone_output(self, backbone_output: BatchFeature) -> BatchFeature:
        """Masked VL self-attention refinement.

        CRITICAL: this head inherits the *vendored* ``Gr00tN1d7ActionHead``, whose
        ``process_backbone_output`` calls ``vl_self_attention(backbone_features)``
        WITHOUT an attention mask. With right-padded prompts that lets real tokens
        attend to padding and collapses eval SR (86% -> 34%) — the same padding bug
        fixed in the rlinf base head, which this head does NOT inherit. We override
        it here to pass ``backbone_attention_mask`` (``SelfAttentionTransformer.forward``
        is monkey-patched at import to accept it). No-op when no mask is available.
        """
        if not hasattr(backbone_output, "backbone_features"):
            return backbone_output
        backbone_features = backbone_output.backbone_features
        vlln = getattr(self, "vlln", None)
        if vlln is not None:
            backbone_features = vlln(backbone_features)
        vl_self_attention = getattr(self, "vl_self_attention", None)
        if vl_self_attention is not None:
            vl_attn_mask = getattr(backbone_output, "backbone_attention_mask", None)
            backbone_features = vl_self_attention(
                backbone_features, attention_mask=vl_attn_mask
            )
        backbone_output.backbone_features = backbone_features
        return backbone_output

    def _run_dit(self, sa_embs, vl_embs, vl_attn_mask, image_mask, t_discretized, training):
        """Call the (alternate) VL DiT exactly like the base head."""
        if training:
            if self.config.use_alternate_vl_dit:
                out, _ = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embs,
                    encoder_attention_mask=vl_attn_mask,
                    timestep=t_discretized,
                    return_all_hidden_states=True,
                    image_mask=image_mask,
                    backbone_attention_mask=vl_attn_mask,
                )
            else:
                out, _ = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embs,
                    encoder_attention_mask=vl_attn_mask,
                    timestep=t_discretized,
                    return_all_hidden_states=True,
                )
            return out
        if self.config.use_alternate_vl_dit:
            return self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                timestep=t_discretized,
                image_mask=image_mask,
                backbone_attention_mask=vl_attn_mask,
            )
        return self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            timestep=t_discretized,
        )

    def _predict_velocity(
        self,
        vl_embs,
        vl_attn_mask,
        image_mask,
        state_features,
        action_features,
        t_discretized,
        embodiment_id,
        action_len,
        training,
    ):
        """Run the DiT + decoder and slice out the action-token velocities."""
        sa_embs = torch.cat((state_features, action_features), dim=1)
        model_output = self._run_dit(
            sa_embs, vl_embs, vl_attn_mask, image_mask, t_discretized, training
        )
        pred = self.action_decoder(model_output, embodiment_id)
        return pred[:, -action_len:]

    @staticmethod
    def _flatten_state(state: torch.Tensor) -> torch.Tensor:
        """(B, state_history_length, max_state_dim) -> (B, 1, history*dim)."""
        return state.reshape(state.shape[0], 1, -1)

    def _forward_cfg_text(
        self, vl_embs, vl_attn_mask, image_mask, state_features, action_features,
        t_discretized, embodiment_id, actions, velocity, action_mask, advantage,
    ):
        """Text-conditioning training step: base flow-matching loss (the advantage
        is in the prompt). Metrics report loss split by advantage label."""
        pred = self._predict_velocity(
            vl_embs, vl_attn_mask, image_mask, state_features, action_features,
            t_discretized, embodiment_id, actions.shape[1], training=True,
        )
        per_elem = F.mse_loss(pred, velocity, reduction="none") * action_mask
        loss = per_elem.sum() / action_mask.sum().clamp_min(1.0)
        denom = action_mask.sum(dim=tuple(range(1, action_mask.ndim))).clamp_min(1.0)
        per_sample_loss = per_elem.sum(dim=tuple(range(1, per_elem.ndim))) / denom
        pos = advantage.to(per_sample_loss.dtype)
        neg = (~advantage).to(per_sample_loss.dtype)
        metrics = {
            "positive_label_count": advantage.sum(),
            "negative_label_count": (~advantage).sum(),
            "positive_loss_sum": (per_sample_loss * pos).sum(),
            "negative_loss_sum": (per_sample_loss * neg).sum(),
        }
        metrics = {k: float(v.detach().cpu().item()) for k, v in metrics.items()}
        return loss, metrics

    # ------------------------------------------------------------------
    # Training forward (RECAP via CFG routing)
    # ------------------------------------------------------------------
    def forward_cfg(self, backbone_output, action_input, advantage):
        """One CFG training step. Returns (loss, metrics) for FSDPCfgWorker."""
        self.set_frozen_modules_to_eval_mode()
        backbone_output = self.process_backbone_output(backbone_output)

        vl_embs = backbone_output.backbone_features
        vl_attn_mask = getattr(backbone_output, "backbone_attention_mask", None)
        image_mask = getattr(backbone_output, "image_mask", None)
        device = vl_embs.device
        embodiment_id = action_input.embodiment_id
        compute_dtype = vl_embs.dtype

        state = self._flatten_state(action_input.state.to(compute_dtype))
        state_features = self.state_encoder(state, embodiment_id)

        actions = action_input.action.to(compute_dtype)
        action_mask = action_input.action_mask
        noise = torch.randn(actions.shape, device=device, dtype=compute_dtype)
        # Beta-distributed time, matching the base head (fp32 Beta: dirichlet has no
        # bf16 backend). t = (1 - Beta(alpha, beta)) * noise_s.
        beta_dist = torch.distributions.Beta(
            torch.tensor(float(self.config.noise_beta_alpha)),
            torch.tensor(float(self.config.noise_beta_beta)),
        )
        beta_sample = beta_dist.sample((actions.shape[0],)).to(device)
        t = (1.0 - beta_sample) * self.config.noise_s
        t = t.to(actions.dtype)[:, None, None]
        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id)
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            action_features = action_features + self.position_embedding(pos_ids).unsqueeze(0)

        advantage = advantage.to(device=device, dtype=torch.bool)

        if self.conditioning == "text":
            # Advantage already injected into the prompt by the dataloader (incl.
            # CFG dropout); the head just runs the base flow-matching loss.
            return self._forward_cfg_text(
                vl_embs, vl_attn_mask, image_mask, state_features, action_features,
                t_discretized, embodiment_id, actions, velocity, action_mask, advantage,
            )

        masks = compute_cfg_routing_masks(
            advantage,
            positive_only_conditional=self.positive_only_conditional,
            unconditional_prob=self.advantage_cfg_dropout_prob,
        )
        labels = torch.full_like(advantage, AdvantageEmbedding.NULL_IDX, dtype=torch.long)
        labels = torch.where(
            masks["positive_conditional_mask"],
            torch.full_like(labels, AdvantageEmbedding.POS_IDX),
            labels,
        )
        labels = torch.where(
            masks["negative_conditional_mask"],
            torch.full_like(labels, AdvantageEmbedding.NEG_IDX),
            labels,
        )

        vl_use, mask_use, img_use = self._condition(vl_embs, vl_attn_mask, image_mask, labels)
        try:
            pred = self._predict_velocity(
                vl_use, mask_use, img_use, state_features, action_features,
                t_discretized, embodiment_id, actions.shape[1], training=True,
            )
        finally:
            self._adv_labels = None

        per_elem = F.mse_loss(pred, velocity, reduction="none") * action_mask
        loss = per_elem.sum() / action_mask.sum().clamp_min(1.0)
        per_sample_denom = action_mask.sum(
            dim=tuple(range(1, action_mask.ndim))
        ).clamp_min(1.0)
        per_sample_loss = (
            per_elem.sum(dim=tuple(range(1, per_elem.ndim))) / per_sample_denom
        )

        conditional_mask = masks["conditional_mask"]
        unconditional_mask = ~conditional_mask

        def _loss_sum(mask):
            return (per_sample_loss * mask.to(per_sample_loss.dtype)).sum()

        metrics = {
            "conditional_count": conditional_mask.sum(),
            "unconditional_count": unconditional_mask.sum(),
            "positive_label_count": masks["positive_mask"].sum(),
            "negative_label_count": masks["negative_mask"].sum(),
            "positive_conditional_count": masks["positive_conditional_mask"].sum(),
            "positive_unconditional_count": masks["positive_unconditional_mask"].sum(),
            "negative_conditional_count": masks["negative_conditional_mask"].sum(),
            "negative_unconditional_count": masks["negative_unconditional_mask"].sum(),
            "conditional_loss_sum": _loss_sum(conditional_mask),
            "unconditional_loss_sum": _loss_sum(unconditional_mask),
            "positive_conditional_loss_sum": _loss_sum(masks["positive_conditional_mask"]),
            "positive_unconditional_loss_sum": _loss_sum(masks["positive_unconditional_mask"]),
            "negative_conditional_loss_sum": _loss_sum(masks["negative_conditional_mask"]),
            "negative_unconditional_loss_sum": _loss_sum(masks["negative_unconditional_mask"]),
        }
        metrics = {
            k: (float(v.detach().cpu().item()) if torch.is_tensor(v) else v)
            for k, v in metrics.items()
        }
        return loss, metrics

    # ------------------------------------------------------------------
    # Inference (CFG-guided denoising)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def get_action_cfg(self, backbone_output, action_input):
        """Denoise an action chunk conditioned on the POS advantage.

        ``cfg_guidance_weight == 1`` -> single conditional (POS) pass per step.
        ``w != 1`` -> ``pred = uncond + w * (cond - uncond)`` per step (uncond=NULL).
        """
        backbone_output = self.process_backbone_output(backbone_output)
        vl_embs = backbone_output.backbone_features
        vl_attn_mask = getattr(backbone_output, "backbone_attention_mask", None)
        image_mask = getattr(backbone_output, "image_mask", None)
        embodiment_id = action_input.embodiment_id

        state = self._flatten_state(action_input.state.to(vl_embs.dtype))
        state_features = self.state_encoder(state, embodiment_id)

        bsz = vl_embs.shape[0]
        device = vl_embs.device
        actions = torch.randn(
            size=(bsz, self.config.action_horizon, self.action_dim),
            dtype=vl_embs.dtype,
            device=device,
        )

        w = self.cfg_guidance_weight
        # Text conditioning lives in the prompt (set at inference) -> single base
        # pass; head-level guidance (dual pass) only applies to film/token.
        dual_pass = (w != 1.0) and self.conditioning != "text"
        pos_labels = torch.full((bsz,), AdvantageEmbedding.POS_IDX, dtype=torch.long, device=device)
        null_labels = torch.full((bsz,), AdvantageEmbedding.NULL_IDX, dtype=torch.long, device=device)

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        def velocity(labels):
            vl_use, mask_use, img_use = self._condition(
                vl_embs, vl_attn_mask, image_mask, labels
            )
            try:
                return self._predict_velocity(
                    vl_use, mask_use, img_use, state_features, action_features,
                    timesteps_tensor, embodiment_id, self.config.action_horizon,
                    training=False,
                )
            finally:
                self._adv_labels = None

        for step in range(num_steps):
            t_cont = step / float(num_steps)
            t_discretized = int(t_cont * self.num_timestep_buckets)
            timesteps_tensor = torch.full((bsz,), t_discretized, device=device)

            action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)
            if self.config.add_pos_embed:
                pos_ids = torch.arange(
                    action_features.shape[1], dtype=torch.long, device=device
                )
                action_features = action_features + self.position_embedding(pos_ids).unsqueeze(0)

            pred = velocity(pos_labels)
            if dual_pass:
                pred_uncond = velocity(null_labels)
                pred = pred_uncond + w * (pred - pred_uncond)

            actions = actions + dt * pred
        return BatchFeature(data={"action_pred": actions})
