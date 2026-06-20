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

"""Advantage-conditioned flow-matching action head for GR00T N1.5 (RECAP stage 4).

Implements classifier-free-guidance (CFG / CFGRL, arXiv:2505.23458) training for
the GR00T flow-matching policy, used by RECAP (arXiv:2511.14759 §IV-B) policy
optimization. A single learned advantage-indicator token (NULL / NEG / POS) is
appended to the DiT cross-attention context so the velocity field can be
conditioned on optimality.

Design choices (intentionally aligned with ``openpi_cfg`` so this head is a
drop-in for ``FSDPCfgWorker``):

* Per-sample CFG dropout *routing* (``compute_cfg_routing_masks``) rather than a
  dual unconditional+conditional loss on every sample. Each sample is trained
  either conditionally (its POS/NEG token) or unconditionally (NULL token),
  selected by ``advantage_cfg_dropout_prob``.
* The per-step advantage label (positive vs negative) is supplied by the caller
  as a boolean ``advantage`` tensor — produced offline by RECAP stages 1-3 and
  read from ``meta/advantages_{tag}.parquet`` — NOT recomputed from a value head
  at train time.
* Training ``forward`` returns ``(loss, metrics)`` with the exact metric keys
  aggregated by ``FSDPCfgWorker.run_training``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from gr00t.model.action_head.flow_matching_action_head import FlowmatchingActionHead
from transformers.feature_extraction_utils import BatchFeature

from rlinf.models.embodiment.gr00t.gr00t_n1d5_cfg.cfg_utils import (
    compute_cfg_routing_masks,
)


class AdvantageEmbedding(nn.Module):
    """Encodes the advantage indicator token appended to the cross-attention context.

    Three indices:
        NULL_IDX (0): unconditional token (CFG dropout / unconditional pass).
        NEG_IDX  (1): A(o, a) <= eps_l  -> "advantage negative".
        POS_IDX  (2): A(o, a)  > eps_l  -> "advantage positive".
    """

    NULL_IDX: int = 0
    NEG_IDX: int = 1
    POS_IDX: int = 2

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(3, embedding_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        """labels: (B,) long in {NULL,NEG,POS} -> (B, 1, embedding_dim)."""
        return self.embedding(labels).unsqueeze(1)


class CFGFlowmatchingActionHead(FlowmatchingActionHead):
    """GR00T flow-matching head with advantage (CFG) conditioning.

    Subclasses the official ``FlowmatchingActionHead`` and reuses its encoders,
    DiT, decoder, VL layernorm/self-attention, and ``sample_time``. Adds a single
    advantage token to ``encoder_hidden_states`` (and extends the attention mask
    by one) so the DiT cross-attention conditions on optimality.
    """

    def __init__(
        self,
        config,
        advantage_cfg_dropout_prob: float = 0.3,
        cfg_guidance_weight: float = 1.0,
        positive_only_conditional: bool = False,
    ):
        super().__init__(config)
        # CFG dropout probability p: fraction of samples routed unconditional.
        self.advantage_cfg_dropout_prob = advantage_cfg_dropout_prob
        # Guidance scale w at inference: w=1 samples the conditional policy with a
        # single forward; w!=1 runs a second unconditional pass per denoise step.
        self.cfg_guidance_weight = cfg_guidance_weight
        self.positive_only_conditional = positive_only_conditional
        # The advantage token is concatenated to the VL features along the
        # sequence dim, so it must match the cross-attention context width.
        self.advantage_embedding = AdvantageEmbedding(config.backbone_embedding_dim)

    # ------------------------------------------------------------------
    # Conditioning helpers
    # ------------------------------------------------------------------
    def _append_advantage_token(
        self,
        vl_embs: torch.Tensor,
        vl_attn_mask: torch.Tensor | None,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Append one advantage token to the cross-attention context (B, S+1, D)."""
        bsz = vl_embs.shape[0]
        adv_token = self.advantage_embedding(labels).to(vl_embs.dtype)  # (B,1,D)
        vl_embs_aug = torch.cat([vl_embs, adv_token], dim=1)
        if vl_attn_mask is not None:
            extra = torch.ones(
                bsz, 1, dtype=vl_attn_mask.dtype, device=vl_attn_mask.device
            )
            vl_attn_mask_aug = torch.cat([vl_attn_mask, extra], dim=1)
        else:
            vl_attn_mask_aug = None
        return vl_embs_aug, vl_attn_mask_aug

    def _predict_velocity(
        self,
        vl_embs: torch.Tensor,
        vl_attn_mask: torch.Tensor | None,
        state_features: torch.Tensor,
        action_features: torch.Tensor,
        t_discretized: torch.Tensor,
        embodiment_id: torch.Tensor,
        action_len: int,
    ) -> torch.Tensor:
        """Run the DiT + decoder and slice out the action-token velocities."""
        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(
            vl_embs.shape[0], -1, -1
        )
        sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)
        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            encoder_attention_mask=vl_attn_mask,
            timestep=t_discretized,
            return_all_hidden_states=False,
        )
        pred = self.action_decoder(model_output, embodiment_id)
        return pred[:, -action_len:]

    # ------------------------------------------------------------------
    # Training forward (RECAP Eq. 3 via CFG routing)
    # ------------------------------------------------------------------
    def forward_cfg(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        advantage: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """One CFG training step.

        Args:
            backbone_output: VL features from the GR00T backbone.
            action_input: BatchFeature with ``state``, ``action``,
                ``action_mask``, ``embodiment_id``.
            advantage: (B,) boolean tensor, True for positive-advantage samples.

        Returns:
            (loss, metrics) where metrics carry the per-route counts and loss
            sums consumed by ``FSDPCfgWorker.run_training``.
        """
        self.set_frozen_modules_to_eval_mode()
        backbone_output = self.process_backbone_output(backbone_output)

        vl_embs = backbone_output.backbone_features
        vl_attn_mask = backbone_output.backbone_attention_mask
        device = vl_embs.device
        embodiment_id = action_input.embodiment_id

        # Build the shared noised flow-matching trajectory (GR00T: x0 noise, x1 data).
        # Cast inputs to the backbone's compute dtype (bf16 under FSDP mixed
        # precision, fp32 in eval) so the action head sees a consistent dtype.
        compute_dtype = vl_embs.dtype
        actions = action_input.action.to(compute_dtype)
        action_mask = action_input.action_mask
        noise = torch.randn(actions.shape, device=device, dtype=compute_dtype)
        # Flow-matching timestep ~ Beta (matches the base head's sample_time), but
        # sampled in fp32 here: when the module is cast to bf16 the stored
        # beta_dist concentration becomes bf16, and Beta/Dirichlet sampling is not
        # implemented for bf16.
        beta_dist = torch.distributions.Beta(
            torch.tensor(float(self.config.noise_beta_alpha)),
            torch.tensor(float(self.config.noise_beta_beta)),
        )
        beta_sample = beta_dist.sample((actions.shape[0],)).to(device)
        t = (self.config.noise_s - beta_sample) / self.config.noise_s
        # Beta samples near 1 can push t slightly <0 (or to 1), giving negative
        # timestep-embedding indices / extrapolated trajectories that produce NaN
        # gradients in bf16. Clamp to a safe open interval.
        t = t.clamp(1e-4, 1.0 - 1e-4)
        t = t.to(actions.dtype)[:, None, None]
        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(
            noisy_trajectory, t_discretized, embodiment_id
        )
        if self.config.add_pos_embed:
            pos_ids = torch.arange(
                action_features.shape[1], dtype=torch.long, device=device
            )
            action_features = action_features + self.position_embedding(pos_ids).unsqueeze(0)
        state_features = self.state_encoder(
            action_input.state.to(compute_dtype), embodiment_id
        )

        # Route each sample to conditional (POS/NEG token) or unconditional (NULL).
        advantage = advantage.to(device=device, dtype=torch.bool)
        masks = compute_cfg_routing_masks(
            advantage,
            positive_only_conditional=self.positive_only_conditional,
            unconditional_prob=self.advantage_cfg_dropout_prob,
        )
        labels = torch.full_like(
            advantage, AdvantageEmbedding.NULL_IDX, dtype=torch.long
        )
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

        vl_aug, mask_aug = self._append_advantage_token(vl_embs, vl_attn_mask, labels)
        pred = self._predict_velocity(
            vl_aug,
            mask_aug,
            state_features,
            action_features,
            t_discretized,
            embodiment_id,
            actions.shape[1],
        )

        # --- NaN diagnostic (temporary) ---
        import os as _os
        if _os.environ.get("CFG_NAN_DEBUG"):
            def _bad(name, x):
                if torch.is_tensor(x) and not torch.isfinite(x).all():
                    n_nan = torch.isnan(x).sum().item()
                    n_inf = torch.isinf(x).sum().item()
                    print(f"[CFG_NAN_DEBUG] {name}: nan={n_nan} inf={n_inf} "
                          f"max={x[torch.isfinite(x)].abs().max().item() if torch.isfinite(x).any() else 'NA'} "
                          f"dtype={x.dtype} shape={tuple(x.shape)}", flush=True)
                    return True
                return False
            anybad = False
            anybad |= _bad("vl_embs", vl_embs)
            anybad |= _bad("state_features", state_features)
            anybad |= _bad("action_features", action_features)
            anybad |= _bad("noisy_trajectory", noisy_trajectory)
            anybad |= _bad("velocity", velocity)
            anybad |= _bad("adv_embed.weight", self.advantage_embedding.embedding.weight
                           if hasattr(self.advantage_embedding, "embedding") else
                           next(self.advantage_embedding.parameters()))
            anybad |= _bad("pred", pred)
            if anybad:
                print(f"[CFG_NAN_DEBUG] t.min={t.min().item():.5f} t.max={t.max().item():.5f} "
                      f"action_mask.sum={action_mask.sum().item()}", flush=True)

        # Per-element masked MSE, then reduce to global loss and per-sample losses.
        per_elem = F.mse_loss(pred, velocity, reduction="none") * action_mask
        loss = per_elem.sum() / action_mask.sum().clamp_min(1.0)
        per_sample_denom = action_mask.sum(dim=tuple(range(1, action_mask.ndim))).clamp_min(1.0)
        per_sample_loss = (
            per_elem.sum(dim=tuple(range(1, per_elem.ndim))) / per_sample_denom
        )  # (B,)

        conditional_mask = masks["conditional_mask"]
        unconditional_mask = ~conditional_mask

        def _loss_sum(mask: torch.Tensor) -> torch.Tensor:
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
        # Worker aggregates metrics with numpy (np.sum/np.mean), so return host
        # python floats rather than CUDA tensors.
        metrics = {
            k: (float(v.detach().cpu().item()) if torch.is_tensor(v) else v)
            for k, v in metrics.items()
        }
        return loss, metrics

    # ------------------------------------------------------------------
    # Inference (CFG-guided denoising)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def get_action_cfg(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
    ) -> BatchFeature:
        """Denoise an action chunk conditioned on the positive-advantage token.

        With ``cfg_guidance_weight == 1`` this is a single conditional pass per
        step. With ``w != 1`` it runs the standard CFG combination
        ``pred = uncond + w * (cond - uncond)`` per denoise step.
        """
        backbone_output = self.process_backbone_output(backbone_output)
        vl_embs = backbone_output.backbone_features
        vl_attn_mask = backbone_output.backbone_attention_mask
        embodiment_id = action_input.embodiment_id
        state_features = self.state_encoder(action_input.state, embodiment_id)

        bsz = vl_embs.shape[0]
        device = vl_embs.device
        actions = torch.randn(
            size=(bsz, self.config.action_horizon, self.config.action_dim),
            dtype=vl_embs.dtype,
            device=device,
        )

        w = self.cfg_guidance_weight
        dual_pass = w != 1.0
        pos_labels = torch.full(
            (bsz,), AdvantageEmbedding.POS_IDX, dtype=torch.long, device=device
        )
        # Keep inference attention-masking CONSISTENT with forward_cfg training
        # (which passes the augmented backbone_attention_mask). The fine-tuned
        # head specializes to that masked cross-attention pattern, so dropping the
        # mask at inference (as the base get_action does) hurts more.
        vl_cond, mask_cond = self._append_advantage_token(
            vl_embs, vl_attn_mask, pos_labels
        )
        if dual_pass:
            null_labels = torch.full(
                (bsz,), AdvantageEmbedding.NULL_IDX, dtype=torch.long, device=device
            )
            vl_null, mask_null = self._append_advantage_token(
                vl_embs, vl_attn_mask, null_labels
            )

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps
        for step in range(num_steps):
            t_cont = step / float(num_steps)
            t_discretized = int(t_cont * self.num_timestep_buckets)
            timesteps_tensor = torch.full((bsz,), t_discretized, device=device)

            action_features = self.action_encoder(
                actions, timesteps_tensor, embodiment_id
            )
            if self.config.add_pos_embed:
                pos_ids = torch.arange(
                    action_features.shape[1], dtype=torch.long, device=device
                )
                action_features = action_features + self.position_embedding(pos_ids).unsqueeze(0)

            pred = self._predict_velocity(
                vl_cond,
                mask_cond,
                state_features,
                action_features,
                timesteps_tensor,
                embodiment_id,
                self.config.action_horizon,
            )
            if dual_pass:
                pred_uncond = self._predict_velocity(
                    vl_null,
                    mask_null,
                    state_features,
                    action_features,
                    timesteps_tensor,
                    embodiment_id,
                    self.config.action_horizon,
                )
                pred = pred_uncond + w * (pred - pred_uncond)

            actions = actions + dt * pred
        return BatchFeature(data={"action_pred": actions})
