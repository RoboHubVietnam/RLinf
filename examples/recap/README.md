# RECAP for GR00T N1.7 on LIBERO-Long

Working implementation of RECAP (advantage-conditioned RL fine-tuning for
VLAs, [arXiv:2511.14759](https://arxiv.org/abs/2511.14759)) for GR00T N1.7
(Cosmos-Reason2-2B backbone) in RLinf, validated on LIBERO-Long on a single
RTX 5090.

## Headline result (n=500 episodes per arm, 5 seeds, randomized resets)

| Policy | Success rate |
|---|---|
| SFT base (20k steps from GR00T-N1.7-3B-base) | 42.8% (214/500) |
| **+ RECAP conditioned SFT, w=1.0** | **65.4% (327/500)** |
| + RECAP conditioned SFT, w=2.0 | 63.4% (317/500) |

**+22.6 points (>7 SE) from advantage-conditioned training.** CFG guidance
w>1 adds nothing over w=1.0 at n=500 — operate the policy at w=1.0 (single
pass, on-manifold, 2x cheaper). Full experiment record:
[`results/n1d7/RECAP_from44_final_results.md`](../../results/n1d7/RECAP_from44_final_results.md);
wandb project `recap-n1d7-from-base`.

## The recipe that works

**Advantage conditioning is an SFT-time ingredient, not a bolt-on.** Train
the CFG head jointly with the action head at SFT learning rate:

```bash
# 1. Value critic on rollouts -> 2. advantages (per-type quantiles) ->
# 3. conditioned SFT. See sft_from_base/run_phase2_condsft.sh for the
# validated end-to-end driver. The core training call:
python examples/recap/cfg/train_cfg.py --config-name libero_cfg_gr00t_n1d7 \
  actor.model.model_path=<SFT_CKPT_HF_DIR> \
  "data.train_data_paths=[{dataset_path:<DEMOS>,type:sft,weight:1.0}]" \
  data.advantage_tag=recal \
  actor.optim.lr=1.0e-4 \          # SFT lr, JOINT training — the key
  runner.max_steps=5000
```

Pipeline stages (each has a driver under `sft_from_base/`):
collect (w=1.0) -> convert (`process/convert_rollouts_to_lerobot.py`) ->
returns -> value critic -> advantages (`process/compute_advantages.py`) ->
conditioned SFT -> eval.

## Four rules (each violated once, each costing a failed run)

1. **Train conditioning jointly at SFT lr (1e-4).** At fine-tune lr (2e-6)
   the advantage embedding never trains — CFG guidance then extrapolates
   along a frozen random direction (looks like noisy, non-monotonic w
   response). Embedding-only training (head frozen) learns a destructive
   common-mode bias instead. Both bolt-on variants fail.
2. **Calibrate labels per data type** (`advantage.positive_quantile_by_type`:
   rollouts ~40%, demos ~30% positive). Forcing demos 100% positive makes
   POS ≈ "be an SFT policy" and guidance has nothing to sharpen toward.
3. **Collect at w=1.0 (β=1) only.** w>1 actions are CFG extrapolations —
   off-manifold and "overly aggressive"; training on them produced a 0/100
   policy. Guidance is inference-only (and per Phase 3, not even needed).
4. **Convert rollout actions back to the model convention.** The collector
   records env-executed actions — after `prepare_actions_for_libero` flips
   the gripper to {-1,+1}; demos store raw {0,1}. Mixed-convention training
   at real lr yields a policy that never grasps (0/100).
   `convert_rollouts_to_lerobot.py` maps back by default
   (`--no-gripper_from_env` to disable). Rollout datasets converted before
   2026-07-03 carry the env convention — do not train on them unmapped.

## Known negative result: one RL iteration did not compound

Collecting 200 episodes at w=1.0 from the 65% policy and retraining from the
SFT anchor on a 50/50 rollout+demo mix scored 52-54% — functional but below
demos-only conditioned SFT. The mediocre-success rollout mix dilutes the
expert anchor (CFG dropout routes failure actions into the unconditional
class by design). Untried levers: demo-heavier weighting, retraining from
the conditioned checkpoint, larger/success-biased collection, stronger base.

## Code map

- Model: `rlinf/models/embodiment/gr00t/gr00t_n1d7_cfg/` (CFG head: film /
  token / text conditioning; zero-init advantage embedding; masked
  vl_self_attention). Architecture diagram:
  `results/n1d7/recap_gr00t_value_attach.mmd`.
- Data: `rlinf/data/datasets/recap/gr00t_n1d7_cfg.py` (per-step advantage
  labels from `meta/advantages_{tag}.parquet`).
- Optimizer: `optim.param_group_overrides` in
  `rlinf/hybrid_engines/fsdp/fsdp_model_manager.py` for per-param-group lr.
- Value critic (separate model, never attached to GR00T): SigLIP2 + Gemma3,
  `value/train_value.py`, config `value/config/libero_sft_value_gr00t_n1d7.yaml`.
- Best checkpoint: `examples/recap/results/recap20k_phase2_condsft/checkpoints/global_step_5000`
  (local only; `examples/recap/results/` is gitignored).
