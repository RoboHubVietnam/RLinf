# RECAP for GR00T N1.7 on LIBERO-Long

Implementation of RECAP (advantage-conditioned RL fine-tuning for VLAs,
[arXiv:2511.14759](https://arxiv.org/abs/2511.14759)) for GR00T N1.7
(Cosmos-Reason2-2B backbone) in RLinf, run on LIBERO-Long on a single RTX 5090.

## Status: pipeline runs end to end; no demonstrated gain from conditioning

> **Correction (control experiment of 2026-07-31).** An earlier version of
> this page reported "+22.6 points from advantage-conditioned training"
> (42.8% -> 65.4%). That claim is **retracted**. Serving the same checkpoint
> with the conditioning switched off, and with the label inverted, gives the
> same success rate:

| Arm (same checkpoint, n=500 each, same session) | Success rate |
|---|---|
| SFT base, served through the CFG path | 44.2% (221/500) |
| Conditioned SFT, conditioning off (w=0) | 58.4% (292/500) |
| Conditioned SFT, positive label (w=1) | 58.0% (290/500) |
| Conditioned SFT, **negative** label (w=1) | 59.8% (299/500) |

What this shows:

- **The gain is extra SFT, not conditioning.** 5,000 more steps at lr 1e-4
  lift the 20k-step base by about 14 points whether or not the label is used.
- **The labels carried no outcome signal.** The demo labels used for that run
  were a threshold on critic error (all demos succeed, so the true advantage
  is constant). On the rollout labels of the same period, 40.8% of frames in
  successful episodes are positive against 40.4% in failed ones.
- **The 65.4% figure does not reproduce** (58.0% on re-measurement of the
  same checkpoint); the evaluation noise was unseeded at the time.

Whether advantage conditioning helps when the labels are informative is
still open and is being tested with known-correct labels and matched
unconditioned controls. Treat everything below as engineering notes for
running the pipeline, not as a validated recipe.

## How the conditioned-SFT stage is run

The advantage embedding only trains when the CFG head is trained jointly with
the action head at SFT learning rate:

```bash
# 1. Value critic on rollouts -> 2. advantages (per-type quantiles) ->
# 3. conditioned SFT. See sft_from_base/run_phase2_condsft.sh for the
# end-to-end driver. The core training call:
python examples/recap/cfg/train_cfg.py --config-name libero_cfg_gr00t_n1d7 \
  actor.model.model_path=<SFT_CKPT_HF_DIR> \
  "data.train_data_paths=[{dataset_path:<DEMOS>,type:sft,weight:1.0}]" \
  data.advantage_tag=recal \
  actor.optim.lr=1.0e-4 \          # SFT lr, joint training
  runner.max_steps=5000
```

Pipeline stages (each has a driver under `sft_from_base/`):
collect (w=1.0) -> convert (`process/convert_rollouts_to_lerobot.py`) ->
returns -> value critic -> advantages (`process/compute_advantages.py`) ->
conditioned SFT -> eval.

## Four engineering rules (each violated once, each costing a failed run)

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
   policy. Guidance is inference-only.
4. **Convert rollout actions back to the model convention.** The collector
   records env-executed actions — after `prepare_actions_for_libero` flips
   the gripper to {-1,+1}; demos store raw {0,1}. Mixed-convention training
   at real lr yields a policy that never grasps (0/100).
   `convert_rollouts_to_lerobot.py` maps back by default
   (`--no-gripper_from_env` to disable). Rollout datasets converted before
   2026-07-03 carry the env convention — do not train on them unmapped.

## Known data issue: rollouts are stored at a fixed 512 frames

The LIBERO collection config runs with `ignore_terminations: True`, so every
rollout episode is 512 frames long and `is_success` is an episode-level flag
with no record of when the task was completed (demos: median 259 frames).
Successful rollouts therefore contain post-task motion, and the return target
is identical for a fast and a slow success. Truncate at first success (and
apply rule 4) before using rollouts for value training, conditioned SFT, or
success-filtered fine-tuning.

## Known negative result: one RL iteration did not compound

Collecting 200 episodes at w=1.0 from the conditioned-SFT policy and retraining
from the SFT anchor on a 50/50 rollout+demo mix scored 52-54% (n=100) — below
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
- Checkpoint behind the table above:
  `examples/recap/results/recap20k_phase2_condsft/checkpoints/global_step_5000`
  (local only; `examples/recap/results/` is gitignored).
