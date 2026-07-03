# RECAP on GR00T N1.7 / LIBERO-Long from a 44% SFT base — final results

**Headline (n=500 per arm, 5 seeds x 100 episodes, randomized resets):**

| Arm | SR | 95% CI |
|---|---|---|
| SFT base (checkpoint-20000) | **42.8%** (214/500) | ±4.3 |
| RECAP conditioned SFT, w=1.0 (β=1) | **65.4%** (327/500) | ±4.2 |
| RECAP conditioned SFT, w=2.0 | 63.4% (317/500) | ±4.2 |

**RECAP conditioned SFT lifts LIBERO-Long success 42.8% → 65.4% (+22.6 pts,
>7 SE).** CFG guidance w>1 adds nothing over w=1.0 at n=500 — the gain comes
from advantage-conditioned training, not inference-time extrapolation
(consistent with the paper's β=1 default).

## The working recipe (examples/recap/sft_from_base/run_phase2_condsft.sh)

Continue from the SFT checkpoint with the CFG head, **jointly at SFT lr
(1e-4)**, on demos only, with paper-calibrated labels (~30% of demo steps
positive via per-type quantiles), CFG dropout 0.3, 5000 steps. Policy:
`recap20k_phase2_condsft/checkpoints/global_step_5000`.

## What did NOT work (all tested, 100-ep evals)

1. **Bolt-on CFG at fine-tune lr 2e-6** (any label calibration, any init):
   the advantage embedding NEVER trains (|POS−NULL|=0.0013 after 3k steps);
   every historical "guidance effect" was noise along a frozen random
   direction. w=1.0 lands below base (head forgets on rollout-heavy data).
2. **Embedding-only training at lr=1e-3** (head frozen): embedding trains but
   learns a destructive common-mode temb bias → 6-12% SR.
3. **Training on CFG-extrapolated (w=2.0) rollout actions**: 0/100.
   Guidance actions are off-manifold; collect at β=1 only.
4. **Gripper convention mismatch (CRITICAL BUG, fixed)**: the collector saves
   env-executed actions, i.e. after `prepare_actions_for_libero` flips the
   gripper to {-1,+1}; demos store raw {0,1}. Mixed training at lr=1e-4
   learns the wrong convention → never grasps → 0/100. All rollout datasets
   converted before 2026-07-03 carry env-convention grippers.
   `convert_rollouts_to_lerobot.py` now maps back by default
   (g_raw = (1 − g_env)/2).
5. **One RL iteration (collect 200 eps at w=1.0 from the 65% policy, retrain
   from SFT anchor on 50/50 rollouts+demos)**: 52-54% at all w — functional
   (post gripper fix) but BELOW demos-only conditioned SFT. The 57.5%-success
   rollout mix dilutes the expert anchor; CFG dropout routes failures into
   the unconditional class by design. Iteration levers not yet tried:
   demo-heavier weighting, retraining from the conditioned ckpt,
   larger/success-biased collection, stronger base.

## Operating rules distilled

- Advantage conditioning is a **pretraining/SFT-time ingredient**: train it
  jointly with the head at SFT lr. Never bolt it onto a finished policy at
  fine-tune lr.
- Calibrate labels per data type (rollouts ~40% / demos ~30% positive), do
  not force demos all-positive.
- **Collect at β=1 (w=1.0)**; w>1 is inference-only (and at n=500 it wasn't
  even better at inference).
- Rollout action recording is in env convention; convert back to the model
  convention before training (gripper!).

All runs in wandb project `recap-n1d7-from-base` (phase3_* experiments are
the n=500 arms).
