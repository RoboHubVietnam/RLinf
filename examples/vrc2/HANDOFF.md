# GR00T-N1.7 SFT on VR_C2 cup-stacking — Handoff / Notes (2026-07-09)

Quick project: fine-tune GR00T-N1.7 on a new-embodiment real-robot dataset and
measure held-out imitation quality. **Status: DONE, successful.**

## TL;DR result

SFT'd GR00T-N1.7 on ~44 demos of the **VR_C2** dual-arm mobile manipulator
stacking cups. On **5 held-out (unseen) episodes**, open-loop action error:

| checkpoint | action MSE | action MAE |
|---|---|---|
| 1000 | 0.0044 | 0.027 |
| 3000 | 0.0034 | 0.020 |
| **5000 (final)** | 0.0035 | **0.019** |

The model learned the novel embodiment + task and **generalizes** to held-out
episodes (MAE ~0.019/joint). Clear improvement 1000→3000 (MSE −23%, MAE −26%),
then **converged** (train loss 0.045→0.011 plateau; 3000≈5000). Best
checkpoint ~3000–5000.

## Key findings / decisions

- **Zero-shot is NOT possible for a brand-new embodiment.** GR00T's base
  checkpoint processor only carries its 8 pretrain embodiments — no action
  head, normalization, or `embodiment_id` for VR_C2's 24-dim space, and
  `Gr00tPolicy` raises on an unknown tag. Honest "before" = an early
  checkpoint, not a true zero-shot number.
- **RECAP (the RL method) is NOT runnable on this data** — it needs an
  environment + reward for rollouts/advantages; this is offline expert demos
  with no VR_C2 simulator and no reward. RLinf has no VR_C2 env. RECAP would
  require a sim or real-robot rollout loop first.
- **Metric = open-loop action MSE/MAE (data-only).** No VR_C2 sim → task
  success NOT measurable. Low MSE = faithful demo imitation, not a guarantee
  of real-robot success.
- Learning rides on the trainable **action head (diffusion) + projector**;
  the Cosmos-2B backbone is frozen.

## Dataset (VR_C2)

`locht131/vrc2_stack_cup_joint_states` (HF, public, LeRobot v2.1). Robot
**VR_C2**: 2 wheels + 3 column + waist + 7 left arm + 7 right arm + 2 head + 2
grippers = **24-dim action**; **72-dim state** = [pos(24), vel(24), effort(24)]
(state[:24] = joint pos, matches action). 4 cameras (head/head_right 600x960,
left/right 480x640), IMU. 49 episodes, 32399 frames, 25 fps, single task
"Stack the three cups together." No reward/success labels.

Downloaded: `/data/datasets/vrc2_stack_cup`. Split (renumbered):
`/data/datasets/vrc2_stack_cup_train` (ep 0-43) and `_val` (ep 44-48 → 0-4).

## Files & artifacts

- **Split script** (repo): `examples/vrc2/split_lerobot.py` — renumbers
  episode_index/index, symlinks videos, rebuilds meta. Reusable.
- **modality.json** (in each dataset `meta/`): `state.joint_pos [0,24]`,
  `action.joints [0,24]`, video `head/left_wrist/right_wrist`, language
  `annotation.human.action.task_description`.
- **GR00T modality config** (in the venv gr00t checkout, NOT RLinf git):
  `vla-rlft-n1d7/gr00t/examples/VR_C2/vrc2_config.py` — registers
  `EmbodimentTag.NEW_EMBODIMENT`; action is RELATIVE joints with
  `state_key="joint_pos"` (required so relative-action stats find the state).
- **Checkpoints**: `/data/checkpoints/vrc2_stack_cup_ft/checkpoint-{1000..5000}`
  (each self-contained: weights + processor_config.json + statistics.json +
  embodiment_id.json + config with LOCAL cosmos path). wandb project
  `vrc2-stack-cup`.
- **Eval plots** (GT vs predicted): `/data/checkpoints/vrc2_stack_cup_ft/
  eval_plots_{1000,3000,5000}.png`.

## gr00t PACKAGE PATCHES (must re-apply if the venv is rebuilt — NOT tracked)

Edits to `vla-rlft-n1d7/gr00t/gr00t/experiment/launch_finetune.py` (needed to
run on a single 32GB GPU, offline):
1. line ~88: `config.model.load_bf16 = True` (was False → fp32 12GB; bf16 = 6GB)
2. line ~92-93: point backbone to local `/data/checkpoints/nvidia/Cosmos-Reason2-2B`
   (was hardcoded to gated hub name → fails under HF_HUB_OFFLINE)
3. line ~99: `config.training.optim = "paged_adamw_8bit"` (was adamw_torch;
   8-bit paged Adam ~4x less optimizer memory + CPU paging — fixed the
   optimizer-step OOM)

Env for all gr00t commands: `USE_TF=0 TRANSFORMERS_NO_TF=1` (transformers tries
to import TF/Keras bindings otherwise), `HF_HUB_OFFLINE=1`,
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `CUDA_VISIBLE_DEVICES=0`.

## Reproduce (from `vla-rlft-n1d7/gr00t`, venv activated)

```bash
# 1. split (from RLinf repo root)
python examples/vrc2/split_lerobot.py --src /data/datasets/vrc2_stack_cup \
  --train_out /data/datasets/vrc2_stack_cup_train --train_eps 0-43 \
  --val_out /data/datasets/vrc2_stack_cup_val --val_eps 44-48

# 2. stats (per split) — validates the data path
python gr00t/data/stats.py --dataset-path /data/datasets/vrc2_stack_cup_train \
  --embodiment-tag NEW_EMBODIMENT --modality-config-path examples/VR_C2/vrc2_config.py
python gr00t/data/stats.py --dataset-path /data/datasets/vrc2_stack_cup_val \
  --embodiment-tag NEW_EMBODIMENT --modality-config-path examples/VR_C2/vrc2_config.py

# 3. SFT (single 32GB GPU): per-device batch 2 (global 16, accum 8)
python gr00t/experiment/launch_finetune.py \
  --base-model-path /data/checkpoints/GR00T-N1.7-3B-base \
  --dataset-path /data/datasets/vrc2_stack_cup_train \
  --embodiment-tag NEW_EMBODIMENT --modality-config-path examples/VR_C2/vrc2_config.py \
  --num-gpus 1 --output-dir /data/checkpoints/vrc2_stack_cup_ft \
  --global-batch-size 16 --gradient-accumulation-steps 8 \
  --learning-rate 1e-4 --max-steps 5000 --save-steps 1000 --save-total-limit 6 \
  --warmup-ratio 0.05 --dataloader-num-workers 4 --use-wandb --wandb-project vrc2-stack-cup
# ~3.5h on one 32GB card @ ~2.4s/it.

# 4. held-out open-loop eval (val episodes 0-4)
python gr00t/eval/open_loop_eval.py --dataset-path /data/datasets/vrc2_stack_cup_val \
  --embodiment-tag NEW_EMBODIMENT --model-path /data/checkpoints/vrc2_stack_cup_ft/checkpoint-5000 \
  --traj-ids 0 1 2 3 4 --action-horizon 16 --steps 300 --modality-keys joints \
  --save-plot-path /data/checkpoints/vrc2_stack_cup_ft/eval_plots_5000
```

## Possible next steps (if resumed)

- Use more/all 4 cameras (dropped `head_right` for memory; both wrists kept).
- Longer/lower-lr run or tune the diffusion head only if overfitting the small set.
- A true "before" baseline: eval a `--max-steps 1` checkpoint (processor built,
  head untrained) to bracket the improvement.
- Closed-loop success needs a VR_C2 simulator or real-robot deployment (then
  RECAP becomes possible).
