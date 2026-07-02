#!/usr/bin/env bash
# 100-ep confirm of the mixed RECAP-film checkpoint at the two tied 20-ep leaders
# (w=1.0 no-forgetting floor, w=1.5 guidance point) vs the 44% SFT base.
set -o pipefail
REPO=/home/duynguyen/Desktop/RESEARCH/VR/RLinf
cd "$REPO"; source "$REPO/vla-rlft-n1d7/bin/activate"
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1
CKPT="$REPO/examples/recap/results/recap20k_iter1_film_mixed/checkpoints/global_step_3000/actor/model_state_dict/full_weights.pt"
STEPS=2560   # 100 episodes (10/task)
STATUS=/tmp/recap20k_mixed_confirm.log
echo "CONFIRM_START $(date)" > "$STATUS"
for W in 1.0 1.5; do
  LOG="/tmp/n17_mixed_confirm_w${W}.log"
  echo "=== $(date +%H:%M:%S) | 100-ep confirm w=$W ===" | tee -a "$STATUS"
  bash "$REPO/evaluations/run_eval.sh" libero libero_10_gr00t_n1d7_cfg_eval \
    "+runner.ckpt_path=${CKPT}" "rollout.model.cfg_guidance_weight=${W}" \
    "+rollout.model.conditioning=film" "env.eval.max_steps_per_rollout_epoch=${STEPS}" \
    "runner.logger.experiment_name=mixed_confirm_w${W}" \
    "runner.logger.logger_backends=[wandb,tensorboard]" \
    "runner.logger.project_name=recap-n1d7-from-base" 2>&1 | tee "$LOG" | tail -2
  s=$(grep -cE "task_id=.*success=True" "$LOG"); t=$(grep -cE "libero eval\] task_id=" "$LOG")
  echo "=== $(date +%H:%M:%S) | CONFIRM w=$W -> ${s}/${t} ===" | tee -a "$STATUS"
done
echo "CONFIRM COMPLETE (base SFT = 44/100)" | tee -a "$STATUS"
