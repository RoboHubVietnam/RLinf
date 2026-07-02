#!/usr/bin/env bash
# Eval a plain GR00T N1.7 SFT checkpoint on LIBERO-Long.
# Usage: eval_n1d7_sft_ckpt.sh <ckpt_dir> <tag> [rollout_steps]
#   rollout_steps: max_steps_per_rollout_epoch.
#     2560 = full 100 episodes (10/task)   [default]
#      512 = quick 20 episodes (2/task)     -> bracketing sweep (~6 min)
set -o pipefail
REPO=/home/duynguyen/Desktop/RESEARCH/VR/RLinf
cd "$REPO"
source "$REPO/vla-rlft-n1d7/bin/activate"

CKPT="$1"
TAG="${2:-sft}"
STEPS="${3:-2560}"

export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1

LOG="/tmp/n17_sft_eval_${TAG}.log"
echo "=== SFT eval ${TAG} ckpt=${CKPT} rollout_steps=${STEPS} $(date) ===" | tee "$LOG"
bash evaluations/run_eval.sh libero libero_10_gr00t_n1d7_sft_eval \
  "rollout.model.model_path=${CKPT}" \
  "env.eval.max_steps_per_rollout_epoch=${STEPS}" \
  "runner.logger.experiment_name=eval_${TAG}" \
  "runner.logger.logger_backends=[tensorboard,wandb]" 2>&1 | tee -a "$LOG"
echo "SFT_EVAL_EXIT=${PIPESTATUS[0]}" | tee -a "$LOG"

# 20 envs, 512 steps/episode, 10 tasks interleaved trial-major:
# episodes = 20*(STEPS/512); trials/task = episodes/10 = STEPS/256.
RUNS=$(( STEPS / 256 )); [ "$RUNS" -lt 1 ] && RUNS=1
echo "=== SR (runs/task=${RUNS}) ===" | tee -a "$LOG"
python evaluations/libero/parse_per_task_sr.py "$LOG" --runs "$RUNS" --label "N1.7 SFT ${TAG}" 2>&1 | tee -a "$LOG"
