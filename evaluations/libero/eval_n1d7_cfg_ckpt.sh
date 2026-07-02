#!/bin/bash
# Eval a trained GR00T N1.7 CFG checkpoint on LIBERO-Long, 10 runs/task.
# Usage: eval_n1d7_cfg_ckpt.sh <full_weights.pt> <log_tag> [cfg_guidance_weight] [conditioning]
#   conditioning: "film" (default) or "token" — MUST match how the ckpt was trained.
set -o pipefail

REPO="/home/duynguyen/Desktop/RESEARCH/VR/RLinf"
cd "$REPO"
# shellcheck disable=SC1091
source "$REPO/vla-rlft-n1d7/bin/activate"

CKPT="$1"
TAG="${2:-cfg}"
W="${3:-1.0}"
COND="${4:-film}"

export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1

LOG="/tmp/n17_cfg_eval_${TAG}.log"
echo "=== CFG eval ${TAG} ckpt=${CKPT} w=${W} cond=${COND} $(date) ===" | tee "$LOG"
bash evaluations/run_eval.sh libero libero_10_gr00t_n1d7_cfg_eval \
  "+runner.ckpt_path=${CKPT}" \
  "rollout.model.cfg_guidance_weight=${W}" \
  "+rollout.model.conditioning=${COND}" \
  "runner.logger.experiment_name=eval_${TAG}" 2>&1 | tee -a "$LOG"
echo "CFG_EVAL_EXIT=${PIPESTATUS[0]}" | tee -a "$LOG"
