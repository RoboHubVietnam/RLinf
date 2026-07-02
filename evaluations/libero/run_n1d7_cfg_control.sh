#!/bin/bash
# Task 1: GR00T N1.7 SFT eval on LIBERO-Long, 10 runs/task, per-task SR.
set -o pipefail

REPO="/home/duynguyen/Desktop/RESEARCH/VR/RLinf"
cd "$REPO"
# shellcheck disable=SC1091
source "$REPO/vla-rlft-n1d7/bin/activate"

export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1

LOG="/tmp/n17_cfg_control_eval.log"
echo "=== CFG control launch $(date) ===" | tee "$LOG"
bash evaluations/run_eval.sh libero libero_10_gr00t_n1d7_cfg_eval 2>&1 | tee -a "$LOG"
echo "EVAL_EXIT=${PIPESTATUS[0]}" | tee -a "$LOG"
