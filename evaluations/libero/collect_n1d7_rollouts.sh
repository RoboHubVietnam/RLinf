#!/bin/bash
# Task 2 stage 1: collect diverse GR00T N1.7 LIBERO-Long rollouts (successes+failures)
# for RECAP. Runs N passes with different seeds + random reset states so each pass
# yields different episodes. Pickles accumulate under per-run dirs.
set -o pipefail

REPO="/home/duynguyen/Desktop/RESEARCH/VR/RLinf"
cd "$REPO"
# shellcheck disable=SC1091
source "$REPO/vla-rlft-n1d7/bin/activate"

export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

N_PASSES=${1:-6}
OUT_ROOT=/data/datasets/n1d7_rollouts_pkl
LOG=/tmp/n17_collect.log
mkdir -p "$OUT_ROOT"
echo "=== N1.7 rollout collection start $(date), passes=$N_PASSES ===" | tee "$LOG"

for k in $(seq 0 $((N_PASSES-1))); do
  echo "=== collection pass $k (seed=$k) ===" | tee -a "$LOG"
  bash evaluations/run_eval.sh libero libero_10_gr00t_n1d7_collect \
    env.eval.seed="$k" \
    env.eval.data_collection.save_dir="$OUT_ROOT/run_$k" \
    runner.logger.log_path="$OUT_ROOT/_logs/run_$k" 2>&1 | tee -a "$LOG"
  echo "pass $k done; pickles so far: $(find "$OUT_ROOT" -name '*.pkl' | wc -l)" | tee -a "$LOG"
done

echo "TOTAL pickles: $(find "$OUT_ROOT" -name '*.pkl' | wc -l)" | tee -a "$LOG"
echo "  success: $(find "$OUT_ROOT" -name '*success.pkl' | wc -l)  fail: $(find "$OUT_ROOT" -name '*fail.pkl' | wc -l)" | tee -a "$LOG"
echo "COLLECT_EXIT=0" | tee -a "$LOG"
