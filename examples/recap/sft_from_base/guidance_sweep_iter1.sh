#!/usr/bin/env bash
# Guidance sweep on the round-1 RECAP-film checkpoint (from the 44% base).
# Evals w in {1.0,1.5,2.0,2.5} at 20 episodes (2/task) to see whether RECAP
# helped at ANY guidance level vs the 44% SFT base, and isolate whether a
# degradation is from over-extrapolation (w>1) or from CFG training itself (w=1.0).
set -o pipefail
REPO=/home/duynguyen/Desktop/RESEARCH/VR/RLinf
cd "$REPO"
source "$REPO/vla-rlft-n1d7/bin/activate"
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1

CKPT="$REPO/examples/recap/results/recap20k_iter1_film/checkpoints/global_step_3000/actor/model_state_dict/full_weights.pt"
STEPS=512   # 20 episodes (2/task)
SUM=/tmp/recap20k_guidance_sweep.txt
: > "$SUM"

for W in 1.0 1.5 2.0 2.5; do
  TAG="g${W}"
  LOG="/tmp/n17_gsweep_${TAG}.log"
  echo "######## guidance w=$W ########"
  bash evaluations/run_eval.sh libero libero_10_gr00t_n1d7_cfg_eval \
    "+runner.ckpt_path=${CKPT}" \
    "rollout.model.cfg_guidance_weight=${W}" \
    "+rollout.model.conditioning=film" \
    "env.eval.max_steps_per_rollout_epoch=${STEPS}" \
    "runner.logger.experiment_name=gsweep_iter1_w${W}" \
    "runner.logger.logger_backends=[tensorboard]" 2>&1 | tee "$LOG" | tail -2
  succ=$(grep -cE "libero eval\] task_id=.*success=True" "$LOG" 2>/dev/null)
  tot=$(grep -cE "libero eval\] task_id=" "$LOG" 2>/dev/null)
  echo "w=$W -> ${succ}/${tot}" | tee -a "$SUM"
done
echo "GSWEEP_DONE" | tee -a "$SUM"
echo "==== SUMMARY (base SFT = 44/100 = 44%) ===="; cat "$SUM"
