#!/usr/bin/env bash
# RECAP Phase 3 — 500-episode confirmatory evaluation (SE ~2%).
#
# Final claim numbers for RECAP on GR00T N1.7 / LIBERO-Long from the 44% SFT
# base. Three arms, 5 seeds x 100 episodes each (randomized reset states):
#   A. SFT base ckpt (no conditioning)          — baseline
#   B. phase2 cond-SFT, w=1.0 (beta=1, paper default operating point)
#   C. phase2 cond-SFT, w=2.0 (best guidance point, 71/100 at n=100)
# 100-ep history: base 44; phase2 63/55/71/58/66 (w=1.0/1.5/2.0/2.5/3.0);
# iter1b mixed retrain 52/52/54 (did not compound past phase2).
set -o pipefail
REPO=/home/duynguyen/Desktop/RESEARCH/VR/RLinf
cd "$REPO"
source "$REPO/vla-rlft-n1d7/bin/activate"
export REPO_PATH="$REPO" EMBODIED_PATH="$REPO/examples/embodiment" PYTHONPATH="$REPO:${PYTHONPATH:-}"
export HF_LEROBOT_HOME=/data/datasets HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1 MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

P2CKPT="$REPO/examples/recap/results/recap20k_phase2_condsft/checkpoints/global_step_5000/actor/model_state_dict/full_weights.pt"
SEEDS="0 1 2 3 4"
STATUS=/tmp/recap20k_phase3_status.log
mark(){ echo "=== $(date +%H:%M:%S) | $* ===" | tee -a "$STATUS"; }
echo "RUN_PHASE3_START $(date)" > "$STATUS"

run_arm() {  # name ckpt guidance
  local NAME=$1 CKPT=$2 W=$3 TOT=0 SUC=0
  for S in $SEEDS; do
    local LOG="/tmp/n17_phase3_${NAME}_s${S}.log"
    if ! grep -qE "libero eval\] task_id=" "$LOG" 2>/dev/null; then
      bash "$REPO/evaluations/run_eval.sh" libero libero_10_gr00t_n1d7_cfg_eval \
        "+runner.ckpt_path=${CKPT}" "rollout.model.cfg_guidance_weight=${W}" \
        "+rollout.model.conditioning=film" \
        "env.eval.use_ordered_reset_state_ids=false" "env.eval.seed=$S" \
        "env.eval.max_steps_per_rollout_epoch=2560" \
        "runner.logger.experiment_name=phase3_${NAME}_s${S}" \
        "runner.logger.logger_backends=[wandb,tensorboard]" \
        "runner.logger.project_name=recap-n1d7-from-base" 2>&1 | tee "$LOG" | tail -1
    fi
    local s t
    s=$(grep -cE "task_id=.*success=True" "$LOG"); t=$(grep -cE "libero eval\] task_id=" "$LOG")
    SUC=$((SUC + s)); TOT=$((TOT + t))
    mark "PHASE3 $NAME seed=$S -> ${s}/${t} (cum ${SUC}/${TOT})"
  done
  mark "PHASE3 ARM $NAME TOTAL ${SUC}/${TOT}"
}

# Arm B/C first (the numbers we care most about), baseline last.
run_arm "p2w2.0" "$P2CKPT" 2.0
run_arm "p2w1.0" "$P2CKPT" 1.0
# Arm A: SFT base through the same CFG eval path (w=1.0 single POS pass is a
# no-op for an unconditioned ckpt: the zero-init embedding is re-initialized
# at load and the head never trained to read it, so this measures plain SFT).
SFT_HF=/data/checkpoints/n1d7_sft_from_base_20k/n1d7_sft_from_base_20k/checkpoint-20000
for S in $SEEDS; do
  LOG="/tmp/n17_phase3_base_s${S}.log"
  if ! grep -qE "libero eval\] task_id=" "$LOG" 2>/dev/null; then
    bash "$REPO/evaluations/run_eval.sh" libero libero_10_gr00t_n1d7_sft_eval \
      "rollout.model.model_path=$SFT_HF" \
      "env.eval.use_ordered_reset_state_ids=false" "env.eval.seed=$S" \
      "env.eval.max_steps_per_rollout_epoch=2560" \
      "runner.logger.experiment_name=phase3_base_s${S}" \
      "runner.logger.logger_backends=[wandb,tensorboard]" \
      "runner.logger.project_name=recap-n1d7-from-base" 2>&1 | tee "$LOG" | tail -1
  fi
  s=$(grep -cE "task_id=.*success=True" "$LOG"); t=$(grep -cE "libero eval\] task_id=" "$LOG")
  mark "PHASE3 base seed=$S -> ${s}/${t}"
done
BS=$(cat /tmp/n17_phase3_base_s*.log | grep -cE "task_id=.*success=True")
BT=$(cat /tmp/n17_phase3_base_s*.log | grep -cE "libero eval\] task_id=")
mark "PHASE3 ARM base TOTAL ${BS}/${BT}"
mark "RUN_PHASE3 COMPLETE"
echo "RUN_PHASE3_EXIT=0" | tee -a "$STATUS"
