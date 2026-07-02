#!/usr/bin/env bash
# RECAP Phase 2 — advantage-CONDITIONED SFT (the paper's pretraining stage).
#
# Phase 1 + 1b established that bolt-on CFG cannot work here:
#   - head fine-tune at lr=2e-6: embedding never trains (|POS-NULL|=0.0013),
#     rollout data drags w=1.0 to 35-38 vs 44 base (forgetting)
#   - embedding-only at lr=1e-3: embedding trains but learns a destructive
#     common-mode temb bias (norm 7.5) -> 12/10/6/7 across w
# The missing ingredient is JOINT training: the head must learn to consume the
# conditioning while it trains, at SFT-scale lr, anchored on expert data —
# exactly RECAP's advantage-conditioned pretraining (arXiv:2511.14759).
#
# This run: continue from the 44% SFT ckpt on DEMOS ONLY (paper pretrains on
# the demo corpus; rollouts enter in later RL iterations), recalibrated labels
# (30% of demo steps positive), CFG dropout 0.3, joint lr=1e-4 (the SFT lr —
# on demo data this is plain continued SFT plus conditioning), 5000 steps.
#
# Read: w=1.0 should hold >= ~44 (it is just more SFT); a w>1 lift above the
# w=1.0 point = conditioning works -> proceed to RL iteration + Phase 3.
set -o pipefail
REPO=/home/duynguyen/Desktop/RESEARCH/VR/RLinf
cd "$REPO"
N1D7_ACT="$REPO/vla-rlft-n1d7/bin/activate"

SFT_CKPT=/data/checkpoints/n1d7_sft_from_base_20k/n1d7_sft_from_base_20k/checkpoint-20000
DEMOS=/data/datasets/libero_10_no_noops_lerobot
CFGEXP=recap20k_phase2_condsft
CFGOUT="$REPO/examples/recap/results/${CFGEXP}/checkpoints"
NEWCK="$CFGOUT/global_step_5000/actor/model_state_dict/full_weights.pt"
STATUS=/tmp/recap20k_phase2_status.log

common_env() {
  export REPO_PATH="$REPO" EMBODIED_PATH="$REPO/examples/embodiment" PYTHONPATH="$REPO:${PYTHONPATH:-}"
  export HF_LEROBOT_HOME=/data/datasets HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  export TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1 MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
}
mark(){ echo "=== $(date +%H:%M:%S) | $* ===" | tee -a "$STATUS"; }

echo "RUN_PHASE2_START $(date)" > "$STATUS"
mark "config: conditioned SFT, demos-only, recal labels (30% pos), joint lr=1e-4, 5000 steps"

[ -f "$DEMOS/meta/advantages_recal.parquet" ] || { mark "ABORT: demos advantages_recal missing"; exit 1; }

# ---- 1. Conditioned-SFT train (joint head + embedding, demos only) ----
if [ -f "$NEWCK" ]; then
  mark "STAGE 1 cond-SFT SKIP (ckpt exists)"
else
  mark "STAGE 1 cond-SFT (film) joint train on demos, lr=1e-4 (5000 steps)"
  ( source "$N1D7_ACT"; common_env; cd "$REPO/examples/recap/cfg"
    python train_cfg.py --config-name libero_cfg_gr00t_n1d7 \
      "actor.model.model_path=$SFT_CKPT" \
      "data.train_data_paths=[{dataset_path:$DEMOS,type:sft,weight:1.0}]" \
      data.advantage_tag=recal \
      "runner.logger.experiment_name=$CFGEXP" \
      "runner.logger.project_name=recap-n1d7-from-base" \
      "runner.max_steps=5000" "runner.save_interval=1000" \
      "actor.optim.lr=1.0e-4" \
      "actor.optim.total_training_steps=5000" "actor.optim.lr_warmup_steps=200" 2>&1 | tail -4 )
  if [ ! -f "$NEWCK" ]; then mark "ABORT: cond-SFT ckpt missing ($NEWCK)"; exit 1; fi
fi
mark "STAGE 1 cond-SFT done: $NEWCK"

# Diagnostic: embedding magnitudes (expect moderate norms and real separation)
( source "$N1D7_ACT"; python - "$NEWCK" <<'EOF'
import sys, torch
sd = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
k = [x for x in sd if "advantage_embedding" in x][0]
w = sd[k].float()
print(f"EMBDIAG norms NULL={w[0].norm():.4f} NEG={w[1].norm():.4f} POS={w[2].norm():.4f} "
      f"|POS-NULL|={(w[2]-w[0]).norm():.4f}")
EOF
) 2>&1 | grep EMBDIAG | while read -r L; do mark "$L"; done

# ---- 2. 100-ep confirm at w=1.0/1.5/2.0 (wandb) ----
mark "STAGE 2 100-ep confirm (w=1.0/1.5/2.0)"
( source "$N1D7_ACT"; common_env
  for W in 1.0 1.5 2.0; do
    LOG="/tmp/n17_phase2_confirm_w${W}.log"
    bash "$REPO/evaluations/run_eval.sh" libero libero_10_gr00t_n1d7_cfg_eval \
      "+runner.ckpt_path=${NEWCK}" "rollout.model.cfg_guidance_weight=${W}" \
      "+rollout.model.conditioning=film" "env.eval.max_steps_per_rollout_epoch=2560" \
      "runner.logger.experiment_name=phase2_condsft_w${W}" \
      "runner.logger.logger_backends=[wandb,tensorboard]" \
      "runner.logger.project_name=recap-n1d7-from-base" 2>&1 | tee "$LOG" | tail -2
    s=$(grep -cE "task_id=.*success=True" "$LOG"); t=$(grep -cE "libero eval\] task_id=" "$LOG")
    mark "PHASE2 CONFIRM w=$W -> ${s}/${t}"
  done )
mark "RUN_PHASE2 COMPLETE (base=44; bolt-on attempts: 35-38 head-tune, 12 emb-only)"
echo "RUN_PHASE2_EXIT=0" | tee -a "$STATUS"
