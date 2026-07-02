#!/usr/bin/env bash
# RECAP Phase 1b — embedding-only (prompt-tuning-style) CFG conditioning.
#
# Phase 1 diagnosis: at lr=2e-6 the advantage embedding NEVER trains
# (|POS-NULL| = 0.0013 after 3000 steps from zero-init; the old std=0.02 runs
# kept their random init exactly), so all previous guidance responses were
# noise along a frozen random direction. Meanwhile fine-tuning the head on
# rollout-heavy data drags w=1.0 to 35-38 vs the 44 base (forgetting).
#
# Phase 1b removes both confounders in one experiment:
#   - train ONLY the advantage embedding (+actor.model.train_advantage_embedding_only)
#     -> the base flow field cannot be forgotten, by construction
#   - lr=1e-3 on that one 3x1536 param -> the conditioning actually trains
#   - same recalibrated labels (advantages_recal: rollouts ~40%, demos 30%)
#
# Interpretation:
#   - w=1.0 should sit near the 44 base (POS bias is the only delta)
#   - if w>1 climbs above 44 -> FIRST clean guidance win -> green-light Phase 2
#   - if flat at ~44 -> temb-bias capacity insufficient; Phase 2 must train
#     conditioning during SFT (or switch to token mode)
set -o pipefail
REPO=/home/duynguyen/Desktop/RESEARCH/VR/RLinf
cd "$REPO"
N1D7_ACT="$REPO/vla-rlft-n1d7/bin/activate"

SFT_CKPT=/data/checkpoints/n1d7_sft_from_base_20k/n1d7_sft_from_base_20k/checkpoint-20000
DEMOS=/data/datasets/libero_10_no_noops_lerobot
DS1=/data/datasets/recap20k_iter1
DS2=/data/datasets/recap20k_iter2
CFGEXP=recap20k_phase1b_embonly
CFGOUT="$REPO/examples/recap/results/${CFGEXP}/checkpoints"
NEWCK="$CFGOUT/global_step_3000/actor/model_state_dict/full_weights.pt"
STATUS=/tmp/recap20k_phase1b_status.log

common_env() {
  export REPO_PATH="$REPO" EMBODIED_PATH="$REPO/examples/embodiment" PYTHONPATH="$REPO:${PYTHONPATH:-}"
  export HF_LEROBOT_HOME=/data/datasets HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  export TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1 MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
}
mark(){ echo "=== $(date +%H:%M:%S) | $* ===" | tee -a "$STATUS"; }

echo "RUN_PHASE1B_START $(date)" > "$STATUS"
mark "config: embedding-only (frozen head), lr=1e-3, recal labels, aggregated pool"

for F in "$DS1/meta/advantages_recal.parquet" "$DS2/meta/advantages_recal.parquet" \
         "$DEMOS/meta/advantages_recal.parquet"; do
  [ -f "$F" ] || { mark "ABORT: missing $F (run Phase 1 stage 1 first)"; exit 1; }
done

# ---- 1. CFG train, advantage embedding only ----
if [ -f "$NEWCK" ]; then
  mark "STAGE 1 CFG SKIP (ckpt exists)"
else
  mark "STAGE 1 CFG (film) train, embedding-only, lr=1e-3 (3000 steps)"
  ( source "$N1D7_ACT"; common_env; cd "$REPO/examples/recap/cfg"
    python train_cfg.py --config-name libero_cfg_gr00t_n1d7 \
      "actor.model.model_path=$SFT_CKPT" \
      "+actor.model.train_advantage_embedding_only=true" \
      "data.train_data_paths=[{dataset_path:$DS1,type:rollout,weight:1.0},{dataset_path:$DS2,type:rollout,weight:1.0},{dataset_path:$DEMOS,type:sft,weight:1.0}]" \
      data.advantage_tag=recal \
      "runner.logger.experiment_name=$CFGEXP" \
      "runner.logger.project_name=recap-n1d7-from-base" \
      "runner.max_steps=3000" "runner.save_interval=500" \
      "actor.optim.lr=1.0e-3" \
      "actor.optim.total_training_steps=3000" "actor.optim.lr_warmup_steps=100" 2>&1 | tail -4 )
  if [ ! -f "$NEWCK" ]; then mark "ABORT: CFG ckpt missing ($NEWCK)"; exit 1; fi
fi
mark "STAGE 1 CFG done: $NEWCK"

# Diagnostic: confirm the embedding actually trained this time
( source "$N1D7_ACT"; python - "$NEWCK" <<'EOF'
import sys, torch
sd = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
k = [x for x in sd if "advantage_embedding" in x][0]
w = sd[k].float()
print(f"EMBDIAG norms NULL={w[0].norm():.4f} NEG={w[1].norm():.4f} POS={w[2].norm():.4f} "
      f"|POS-NULL|={(w[2]-w[0]).norm():.4f}")
EOF
) 2>&1 | grep EMBDIAG | while read -r L; do mark "$L"; done

# ---- 2. 100-ep confirm at w=1.0/1.5/2.0/3.0 (wandb) ----
mark "STAGE 2 100-ep confirm (w=1.0/1.5/2.0/3.0)"
( source "$N1D7_ACT"; common_env
  for W in 1.0 1.5 2.0 3.0; do
    LOG="/tmp/n17_phase1b_confirm_w${W}.log"
    bash "$REPO/evaluations/run_eval.sh" libero libero_10_gr00t_n1d7_cfg_eval \
      "+runner.ckpt_path=${NEWCK}" "rollout.model.cfg_guidance_weight=${W}" \
      "+rollout.model.conditioning=film" "env.eval.max_steps_per_rollout_epoch=2560" \
      "runner.logger.experiment_name=phase1b_embonly_w${W}" \
      "runner.logger.logger_backends=[wandb,tensorboard]" \
      "runner.logger.project_name=recap-n1d7-from-base" 2>&1 | tee "$LOG" | tail -2
    s=$(grep -cE "task_id=.*success=True" "$LOG"); t=$(grep -cE "libero eval\] task_id=" "$LOG")
    mark "PHASE1B CONFIRM w=$W -> ${s}/${t}"
  done )
mark "RUN_PHASE1B COMPLETE (base=44; phase1 head-tune w1.0/1.5/2.0=35/36/37)"
echo "RUN_PHASE1B_EXIT=0" | tee -a "$STATUS"
