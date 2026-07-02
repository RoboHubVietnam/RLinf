#!/usr/bin/env bash
# RECAP round-1, MIXED variant: CFG-train the 44% base on rollouts + expert SFT
# demos, to counter the CFG forgetting that flatlined the rollouts-only run.
#
# Stages (resumable — each skips if its output exists):
#   1. prep    : give the demo dataset a returns_rollout sidecar (= its returns_sft)
#   2. advantages: recompute advantages over BOTH datasets together (global 30% threshold,
#                  round-1 value head, workers=0) -> demos land positive, rollout-fails negative
#   3. cfg     : train CFG-film from checkpoint-20000 on the mixed data (3000 steps, save 500)
#   4. sweep   : guidance sweep (w=1.0/1.5/2.0/2.5, 20 ep) on the result
set -o pipefail
REPO=/home/duynguyen/Desktop/RESEARCH/VR/RLinf
cd "$REPO"
N1D7_ACT="$REPO/vla-rlft-n1d7/bin/activate"
OPENPI_ACT="$REPO/vla-rlft-openpi/bin/activate"
ROLLOUTS=/data/datasets/recap20k_iter1
DEMOS=/data/datasets/libero_10_no_noops_lerobot
VALCK=/data/checkpoints/recap20k_iter1_value/value_iter1/checkpoints/global_step_1500
SFT_CKPT=/data/checkpoints/n1d7_sft_from_base_20k/n1d7_sft_from_base_20k/checkpoint-20000
CFGEXP=recap20k_iter1_film_mixed
CFGOUT="$REPO/examples/recap/results/${CFGEXP}/checkpoints"
NEWCK="$CFGOUT/global_step_3000/actor/model_state_dict/full_weights.pt"
STATUS=/tmp/recap20k_mixed_status.log

common_env() {
  export REPO_PATH="$REPO" EMBODIED_PATH="$REPO/examples/embodiment" PYTHONPATH="$REPO:${PYTHONPATH:-}"
  export HF_LEROBOT_HOME=/data/datasets HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  export TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1 MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
}
mark(){ echo "=== $(date +%H:%M:%S) | $* ===" | tee -a "$STATUS"; }

echo "RUN_MIXED_START $(date)" > "$STATUS"

# ---- 1. prep demo returns_rollout sidecar (returns are trajectory-fixed: = returns_sft) ----
if [ -f "$DEMOS/meta/returns_rollout.parquet" ]; then
  mark "STAGE 1 prep SKIP (demo returns_rollout exists)"
else
  mark "STAGE 1 prep: demo returns_sft -> returns_rollout"
  cp "$DEMOS/meta/returns_sft.parquet" "$DEMOS/meta/returns_rollout.parquet"
fi

# ---- 2. recompute advantages over BOTH datasets (global threshold) ----
# back up the rollouts-only advantages first (regenerable, but keep for comparison)
[ -f "$ROLLOUTS/meta/advantages_rollout.parquet" ] && [ ! -f "$ROLLOUTS/meta/advantages_rollout.rollouts_only.bak" ] && \
  cp "$ROLLOUTS/meta/advantages_rollout.parquet" "$ROLLOUTS/meta/advantages_rollout.rollouts_only.bak"
if [ -f "$DEMOS/meta/advantages_rollout.parquet" ] && [ -f "$ROLLOUTS/meta/advantages_rollout.mixed.done" ]; then
  mark "STAGE 2 advantages SKIP (mixed advantages exist)"
else
  mark "STAGE 2 advantages: compute over rollouts+demos (workers=0, value=round1)"
  ( source "$OPENPI_ACT"; common_env; cd "$REPO/examples/recap/process"
    torchrun --nproc_per_node=1 compute_advantages.py --config-name compute_advantages_n1d7_mixed \
      "advantage.value_checkpoint=$VALCK" "advantage.tag=rollout" "advantage.returns_tag=rollout" \
      "advantage.num_dataloader_workers_per_gpu=0" \
      "data.return_min=-811.0" "data.return_max=0.0" 2>&1 | tail -12 )
  if [ ! -f "$DEMOS/meta/advantages_rollout.parquet" ]; then mark "ABORT: demo advantages missing"; exit 1; fi
  touch "$ROLLOUTS/meta/advantages_rollout.mixed.done"
  mark "STAGE 2 advantages done"
fi

# ---- 3. CFG train on mixed data from checkpoint-20000 ----
if [ -f "$NEWCK" ]; then
  mark "STAGE 3 CFG SKIP (ckpt exists)"
else
  mark "STAGE 3 CFG (film) train on MIXED data (3000 steps, save 500)"
  ( source "$N1D7_ACT"; common_env; cd "$REPO/examples/recap/cfg"
    python train_cfg.py --config-name libero_cfg_gr00t_n1d7 \
      "actor.model.model_path=$SFT_CKPT" \
      "data.train_data_paths=[{dataset_path:$ROLLOUTS,type:rollout,weight:1.0},{dataset_path:$DEMOS,type:sft,weight:1.0}]" \
      data.advantage_tag=rollout \
      "runner.logger.experiment_name=$CFGEXP" \
      "runner.max_steps=3000" "runner.save_interval=500" \
      "actor.optim.total_training_steps=3000" "actor.optim.lr_warmup_steps=300" 2>&1 | tail -4 )
  if [ ! -f "$NEWCK" ]; then mark "ABORT: CFG ckpt missing ($NEWCK)"; exit 1; fi
  mark "STAGE 3 CFG done: $NEWCK"
fi

# ---- 4. guidance sweep on the mixed result ----
mark "STAGE 4 guidance sweep (w=1.0/1.5/2.0/2.5, 20 ep)"
( source "$N1D7_ACT"; common_env
  for W in 1.0 1.5 2.0 2.5; do
    LOG="/tmp/n17_mixed_gsweep_w${W}.log"
    bash "$REPO/evaluations/run_eval.sh" libero libero_10_gr00t_n1d7_cfg_eval \
      "+runner.ckpt_path=${NEWCK}" "rollout.model.cfg_guidance_weight=${W}" \
      "+rollout.model.conditioning=film" "env.eval.max_steps_per_rollout_epoch=512" \
      "runner.logger.experiment_name=mixed_iter1_w${W}" \
      "runner.logger.logger_backends=[tensorboard]" 2>&1 | tee "$LOG" | tail -2
    s=$(grep -cE "task_id=.*success=True" "$LOG" 2>/dev/null); t=$(grep -cE "libero eval\] task_id=" "$LOG" 2>/dev/null)
    mark "MIXED w=$W -> ${s}/${t}"
  done )
mark "RUN_MIXED COMPLETE"
echo "RUN_MIXED_EXIT=0" | tee -a "$STATUS"
