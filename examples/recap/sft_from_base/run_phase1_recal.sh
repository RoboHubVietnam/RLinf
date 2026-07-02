#!/usr/bin/env bash
# RECAP Phase 1 — "proper RECAP" fixes on existing data (no new collection):
#   fix A: zero-init advantage embedding (adaLN-zero) — patched in
#          rlinf/models/embodiment/gr00t/gr00t_n1d7_cfg/ (and n1d5_cfg)
#   fix B: paper-faithful advantage calibration — demos thresholded to ~30%
#          positive, rollouts to ~40% (was: demos forced 100%, rollouts 19%)
#   fix C: train on the AGGREGATED pool (iter1 + iter2 rollouts + demos)
#
# Stages (resumable — each skips if its output exists):
#   1. advantages: recompute with per-type quantiles -> advantages_recal.parquet
#   2. cfg       : CFG-film train from SFT base on aggregated pool (3000 steps)
#   3. confirm   : 100-ep eval at w=1.0/1.5/2.0 (wandb)
#
# Success criteria vs history (base=44, iter1 w1.0/1.5=38/49, iter2 w1.0/1.5/2.0=38/42/45):
#   - w=1.0 >= ~44 (forgetting gone: unconditional pass preserved by zero-init)
#   - monotone-ish w response (guidance direction learned, not random)
set -o pipefail
REPO=/home/duynguyen/Desktop/RESEARCH/VR/RLinf
cd "$REPO"
N1D7_ACT="$REPO/vla-rlft-n1d7/bin/activate"
OPENPI_ACT="$REPO/vla-rlft-openpi/bin/activate"

SFT_CKPT=/data/checkpoints/n1d7_sft_from_base_20k/n1d7_sft_from_base_20k/checkpoint-20000
DEMOS=/data/datasets/libero_10_no_noops_lerobot
DS1=/data/datasets/recap20k_iter1
DS2=/data/datasets/recap20k_iter2
CFGEXP=recap20k_phase1_recal
CFGOUT="$REPO/examples/recap/results/${CFGEXP}/checkpoints"
NEWCK="$CFGOUT/global_step_3000/actor/model_state_dict/full_weights.pt"
STATUS=/tmp/recap20k_phase1_status.log

common_env() {
  export REPO_PATH="$REPO" EMBODIED_PATH="$REPO/examples/embodiment" PYTHONPATH="$REPO:${PYTHONPATH:-}"
  export HF_LEROBOT_HOME=/data/datasets HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  export TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1 MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
}
mark(){ echo "=== $(date +%H:%M:%S) | $* ===" | tee -a "$STATUS"; }

echo "RUN_PHASE1_START $(date)" > "$STATUS"
mark "config: zero-init + per-type quantiles (rollout 0.4 / sft 0.3) + aggregated pool"

# ---- 1. ADVANTAGES recompute (per-type calibration, tag=recal) ----
if [ -f "$DS1/meta/advantages_recal.parquet" ] && [ -f "$DS2/meta/advantages_recal.parquet" ] \
   && [ -f "$DEMOS/meta/advantages_recal.parquet" ]; then
  mark "STAGE 1 advantages SKIP (advantages_recal.parquet exist)"
else
  mark "STAGE 1 advantages: recalibrate over iter1+iter2+demos (value=iter2)"
  ( source "$OPENPI_ACT"; common_env; cd "$REPO/examples/recap/process"
    torchrun --nproc_per_node=1 compute_advantages.py \
      --config-name compute_advantages_n1d7_phase1 2>&1 \
      | tee /tmp/recap20k_phase1_adv.log | tail -20 )
  if [ ! -f "$DEMOS/meta/advantages_recal.parquet" ]; then
    mark "ABORT: advantages_recal.parquet missing"; exit 1
  fi
fi
# Sanity: report positive fractions (paper targets: rollouts ~40%, demos ~30%)
( source "$OPENPI_ACT"
  for D in "$DS1" "$DS2" "$DEMOS"; do
    python - "$D" <<'EOF'
import sys, pandas as pd
d = sys.argv[1]
df = pd.read_parquet(f"{d}/meta/advantages_recal.parquet")
print(f"POSFRAC {d.split('/')[-1]}: {df['advantage'].mean()*100:.1f}% of {len(df)}")
EOF
  done ) 2>&1 | grep POSFRAC | while read -r L; do mark "$L"; done
mark "STAGE 1 advantages done"

# ---- 2. CFG train (film, zero-init) from SFT base on AGGREGATED pool ----
if [ -f "$NEWCK" ]; then
  mark "STAGE 2 CFG SKIP (ckpt exists)"
else
  mark "STAGE 2 CFG (film, zero-init) train on iter1+iter2+demos, tag=recal (3000 steps)"
  ( source "$N1D7_ACT"; common_env; cd "$REPO/examples/recap/cfg"
    python train_cfg.py --config-name libero_cfg_gr00t_n1d7 \
      "actor.model.model_path=$SFT_CKPT" \
      "data.train_data_paths=[{dataset_path:$DS1,type:rollout,weight:1.0},{dataset_path:$DS2,type:rollout,weight:1.0},{dataset_path:$DEMOS,type:sft,weight:1.0}]" \
      data.advantage_tag=recal \
      "runner.logger.experiment_name=$CFGEXP" \
      "runner.logger.project_name=recap-n1d7-from-base" \
      "runner.max_steps=3000" "runner.save_interval=500" \
      "actor.optim.total_training_steps=3000" "actor.optim.lr_warmup_steps=300" 2>&1 | tail -4 )
  if [ ! -f "$NEWCK" ]; then mark "ABORT: CFG ckpt missing ($NEWCK)"; exit 1; fi
fi
mark "STAGE 2 CFG done: $NEWCK"

# ---- 3. 100-ep confirm at w=1.0/1.5/2.0 (wandb) ----
mark "STAGE 3 100-ep confirm (w=1.0/1.5/2.0)"
( source "$N1D7_ACT"; common_env
  for W in 1.0 1.5 2.0; do
    LOG="/tmp/n17_phase1_confirm_w${W}.log"
    bash "$REPO/evaluations/run_eval.sh" libero libero_10_gr00t_n1d7_cfg_eval \
      "+runner.ckpt_path=${NEWCK}" "rollout.model.cfg_guidance_weight=${W}" \
      "+rollout.model.conditioning=film" "env.eval.max_steps_per_rollout_epoch=2560" \
      "runner.logger.experiment_name=phase1_recal_w${W}" \
      "runner.logger.logger_backends=[wandb,tensorboard]" \
      "runner.logger.project_name=recap-n1d7-from-base" 2>&1 | tee "$LOG" | tail -2
    s=$(grep -cE "task_id=.*success=True" "$LOG"); t=$(grep -cE "libero eval\] task_id=" "$LOG")
    mark "PHASE1 CONFIRM w=$W -> ${s}/${t}"
  done )
mark "RUN_PHASE1 COMPLETE (history: base=44, iter1 w1.0/1.5=38/49, iter2 w1.0/1.5/2.0=38/42/45)"
echo "RUN_PHASE1_EXIT=0" | tee -a "$STATUS"
