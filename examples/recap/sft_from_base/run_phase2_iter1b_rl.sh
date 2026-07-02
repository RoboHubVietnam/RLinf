#!/usr/bin/env bash
# RECAP Phase 2, RL iteration 1b — corrected collection guidance.
#
# Iteration 1a FAILED (0/100 at all w): collection ran at w=2.0 (CFG
# extrapolation), producing off-manifold "overly aggressive" actions
# (translation std up to +70% vs w=1.5 data, saturated at +/-1 — exactly the
# failure mode the RECAP paper warns about for high beta). Training jointly at
# lr=1e-4 on 50% such data taught the policy saturated garbage. Post-mortems
# ruled out: NaN weights (none), temporal misalignment (lag-0 peak), eval
# infra (phase2 evals fine).
#
# Fix, per paper (arXiv:2511.14759): COLLECT AT beta=1 (w=1.0, single positive
# -conditioned pass, on-manifold sampling). The phase2 policy at w=1.0 is 63%
# — still far better data than any earlier round. Guidance w>1 remains an
# INFERENCE-only lever; never train on w>1 actions.
#
# Stages (resumable): collect(w=1.0) -> convert -> returns -> value ->
# advantages(recalp2b) -> cond-SFT retrain from SFT anchor -> 100-ep confirm.
set -o pipefail
REPO=/home/duynguyen/Desktop/RESEARCH/VR/RLinf
cd "$REPO"
N1D7_ACT="$REPO/vla-rlft-n1d7/bin/activate"
OPENPI_ACT="$REPO/vla-rlft-openpi/bin/activate"

SFT_CKPT=/data/checkpoints/n1d7_sft_from_base_20k/n1d7_sft_from_base_20k/checkpoint-20000
P2CKPT="$REPO/examples/recap/results/recap20k_phase2_condsft/checkpoints/global_step_5000/actor/model_state_dict/full_weights.pt"
DEMOS=/data/datasets/libero_10_no_noops_lerobot
COLLECT_W=1.0
PKL=/data/datasets/recap20k_p2iter1b_pkl
REPOID=recap20k_p2iter1b
DS=/data/datasets/$REPOID
VALDIR=/data/checkpoints/recap20k_p2iter1b_value
CFGEXP=recap20k_phase2_iter1b
CFGOUT="$REPO/examples/recap/results/${CFGEXP}/checkpoints"
NEWCK="$CFGOUT/global_step_5000/actor/model_state_dict/full_weights.pt"
COLLECT_SEEDS="0 1"
STATUS=/tmp/recap20k_p2iter1b_status.log

common_env() {
  export REPO_PATH="$REPO" EMBODIED_PATH="$REPO/examples/embodiment" PYTHONPATH="$REPO:${PYTHONPATH:-}"
  export HF_LEROBOT_HOME=/data/datasets HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  export TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1 MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
}
mark(){ echo "=== $(date +%H:%M:%S) | $* ===" | tee -a "$STATUS"; }

echo "RUN_P2ITER1B_START $(date)" > "$STATUS"
mark "config: collect at w=$COLLECT_W (paper beta=1, on-manifold), retrain joint lr=1e-4"

# ---- 1. COLLECT from the conditioned policy at w=1.0 ----
NP=$(find "$PKL" -name '*.pkl' 2>/dev/null | wc -l)
if [ "$NP" -ge 100 ]; then
  mark "STAGE 1 collect SKIP (found $NP pkls)"
else
  mark "STAGE 1 collect: phase2 cond-SFT policy (film, w=$COLLECT_W)"
  ( source "$N1D7_ACT"; common_env
    for S in $COLLECT_SEEDS; do
      bash "$REPO/evaluations/run_eval.sh" libero libero_10_gr00t_n1d7_cfg_eval \
        "+runner.ckpt_path=$P2CKPT" \
        "+rollout.model.conditioning=film" \
        "rollout.model.cfg_guidance_weight=$COLLECT_W" \
        "env.eval.use_ordered_reset_state_ids=false" \
        "env.eval.seed=$S" \
        "env.eval.total_num_envs=20" \
        "env.eval.max_steps_per_rollout_epoch=2560" \
        "+env.eval.data_collection.enabled=true" \
        "+env.eval.data_collection.save_dir=$PKL/run_$S" \
        "+env.eval.data_collection.export_format=pickle" \
        "+env.eval.data_collection.robot_type=libero" \
        "+env.eval.data_collection.only_success=false" \
        "+env.eval.data_collection.fps=20" \
        "+env.eval.data_collection.finalize_interval=50" \
        "runner.logger.log_path=$PKL/_logs/run_$S" 2>&1 | tail -2
    done )
  NP=$(find "$PKL" -name '*.pkl' 2>/dev/null | wc -l)
  NS=$(find "$PKL" -name '*success*.pkl' 2>/dev/null | wc -l)
  mark "STAGE 1 collect done: $NP episodes ($NS success)"
fi
if [ "$NP" -lt 20 ]; then mark "ABORT: too few rollouts ($NP)"; exit 1; fi

# ---- 2. CONVERT ----
if [ -f "$DS/meta/episodes.jsonl" ]; then
  mark "STAGE 2 convert SKIP (dataset exists)"
else
  mark "STAGE 2 convert -> LeRobot v2.1"
  ( source "$OPENPI_ACT"; common_env
    python "$REPO/examples/recap/process/convert_rollouts_to_lerobot.py" \
      --pkl_dir "$PKL" --repo_id "$REPOID" --sft_modality "$DEMOS/meta/modality.json" 2>&1 | tail -2 )
  if [ ! -f "$DS/meta/episodes.jsonl" ]; then mark "ABORT: convert produced no dataset"; exit 1; fi
fi

# ---- 3. RETURNS ----
mark "STAGE 3 compute_returns"
( source "$OPENPI_ACT"; common_env; cd "$REPO/examples/recap/process"
  python compute_returns.py --config-name compute_returns_rollout \
    "data.train_data_paths.0.dataset_path=$DS" data.tag=rollout 2>&1 | tee /tmp/recap20k_p2iter1b_returns.log | tail -2 )
RMIN=$(grep -oE "return: min=-?[0-9.]+" /tmp/recap20k_p2iter1b_returns.log | tail -1 | grep -oE -- "-?[0-9.]+")
RMIN=${RMIN:--811.0}
mark "STAGE 3 returns done: return_min=$RMIN"

# ---- 4. VALUE critic ----
VALCK=$(ls -d "$VALDIR"/value_p2iter1b/checkpoints/global_step_* 2>/dev/null | sort -t_ -k3 -n | tail -1)
if [ -n "$VALCK" ] && [ -f "$VALCK/actor/model_state_dict/full_weights.pt" ]; then
  mark "STAGE 4 value SKIP (found $VALCK)"
else
  mark "STAGE 4 train value critic (max_steps=1500)"
  ( source "$OPENPI_ACT"; common_env; cd "$REPO/examples/recap/value"
    python train_value.py --config-path config --config-name libero_sft_value_gr00t_n1d7 \
      "data.train_data_paths.0.dataset_path=$DS" data.tag=rollout \
      "data.return_min=$RMIN" "data.return_max=0.0" \
      "runner.logger.log_path=$VALDIR" "runner.logger.experiment_name=value_p2iter1b" \
      "runner.max_steps=1500" "runner.save_interval=1500" \
      "actor.optim.total_training_steps=1500" 2>&1 | tail -3 )
  VALCK=$(ls -d "$VALDIR"/value_p2iter1b/checkpoints/global_step_* 2>/dev/null | sort -t_ -k3 -n | tail -1)
fi
if [ -z "$VALCK" ]; then mark "ABORT: value checkpoint missing"; exit 1; fi
mark "STAGE 4 value ckpt: $VALCK"

# ---- 5. ADVANTAGES (per-type quantiles, tag=recalp2b) ----
if [ -f "$DS/meta/advantages_recalp2b.parquet" ] && [ -f "$DEMOS/meta/advantages_recalp2b.parquet" ]; then
  mark "STAGE 5 advantages SKIP (recalp2b exists)"
else
  mark "STAGE 5 advantages: p2iter1b rollouts + demos (rollout 0.4 / sft 0.3)"
  ( source "$OPENPI_ACT"; common_env; cd "$REPO/examples/recap/process"
    torchrun --nproc_per_node=1 compute_advantages.py --config-name compute_advantages_n1d7_phase2 \
      "data.train_data_paths.0.dataset_path=$DS" \
      "advantage.value_checkpoint=$VALCK" "advantage.tag=recalp2b" \
      "data.return_min=$RMIN" "data.return_max=0.0" 2>&1 | tail -12 )
  if [ ! -f "$DS/meta/advantages_recalp2b.parquet" ]; then mark "ABORT: advantages missing"; exit 1; fi
fi
mark "STAGE 5 advantages done"

# ---- 6. Conditioned-SFT retrain from SFT anchor on rollouts + demos ----
if [ -f "$NEWCK" ]; then
  mark "STAGE 6 CFG SKIP (ckpt exists)"
else
  mark "STAGE 6 cond-SFT retrain (film, joint lr=1e-4, 5000 steps) on p2iter1b+demos"
  ( source "$N1D7_ACT"; common_env; cd "$REPO/examples/recap/cfg"
    python train_cfg.py --config-name libero_cfg_gr00t_n1d7 \
      "actor.model.model_path=$SFT_CKPT" \
      "data.train_data_paths=[{dataset_path:$DS,type:rollout,weight:1.0},{dataset_path:$DEMOS,type:sft,weight:1.0}]" \
      data.advantage_tag=recalp2b \
      "runner.logger.experiment_name=$CFGEXP" \
      "runner.logger.project_name=recap-n1d7-from-base" \
      "runner.max_steps=5000" "runner.save_interval=1000" \
      "actor.optim.lr=1.0e-4" \
      "actor.optim.total_training_steps=5000" "actor.optim.lr_warmup_steps=200" 2>&1 | tail -4 )
  if [ ! -f "$NEWCK" ]; then mark "ABORT: CFG ckpt missing ($NEWCK)"; exit 1; fi
fi
mark "STAGE 6 CFG done: $NEWCK"

# ---- 7. 100-ep confirm at w=1.0/1.5/2.0/2.5 ----
mark "STAGE 7 100-ep confirm (w=1.0/1.5/2.0/2.5)"
( source "$N1D7_ACT"; common_env
  for W in 1.0 1.5 2.0 2.5; do
    LOG="/tmp/n17_p2iter1b_confirm_w${W}.log"
    bash "$REPO/evaluations/run_eval.sh" libero libero_10_gr00t_n1d7_cfg_eval \
      "+runner.ckpt_path=${NEWCK}" "rollout.model.cfg_guidance_weight=${W}" \
      "+rollout.model.conditioning=film" "env.eval.max_steps_per_rollout_epoch=2560" \
      "runner.logger.experiment_name=p2iter1b_confirm_w${W}" \
      "runner.logger.logger_backends=[wandb,tensorboard]" \
      "runner.logger.project_name=recap-n1d7-from-base" 2>&1 | tee "$LOG" | tail -2
    s=$(grep -cE "task_id=.*success=True" "$LOG"); t=$(grep -cE "libero eval\] task_id=" "$LOG")
    mark "P2ITER1B CONFIRM w=$W -> ${s}/${t}"
  done )
mark "RUN_P2ITER1B COMPLETE (phase2 cond-SFT: 63/55/71/58/66 at w=1.0-3.0; base=44)"
echo "RUN_P2ITER1B_EXIT=0" | tee -a "$STATUS"
