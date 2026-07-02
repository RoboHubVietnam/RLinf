#!/usr/bin/env bash
# RECAP round-2, MIXED variant. Tests whether RECAP gains COMPOUND across
# iterations: collect fresh rollouts from the improved iter1 policy (film, w=1.5,
# the 49% operating point), then repeat the mixed recipe (rollouts + expert demos).
#
# Full loop (each stage resumable — skips if its output exists):
#   1. collect  : run iter1 mixed policy (film, w=1.5) in LIBERO sim -> pkls
#   2. convert  : pkl -> LeRobot v2.1 (generic convert, same as iter1)
#   3. returns  : Monte-Carlo returns on the iter2 rollouts -> RMIN
#   4. value    : train iter2 value critic on iter2 returns
#   5. advantages: recompute over iter2 rollouts + demos (global threshold, value=iter2)
#   6. cfg      : CFG-film train from SFT base ckpt on mixed iter2 data (3000 steps, save 500)
#   7. confirm  : 100-ep eval at w=1.0/1.5/2.0 vs the 44% base and iter1's 49%
set -o pipefail
REPO=/home/duynguyen/Desktop/RESEARCH/VR/RLinf
cd "$REPO"
N1D7_ACT="$REPO/vla-rlft-n1d7/bin/activate"
OPENPI_ACT="$REPO/vla-rlft-openpi/bin/activate"
N1D7_PY="$REPO/vla-rlft-n1d7/bin/python"

SFT_CKPT=/data/checkpoints/n1d7_sft_from_base_20k/n1d7_sft_from_base_20k/checkpoint-20000
BEST_POLICY="$REPO/examples/recap/results/recap20k_iter1_film_mixed/checkpoints/global_step_3000/actor/model_state_dict/full_weights.pt"
DEMOS=/data/datasets/libero_10_no_noops_lerobot
MODALITY="$DEMOS/meta/modality.json"
GUIDANCE=1.5
CONDITIONING=film
COLLECT_SEEDS="0 1"

PKL=/data/datasets/recap20k_iter2_pkl
REPOID=recap20k_iter2
DS=/data/datasets/$REPOID
VALDIR=/data/checkpoints/recap20k_iter2_value
CFGEXP=recap20k_iter2_film_mixed
CFGOUT="$REPO/examples/recap/results/${CFGEXP}/checkpoints"
NEWCK="$CFGOUT/global_step_3000/actor/model_state_dict/full_weights.pt"
STATUS=/tmp/recap20k_iter2_status.log

common_env() {
  export REPO_PATH="$REPO" EMBODIED_PATH="$REPO/examples/embodiment" PYTHONPATH="$REPO:${PYTHONPATH:-}"
  export HF_LEROBOT_HOME=/data/datasets HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  export TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1 MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
}
mark(){ echo "=== $(date +%H:%M:%S) | $* ===" | tee -a "$STATUS"; }

echo "RUN_ITER2_START $(date)" > "$STATUS"
mark "config: best_policy=iter1_mixed w=$GUIDANCE cond=$CONDITIONING seeds=[$COLLECT_SEEDS]"

# ---- 1. COLLECT from iter1 mixed policy (conditioned, guided) ----
NP=$(find "$PKL" -name '*.pkl' 2>/dev/null | wc -l)
if [ "$NP" -ge 100 ]; then
  mark "STAGE 1 collect SKIP (found $NP pkls)"
else
  mark "STAGE 1 collect: iter1 mixed policy (film, w=$GUIDANCE) in LIBERO sim"
  ( source "$N1D7_ACT"; common_env
    for S in $COLLECT_SEEDS; do
      bash "$REPO/evaluations/run_eval.sh" libero libero_10_gr00t_n1d7_cfg_eval \
        "+runner.ckpt_path=$BEST_POLICY" \
        "+rollout.model.conditioning=$CONDITIONING" \
        "rollout.model.cfg_guidance_weight=$GUIDANCE" \
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

# ---- 2. CONVERT pkl -> LeRobot v2.1 (generic convert, matches iter1) ----
if [ -f "$DS/meta/episodes.jsonl" ]; then
  mark "STAGE 2 convert SKIP (dataset exists)"
else
  mark "STAGE 2 convert -> LeRobot v2.1"
  ( source "$OPENPI_ACT"; common_env
    python "$REPO/examples/recap/process/convert_rollouts_to_lerobot.py" \
      --pkl_dir "$PKL" --repo_id "$REPOID" --sft_modality "$MODALITY" 2>&1 | tail -2 )
  if [ ! -f "$DS/meta/episodes.jsonl" ]; then mark "ABORT: convert produced no dataset"; exit 1; fi
fi

# ---- 3. RETURNS (Monte-Carlo) -> RMIN ----
mark "STAGE 3 compute_returns"
( source "$OPENPI_ACT"; common_env; cd "$REPO/examples/recap/process"
  python compute_returns.py --config-name compute_returns_rollout \
    "data.train_data_paths.0.dataset_path=$DS" data.tag=rollout 2>&1 | tee /tmp/recap20k_iter2_returns.log | tail -2 )
RMIN=$(grep -oE "return: min=-?[0-9.]+" /tmp/recap20k_iter2_returns.log | tail -1 | grep -oE -- "-?[0-9.]+")
RMIN=${RMIN:--811.0}
mark "STAGE 3 returns done: return_min=$RMIN"

# ---- 4. VALUE critic on iter2 rollouts ----
VALCK=$(ls -d "$VALDIR"/value_iter2/checkpoints/global_step_* 2>/dev/null | sort -t_ -k3 -n | tail -1)
if [ -n "$VALCK" ] && [ -f "$VALCK/actor/model_state_dict/full_weights.pt" ]; then
  mark "STAGE 4 value SKIP (found $VALCK)"
else
  mark "STAGE 4 train value critic (max_steps=1500)"
  ( source "$OPENPI_ACT"; common_env; cd "$REPO/examples/recap/value"
    python train_value.py --config-path config --config-name libero_sft_value_gr00t_n1d7 \
      "data.train_data_paths.0.dataset_path=$DS" data.tag=rollout \
      "data.return_min=$RMIN" "data.return_max=0.0" \
      "runner.logger.log_path=$VALDIR" "runner.logger.experiment_name=value_iter2" \
      "runner.max_steps=1500" "runner.save_interval=1500" \
      "actor.optim.total_training_steps=1500" 2>&1 | tail -3 )
  VALCK=$(ls -d "$VALDIR"/value_iter2/checkpoints/global_step_* 2>/dev/null | sort -t_ -k3 -n | tail -1)
fi
if [ -z "$VALCK" ]; then mark "ABORT: value checkpoint missing"; exit 1; fi
mark "STAGE 4 value ckpt: $VALCK"

# ---- 5. ADVANTAGES over iter2 rollouts + demos (global threshold, value=iter2) ----
if [ -f "$DS/meta/advantages_rollout.parquet" ] && [ -f "$DS/meta/advantages_rollout.mixed.done" ]; then
  mark "STAGE 5 advantages SKIP (mixed advantages exist)"
else
  mark "STAGE 5 advantages: iter2 rollouts + demos (workers=0, value=iter2)"
  ( source "$OPENPI_ACT"; common_env; cd "$REPO/examples/recap/process"
    torchrun --nproc_per_node=1 compute_advantages.py --config-name compute_advantages_n1d7_mixed \
      "data.train_data_paths.0.dataset_path=$DS" \
      "advantage.value_checkpoint=$VALCK" "advantage.tag=rollout" "advantage.returns_tag=rollout" \
      "advantage.num_dataloader_workers_per_gpu=0" \
      "data.return_min=$RMIN" "data.return_max=0.0" 2>&1 | tail -12 )
  if [ ! -f "$DS/meta/advantages_rollout.parquet" ]; then mark "ABORT: iter2 advantages missing"; exit 1; fi
  touch "$DS/meta/advantages_rollout.mixed.done"
  mark "STAGE 5 advantages done"
fi

# ---- 6. CFG train MIXED from SFT base on iter2 data ----
if [ -f "$NEWCK" ]; then
  mark "STAGE 6 CFG SKIP (ckpt exists)"
else
  mark "STAGE 6 CFG (film) train MIXED (iter2 rollouts + demos, 3000 steps, save 500)"
  ( source "$N1D7_ACT"; common_env; cd "$REPO/examples/recap/cfg"
    python train_cfg.py --config-name libero_cfg_gr00t_n1d7 \
      "actor.model.model_path=$SFT_CKPT" \
      "data.train_data_paths=[{dataset_path:$DS,type:rollout,weight:1.0},{dataset_path:$DEMOS,type:sft,weight:1.0}]" \
      data.advantage_tag=rollout \
      "runner.logger.experiment_name=$CFGEXP" \
      "runner.max_steps=3000" "runner.save_interval=500" \
      "actor.optim.total_training_steps=3000" "actor.optim.lr_warmup_steps=300" 2>&1 | tail -4 )
  if [ ! -f "$NEWCK" ]; then mark "ABORT: CFG ckpt missing ($NEWCK)"; exit 1; fi
  mark "STAGE 6 CFG done: $NEWCK"
fi

# ---- 7. 100-ep confirm at w=1.0/1.5/2.0 (wandb) ----
mark "STAGE 7 100-ep confirm (w=1.0/1.5/2.0)"
( source "$N1D7_ACT"; common_env
  for W in 1.0 1.5 2.0; do
    LOG="/tmp/n17_iter2_confirm_w${W}.log"
    bash "$REPO/evaluations/run_eval.sh" libero libero_10_gr00t_n1d7_cfg_eval \
      "+runner.ckpt_path=${NEWCK}" "rollout.model.cfg_guidance_weight=${W}" \
      "+rollout.model.conditioning=film" "env.eval.max_steps_per_rollout_epoch=2560" \
      "runner.logger.experiment_name=iter2_confirm_w${W}" \
      "runner.logger.logger_backends=[wandb,tensorboard]" \
      "runner.logger.project_name=recap-n1d7-from-base" 2>&1 | tee "$LOG" | tail -2
    s=$(grep -cE "task_id=.*success=True" "$LOG"); t=$(grep -cE "libero eval\] task_id=" "$LOG")
    mark "ITER2 CONFIRM w=$W -> ${s}/${t}"
  done )
mark "RUN_ITER2 COMPLETE (base=44/100, iter1 mixed w=1.5=49/100)"
echo "RUN_ITER2_EXIT=0" | tee -a "$STATUS"
