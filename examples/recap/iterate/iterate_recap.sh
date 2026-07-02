#!/bin/bash
# Iterated sim-in-the-loop RECAP for GR00T N1.7 on LIBERO-Long.
# Each round: collect rollouts from the CURRENT policy in LIBERO sim -> convert ->
# returns -> value -> advantages -> CFG (text) train from SFT ckpt -> eval -> log SR+video.
# Sim replaces the real robot in the RECAP loop. Runs autonomously across rounds.
#
# Usage: iterate_recap.sh <start_policy_ckpt.pt|SFT> <num_rounds> [collect_seeds]
# Resumable: each stage is skipped if its output already exists, so a killed run
# can be relaunched and it picks up where it stopped. Set FORCE_FRESH=1 to wipe a
# round's artifacts and redo it from scratch.
# NOTE: no `set -u` — venv activate scripts reference unbound vars in non-interactive shells.
set -o pipefail

REPO="/home/duynguyen/Desktop/RESEARCH/VR/RLinf"
N1D7="$REPO/vla-rlft-n1d7/bin/python"
OPENPI_ACT="$REPO/vla-rlft-openpi/bin/activate"
N1D7_ACT="$REPO/vla-rlft-n1d7/bin/activate"
SFT_CKPT="/data/checkpoints/GR00T-N1.7-LIBERO/libero_10"
BACKBONE="/data/checkpoints/Cosmos-Reason2-2B"
MODALITY="/data/datasets/libero_10_no_noops_lerobot/meta/modality.json"
STATUS=/tmp/recap_iter_status.log

POLICY_CKPT="$1"          # starting policy ("SFT" for round 1 = plain SFT weights)
NUM_ROUNDS="${2:-3}"
COLLECT_SEEDS="${3:-0 1}" # seeds for diverse collection
FORCE_FRESH="${FORCE_FRESH:-0}"
VALUE_STEPS="${VALUE_STEPS:-1500}"  # hard stop for value training (runner.max_steps)
CFG_STEPS="${CFG_STEPS:-3000}"      # hard stop for CFG training (runner.max_steps)
CONDITIONING="${CONDITIONING:-film}"  # film|token support guidance (w>1); text does NOT
GUIDANCE="${GUIDANCE:-1.5}"           # CFG guidance weight for collection + eval (film/token only)
# CFG training config per conditioning: film/token share the film config, text uses _text.
if [ "$CONDITIONING" = "text" ]; then CFG_CONFIG="libero_cfg_gr00t_n1d7_text"; else CFG_CONFIG="libero_cfg_gr00t_n1d7"; fi

mark() { echo "=== $(date +%H:%M:%S) | $* ===" | tee -a "$STATUS"; }

common_env() {
  export REPO_PATH="$REPO" EMBODIED_PATH="$REPO/examples/embodiment"
  export PYTHONPATH="$REPO:${PYTHONPATH:-}"
  export HF_LEROBOT_HOME=/data/datasets HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  export TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1
  export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
}

echo "ITERATE_RECAP_START $(date)" > "$STATUS"
mark "config: start_ckpt=$POLICY_CKPT rounds=$NUM_ROUNDS seeds=[$COLLECT_SEEDS] value_steps=$VALUE_STEPS cfg_steps=$CFG_STEPS cond=$CONDITIONING w=$GUIDANCE fresh=$FORCE_FRESH"

for R in $(seq 1 "$NUM_ROUNDS"); do
  mark "ROUND $R BEGIN (policy=$POLICY_CKPT)"
  PKL="/data/datasets/recap_iter${R}_pkl"
  REPOID="recap_iter${R}"
  DS="/data/datasets/${REPOID}"
  VALDIR="/data/checkpoints/recap_iter${R}_value"
  CFGEXP="recap_iter${R}_${CONDITIONING}"
  CFGOUT="$REPO/examples/recap/results/${CFGEXP}/checkpoints"
  NEWCK="$CFGOUT/global_step_${CFG_STEPS}/actor/model_state_dict/full_weights.pt"
  EVLOG="/tmp/n17_cfg_eval_iter${R}.log"
  if [ "$FORCE_FRESH" = "1" ]; then
    rm -rf "$PKL" "$DS" "$VALDIR" "$REPO/examples/recap/results/${CFGEXP}" "$EVLOG" 2>/dev/null
  fi

  # ---- 1. COLLECT (n1d7 venv): run current policy in LIBERO sim ----
  NP=$(find "$PKL" -name '*.pkl' 2>/dev/null | wc -l)
  if [ "$NP" -ge 100 ]; then
    mark "ROUND $R STAGE 1/7 collect SKIP (found $NP pkls)"
  else
    mark "ROUND $R STAGE 1/7 collect (sim rollouts from current policy)"
    if [ "$POLICY_CKPT" = "SFT" ]; then
      # Round 1: collect from the PLAIN, UNCONDITIONED SFT policy. SFT has no
      # advantage conditioning, so appending "Advantage: positive" (the CFG-text
      # path) is OOD and suppresses it (observed: 38% vs SFT's true 86%). Use the
      # plain gr00t_n1d7 collect config; bump envs/steps to ~100 episodes/seed.
      ( source "$N1D7_ACT"; common_env
        for S in $COLLECT_SEEDS; do
          bash "$REPO/evaluations/run_eval.sh" libero libero_10_gr00t_n1d7_collect \
            "env.eval.seed=$S" \
            "env.eval.total_num_envs=20" \
            "env.eval.max_steps_per_rollout_epoch=2560" \
            "env.eval.data_collection.save_dir=$PKL/run_$S" \
            "env.eval.data_collection.only_success=false" \
            "runner.logger.log_path=$PKL/_logs/run_$S" 2>&1 | tail -2
        done )
    else
      # Rounds 2+: collect from the advantage-conditioned policy with guidance
      # (w>1 sharpens toward positive advantage -> better-than-SFT data each round).
      ( source "$N1D7_ACT"; common_env
        for S in $COLLECT_SEEDS; do
          bash "$REPO/evaluations/run_eval.sh" libero libero_10_gr00t_n1d7_cfg_eval \
            "+runner.ckpt_path=$POLICY_CKPT" \
            "+rollout.model.conditioning=$CONDITIONING" \
            "rollout.model.cfg_guidance_weight=$GUIDANCE" \
            "env.eval.use_ordered_reset_state_ids=false" \
            "env.eval.seed=$S" \
            "+env.eval.data_collection.enabled=true" \
            "+env.eval.data_collection.save_dir=$PKL/run_$S" \
            "+env.eval.data_collection.export_format=pickle" \
            "+env.eval.data_collection.robot_type=libero" \
            "+env.eval.data_collection.only_success=false" \
            "+env.eval.data_collection.fps=20" \
            "+env.eval.data_collection.finalize_interval=50" \
            "runner.logger.log_path=$PKL/_logs/run_$S" 2>&1 | tail -2
        done )
    fi
    NP=$(find "$PKL" -name '*.pkl' 2>/dev/null | wc -l)
    NS=$(find "$PKL" -name '*success*.pkl' 2>/dev/null | wc -l)
    mark "ROUND $R collect done: $NP episodes ($NS success)"
  fi
  if [ "$NP" -lt 20 ]; then mark "ROUND $R ABORT: too few rollouts"; exit 1; fi

  # ---- 2. CONVERT (openpi venv): pickle -> LeRobot v2.1 ----
  if [ -f "$DS/meta/episodes.jsonl" ]; then
    mark "ROUND $R STAGE 2/7 convert SKIP (dataset exists)"
  else
    mark "ROUND $R STAGE 2/7 convert -> LeRobot v2.1"
    ( source "$OPENPI_ACT"; common_env
      python "$REPO/examples/recap/process/convert_rollouts_to_lerobot.py" \
        --pkl_dir "$PKL" --repo_id "$REPOID" --sft_modality "$MODALITY" 2>&1 | tail -2 )
    if [ ! -f "$DS/meta/episodes.jsonl" ]; then mark "ROUND $R ABORT: convert produced no dataset"; exit 1; fi
  fi

  # ---- 3. RETURNS (openpi venv) ---- (fast; always run to recover RMIN) ----
  mark "ROUND $R STAGE 3/7 compute_returns"
  ( source "$OPENPI_ACT"; common_env; cd "$REPO/examples/recap/process"
    python compute_returns.py --config-name compute_returns_rollout \
      "data.train_data_paths.0.dataset_path=$DS" data.tag=rollout 2>&1 | tee /tmp/recap_iter${R}_returns.log | tail -2 )
  RMIN=$(grep -oE "return: min=-?[0-9.]+" /tmp/recap_iter${R}_returns.log | tail -1 | grep -oE -- "-?[0-9.]+")
  RMIN=${RMIN:--811.0}
  mark "ROUND $R returns done: return_min=$RMIN"

  # ---- 4. VALUE (openpi venv): SigLIP2+Gemma3 critic on returns ----
  VALCK=$(ls -d "$VALDIR"/value_iter${R}/checkpoints/global_step_* 2>/dev/null | sort -t_ -k3 -n | tail -1)
  if [ -n "$VALCK" ] && [ -f "$VALCK/actor/model_state_dict/full_weights.pt" ]; then
    mark "ROUND $R STAGE 4/7 value SKIP (found $VALCK)"
  else
    mark "ROUND $R STAGE 4/7 train value model (max_steps=$VALUE_STEPS)"
    ( source "$OPENPI_ACT"; common_env; cd "$REPO/examples/recap/value"
      python train_value.py --config-path config --config-name libero_sft_value_gr00t_n1d7 \
        "data.train_data_paths.0.dataset_path=$DS" data.tag=rollout \
        "data.return_min=$RMIN" "data.return_max=0.0" \
        "runner.logger.log_path=$VALDIR" "runner.logger.experiment_name=value_iter${R}" \
        "runner.max_steps=$VALUE_STEPS" "runner.save_interval=$VALUE_STEPS" \
        "actor.optim.total_training_steps=$VALUE_STEPS" 2>&1 | tail -3 )
    VALCK=$(ls -d "$VALDIR"/value_iter${R}/checkpoints/global_step_* 2>/dev/null | sort -t_ -k3 -n | tail -1)
  fi
  if [ -z "$VALCK" ]; then mark "ROUND $R ABORT: value checkpoint missing"; exit 1; fi
  mark "ROUND $R value ckpt: $VALCK"

  # ---- 5. ADVANTAGES (openpi venv) ----
  if [ -f "$DS/meta/advantages_rollout.parquet" ]; then
    mark "ROUND $R STAGE 5/7 advantages SKIP (parquet exists)"
  else
    mark "ROUND $R STAGE 5/7 compute_advantages"
    ( source "$OPENPI_ACT"; common_env; cd "$REPO/examples/recap/process"
      torchrun --nproc_per_node=1 compute_advantages.py --config-name compute_advantages_n1d7 \
        "advantage.value_checkpoint=$VALCK" "advantage.tag=rollout" "advantage.returns_tag=rollout" \
        "data.train_data_paths.0.dataset_path=$DS" "data.return_min=$RMIN" "data.return_max=0.0" 2>&1 | tail -3 )
    if [ ! -f "$DS/meta/advantages_rollout.parquet" ]; then mark "ROUND $R ABORT: advantages parquet missing"; exit 1; fi
  fi

  # ---- 6. CFG TRAIN (n1d7 venv): advantage conditioning, from SFT ckpt, on this round's advantages ----
  if [ -f "$NEWCK" ]; then
    mark "ROUND $R STAGE 6/7 CFG train SKIP (ckpt exists)"
  else
    mark "ROUND $R STAGE 6/7 CFG ($CONDITIONING) train from SFT ckpt (max_steps=$CFG_STEPS)"
    ( source "$N1D7_ACT"; common_env; cd "$REPO/examples/recap/cfg"
      python train_cfg.py --config-name "$CFG_CONFIG" \
        "data.train_data_paths.0.dataset_path=$DS" data.advantage_tag=rollout \
        "runner.logger.experiment_name=$CFGEXP" \
        "runner.max_steps=$CFG_STEPS" \
        "actor.optim.total_training_steps=$CFG_STEPS" "actor.optim.lr_warmup_steps=300" 2>&1 | tail -3 )
    if [ ! -f "$NEWCK" ]; then mark "ROUND $R ABORT: CFG ckpt missing ($NEWCK)"; exit 1; fi
    mark "ROUND $R CFG train done: $NEWCK"
  fi

  # ---- 7. EVAL (n1d7 venv) + wandb log ----
  if grep -q "SUMMARY total_sr" "$EVLOG" 2>/dev/null; then
    mark "ROUND $R STAGE 7/7 eval SKIP (log exists)"
  else
    mark "ROUND $R STAGE 7/7 eval policy"
    ( source "$N1D7_ACT"; common_env
      bash "$REPO/evaluations/libero/eval_n1d7_cfg_ckpt.sh" "$NEWCK" "iter${R}" "$GUIDANCE" "$CONDITIONING" 2>&1 | tail -2 )
    # eval writes videos under a fresh logs/<ts>-libero_10_gr00t_n1d7_cfg_eval/video; grab newest.
    VID=$(find "$REPO/logs" -name "*.mp4" -newermt "-25 min" 2>/dev/null | head -1)
    ( source "$N1D7_ACT"; common_env
      python "$REPO/examples/recap/iterate/log_round_wandb.py" --round "$R" --eval_log "$EVLOG" ${VID:+--video "$VID"} 2>&1 | tail -2 )
  fi
  SR=$("$N1D7" "$REPO/evaluations/libero/parse_per_task_sr.py" "$EVLOG" --runs 10 2>/dev/null | grep -oE "total_sr=[0-9.]+" | head -1)
  mark "ROUND $R DONE: $SR  (policy for next round: $NEWCK)"

  POLICY_CKPT="$NEWCK"  # next round collects from the improved policy
done

mark "ITERATE_RECAP COMPLETE ($NUM_ROUNDS rounds)"
echo "ITERATE_RECAP_EXIT=0" | tee -a "$STATUS"
