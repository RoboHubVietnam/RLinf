#!/bin/bash
# Run ONE embodied eval (pre- or post-RECAP) over a set of seeds and report the
# aggregated success rate. Pre/post use the SAME ordered reset states so videos
# pair 1:1 per seed for side-by-side comparison.
#
# Usage:
#   eval_one.sh <out_dir> <model_type> <guidance_weight|-> <ckpt_path|-> <seeds...>
#     out_dir         : where eval logs/videos go
#     model_type      : gr00t (pre/base SFT) or gr00t_cfg (post-RECAP)
#     guidance_weight : CFG weight (only used for gr00t_cfg); pass - to skip
#     ckpt_path       : trained CFG full_weights.pt (post); pass - for base SFT
#     seeds           : one eval per seed (16 envs each)
cd /home/duynguyen/Desktop/RESEARCH/VR/RLinf
source vla-rlft/bin/activate
export REPO_PATH=$(pwd) EMBODIED_PATH=$(pwd)/examples/embodiment PYTHONPATH=$(pwd) HF_HOME=/data/hf_cache
# OSMesa software rendering: EGL needs /dev/dri device permissions that aren't
# reliably available here (libEGL "Permission denied"). OSMesa is slower but
# headless-safe and is what rollout collection used successfully.
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa ROBOT_PLATFORM=LIBERO LIBERO_TYPE=standard
export HYDRA_FULL_ERROR=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUT_DIR=$1; MODEL_TYPE=$2; GW=$3; CKPT=$4; shift 4; SEEDS="$@"
SFT=/data/checkpoints/RLinf-Gr00t-SFT-10
mkdir -p "$OUT_DIR"  # shell redirect below writes $OUT_DIR/seed_N.log before python creates the dir

EXTRA=()
if [ "$MODEL_TYPE" = "gr00t_cfg" ]; then
  # Only actor.model.model_type is a real key; the rollout model is a deepcopy of
  # actor.model, so setting it here is enough (and rollout.model.model_type does
  # not exist in the eval config). cfg_guidance_weight is a new key -> needs '+'.
  EXTRA+=("actor.model.model_type=gr00t_cfg")
  [ "$GW" != "-" ] && EXTRA+=("+actor.model.cfg_guidance_weight=$GW")
fi
[ "$CKPT" != "-" ] && EXTRA+=("+runner.ckpt_path=$CKPT")

for s in $SEEDS; do
  echo "--- eval seed $s -> $OUT_DIR/seed_$s ---"
  python examples/embodiment/eval_embodied_agent.py --config-name libero_10_ppo_gr00t \
    actor.model.model_path=$SFT rollout.model.model_path=$SFT \
    env.eval.total_num_envs=16 \
    env.eval.use_ordered_reset_state_ids=True \
    env.eval.seed=$s \
    "${EXTRA[@]}" \
    runner.logger.log_path=$OUT_DIR/seed_$s > $OUT_DIR/seed_$s.log 2>&1 \
    && grep -E "success_once|num_trajectories" $OUT_DIR/seed_$s.log | tail -1 \
    || { echo "SEED $s FAILED — tail:"; tail -8 $OUT_DIR/seed_$s.log; }
done

# Aggregate SR across seeds
TBDIRS=""
for s in $SEEDS; do TBDIRS="$TBDIRS $OUT_DIR/seed_$s/tensorboard"; done
SR=$(python examples/recap/eval/read_sr.py $TBDIRS)
echo "EVAL_RESULT out_dir=$OUT_DIR success_once_at_end_ntraj=[$SR]"
