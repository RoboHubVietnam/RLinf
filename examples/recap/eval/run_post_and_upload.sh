#!/bin/bash
# Post-RECAP eval over all CFG checkpoints + side-by-side wandb upload.
# For each checkpoint: eval (gr00t_cfg, w=1.0) over the same seeds as the pre
# baseline, parse SR, then compose pre(left)|post(right) videos and log to wandb
# at step=<checkpoint step>.
#
# Usage: run_post_and_upload.sh <pre_dir> <pre_once> <ckpt_root> <out_root> <seeds_csv> <steps_csv>
cd /home/duynguyen/Desktop/RESEARCH/VR/RLinf
source vla-rlft/bin/activate
export REPO_PATH=$(pwd) EMBODIED_PATH=$(pwd)/examples/embodiment PYTHONPATH=$(pwd) HF_HOME=/data/hf_cache
export WANDB_PROJECT=rlinf-recap-gr00t WANDB_ENTITY=buzinguyen

PRE_DIR=$1; PRE_ONCE=$2; CKPT_ROOT=$3; OUT_ROOT=$4
SEEDS=$(echo $5 | tr ',' ' '); STEPS=$(echo $6 | tr ',' ' ')

for step in $STEPS; do
  CKPT=$CKPT_ROOT/global_step_$step/actor/model_state_dict/full_weights.pt
  OUT=$OUT_ROOT/post_step$step
  echo "===== POST eval step $step ====="
  bash examples/recap/eval/eval_one.sh $OUT gr00t_cfg 1.0 $CKPT $SEEDS
  TBDIRS=""; for s in $SEEDS; do TBDIRS="$TBDIRS $OUT/seed_$s/tensorboard"; done
  POST_ONCE=$(python examples/recap/eval/read_sr.py $TBDIRS | tail -1 | awk '{print $1}')
  echo "step $step  pre_once=$PRE_ONCE  post_once=$POST_ONCE"
  python examples/recap/eval/compare_videos_wandb.py \
    --pre_dir $PRE_DIR --post_dir $OUT --step $step \
    --pre_sr $PRE_ONCE --post_sr $POST_ONCE \
    --wandb_project rlinf-recap-gr00t --wandb_run recap_gr00t_libero10_v2 \
    --wandb_entity buzinguyen 2>&1 | tail -3
done
echo "POST_UPLOAD_DONE"
