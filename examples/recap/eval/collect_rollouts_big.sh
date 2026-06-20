#!/bin/bash
# Collect many DIVERSE GR00T rollouts on LIBERO-Long for the large RECAP run.
# Runs N passes with different seeds + random reset states (use_ordered=False)
# so each pass yields different episodes. Pickles accumulate under per-run dirs.
set -e
cd /home/duynguyen/Desktop/RESEARCH/VR/RLinf
source vla-rlft/bin/activate
export REPO_PATH=$(pwd) EMBODIED_PATH=$(pwd)/examples/embodiment PYTHONPATH=$(pwd) HF_HOME=/data/hf_cache
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa ROBOT_PLATFORM=LIBERO LIBERO_TYPE=standard
export HYDRA_FULL_ERROR=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

N_PASSES=${1:-8}
OUT_ROOT=/data/datasets/rollouts_big_pkl
mkdir -p "$OUT_ROOT"

for k in $(seq 0 $((N_PASSES-1))); do
  echo "=== collection pass $k (seed=$k) ==="
  python examples/embodiment/eval_embodied_agent.py --config-name libero_10_ppo_gr00t \
    actor.model.model_path=/data/checkpoints/RLinf-Gr00t-SFT-10 \
    rollout.model.model_path=/data/checkpoints/RLinf-Gr00t-SFT-10 \
    env.eval.total_num_envs=16 \
    env.eval.use_ordered_reset_state_ids=False \
    env.eval.seed=$k \
    +env.eval.data_collection.enabled=true \
    +env.eval.data_collection.save_dir=$OUT_ROOT/run_$k \
    +env.eval.data_collection.export_format=pickle \
    +env.eval.data_collection.robot_type=libero \
    +env.eval.data_collection.only_success=false \
    +env.eval.data_collection.fps=20 \
    +env.eval.data_collection.finalize_interval=50 \
    runner.logger.log_path=$OUT_ROOT/_logs/run_$k 2>&1 | tail -3
done

echo "TOTAL pickles: $(find $OUT_ROOT -name '*.pkl' | wc -l)"
echo "  success: $(find $OUT_ROOT -name '*success.pkl' | wc -l)  fail: $(find $OUT_ROOT -name '*fail.pkl' | wc -l)"
echo "COLLECT_BIG_EXIT=0"
