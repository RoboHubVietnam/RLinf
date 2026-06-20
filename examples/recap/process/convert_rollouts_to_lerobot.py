"""Convert GR00T rollout pickles (CollectEpisode pickle export) into a LeRobot
dataset matching the GR00T LIBERO SFT schema, for RECAP's rollout dataset.

Run in the openpi venv (vla-rlft-openpi) which has lerobot installed:
    LEROBOT_HOME=/data/datasets HF_LEROBOT_HOME=/data/datasets \
    python examples/recap/process/convert_rollouts_to_lerobot.py \
        --pkl_dir /data/datasets/libero_10_gr00t_rollouts_pkl \
        --repo_id libero_10_gr00t_rollouts \
        --sft_modality /data/datasets/libero_10_no_noops_lerobot/meta/modality.json
"""

import argparse
import glob
import os
import pickle
import shutil

import numpy as np

from rlinf.data.lerobot_writer import LeRobotDatasetWriter

FEATURES = {
    # video dtype (mp4) to match the SFT dataset + GR00T's video reader used in
    # CFG training. lerobot encodes the per-frame arrays to mp4 on save_episode.
    "observation.images.image": {
        "dtype": "video",
        "shape": [256, 256, 3],
        "names": ["height", "width", "channel"],
    },
    "observation.images.wrist_image": {
        "dtype": "video",
        "shape": [256, 256, 3],
        "names": ["height", "width", "channel"],
    },
    "observation.state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
    "action": {"dtype": "float32", "shape": (7,), "names": ["action"]},
    "is_success": {"dtype": "bool", "shape": (1,), "names": ["is_success"]},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl_dir", required=True)
    ap.add_argument("--repo_id", default="libero_10_gr00t_rollouts")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--sft_modality", default=None)
    args = ap.parse_args()

    # recursive so it also picks up per-pass subdirs (run_0/, run_1/, ...)
    pkls = sorted(glob.glob(os.path.join(args.pkl_dir, "**", "*.pkl"), recursive=True))
    assert pkls, f"No pickles in {args.pkl_dir}"
    print(f"Found {len(pkls)} episode pickles")

    writer = LeRobotDatasetWriter()
    writer.create(
        repo_id=args.repo_id,
        robot_type="libero",
        fps=args.fps,
        features=FEATURES,
    )

    n_succ = n_fail = 0
    for p in pkls:
        d = pickle.load(open(p, "rb"))
        obs = d["observations"]
        actions = d["actions"]
        success = bool(d["success"])
        n_succ += success
        n_fail += not success
        task = obs[0]["task_descriptions"]
        n_frames = len(actions)  # obs has one more (terminal); align to actions
        episode = []
        for i in range(n_frames):
            episode.append(
                {
                    "observation.images.image": np.asarray(obs[i]["main_images"]),
                    "observation.images.wrist_image": np.asarray(obs[i]["wrist_images"]),
                    "observation.state": np.asarray(obs[i]["states"], dtype=np.float32),
                    "action": np.asarray(actions[i], dtype=np.float32),
                    "is_success": np.asarray([success], dtype=bool),
                    "task": task,
                }
            )
        writer.add_episode(episode)

    writer.finalize()
    print(f"Converted {len(pkls)} episodes: {n_succ} success / {n_fail} fail")

    # Copy the SFT modality.json so GR00T's reader maps the same keys.
    if args.sft_modality:
        home = os.environ.get("LEROBOT_HOME") or os.environ.get(
            "HF_LEROBOT_HOME", os.path.expanduser("~/.cache/huggingface/lerobot")
        )
        dst = os.path.join(home, args.repo_id, "meta", "modality.json")
        shutil.copy(args.sft_modality, dst)
        print(f"Copied modality.json -> {dst}")


if __name__ == "__main__":
    main()
