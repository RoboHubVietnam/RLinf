"""Convert GR00T N1.7 rollout pickles (CollectEpisode pickle export) into a
LeRobot v3 dataset readable by GR00T N1.7's LeRobotEpisodeLoader (lerobot 0.4.4).

Unlike the N1.5 path (rlinf.data.lerobot_writer -> lerobot 0.1.0 / common.datasets),
this uses the lerobot 0.4.4 API (lerobot.datasets, add_frame/save_episode) so the
written dataset matches the version the N1.7 reader uses.

Run in vla-rlft-n1d7:
    HF_LEROBOT_HOME=/data/datasets \
    python examples/recap/process/convert_n1d7_rollouts_to_lerobot.py \
        --pkl_dir /data/datasets/n1d7_rollouts_pkl \
        --repo_id n1d7_libero_rollouts \
        --sft_modality /data/datasets/libero_10_no_noops_lerobot/meta/modality.json
"""

import argparse
import glob
import os
import pickle
import shutil

import numpy as np

FEATURES = {
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
    # NOTE: non-image feature shapes MUST be tuples — lerobot 0.4.4 validate_frame
    # compares value.shape (tuple) != feature["shape"] directly, so a list never matches.
    "observation.state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
    "action": {"dtype": "float32", "shape": (7,), "names": ["action"]},
    "is_success": {"dtype": "bool", "shape": (1,), "names": ["is_success"]},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl_dir", required=True)
    ap.add_argument("--repo_id", default="n1d7_libero_rollouts")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--sft_modality", default=None)
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    home = os.environ.get("HF_LEROBOT_HOME") or os.environ.get(
        "LEROBOT_HOME", os.path.expanduser("~/.cache/huggingface/lerobot")
    )
    out_root = os.path.join(home, args.repo_id)
    if os.path.exists(out_root):
        print(f"Removing existing dataset at {out_root}")
        shutil.rmtree(out_root)

    pkls = sorted(glob.glob(os.path.join(args.pkl_dir, "**", "*.pkl"), recursive=True))
    assert pkls, f"No pickles in {args.pkl_dir}"
    print(f"Found {len(pkls)} episode pickles")

    ds = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        features=FEATURES,
        robot_type="libero",
        use_videos=True,
        root=out_root,
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
        if isinstance(task, (list, tuple, np.ndarray)):
            task = task[0]
        n_frames = len(actions)  # obs has one extra (terminal); align to actions
        for i in range(n_frames):
            ds.add_frame(
                {
                    "observation.images.image": np.asarray(obs[i]["main_images"]),
                    "observation.images.wrist_image": np.asarray(obs[i]["wrist_images"]),
                    "observation.state": np.asarray(obs[i]["states"], dtype=np.float32),
                    "action": np.asarray(actions[i], dtype=np.float32),
                    "is_success": np.asarray([success], dtype=bool),
                    "task": str(task),
                }
            )
        ds.save_episode()

    print(f"Converted {len(pkls)} episodes: {n_succ} success / {n_fail} fail")

    if args.sft_modality:
        dst = os.path.join(out_root, "meta", "modality.json")
        shutil.copy(args.sft_modality, dst)
        print(f"Copied modality.json -> {dst}")


if __name__ == "__main__":
    main()
