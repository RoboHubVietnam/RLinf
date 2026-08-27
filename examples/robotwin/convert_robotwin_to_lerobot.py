"""Convert RoboTwin hdf5 demos -> LeRobot **v2.1** for GR00T-N1.7 SFT.

GR00T-N1.7's loader expects LeRobot v2.1 (per-episode parquet + per-episode
per-camera mp4 + meta/{info.json, episodes.jsonl, tasks.jsonl,
episodes_stats.jsonl, modality.json}) — confirmed against the working VR_C2
dataset. lerobot 0.4.4 writes the newer v3.0 layout, so we write v2.1
directly here.

RoboTwin hdf5 (per episode):
  joint_action/vector  (T, 14) -> [left_arm(6),left_grip,right_arm(6),right_grip]
  observation/{head,left,right}_camera/rgb (T,) encoded jpg -> (240,320,3)
  instructions/episodeN.json {"seen":[...],"unseen":[...]}

Mapping (mirrors examples/vrc2, ABSOLUTE actions — position control):
  observation.state = action = joint_action.vector[t]  (14-dim)
  observation.images.{head,left_wrist,right_wrist} = the 3 cameras
  task = a random "seen" instruction per episode

Usage:
  python examples/robotwin/convert_robotwin_to_lerobot.py \
    --src .../aloha-agilex_randomized_500 --repo_id robotwin_bowls_aloha --fps 30
"""
import argparse
import glob
import json
import os

import cv2
import h5py
import imageio.v2 as imageio
import numpy as np
import pandas as pd

H, W = 240, 320
DIM = 14
CAMS = {"head": "head_camera", "left_wrist": "left_camera", "right_wrist": "right_camera"}

VIDEO_INFO = {
    "dtype": "video", "shape": [H, W, 3],
    "names": ["height", "width", "channels"],
    "info": {"video.height": H, "video.width": W, "video.codec": "h264",
             "video.pix_fmt": "yuv420p", "video.is_depth_map": False,
             "video.fps": 30, "video.channels": 3, "has_audio": False},
}
MODALITY = {
    "state": {"joint_pos": {"start": 0, "end": DIM}},
    "action": {"joints": {"start": 0, "end": DIM}},
    "video": {k: {"original_key": f"observation.images.{k}"} for k in CAMS},
    "annotation": {"human.action.task_description": {"original_key": "task_index"}},
}


def decode_cam(h5, cam):
    blobs = h5[f"observation/{cam}/rgb"]
    out = np.empty((len(blobs), H, W, 3), np.uint8)
    for i, b in enumerate(blobs):
        img = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
        out[i] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return out


def feat_stats(arr):  # arr (N, D) -> per-dim stats dict (lerobot v2.1 schema)
    a = np.asarray(arr, np.float64)
    return {"min": a.min(0).tolist(), "max": a.max(0).tolist(),
            "mean": a.mean(0).tolist(), "std": a.std(0).tolist(),
            "q01": np.quantile(a, 0.01, 0).tolist(),
            "q99": np.quantile(a, 0.99, 0).tolist(), "count": [int(len(a))]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--repo_id", required=True)
    ap.add_argument("--root", default="/data/datasets")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--max_eps", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    out = os.path.join(args.root, args.repo_id)
    for sub in ["data/chunk-000", "meta"] + [f"videos/chunk-000/observation.images.{k}" for k in CAMS]:
        os.makedirs(os.path.join(out, sub), exist_ok=True)

    hdf5s = sorted(glob.glob(os.path.join(args.src, "data", "*.hdf5")),
                   key=lambda p: int("".join(c for c in os.path.basename(p) if c.isdigit())))
    if args.max_eps > 0:
        hdf5s = hdf5s[: args.max_eps]
    print(f"{len(hdf5s)} episodes -> {out}")

    task_to_idx, tasks_list = {}, []
    ep_lines, epstats_lines = [], []
    global_index = 0
    all_state, all_action = [], []

    for ep, fp in enumerate(hdf5s):
        with h5py.File(fp, "r") as h:
            vec = np.asarray(h["joint_action/vector"], np.float32)  # (T,14) commands
            # re-rendered datasets carry measured qpos; use it as state so
            # state != action (no proprio->action leakage)
            if "observation/qpos_measured" in h:
                state = np.asarray(h["observation/qpos_measured"], np.float32)
            else:
                state = vec
            # rollout collections carry a real success flag (RECAP needs it)
            ep_success = bool(h.attrs.get("success", True))
            frames = {k: decode_cam(h, cam) for k, cam in CAMS.items()}
        n = len(vec)
        idx = int("".join(c for c in os.path.basename(fp) if c.isdigit()))
        with h5py.File(fp, "r") as h:
            attr_instr = h.attrs.get("instruction")
        ip = os.path.join(args.src, "instructions", f"episode{idx}.json")
        cjson = os.path.join(args.src, "colors", f"episode{idx}.json")
        if os.path.exists(cjson):           # color-conditioned cup task
            c = json.load(open(cjson))["target_color"]
            task = f"stack the other cups into the {c} cup"
        elif attr_instr is not None:        # rollout collections carry theirs
            task = str(attr_instr)
        else:
            seen = json.load(open(ip))["seen"] if os.path.exists(ip) else \
                ["stack the three bowls by nesting them into one another"]
            task = seen[int(rng.integers(len(seen)))]
        if task not in task_to_idx:
            task_to_idx[task] = len(tasks_list)
            tasks_list.append(task)
        tidx = task_to_idx[task]

        # per-camera mp4
        for k in CAMS:
            vp = os.path.join(out, f"videos/chunk-000/observation.images.{k}",
                              f"episode_{ep:06d}.mp4")
            w = imageio.get_writer(vp, fps=args.fps, codec="libx264",
                                   format="ffmpeg", pixelformat="yuv420p",
                                   macro_block_size=1)
            for t in range(n):
                w.append_data(frames[k][t])
            w.close()

        # parquet (vector features + index cols; images are video-referenced)
        df = pd.DataFrame({
            "action": list(vec),
            "observation.state": list(state),
            "is_success": [np.array([ep_success]) for _ in range(n)],
            "timestamp": (np.arange(n) / args.fps).astype(np.float32),
            "frame_index": np.arange(n, dtype=np.int64),
            "episode_index": np.full(n, ep, np.int64),
            "index": np.arange(global_index, global_index + n, dtype=np.int64),
            "task_index": np.full(n, tidx, np.int64),
        })
        df.to_parquet(os.path.join(out, f"data/chunk-000/episode_{ep:06d}.parquet"), index=False)
        global_index += n
        all_state.append(state); all_action.append(vec)

        ep_lines.append({"episode_index": ep, "tasks": [task], "length": n})
        epstats_lines.append({"episode_index": ep, "stats": {
            "observation.state": feat_stats(state), "action": feat_stats(vec)}})
        if (ep + 1) % 25 == 0:
            print(f"  {ep + 1}/{len(hdf5s)}", flush=True)

    # ---- meta ----
    features = {
        "action": {"dtype": "float32", "shape": [DIM], "names": ["action"]},
        "observation.state": {"dtype": "float32", "shape": [DIM], "names": ["state"]},
        "is_success": {"dtype": "bool", "shape": [1], "names": ["is_success"]},
        **{f"observation.images.{k}": VIDEO_INFO for k in CAMS},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    info = {
        "codebase_version": "v2.1", "robot_type": "aloha_agilex",
        "total_episodes": len(hdf5s), "total_frames": global_index,
        "total_tasks": len(tasks_list), "total_videos": len(hdf5s) * len(CAMS),
        "total_chunks": 1, "chunks_size": 1000, "fps": args.fps,
        "splits": {"train": f"0:{len(hdf5s)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }
    json.dump(info, open(f"{out}/meta/info.json", "w"), indent=4)
    with open(f"{out}/meta/episodes.jsonl", "w") as f:
        for e in ep_lines: f.write(json.dumps(e) + "\n")
    with open(f"{out}/meta/tasks.jsonl", "w") as f:
        for i, t in enumerate(tasks_list): f.write(json.dumps({"task_index": i, "task": t}) + "\n")
    with open(f"{out}/meta/episodes_stats.jsonl", "w") as f:
        for e in epstats_lines: f.write(json.dumps(e) + "\n")
    S = np.concatenate(all_state); A = np.concatenate(all_action)
    json.dump({"observation.state": feat_stats(S), "action": feat_stats(A)},
              open(f"{out}/meta/stats.json", "w"), indent=2)
    json.dump(MODALITY, open(f"{out}/meta/modality.json", "w"), indent=2)
    print(f"CONVERT_DONE -> {out} ({len(hdf5s)} eps, {global_index} frames)")


if __name__ == "__main__":
    main()
