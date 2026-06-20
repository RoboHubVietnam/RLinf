"""Compose pre-RECAP (left) vs post-RECAP (right) eval videos side-by-side and
upload to wandb. Called once per checkpoint with that checkpoint's eval videos.

Pre and post evals use the SAME seed / fixed reset states, so videos with the
same index correspond to the same LIBERO task/scene and pair directly.

Usage:
    python examples/recap/eval/compare_videos_wandb.py \
        --pre_dir  <pre_video_base_dir> \
        --post_dir <post_video_base_dir> \
        --step <checkpoint_step> \
        --pre_sr 0.62 --post_sr 0.71 \
        --wandb_project rlinf-recap-gr00t --wandb_run recap_gr00t_libero10
"""

import argparse
import glob
import os

import imageio.v2 as imageio
import numpy as np

LABEL_H = 24  # pixels reserved at top for a text-free colored banner separator
SEP_W = 6     # vertical separator width between the two clips


def _read(path):
    r = imageio.get_reader(path)
    frames = [np.asarray(f)[:, :, :3] for f in r]
    r.close()
    return frames


def _find_mp4s(base_dir):
    # Eval videos are saved at {base}/seed_*/[sub/]{idx}.mp4
    mp4s = sorted(glob.glob(os.path.join(base_dir, "**", "*.mp4"), recursive=True))
    return mp4s


def _side_by_side(left, right):
    """Horizontally stack two clips (pad shorter by repeating its last frame)."""
    n = max(len(left), len(right))
    if not left or not right:
        return None
    h = max(left[0].shape[0], right[0].shape[0])

    def fit(fr):
        if fr.shape[0] != h:
            pad = np.zeros((h - fr.shape[0], fr.shape[1], 3), dtype=fr.dtype)
            fr = np.concatenate([fr, pad], axis=0)
        return fr

    sep = np.zeros((h, SEP_W, 3), dtype=np.uint8)
    sep[:, :, :] = (60, 60, 60)
    out = []
    for i in range(n):
        lf = fit(left[min(i, len(left) - 1)])
        rf = fit(right[min(i, len(right) - 1)])
        out.append(np.concatenate([lf, sep, rf], axis=1))
    return np.stack(out)  # (T, H, W, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre_dir", required=True)
    ap.add_argument("--post_dir", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--pre_sr", type=float, default=None)
    ap.add_argument("--post_sr", type=float, default=None)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--max_videos", type=int, default=10)
    ap.add_argument("--wandb_project", default="rlinf-recap-gr00t")
    ap.add_argument("--wandb_run", default="recap_gr00t_libero10")
    ap.add_argument("--wandb_entity", default=None)
    ap.add_argument("--out_dir", default="/data/checkpoints/recap_compare_videos")
    args = ap.parse_args()

    import wandb

    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run,
        entity=args.wandb_entity,
        id=args.wandb_run,
        resume="allow",  # same run across checkpoints; log at each step
    )

    pre = _find_mp4s(args.pre_dir)
    post = _find_mp4s(args.post_dir)
    n = min(len(pre), len(post), args.max_videos)
    print(f"pairing {n} videos (pre={len(pre)}, post={len(post)}) at step {args.step}")

    os.makedirs(args.out_dir, exist_ok=True)
    log = {}
    if args.pre_sr is not None:
        log["eval/pre_recap_success_rate"] = args.pre_sr
    if args.post_sr is not None:
        log["eval/post_recap_success_rate"] = args.post_sr
    if args.pre_sr is not None and args.post_sr is not None:
        log["eval/success_rate_delta"] = args.post_sr - args.pre_sr

    for i in range(n):
        clip = _side_by_side(_read(pre[i]), _read(post[i]))
        if clip is None:
            continue
        out_mp4 = os.path.join(args.out_dir, f"step{args.step}_cmp_{i}.mp4")
        imageio.mimwrite(out_mp4, clip, fps=args.fps, macro_block_size=1)
        log[f"compare/episode_{i}"] = wandb.Video(
            out_mp4, caption=f"step {args.step} | LEFT=pre-RECAP  RIGHT=post-RECAP", fps=args.fps
        )

    wandb.log(log, step=args.step)
    print(f"Logged {n} side-by-side videos + success rates to wandb at step {args.step}")
    run.finish()


if __name__ == "__main__":
    main()
