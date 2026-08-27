"""Split a LeRobot v2.1 dataset into train/val by episode, renumbering indices.

Val episodes are renumbered to a fresh 0-based range so GR00T's loader sees a
self-contained dataset. Parquets are rewritten (episode_index/index columns);
videos are symlinked under their new names (large files, no copy).

Usage:
  python split_lerobot.py --src /data/datasets/vrc2_stack_cup \
      --train_out /data/datasets/vrc2_stack_cup_train --train_eps 0-43 \
      --val_out /data/datasets/vrc2_stack_cup_val --val_eps 44-48
"""
import argparse, json, os, shutil, glob
import pandas as pd


def parse_eps(s):
    a, b = s.split("-")
    return list(range(int(a), int(b) + 1))


def build_split(src, out, eps):
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(f"{out}/data/chunk-000")
    os.makedirs(f"{out}/meta")
    info = json.load(open(f"{src}/meta/info.json"))
    chunks_size = info.get("chunks_size", 1000)
    data_tpl = info["data_path"]
    video_tpl = info["video_path"]
    video_keys = [k for k, v in info["features"].items() if v["dtype"] == "video"]

    # meta line lookups
    ep_meta = {json.loads(l)["episode_index"]: json.loads(l)
               for l in open(f"{src}/meta/episodes.jsonl")}
    stats_meta = {json.loads(l)["episode_index"]: json.loads(l)
                  for l in open(f"{src}/meta/episodes_stats.jsonl")}

    global_idx = 0
    new_episodes, new_stats = [], []
    total_frames = 0
    for new_ei, old_ei in enumerate(eps):
        # rewrite parquet
        src_pq = f"{src}/" + data_tpl.format(episode_chunk=old_ei // chunks_size,
                                             episode_index=old_ei)
        df = pd.read_parquet(src_pq)
        n = len(df)
        df["episode_index"] = new_ei
        df["index"] = range(global_idx, global_idx + n)
        global_idx += n
        total_frames += n
        dst_pq = f"{out}/" + data_tpl.format(episode_chunk=new_ei // chunks_size,
                                            episode_index=new_ei)
        os.makedirs(os.path.dirname(dst_pq), exist_ok=True)
        df.to_parquet(dst_pq, index=False)
        # symlink videos (renamed)
        for vk in video_keys:
            src_v = f"{src}/" + video_tpl.format(episode_chunk=old_ei // chunks_size,
                                                video_key=vk, episode_index=old_ei)
            dst_v = f"{out}/" + video_tpl.format(episode_chunk=new_ei // chunks_size,
                                                video_key=vk, episode_index=new_ei)
            os.makedirs(os.path.dirname(dst_v), exist_ok=True)
            if os.path.exists(src_v):
                os.symlink(os.path.abspath(src_v), dst_v)
        # meta lines (renumbered)
        em = dict(ep_meta[old_ei]); em["episode_index"] = new_ei
        new_episodes.append(em)
        if old_ei in stats_meta:
            sm = dict(stats_meta[old_ei]); sm["episode_index"] = new_ei
            new_stats.append(sm)

    # write meta
    info2 = dict(info)
    info2["total_episodes"] = len(eps)
    info2["total_frames"] = total_frames
    info2["total_videos"] = len(eps) * len(video_keys)
    if "splits" in info2:
        info2["splits"] = {"train": f"0:{len(eps)}"}
    json.dump(info2, open(f"{out}/meta/info.json", "w"), indent=4)
    with open(f"{out}/meta/episodes.jsonl", "w") as f:
        for e in new_episodes:
            f.write(json.dumps(e) + "\n")
    with open(f"{out}/meta/episodes_stats.jsonl", "w") as f:
        for e in new_stats:
            f.write(json.dumps(e) + "\n")
    shutil.copy(f"{src}/meta/tasks.jsonl", f"{out}/meta/tasks.jsonl")
    shutil.copy(f"{src}/meta/modality.json", f"{out}/meta/modality.json")
    print(f"{out}: {len(eps)} episodes, {total_frames} frames")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--train_out", required=True); ap.add_argument("--train_eps", required=True)
    ap.add_argument("--val_out", required=True); ap.add_argument("--val_eps", required=True)
    a = ap.parse_args()
    build_split(a.src, a.train_out, parse_eps(a.train_eps))
    build_split(a.src, a.val_out, parse_eps(a.val_eps))


if __name__ == "__main__":
    main()
