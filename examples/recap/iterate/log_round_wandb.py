#!/usr/bin/env python3
"""Log one iterated-RECAP round to a single persistent wandb run so the SR
trajectory (and sample eval videos) accumulate across rounds.

Usage:
    python log_round_wandb.py --round N --eval_log <eval.log> [--video <mp4>]
                              [--run_id_file <path>]

Parses success_once / success_at_end + per-task SR from the eval log and logs
them at step=round. Reuses one wandb run (id persisted in --run_id_file) so all
rounds land on the same charts.
"""
import argparse
import os
import re

TASK_RE = re.compile(
    r"\[libero eval\]\s*task_id=(\d+),\s*trial_id=(\d+),\s*success=(True|False)"
)


def parse_sr(path, runs=10):
    seen = {}
    with open(path, errors="ignore") as f:
        for line in f:
            m = TASK_RE.search(line)
            if not m:
                continue
            t, tr, ok = int(m.group(1)), int(m.group(2)), m.group(3) == "True"
            if tr < runs:
                seen.setdefault(t, {}).setdefault(tr, ok)
    per_task, s_tot, n_tot = {}, 0, 0
    for t in range(10):
        tr = seen.get(t, {})
        s, n = sum(tr.values()), len(tr)
        per_task[t] = (100.0 * s / n) if n else 0.0
        s_tot += s
        n_tot += n
    return (100.0 * s_tot / n_tot if n_tot else 0.0), per_task, n_tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--eval_log", required=True)
    ap.add_argument("--video", default=None)
    ap.add_argument("--run_id_file", default="/tmp/recap_iter_wandb_run_id")
    ap.add_argument("--project", default="rlinf")
    ap.add_argument("--name", default="recap_iterations")
    args = ap.parse_args()

    import wandb

    sr, per_task, n = parse_sr(args.eval_log)
    run_id = None
    if os.path.exists(args.run_id_file):
        run_id = open(args.run_id_file).read().strip() or None

    run = wandb.init(
        project=args.project,
        name=args.name,
        id=run_id,
        resume="allow",
        reinit=True,
    )
    if run_id is None:
        with open(args.run_id_file, "w") as f:
            f.write(run.id)

    data = {"recap_iter/success_once": sr, "recap_iter/n_episodes": n}
    for t, v in per_task.items():
        data[f"recap_iter/task_{t}_sr"] = v
    if args.video and os.path.exists(args.video):
        try:
            data["recap_iter/eval_video"] = wandb.Video(args.video, format="mp4")
        except Exception as e:  # noqa: BLE001
            print(f"[wandb] video log failed: {e}", flush=True)
    wandb.log(data, step=args.round)
    print(f"[wandb] round {args.round}: success_once={sr:.1f}% (n={n}) logged.", flush=True)
    wandb.finish()


if __name__ == "__main__":
    main()
