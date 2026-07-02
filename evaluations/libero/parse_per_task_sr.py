#!/usr/bin/env python3
"""Parse per-task success rate for LIBERO-Long from an RLinf eval log.

Reads lines of the form:
    [libero eval] task_id=<tid>, trial_id=<trial>, success=<True|False>
and aggregates success rate per task (capping at --runs trials/task, default 10).

Usage:
    python parse_per_task_sr.py <eval_log> [--runs 10] [--label "GR00T N1.7 SFT"]
"""

import argparse
import re
from collections import defaultdict

TASK_NAMES = {
    0: "put both the alphabet soup and the tomato sauce in the basket",
    1: "put both the cream cheese box and the butter in the basket",
    2: "turn on the stove and put the moka pot on it",
    3: "put the black bowl in the bottom drawer of the cabinet and close it",
    4: "put the white mug on the left plate and the yellow-white mug on the right plate",
    5: "pick up the book and place it in the back compartment of the caddy",
    6: "put the white mug on the plate and the chocolate pudding to the right of the plate",
    7: "put both the alphabet soup and the cream cheese box in the basket",
    8: "put both moka pots on the stove",
    9: "put the yellow and white mug in the microwave and close it",
}

LINE_RE = re.compile(
    r"\[libero eval\]\s*task_id=(\d+),\s*trial_id=(\d+),\s*success=(True|False)"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", help="path to eval log file")
    ap.add_argument("--runs", type=int, default=10, help="trials per task to count")
    ap.add_argument("--label", default="GR00T N1.7 SFT", help="label for the table")
    args = ap.parse_args()

    # task_id -> {trial_id: success}
    seen = defaultdict(dict)
    with open(args.log, "r", errors="ignore") as f:
        for line in f:
            m = LINE_RE.search(line)
            if not m:
                continue
            tid, trial, ok = int(m.group(1)), int(m.group(2)), m.group(3) == "True"
            # keep first occurrence per (task, trial); only first `runs` trials
            if trial < args.runs and trial not in seen[tid]:
                seen[tid][trial] = ok

    tot_succ = tot_n = 0
    rows = []
    for tid in sorted(TASK_NAMES):
        trials = seen.get(tid, {})
        n = len(trials)
        s = sum(trials.values())
        tot_succ += s
        tot_n += n
        sr = (s / n * 100.0) if n else float("nan")
        rows.append((tid, n, s, sr))

    name_w = max(len(v) for v in TASK_NAMES.values())
    print(f"\n=== {args.label} — LIBERO-Long per-task success rate ===\n")
    print(f"{'task':>4}  {'name':<{name_w}}  {'runs':>4}  {'succ':>4}  {'SR%':>6}")
    print("-" * (4 + 2 + name_w + 2 + 4 + 2 + 4 + 2 + 6))
    for tid, n, s, sr in rows:
        print(f"{tid:>4}  {TASK_NAMES[tid]:<{name_w}}  {n:>4}  {s:>4}  {sr:>6.1f}")
    print("-" * (4 + 2 + name_w + 2 + 4 + 2 + 4 + 2 + 6))
    total_sr = (tot_succ / tot_n * 100.0) if tot_n else float("nan")
    print(f"{'TOT':>4}  {'(all tasks)':<{name_w}}  {tot_n:>4}  {tot_succ:>4}  {total_sr:>6.1f}")
    print()
    # Machine-readable summary line
    per_task = {tid: (s, n) for tid, n, s, _ in rows}
    print(f"SUMMARY total_sr={total_sr:.4f} total={tot_succ}/{tot_n} per_task={per_task}")


if __name__ == "__main__":
    main()
