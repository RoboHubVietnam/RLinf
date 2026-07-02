#!/usr/bin/env python3
"""Build the SFT-vs-RECAP(CFG) LIBERO-Long comparison table (per-task + total).

Parses one or more eval logs (each containing '[libero eval] task_id=,trial_id=,
success=' lines) and prints a task-by-task + total success-rate comparison as a
markdown table.

Usage:
    python build_comparison_table.py SFT=<sft.log> RECAP@1000=<cfg1000.log> ...
"""

import re
import sys
from collections import defaultdict

TASK_NAMES = {
    0: "alphabet soup + tomato sauce in basket",
    1: "cream cheese box + butter in basket",
    2: "turn on stove + put moka pot on it",
    3: "black bowl in bottom drawer + close it",
    4: "white mug left plate + yellow/white mug right plate",
    5: "book in back compartment of caddy",
    6: "white mug on plate + chocolate pudding right of plate",
    7: "alphabet soup + cream cheese box in basket",
    8: "both moka pots on stove",
    9: "yellow/white mug in microwave + close it",
}
LINE_RE = re.compile(
    r"\[libero eval\]\s*task_id=(\d+),\s*trial_id=(\d+),\s*success=(True|False)"
)


def parse(path, runs=10):
    seen = defaultdict(dict)
    with open(path, errors="ignore") as f:
        for line in f:
            m = LINE_RE.search(line)
            if not m:
                continue
            tid, trial, ok = int(m.group(1)), int(m.group(2)), m.group(3) == "True"
            if trial < runs and trial not in seen[tid]:
                seen[tid][trial] = ok
    per_task = {}
    ts = tn = 0
    for tid in TASK_NAMES:
        tr = seen.get(tid, {})
        s, n = sum(tr.values()), len(tr)
        per_task[tid] = (s, n)
        ts += s
        tn += n
    return per_task, (ts, tn)


def main():
    cols = []  # (label, per_task, total)
    for arg in sys.argv[1:]:
        label, path = arg.split("=", 1)
        per_task, total = parse(path)
        cols.append((label, per_task, total))
    if not cols:
        print("no inputs")
        return

    labels = [c[0] for c in cols]
    # Header
    hdr = "| Task | Description | " + " | ".join(f"{l} SR%" for l in labels) + " |"
    sep = "|---|---|" + "|".join("---" for _ in labels) + "|"
    print(hdr)
    print(sep)
    for tid in TASK_NAMES:
        cells = []
        for _, per_task, _ in cols:
            s, n = per_task[tid]
            cells.append(f"{(100.0*s/n if n else float('nan')):.0f} ({s}/{n})")
        print(f"| {tid} | {TASK_NAMES[tid]} | " + " | ".join(cells) + " |")
    # Total row
    tcells = []
    for _, _, (ts, tn) in cols:
        tcells.append(f"**{(100.0*ts/tn if tn else float('nan')):.1f}** ({ts}/{tn})")
    print(f"| **TOT** | **all tasks** | " + " | ".join(tcells) + " |")


if __name__ == "__main__":
    main()
