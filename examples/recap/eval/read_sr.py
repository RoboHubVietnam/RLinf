"""Read eval success rates from one or more eval tensorboard dirs and print a
single averaged line: ``success_once success_at_end num_trajectories``.

Used by the RECAP eval driver to aggregate success rates across seeds.
"""

import glob
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def read_dir(tb_dir):
    files = glob.glob(f"{tb_dir}/**/events*", recursive=True)
    once = end = ntraj = None
    for f in files:
        ea = EventAccumulator(f)
        ea.Reload()
        tags = ea.Tags().get("scalars", [])
        if "eval/success_once" in tags:
            once = ea.Scalars("eval/success_once")[-1].value
        if "eval/success_at_end" in tags:
            end = ea.Scalars("eval/success_at_end")[-1].value
        if "eval/num_trajectories" in tags:
            ntraj = ea.Scalars("eval/num_trajectories")[-1].value
    return once, end, ntraj


def main():
    dirs = sys.argv[1:]
    onces, ends, ntrajs = [], [], []
    for d in dirs:
        o, e, n = read_dir(d)
        if o is not None:
            onces.append(o * (n or 1))
            ends.append((e or 0) * (n or 1))
            ntrajs.append(n or 0)
    total = sum(ntrajs)
    if total == 0:
        print("nan nan 0")
        return
    print(f"{sum(onces) / total:.4f} {sum(ends) / total:.4f} {int(total)}")


if __name__ == "__main__":
    main()
