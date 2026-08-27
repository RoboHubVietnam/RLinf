"""Failure-mode diagnostic: where do base-policy cup rollouts actually fail?

The branching pilot's intervention premise assumes failures are second-nest
PRECISION misses (reach stage 1, miss stage 2). E040/E041 reported mean stage
0.97/2, but a mean of ~1 is equally consistent with a success / early-fail MIX.
This measures the real distribution: per episode, the PEAK stage reached and the
final outcome. If failures are stage-0-heavy (never complete the first nest), the
"correct the second nest" intervention is aimed at the wrong window.

No snapshotting, no expert — just the taxonomy. Base policy via the validated CFG
serving path. Run from RoboTwin root in the robotwin venv (launcher starts server).
"""
import argparse
import os
import pickle
import socket
import struct
import sys
from collections import Counter

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from branch_m06_recovery import build_env, cup_stage, _send, _recv  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="stack_cups_three")
    ap.add_argument("--task-config", default="cups_collect")
    ap.add_argument("--seeds", default="1,7,18,24,30,41,42,55,56,63,69,81,82,92,94,95,96,108")
    ap.add_argument("--port", type=int, default=5556)
    ap.add_argument("--action-steps", type=int, default=16)
    args = ap.parse_args()

    TASK_ENV, cargs = build_env(args.task, args.task_config)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(("127.0.0.1", args.port))
    _send(sock, {"cmd": "ping"}); assert _recv(sock).get("ok")

    seeds = [int(x) for x in args.seeds.split(",") if x]
    peak_all, peak_fail = Counter(), Counter()
    nsucc = 0
    for ep, seed in enumerate(seeds):
        TASK_ENV.setup_demo(now_ep_num=ep, seed=seed, **cargs)
        instr = getattr(TASK_ENV, "instruction", "") or "stack the cups"
        peak = 0
        while TASK_ENV.take_action_cnt < TASK_ENV.step_lim and not TASK_ENV.eval_success:
            o = TASK_ENV.get_obs()["observation"]
            _send(sock, {
                "cmd": "act",
                "head": np.asarray(o["head_camera"]["rgb"], np.uint8),
                "left": np.asarray(o["left_camera"]["rgb"], np.uint8),
                "right": np.asarray(o["right_camera"]["rgb"], np.uint8),
                "state": np.asarray(TASK_ENV.get_obs()["joint_action"]["vector"], np.float32),
                "instruction": instr,
            })
            chunk = np.asarray(_recv(sock)["action"], np.float32)[: args.action_steps]
            for a in chunk:
                if TASK_ENV.take_action_cnt >= TASK_ENV.step_lim or TASK_ENV.eval_success:
                    break
                TASK_ENV.take_action(a, action_type="qpos")
            peak = max(peak, cup_stage(TASK_ENV)[0])
            if peak >= 2:
                break
        final = cup_stage(TASK_ENV)[0]
        ok = bool(TASK_ENV.eval_success or TASK_ENV.check_success() or final >= 2)
        nsucc += ok
        peak_all[peak] += 1
        if not ok:
            peak_fail[peak] += 1
        print(f"seed {seed}: success={ok} peak_stage={peak} final_stage={final}", flush=True)

    n = len(seeds)
    print(f"\nbase SR: {nsucc}/{n} = {100*nsucc/n:.0f}%")
    print(f"peak-stage distribution (all):     "
          f"{ {s: peak_all[s] for s in (0,1,2)} }")
    print(f"peak-stage distribution (failures):"
          f"{ {s: peak_fail[s] for s in (0,1,2)} }")
    nf = sum(peak_fail.values())
    if nf:
        s1plus = peak_fail[1] + peak_fail[2]
        print(f"of {nf} failures: stage-0 (never nested) = {peak_fail[0]} "
              f"({100*peak_fail[0]/nf:.0f}%), reached stage>=1 = {s1plus} "
              f"({100*s1plus/nf:.0f}%)")
        print("VERDICT: " + ("STAGE-1 failures dominate -> second-nest intervention window is right"
                             if s1plus >= peak_fail[0] else
                             "STAGE-0 failures dominate -> policy fails EARLY; second-nest premise is wrong"))
    print("DIAG_DONE")


if __name__ == "__main__":
    main()
