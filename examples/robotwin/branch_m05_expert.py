"""M0.5: expert-recovery gate for the branching pilot.

Codex's decisive gate, cheap variant (no policy server). If the scripted MP
expert cannot complete the second nest from a RESTORED pre-contact state, then
restore->expert branching is broken and the collector cannot produce POS
corrections — no label recipe fixes that. This validates expert-resumability from
a restored snapshot in isolation, before we wire in the policy.

Protocol per seed:
  1. setup; run expert cup1 -> cup2 (reach second-nest precontact: two placed).
  2. SNAPSHOT (sim state + las_arm).
  3. expert cup3 -> check_success  [A: completes normally]
  4. RESTORE snapshot.
  5. expert cup3 again -> check_success  [B: completes from restored state]
Gate: B must succeed whenever A does. A==B success also evidences determinism.

Run from RoboTwin repo root in the robotwin venv (see branch_m0_fidelity.py).
"""
import argparse
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from branching import restore_state, snapshot_state  # noqa: E402


def build_env(task, task_config):
    sys.path.insert(0, "script")
    from collect_data import class_decorator, get_embodiment_config
    with open(f"./task_config/{task_config}.yml") as f:
        cargs = yaml.load(f.read(), Loader=yaml.FullLoader)
    emb = cargs["embodiment"]
    with open("./task_config/_embodiment_config.yml") as f:
        et = yaml.load(f.read(), Loader=yaml.FullLoader)
    ef = lambda t: "./" + et[t]["file_path"]
    if len(emb) == 1:
        cargs["left_robot_file"] = cargs["right_robot_file"] = ef(emb[0])
        cargs["dual_arm_embodied"] = True
    else:
        cargs["left_robot_file"] = ef(emb[0]); cargs["right_robot_file"] = ef(emb[1])
        cargs["embodiment_dis"] = emb[2]; cargs["dual_arm_embodied"] = False
    cargs["left_embodiment_config"] = get_embodiment_config(cargs["left_robot_file"])
    cargs["right_embodiment_config"] = get_embodiment_config(cargs["right_robot_file"])
    cargs["task_name"] = task
    cargs["save_path"] = f"./eval_result/{task}/{task_config}"
    cargs["render_freq"] = 0
    cargs["is_test"] = True
    return class_decorator(task), cargs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="stack_cups_three")
    ap.add_argument("--task-config", default="cups_collect")
    ap.add_argument("--seeds", default="1,7,18,24,30")
    args = ap.parse_args()

    TASK_ENV, cargs = build_env(args.task, args.task_config)
    seeds = [int(x) for x in args.seeds.split(",") if x]
    nA = nB = n = 0
    for ep, seed in enumerate(seeds):
        TASK_ENV.setup_demo(now_ep_num=ep, seed=seed, **cargs)
        try:
            # reach second-nest precontact: place cup1, nest cup2
            TASK_ENV.las_arm = None
            TASK_ENV.move_cup(TASK_ENV.cup1, TASK_ENV.cup1_target_pose)
            TASK_ENV.move_cup(TASK_ENV.cup2, TASK_ENV.cup1.get_pose().p + [0, 0, TASK_ENV.NEST_DZ])
        except Exception as e:
            print(f"seed {seed}: precontact setup failed ({e}); skipping", flush=True)
            continue
        n += 1
        snap = snapshot_state(TASK_ENV)
        las = TASK_ENV.las_arm

        def do_third_nest():
            tgt = TASK_ENV.cup2.get_pose().p + [0, 0, TASK_ENV.NEST_DZ]
            try:
                TASK_ENV.move_cup(TASK_ENV.cup3, tgt)
            except Exception as e:
                return False, str(e)
            return bool(TASK_ENV.check_success()), ""

        okA, eA = do_third_nest()

        restore_state(TASK_ENV, snap)
        TASK_ENV.las_arm = las
        okB, eB = do_third_nest()

        nA += okA; nB += okB
        tagA = "OK" if okA else f"fail({eA[:30]})"
        tagB = "OK" if okB else f"fail({eB[:30]})"
        print(f"seed {seed}: normal={tagA}  from-restore={tagB}  "
              f"{'MATCH' if okA == okB else 'MISMATCH'}", flush=True)

    print(f"\nM0.5 expert-recovery: normal {nA}/{n}, from-restore {nB}/{n}")
    verdict = "PASS" if (n > 0 and nB >= max(1, nA - 1)) else "FAIL"
    print(f"M0.5_EXPERT {verdict}"
          f"  (restore-resume {'works' if verdict=='PASS' else 'BROKEN'})")


if __name__ == "__main__":
    main()
