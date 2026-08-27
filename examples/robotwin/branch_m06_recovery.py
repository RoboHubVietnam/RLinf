"""M0.6: policy-in-the-loop expert-recovery gate (Codex's decisive gate).

Runs the BASE policy (via the eval server); at the second-nest precontact (the
last policy chunk where stage==1: first nest done, second pending) it snapshots.
If the policy then FAILS the second nest, it restores that precontact snapshot and
hands control to the scripted MP expert to complete the missing nest.

Read:
  recovery_rate = expert-succeeds / policy-failed-with-a-stage1-snapshot.
  HIGH  -> matched-state corrections exist; build the full collector.
  LOW   -> the pre-miss states are not expert-recoverable (contact/observability
           bottleneck) -> no label recipe fixes it; stop the pilot (Codex).

Needs the eval server up (base cup ckpt, no CFG weights). Run from RoboTwin root
in the robotwin venv; the launcher script starts the server first.
"""
import argparse
import os
import pickle
import socket
import struct
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from branching import restore_state, snapshot_state  # noqa: E402


def _recvall(sock, n):
    buf = b""
    while len(buf) < n:
        c = sock.recv(n - len(buf))
        if not c:
            return None
        buf += c
    return buf


def _send(sock, obj):
    b = pickle.dumps(obj)
    sock.sendall(struct.pack(">I", len(b)) + b)


def _recv(sock):
    raw = _recvall(sock, 4)
    if raw is None:
        return None
    (n,) = struct.unpack(">I", raw)
    return pickle.loads(_recvall(sock, n))


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


def cup_stage(task):
    """(#nests 0/1/2, list of aligned-chain cup objs from bottom)."""
    cups = [task.cup1, task.cup2, task.cup3]
    ps = [np.asarray(c.get_pose().p) for c in cups]
    order = sorted(range(3), key=lambda i: ps[i][2])  # low z first
    chain = [order[0]]
    for a, b in zip(order, order[1:]):
        dz = ps[b][2] - ps[a][2]
        if (abs(ps[a][0] - ps[b][0]) < 0.04 and abs(ps[a][1] - ps[b][1]) < 0.04
                and 0.02 < dz < 0.10):
            chain.append(b)
        else:
            break
    return len(chain) - 1, [cups[i] for i in chain]


def expert_recover(task):
    """From a stage>=1 state, move the un-nested cup onto the top of the chain."""
    stage, chain = cup_stage(task)
    if stage < 1:
        return False, f"stage {stage} (<1), not a second-nest state"
    cups = [task.cup1, task.cup2, task.cup3]
    unnested = [c for c in cups if c not in chain]
    if len(unnested) != 1:
        return False, f"expected 1 un-nested cup, got {len(unnested)}"
    top = chain[-1]
    target = np.asarray(top.get_pose().p) + [0, 0, task.NEST_DZ]
    task.las_arm = None
    try:
        task.move_cup(unnested[0], target)
    except Exception as e:
        return False, f"expert move raised: {str(e)[:40]}"
    return bool(task.check_success()), ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="stack_cups_three")
    ap.add_argument("--task-config", default="cups_collect")
    ap.add_argument("--seeds", default="1,7,18,24,30,41,42,55,56,63")
    ap.add_argument("--port", type=int, default=5556)
    ap.add_argument("--action-steps", type=int, default=16)
    ap.add_argument("--lang-key", default="annotation.human.action.task_description")
    args = ap.parse_args()

    TASK_ENV, cargs = build_env(args.task, args.task_config)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(("127.0.0.1", args.port))
    _send(sock, {"cmd": "ping"}); assert _recv(sock).get("ok")

    seeds = [int(x) for x in args.seeds.split(",") if x]
    pol_succ = 0
    fail_with_snap = 0
    recovered = 0
    irrecoverable = 0
    for ep, seed in enumerate(seeds):
        TASK_ENV.setup_demo(now_ep_num=ep, seed=seed, **cargs)
        instr = getattr(TASK_ENV, "instruction", "") or "stack the cups"
        last_snap = None
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
            resp = _recv(sock)
            chunk = np.asarray(resp["action"], np.float32)[: args.action_steps]
            for a in chunk:
                if TASK_ENV.take_action_cnt >= TASK_ENV.step_lim or TASK_ENV.eval_success:
                    break
                TASK_ENV.take_action(a, action_type="qpos")
            st, _ = cup_stage(TASK_ENV)
            if st == 1:                       # second-nest precontact window
                last_snap = snapshot_state(TASK_ENV)
            elif st >= 2:
                break

        final_stage, _ = cup_stage(TASK_ENV)
        # stage 2 = both nests seated; check_success additionally requires grippers
        # open, which a mid-window policy may not satisfy. For the recovery gate we
        # treat "both nests achieved" as policy-success.
        pol_ok = bool(TASK_ENV.eval_success or TASK_ENV.check_success() or final_stage >= 2)
        pol_succ += pol_ok
        rec_tag = "-"
        if not pol_ok:
            if last_snap is None:
                rec_tag = f"no-stage1-snap (policy peaked at stage {final_stage})"
            else:
                fail_with_snap += 1
                restore_state(TASK_ENV, last_snap)
                ok, why = expert_recover(TASK_ENV)
                recovered += ok
                if not ok and "raised" in why:
                    irrecoverable += 1
                rec_tag = "RECOVERED" if ok else f"expert-fail({why[:30]})"
        print(f"seed {seed}: policy={'OK' if pol_ok else f'FAIL@stage{final_stage}'}  "
              f"recovery={rec_tag}", flush=True)

    print(f"\nM0.6 base-policy SR: {pol_succ}/{len(seeds)}")
    print(f"policy failures with a stage-1 precontact snapshot: {fail_with_snap}")
    print(f"expert recovered from those: {recovered}/{fail_with_snap}"
          f"{' (rate %.0f%%)' % (100*recovered/fail_with_snap) if fail_with_snap else ''}")
    verdict = ("PASS" if fail_with_snap and recovered / fail_with_snap >= 0.4
               else "FAIL" if fail_with_snap else "INCONCLUSIVE")
    print(f"M0.6_RECOVERY {verdict}"
          f"  (>=40% -> matched-state corrections exist; <40% -> pre-miss states "
          f"not expert-recoverable)")


if __name__ == "__main__":
    main()
