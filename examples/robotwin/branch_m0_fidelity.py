"""M0: snapshot/restore fidelity gate for the branching pilot.

Cheapest possible validation before building the collector: if we cannot save the
cup-task sim state and restore it bit-for-bit, matched-state branching is
impossible and the whole pilot is dead. No policy server, no expert — pure sim.

Protocol per seed: setup -> settle -> SNAPSHOT -> step forward (state must
DIVERGE) -> RESTORE -> the live state must return to the snapshot (~0 diff).

Run from the RoboTwin repo root in the robotwin venv:
  cd /home/duynguyen/Desktop/RESEARCH/VR/RoboTwin
  source <robotwin venv>/bin/activate
  ROBOTWIN_DENOISER=optix CUDA_HOME=/usr/local/cuda-12.9 PATH=/usr/local/cuda-12.9/bin:$PATH \
    PYTHONPATH=$PWD python <repo>/examples/robotwin/branch_m0_fidelity.py \
    --task stack_cups_three --task-config cups_collect --seeds 1,7,18
"""
import argparse
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from branching import restore_state, snapshot_state, state_diff  # noqa: E402


def build_env(task, task_config):
    sys.path.insert(0, "script")
    from collect_data import class_decorator, get_embodiment_config

    with open(f"./task_config/{task_config}.yml") as f:
        cargs = yaml.load(f.read(), Loader=yaml.FullLoader)
    emb = cargs["embodiment"]
    with open("./task_config/_embodiment_config.yml") as f:
        emb_types = yaml.load(f.read(), Loader=yaml.FullLoader)
    ef = lambda t: "./" + emb_types[t]["file_path"]
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
    ap.add_argument("--seeds", default="1,7,18")
    ap.add_argument("--settle", type=int, default=40)
    ap.add_argument("--diverge", type=int, default=60)
    args = ap.parse_args()

    TASK_ENV, cargs = build_env(args.task, args.task_config)
    seeds = [int(x) for x in args.seeds.split(",") if x]
    ok_all = True
    for ep, seed in enumerate(seeds):
        TASK_ENV.setup_demo(now_ep_num=ep, seed=seed, **cargs)
        for _ in range(args.settle):
            TASK_ENV.scene.step()

        snap = snapshot_state(TASK_ENV)
        # Perturb so the state genuinely MOVES (a settled scene is static under
        # free stepping): drive every active joint away from its held target, and
        # give each cup a velocity kick. Restore must undo all of it.
        for rs in snap["robot"]:
            art = rs["art"]
            for j, t in zip(art.get_active_joints(), rs["drive_targets"]):
                j.set_drive_target(float(t) + 0.15)
        for a in snap["actors"]:
            a["rb"].set_linear_velocity(np.array([0.3, 0.3, 0.2]))
        for _ in range(args.diverge):
            TASK_ENV.scene.step()
        diverged = state_diff(TASK_ENV, snap)

        restore_state(TASK_ENV, snap)
        TASK_ENV.scene.step()          # one step: dynamics must stay put if restore is faithful
        recovered = state_diff(TASK_ENV, snap)

        # after divergence the state should have moved; after restore it should be ~0
        moved = max(diverged["actor_pos"], diverged["robot_qpos"])
        back = max(recovered["actor_pos"], recovered["robot_qpos"], recovered["robot_qvel"])
        verdict = "PASS" if (moved > 1e-3 and back < 5e-3) else "FAIL"
        ok_all &= verdict == "PASS"
        print(f"seed {seed}: diverged(pos={diverged['actor_pos']:.4f} "
              f"qpos={diverged['robot_qpos']:.4f}) -> restored(pos={recovered['actor_pos']:.2e} "
              f"qpos={recovered['robot_qpos']:.2e} qvel={recovered['robot_qvel']:.2e})  {verdict}",
              flush=True)
    print(f"M0_FIDELITY {'PASS' if ok_all else 'FAIL'}")


if __name__ == "__main__":
    main()
