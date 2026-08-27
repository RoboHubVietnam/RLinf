"""Closed-loop RoboTwin eval client for a GR00T policy (runs in the robotwin
venv). Talks to gr00t_eval_server.py (gr00t venv) over a socket.

Per episode: setup RoboTwin task env (planner-free — the policy acts, the env
steps), set instruction, loop get_obs -> [socket] GR00T action chunk (16,14)
-> take_action(chunk,'qpos') -> check_success. Reports SR over N episodes.

Run FROM the RoboTwin repo root (needs `envs`, `script` on the path):
  cd /home/duynguyen/Desktop/RESEARCH/VR/RoboTwin
  source <robotwin venv>/bin/activate
  PYTHONPATH=$PWD python <repo>/examples/robotwin/robotwin_eval_client.py \
    --task stack_bowls_three --task-config demo_quickvid --episodes 20 --port 5556
"""
import argparse
import os
import pickle
import socket
import struct
import sys

import numpy as np
import yaml
import imageio.v2 as imageio

# Runs from the RoboTwin repo root, so <repo>/examples is not importable via the
# usual package path; add it explicitly for the artifact-root resolver.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from artifact_paths import resolve_artifact_dir  # noqa: E402


def recvall(sock, n):
    buf = b""
    while len(buf) < n:
        c = sock.recv(n - len(buf))
        if not c:
            return None
        buf += c
    return buf


def send_msg(sock, obj):
    d = pickle.dumps(obj)
    sock.sendall(struct.pack(">I", len(d)) + d)


def recv_msg(sock):
    raw = recvall(sock, 4)
    if raw is None:
        return None
    (n,) = struct.unpack(">I", raw)
    return pickle.loads(recvall(sock, n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="stack_bowls_three")
    ap.add_argument("--task-config", default="demo_quickvid")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--port", type=int, default=5556)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--video-dir", default="",
                    help="where to write per-episode mp4s. A RELATIVE value is "
                         "resolved under the labboard-pinned artifact root "
                         "($RLINF_ARTIFACTS, default <repo>/artifacts), so "
                         "'robotwin-cups/e049-eval' is immediately viewable; an "
                         "absolute path is used verbatim")
    ap.add_argument("--action-steps", type=int, default=16,
                    help="steps of the predicted chunk to execute per query")
    ap.add_argument("--seeds", default="",
                    help="comma-separated env seeds (default: 0..episodes-1); "
                         "unstable-scene seeds are skipped with a warning")
    ap.add_argument("--state-from", choices=["measured", "command"], default="measured",
                    help="proprio fed to the policy: measured qpos, or the last "
                         "commanded target (matches converters where state=command)")
    ap.add_argument("--ensemble-query", type=int, default=0,
                    help="ACT-style temporal ensembling: re-query every K steps and "
                         "execute the exp-weighted average of all live chunk "
                         "predictions for the current step (0 = off)")
    ap.add_argument("--color-counterfactual", action="store_true",
                    help="E041: put a DIFFERENT (non-target) cup colour in the "
                         "instruction. If SR and the chosen base cup are "
                         "unchanged, the policy ignores language (colour-blind)")
    ap.add_argument("--collect-dir", default="",
                    help="record each episode as RoboTwin-schema hdf5 (3 cams jpg + "
                         "qpos_measured + executed commands + success attr) for RECAP")
    args = ap.parse_args()

    if args.video_dir:
        # Land inside the labboard-pinned artifact root unless told otherwise,
        # so eval videos are viewable without any copy step.
        args.video_dir = str(resolve_artifact_dir(args.video_dir, create=True))
        print(f"[eval] videos -> {args.video_dir}", flush=True)

    # ---- RoboTwin env setup (mirrors script/collect_data.py main()) ----
    sys.path.insert(0, "script")
    from collect_data import class_decorator, get_embodiment_config

    cfg_path = f"./task_config/{args.task_config}.yml"
    with open(cfg_path) as f:
        cargs = yaml.load(f.read(), Loader=yaml.FullLoader)
    # resolve embodiment config paths (as collect_data does)
    emb = cargs["embodiment"]
    with open("./task_config/_embodiment_config.yml") as f:
        emb_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def emb_file(t):
        return "./" + emb_types[t]["file_path"]

    if len(emb) == 1:
        cargs["left_robot_file"] = cargs["right_robot_file"] = emb_file(emb[0])
        cargs["dual_arm_embodied"] = True
    else:
        cargs["left_robot_file"] = emb_file(emb[0])
        cargs["right_robot_file"] = emb_file(emb[1])
        cargs["embodiment_dis"] = emb[2]
        cargs["dual_arm_embodied"] = False
    cargs["left_embodiment_config"] = get_embodiment_config(cargs["left_robot_file"])
    cargs["right_embodiment_config"] = get_embodiment_config(cargs["right_robot_file"])
    cargs["task_name"] = args.task
    cargs["save_path"] = f"./eval_result/{args.task}/{args.task_config}"
    cargs["render_freq"] = 0
    cargs["is_test"] = True

    TASK_ENV = class_decorator(args.task)

    def connect():
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        s.connect((args.host, args.port))
        send_msg(s, {"cmd": "ping"})
        assert recv_msg(s).get("ok"), "server not responding"
        return s

    sock = connect()
    print("[client] connected to GR00T server", flush=True)

    def query(payload, retries=2):
        """Send one request; on socket failure reconnect and retry."""
        nonlocal sock
        for attempt in range(retries + 1):
            try:
                send_msg(sock, payload)
                resp = recv_msg(sock)
                if resp is not None:
                    return resp
                raise ConnectionError("server closed connection")
            except (ConnectionError, BrokenPipeError, OSError) as e:
                print(f"[client] socket error ({e}); reconnecting "
                      f"({attempt + 1}/{retries})", flush=True)
                try:
                    sock.close()
                except Exception:
                    pass
                if attempt == retries:
                    raise
                import time; time.sleep(3)
                sock = connect()

    from envs.utils.create_actor import UnStableError

    seeds = ([int(x) for x in args.seeds.split(",") if x != ""]
             if args.seeds else list(range(args.episodes)))
    suc = 0
    done = 0
    for ep, seed in enumerate(seeds[: args.episodes]):
        try:
            TASK_ENV.setup_demo(now_ep_num=ep, seed=seed, **cargs)
        except UnStableError as e:
            print(f"[client] seed {seed}: unstable scene, skipping ({e})", flush=True)
            try:  # tear down the half-built env — a stale robot poisons the
                TASK_ENV.close_env()  # next setup_demo (robot.reset path)
            except Exception:
                pass
            continue
        # instruction from the task's own description
        instr = "stack all three bowls by placing one inside another"
        commanded = getattr(TASK_ENV, "target_color", None)
        if commanded:  # color-conditioned cup task
            if args.color_counterfactual:
                others = [c for c in getattr(TASK_ENV, "cup_colors", {}).values()
                          if c != commanded]
                if others:
                    commanded = others[0]
            instr = f"stack the other cups into the {commanded} cup"
        TASK_ENV.set_instruction(instruction=instr)
        frames = []
        last_cmd = None  # for --state-from command
        rec = {"head_camera": [], "left_camera": [], "right_camera": [],
               "state": [], "action": []} if args.collect_dir else None

        def get_chunk():
            obs = TASK_ENV.get_obs()
            o = obs["observation"]
            if args.video_dir:
                frames.append(np.asarray(o["head_camera"]["rgb"], np.uint8))
            resp = query({
                "cmd": "act",
                "head": np.asarray(o["head_camera"]["rgb"], np.uint8),
                "left": np.asarray(o["left_camera"]["rgb"], np.uint8),
                "right": np.asarray(o["right_camera"]["rgb"], np.uint8),
                "state": (last_cmd if args.state_from == "command" and last_cmd is not None
                          else np.asarray(obs["joint_action"]["vector"], np.float32)),
                "instruction": instr,
            })
            if "error" in resp:
                raise RuntimeError(f"server error: {resp['error']}")
            return obs, np.asarray(resp["action"], np.float32)  # obs, (16,14)

        if args.ensemble_query > 0:
            # ACT-style temporal ensembling: re-query every K steps, execute the
            # exp-weighted average of all live predictions for the current step.
            # (--collect-dir is not recorded in this mode.)
            K, H_PRED, M = args.ensemble_query, 16, 0.25
            live, t = [], 0
            try:
                while TASK_ENV.take_action_cnt < TASK_ENV.step_lim and not TASK_ENV.eval_success:
                    if t % K == 0:
                        live.append((t, get_chunk()[1]))
                    live = [(t0, c) for t0, c in live if t - t0 < min(H_PRED, len(c))]
                    preds = [c[t - t0] for t0, c in live]
                    w = np.exp(-M * np.arange(len(preds))[::-1])  # newest heaviest
                    a = np.average(np.stack(preds), axis=0, weights=w).astype(np.float32)
                    TASK_ENV.take_action(a, action_type="qpos")
                    last_cmd = a
                    t += 1
            except RuntimeError as e:
                print(f"[client] {e}", flush=True)

        while args.ensemble_query == 0 and \
                TASK_ENV.take_action_cnt < TASK_ENV.step_lim and not TASK_ENV.eval_success:
            try:
                obs, full_chunk = get_chunk()
            except RuntimeError as e:
                print(f"[client] {e}", flush=True); break
            chunk = full_chunk[: args.action_steps]
            for a in chunk:  # take_action applies ONE (14,) target at a time
                if TASK_ENV.take_action_cnt >= TASK_ENV.step_lim or TASK_ENV.eval_success:
                    break
                if rec is not None:
                    # pre-action obs for THIS step (chunk steps >0 re-fetch)
                    so = obs if not rec["action"] or len(rec["action"]) % args.action_steps == 0 \
                        else TASK_ENV.get_obs()
                    oo = so["observation"]
                    import cv2
                    for cam in ("head_camera", "left_camera", "right_camera"):
                        ok_, buf = cv2.imencode(".jpg", cv2.cvtColor(
                            np.asarray(oo[cam]["rgb"], np.uint8), cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                        rec[cam].append(buf.tobytes())
                    rec["state"].append(np.asarray(so["joint_action"]["vector"], np.float32))
                    rec["action"].append(np.asarray(a, np.float32))
                TASK_ENV.take_action(a, action_type="qpos")
                last_cmd = np.asarray(a, np.float32)

        if args.video_dir and frames:
            import os as _os; _os.makedirs(args.video_dir, exist_ok=True)
            imageio.mimwrite(f"{args.video_dir}/ep{ep}.mp4", frames, fps=25, quality=7)
        ok = bool(TASK_ENV.eval_success or TASK_ENV.check_success())
        try:  # E041 audit: base cup identity/colour + stage score (0/1/2 nests)
            cups = [TASK_ENV.cup1, TASK_ENV.cup2, TASK_ENV.cup3]
            ps = [c.get_pose().p for c in cups]
            order = sorted(range(3), key=lambda i: ps[i][2])   # low -> high
            base_i = order[0]
            cols = getattr(TASK_ENV, "cup_colors", {})
            base_col = cols.get(cups[base_i], "?")
            stages = 0
            for a, b in zip(order, order[1:]):
                dz = ps[b][2] - ps[a][2]
                if (abs(ps[a][0] - ps[b][0]) < 0.04 and abs(ps[a][1] - ps[b][1]) < 0.04
                        and 0.02 < dz < 0.10):
                    stages += 1
                else:
                    break
            print(f"[audit] seed {seed}: commanded={commanded} base_cup=cup{base_i+1} "
                  f"base_colour={base_col} stages={stages}/2 success={ok}", flush=True)
        except Exception as e:
            print(f"[audit] failed: {e}", flush=True)
        if rec is not None and rec["action"]:
            import h5py
            os.makedirs(args.collect_dir, exist_ok=True)
            with h5py.File(f"{args.collect_dir}/episode{ep}.hdf5", "w") as g:
                g.attrs["success"] = ok
                g.attrs["seed"] = seed
                g.attrs["instruction"] = instr
                g.create_dataset("joint_action/vector", data=np.stack(rec["action"]))
                g.create_dataset("observation/qpos_measured", data=np.stack(rec["state"]))
                dt = h5py.special_dtype(vlen=np.dtype("uint8"))
                for cam in ("head_camera", "left_camera", "right_camera"):
                    d = g.create_dataset(f"observation/{cam}/rgb", (len(rec[cam]),), dtype=dt)
                    for i, b in enumerate(rec[cam]):
                        d[i] = np.frombuffer(b, np.uint8)
        suc += ok
        done += 1
        print(f"[{args.task}] ep {ep + 1}/{args.episodes} (seed {seed}): "
              f"{'SUCCESS' if ok else 'fail'}  | SR {suc}/{done} "
              f"({suc / done * 100:.1f}%)", flush=True)
        TASK_ENV.close_env()

    print(f"CLOSED_LOOP_SR {suc}/{done} = {suc / max(done, 1) * 100:.1f}%")
    sock.close()


if __name__ == "__main__":
    main()
