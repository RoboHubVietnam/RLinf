"""GR00T-N1.7 inference server for closed-loop RoboTwin eval (runs in the
gr00t venv, vla-rlft-n1d7). Pairs with robotwin_eval_client.py (robotwin venv).

Cross-venv by design: RoboTwin (sapien/mplib) and GR00T (torch/cosmos) don't
coexist in one venv, so the policy serves over a socket. Length-prefixed
pickle protocol (numpy-friendly).

Protocol (client -> server):
  {"cmd": "act", "head": (H,W,3) uint8, "left": ..., "right": ...,
   "state": (14,) float32, "instruction": str}
    -> {"action": (16,14) float32}
  {"cmd": "ping"} -> {"ok": True}

Run:
  cd vla-rlft-n1d7/gr00t
  PYTHONPATH=$PWD:$PWD/examples USE_TF=0 HF_HUB_OFFLINE=1 \
  python <repo>/examples/robotwin/gr00t_eval_server.py \
    --model-path /data/checkpoints/robotwin_bowls_ft/checkpoint-5000 --port 5556
"""
import argparse
import pickle
import socket
import struct

import numpy as np


def recvall(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def recv_msg(sock):
    raw = recvall(sock, 4)
    if raw is None:
        return None
    (length,) = struct.unpack(">I", raw)
    return pickle.loads(recvall(sock, length))


def send_msg(sock, obj):
    data = pickle.dumps(obj)
    sock.sendall(struct.pack(">I", len(data)) + data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--embodiment-tag", default="new_embodiment")
    ap.add_argument("--port", type=int, default=5556)
    ap.add_argument("--lang-key", default="annotation.human.action.task_description")
    ap.add_argument("--cfg-model-path", default="",
                    help="serve a RECAP advantage-conditioned (CFG) checkpoint: "
                         "build GR00T_N1_7_ForCFG from this path and graft it "
                         "into the Gr00tPolicy wrapper (--model-path = base SFT "
                         "ckpt supplying the processor/transforms)")
    ap.add_argument("--cfg-weights", default="",
                    help="RLinf FSDP save (actor/model_state_dict/full_weights.pt) "
                         "loaded over the CFG model after construction")
    ap.add_argument("--cfg-guidance-weight", type=float, default=1.0)
    ap.add_argument("--cfg-conditioning", default="film")
    ap.add_argument("--cfg-conditioning-label", default="pos",
                    choices=["pos", "neg", "null"],
                    help="which advantage row the conditional branch reads "
                         "(POS default; neg/null for the label-semantics diagnostic)")
    ap.add_argument("--cfg-xt-seed", type=int, default=None,
                    help="fix the initial flow noise x_T for deterministic / "
                         "common-random-number paired evals (None = unseeded)")
    args = ap.parse_args()

    # register the RoboTwin modality config (examples/RoboTwin/robotwin_config.py)
    import examples.RoboTwin.robotwin_config  # noqa: F401
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    policy = Gr00tPolicy(
        embodiment_tag=EmbodimentTag.resolve(args.embodiment_tag),
        model_path=args.model_path,
        device="cuda",
    )
    print(f"[server] Gr00tPolicy loaded from {args.model_path}", flush=True)

    if args.cfg_model_path:
        # Swap in the advantage-conditioned model; the policy wrapper keeps
        # supplying processor transforms (identical modality configs).
        import torch
        from omegaconf import OmegaConf
        from rlinf.models.embodiment.gr00t.gr00t_n1d7_cfg import (
            get_model as get_cfg_model,
        )

        cfg = OmegaConf.create({
            "model_path": args.cfg_model_path,
            "backbone_model_path": "/data/checkpoints/Cosmos-Reason2-2B",
            "embodiment_tag": args.embodiment_tag,
            "num_action_chunks": 16,
            "obs_converter_type": "libero",  # unused: policy wrapper feeds
            "denoising_steps": 4,            # normalized inputs directly
            "precision": "bf16",
            "is_lora": False,
            "load_to_device": True,
            "advantage_cfg_dropout_prob": 0.3,
            "cfg_guidance_weight": args.cfg_guidance_weight,
            "positive_only_conditional": False,
            "conditioning": args.cfg_conditioning,
            "cfg_conditioning_label": args.cfg_conditioning_label,
            "cfg_xt_seed": args.cfg_xt_seed,
            "add_value_head": False,
            "rl_head_config": {
                "joint_logprob": False, "noise_method": "flow_sde",
                "ignore_last": False, "safe_get_logprob": False,
                "noise_anneal": False, "noise_params": [0.7, 0.3, 400],
                "noise_level": 0.5, "action_noise_scale": 0.1,
                "add_value_head": False, "chunk_critic_input": False,
                "detach_critic_input": True, "disable_dropout": True,
                "use_vlm_value": False, "value_vlm_mode": "mean_token",
                "padding_value": 570,
            },
        })
        cfg_model = get_cfg_model(cfg, torch_dtype=torch.bfloat16)
        if args.cfg_weights:
            sd = torch.load(args.cfg_weights, map_location="cpu", weights_only=True)
            missing, unexpected = cfg_model.load_state_dict(sd, strict=False)
            print(f"[server] loaded {args.cfg_weights}: "
                  f"{len(missing)} missing / {len(unexpected)} unexpected keys", flush=True)
            assert not unexpected, f"unexpected keys: {unexpected[:5]}"
        cfg_model.to("cuda")
        cfg_model.eval()  # NOTE: RLinf's eval() override may not return self
        policy.model = cfg_model
        assert policy.model is not None
        print(f"[server] CFG model grafted from {args.cfg_model_path} "
              f"(w={args.cfg_guidance_weight}, {args.cfg_conditioning})", flush=True)

    def build_obs(m):
        # nested {"video":{...},"state":{...},"language":{...}} with (B=1,T=1)
        return {
            "video": {
                "head": m["head"][None, None].astype(np.uint8),
                "left_wrist": m["left"][None, None].astype(np.uint8),
                "right_wrist": m["right"][None, None].astype(np.uint8),
            },
            "state": {"joint_pos": m["state"][None, None].astype(np.float32)},
            "language": {args.lang_key: [[m["instruction"]]]},
        }

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", args.port))
    srv.listen(1)
    print(f"[server] listening on :{args.port}", flush=True)

    while True:
        conn, _ = srv.accept()
        while True:
            msg = recv_msg(conn)
            if msg is None:
                break
            try:
                if msg.get("cmd") == "ping":
                    send_msg(conn, {"ok": True}); continue
                result = policy.get_action(build_obs(msg))
                action = result[0] if isinstance(result, tuple) else result
                joints = action.get("action.joints")
                if joints is None:
                    joints = next(v for k, v in action.items() if "joints" in k)
                joints = np.asarray(joints, np.float32)
                if joints.ndim == 3:      # (B,T,D) -> (T,D)
                    joints = joints[0]
                send_msg(conn, {"action": joints})
            except Exception as e:
                import traceback; traceback.print_exc()
                send_msg(conn, {"error": str(e)})
        conn.close()


if __name__ == "__main__":
    main()
