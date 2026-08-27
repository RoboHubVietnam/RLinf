"""EMBDIAG: verify the RECAP advantage embedding actually trained.

|POS-NULL| >= 0.05 means conditioning is live; ~0.005 means it never trained
(the lr-2e-6 trap that voided E032/E040 — see docs/log/daily/2026-07-25.md).

Usage: python examples/recap/cfg/embdiag.py <ckpt_dir_or_glob>...
"""
import glob
import sys

import torch


def diag(path: str) -> None:
    for fp in sorted(glob.glob(path)):
        sd = torch.load(fp, map_location="cpu", weights_only=False)
        sd = sd.get("model", sd) if isinstance(sd, dict) else sd
        for k in [k for k in sd if "advantage_embedding" in k]:
            w = sd[k].float()
            if w.ndim == 2 and w.shape[0] >= 2:
                # Row order is fixed by AdvantageEmbedding in cfg_action_head.py:
                # NULL_IDX=0, NEG_IDX=1, POS_IDX=2. This printout used to label
                # w[0] "POS" and w[-1] "NULL" — backwards (fixed 2026-07-26).
                # |POS-NULL| is symmetric, so every LIVE/void verdict before the
                # fix stands; only the per-row norms were mislabelled.
                null_v, pos_v = w[0], w[-1]
                sep = (pos_v - null_v).norm().item()
                verdict = "LIVE" if sep >= 0.05 else "NOT TRAINED (void run)"
                neg = f" |NEG|={w[1].norm():.5f}" if w.shape[0] >= 3 else ""
                print(f"{fp}\n  {k}: |NULL|={null_v.norm():.5f}{neg} "
                      f"|POS|={pos_v.norm():.5f} |POS-NULL|={sep:.5f}  -> {verdict}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for a in sys.argv[1:]:
        diag(a if a.endswith(".pt") else f"{a.rstrip('/')}/**/full_weights.pt")
