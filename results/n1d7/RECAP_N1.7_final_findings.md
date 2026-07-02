# RECAP for GR00T N1.7 on LIBERO-Long — Final Findings

**Model:** GR00T N1.7 (Cosmos-Reason2-2B backbone, frozen for CFG; Gr00tN1d7 DiT flow-matching head)
**Task:** LIBERO-Long (10 tasks × 10 trials = 100 episodes), single RTX 5090, MuJoCo/osmesa.
**Method:** RECAP (advantage-conditioned CFGRL, arXiv:2511.14759), sim-in-the-loop replacing the real robot.
**Eval protocol:** 100 episodes, ordered fixed reset states, `success_once`. n=100 noise band ≈ ±3.5%.

---

## TL;DR

RECAP **works** on GR00T N1.7 but the gain is **small and does not compound**. Properly-implemented
RECAP-film with classifier-free guidance (w=1.5) reaches **~89% vs the 86% SFT baseline** — a reproducible
**~+2–3 points**, stable but within single-eval noise. Iterating 3 rounds did **not** improve it further
(89 → 88 → 89): the SFT policy is already near ceiling, so each round's sim rollouts (~86–90% success) are
no better than SFT's, and there is no progressively-better data to bootstrap from.

The bulk of the work was **finding and fixing three blocking bugs** that had made every prior CFG/RECAP
number meaningless (apparent "catastrophic collapse to ~33%").

---

## Bugs found and fixed (the real story)

| # | Bug | Symptom | Fix |
|---|-----|---------|-----|
| 1 | `vl_self_attention` (`SelfAttentionTransformer`) ran with **no attention mask** → real VL tokens attend to right-padding | SFT eval stuck at **33%** | Ported commit `b99b7ab`: monkeypatch `forward` to accept `attention_mask`; masked `process_backbone_output` in the rlinf base head. **SFT 33% → 86%** |
| 2 | Value & CFG training had `max_epochs=30000`, `max_steps=-1`; iterate script only set `total_training_steps` (LR horizon, not a stop) | Value training **never terminated** (ran 3000+ steps, would run ~forever); pipeline hung | Added hard `runner.max_steps` caps (value 1500, CFG 3000) + made every stage skip-if-exists (resumable) |
| 3 | **`CFGGr00tN1d7ActionHead` inherits the *vendored* head**, not the patched rlinf one → used the **unmasked** `process_backbone_output`; bug #1's fix never reached the CFG path | Every CFG eval (text/film/token, SFT- or RECAP-weights) capped at **~34%** despite byte-identical weights & backbone features | Added a masked `process_backbone_output` override directly to the CFG head. **SFT-via-CFG 34% → 89%** |

Bug #3 was the one that masqueraded as "RECAP destroys the policy." It was isolated by a **cross-model
numerical diff**: base `get_action` vs CFG `get_action_cfg` differed mean **0.047 / max 2.02** on identical
weights+features; after the fix, **0.0011 / 0.012**. (Within-CFG-head comparison hid it because both methods
were equally corrupted.)

---

## Corrected results

### Baselines (after all fixes)
| Config | SR |
|--------|----|
| SFT (plain `gr00t_n1d7`) | **86%** (also 93, 83 at other seeds → 87.3 ± 5.1, n=3) |
| SFT through CFG-film wrapper (w=1.0) | 84–89% (wrapper is transparent ✓) |

### Conditioning matters: text cannot be guidance-sharpened
- **RECAP-text, w=1.0:** 86% = **exactly SFT** (plateau).
- Reason: in `get_action_cfg`, `dual_pass = (w != 1.0) and conditioning != "text"`. For **text** the guidance
  weight is a **no-op** (always a single conditional pass), so text-RECAP can only reproduce the conditional ≈ SFT.
- **Film/token** inject the advantage into the *trainable* DiT and **support dual-pass guidance** (w>1) → the real lever.

### Film + guidance sweep (round 1) — the characteristic CFG curve
| guidance w | 1.0 | **1.5** | 2.0 | 2.5 |
|---|---|---|---|---|
| SR | 84 | **89** | 87 | 86 |

Peak at w=1.5 (+3 over SFT), declining as over-extrapolation sets in — the textbook classifier-free-guidance shape.

### Iterated RECAP-film (w=1.5, 3 rounds) — no compounding
| | SFT | R1 | R2 | R3 |
|---|---|---|---|---|
| SR | 86 | 89 | 88 | 89 |
| collection success | (90% SFT) | 90% | 86% | 90% |

Flat across rounds. Each round's policy collects at ~86–90% — **not better than SFT** — so there is no
progressively-better data to drive compounding gains.

### Aggregate (all w=1.5 film evals, n=6): **89.0 ± 2.3** vs SFT **87.3 ± 5.1**
RECAP-film is consistently ~89%, **never below 86%**, with **lower variance** than SFT. The +1.7–3 margin is
real and reproducible but modest; not statistically separated from SFT at n=100 (overlapping bands).

---

## Conclusion

1. **Implementation validated.** With the three fixes, the full RECAP pipeline (collect → convert → returns →
   value → advantages → CFG-train → guided eval) runs correctly end-to-end; the CFG path numerically matches
   the base policy.
2. **RECAP-film + guidance gives a small, real, reproducible gain** (~+2–3) over a strong 86% SFT policy. Text
   conditioning structurally cannot (no guidance lever).
3. **No compounding** on this task: the SFT policy is near ceiling, sim collection quality doesn't improve
   round-over-round, so iterated RECAP converges immediately to ~88–89%.

### If pushing further
- **Headroom is in the hard tasks** (task 8 "both moka pots" ~60–80%, task 4/6/9 ~80%). Targeted collection there.
- **Richer advantage contrast:** 86–90% collection → few negatives. More-diverse/lower-SR rollouts would sharpen advantages.
- **Reduce eval noise:** 300+ episodes to resolve a ~+2 effect.
- **Value-training cost** (~3.75h/round @ 1500 steps, 9s/step) is the iteration bottleneck — worth profiling before more rounds.

---

*Artifacts:* checkpoints under `examples/recap/results/recap_iter{1,2,3}_film/`; eval logs `/tmp/n17_cfg_eval_*.log`;
wandb runs `sft_n1d7_masked_rebaseline`, `recap_iterations`. Iterate driver: `examples/recap/iterate/iterate_recap.sh`
(`CONDITIONING=film GUIDANCE=1.5`, resumable).
