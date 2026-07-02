# RECAP Training Pipeline for GR00T N1.7 — Implementation Report

**Scope:** Port the RECAP (CFGRL advantage-conditioned RL fine-tuning, arXiv:2511.14759)
pipeline from GR00T N1.5 to GR00T N1.7 and evaluate it on LIBERO-Long (`libero_10`)
inside RLinf. Compare against the N1.7 SFT baseline.

**Headline result:** the pipeline runs end-to-end, but RECAP fine-tuning **collapses**
the N1.7 policy (41.0% SFT → 0.0% RECAP). Root cause is the conditioning *injection
site*, not the training loop (see §6).

---

## 1. Architecture choice

GR00T N1.7 = Cosmos-Reason2-2B (Qwen3-VL) backbone + `Gr00tN1d7ActionHead`
(flow-matching DiT; the LIBERO checkpoint uses the `AlternateVLDiT` variant which
alternates cross-attention between image and text tokens).

We reused the official RL model as the base and only swapped the action head:

- `CFGGr00tN1d7ActionHead(Gr00tN1d7ActionHead)` adds a learned **advantage
  embedding** — 3 tokens: NULL (unconditional / CFG-dropout), NEG (A ≤ ε), POS (A > ε).
- `GR00T_N1_7_ForCFG(GR00T_N1_7_ForRLActionPrediction)` builds the full RL model
  (so we inherit the Cosmos backbone loading, `Gr00tN1d7Processor`, obs/action
  conversion, metadata) and then **replaces** `self.action_head` with the CFG head.
  Because the CFG head shares all base submodules (state/action encoder, action
  decoder, DiT, vlln, vl_self_attention), the SFT checkpoint loads cleanly; only
  `advantage_embedding` is new.
- The Cosmos backbone is **frozen**; only the action head + advantage embedding train.

**Conditioning mechanism (the design decision under review):** the advantage token
is **appended to the DiT cross-attention encoder context** (`vl_embs → [vl_embs;
adv_token]`), with `backbone_attention_mask` extended by 1 (valid) and `image_mask`
by 0 (non-image). This mirrors our N1.5 port and the CFGRL reference.

Files: `rlinf/models/embodiment/gr00t/gr00t_n1d7_cfg/{cfg_action_head.py,
gr00t_cfg_model.py,cfg_utils.py,__init__.py}`. Registered as model type
`gr00t_n1d7_cfg` in `SupportedModel`, `models/__init__.py`, and the gr00t dispatcher.

## 2. Loss function

Classifier-free-guidance flow-matching (CFGRL). For each sample:

1. Sample flow time `t = (1 − Beta(α,β)) · noise_s` (α=1.5, β=1.0, noise_s=0.999),
   built in fp32 (torch's Beta/dirichlet is not implemented for bf16).
2. Linear-interpolation noised trajectory `x_t = (1−t)·noise + t·action`; target
   velocity `v = action − noise`.
3. **CFG routing** (`compute_cfg_routing_masks`): with prob `p_drop = 0.3` route a
   sample to **unconditional** (NULL token); otherwise **conditional** with the token
   set by the sample's advantage label (POS / NEG).
4. Loss = masked MSE between the DiT-predicted velocity and `v` (action-mask weighted).

The forward returns `(loss, metrics)` matching the `FSDPCfgWorker` contract; metrics
log conditional/unconditional and POS/NEG ratios for sanity (observed:
unconditional ≈ 0.30 = p_drop, positive ≈ 0.30 = advantage quantile — both correct).

## 3. Dataset processing (RECAP stages 1–3)

The advantage labels come from RECAP's offline value pipeline; it is policy-agnostic
and operates on the LIBERO env observations, so stages 1–3 reuse the existing scripts.

1. **Rollout collection** — run the N1.7 SFT policy on LIBERO-Long with random reset
   states across several seeds; export per-episode pickles (successes + failures).
   192 episodes, 62 success / 130 fail.
2. **Convert → LeRobot v2.1** — `convert_rollouts_to_lerobot.py`. *Critical version
   constraint:* the N1.7 `LeRobotEpisodeLoader` requires lerobot **v2.1**
   (`episodes.jsonl`); lerobot 0.4.4 writes v3.0 (unreadable), so the dataset is
   written with lerobot 0.1.0 — which also matches the value pipeline.
3. **Returns** — `compute_returns` (failure_reward −300, γ=1.0) → returns ∈ [−811, 0].
4. **Value model** — SigLIP2-so400m + Gemma3-270m critic trained to predict
   normalized returns (Spearman **0.88**).
5. **Advantages** — `compute_advantages` with `A = norm(r_{t:t+N}) + γ^N V(o_{t+N}) −
   V(o_t)`, top 30% labelled positive → `advantages_rollout.parquet` (98,304 rows).

**Stage-4 CFG data loader (new, N1.7-specific):**
`rlinf/data/datasets/recap/gr00t_n1d7_cfg.py`. N1.7's data path differs entirely from
N1.5's eagle collator. A per-frame `Dataset` reuses the official
`LeRobotEpisodeLoader` + `extract_step_data` + `Gr00tN1d7Processor` (so state/action
normalization matches the checkpoint exactly), attaches the per-step advantage from
the parquet, and collates with the processor's own `Gr00tN1d7DataCollator` into the
`(observation, actions, advantage)` 3-tuple. Validated against the real dataset
(95,424 frames; correct `input_ids/pixel_values/state/action` shapes). Video decode
required upgrading `torchcodec` 0.2.0 → 0.4.0 for torch-2.7 ABI compatibility.

## 4. Training

- Worker: `FSDPCfgWorker` (added a `gr00t_n1d7_cfg` branch to build the loader).
- FSDP `no_shard`, `use_orig_params=True` (frozen backbone + trainable head trips the
  flat-param view-inplace guard otherwise), `reduce_dtype=fp32` (bf16 grad reduction
  gave NaN norms on N1.5), gradient checkpointing off.
- bf16 master weights; advantage embedding deterministically re-initialized after
  load (HF `from_pretrained` materializes the missing CFG param from uninitialized
  memory → NaN if not re-init — a bug we hit and fixed on N1.5).
- LR 2e-6 (deliberately conservative: on N1.5, lr=1e-5 degraded the policy; 2e-6
  preserved it), cosine, 500 warmup, 5000 steps, micro-batch 2 / global 32.
- Config: `examples/recap/cfg/config/libero_cfg_gr00t_n1d7.yaml`. Logged to
  TensorBoard + Weights & Biases. Loss/grad-norm finite throughout; checkpoints at
  every 1000 steps.

## 5. Evaluation

Standalone eval (`task_type: embodied_eval`) with `model_type=gr00t_n1d7_cfg` and
`runner.ckpt_path=<full_weights.pt>`; CFG-guided denoising (`get_action_cfg`):
the POS advantage token conditions the velocity field, with optional guidance weight
`w` (`pred = uncond + w·(cond − uncond)`; w=1 → pure conditional, w=0 → unconditional).
Protocol: LIBERO-Long, **10 runs/task** (100 episodes; interleaved fixed reset states),
`success_once`. Same harness as the SFT baseline, so the comparison is apples-to-apples.

## 6. Results & diagnosis

| Model | LIBERO-Long SR (10/task) |
|---|---|
| GR00T N1.7 SFT (baseline) | **41.0%** |
| RECAP CFG @ 1000 steps | 0.0% |
| RECAP CFG @ 3000 steps | 0.0% |
| RECAP CFG @ 3000, unconditional (w=0) | 0.0% |
| No-checkpoint control (SFT weights + untrained token) | 0.0% |

RECAP collapses the policy. Diagnosis isolates the cause to the **conditioning
injection site, not the training**:
- A single-step numerical test shows `get_action_cfg` **without** the appended token
  is byte-identical to the SFT denoiser (which scores 41%).
- Collapse occurs for **any** appended token — POS or NULL, trained or untrained.
- Therefore appending an extra token to N1.7's `AlternateVLDiT` cross-attention
  context destabilizes the *closed-loop* policy (a ~0.003 per-step action perturbation
  compounds catastrophically over a 512-step rollout). N1.5's Eagle backbone tolerated
  the same mechanism (RECAP only degraded there); N1.7's alternating-attention DiT
  does not.

**Recommended next step:** change the conditioning injection from "append a token to
the cross-attention context" to an **additive** scheme — add the advantage embedding
to the timestep/state embedding, or FiLM-modulate the DiT — which does not alter the
encoder token sequence. The data pipeline, value model, advantages, training loop, and
eval are all built and reusable; only the injection site needs to change.

## 7. Artifacts

- Code: `gr00t_n1d7_cfg/` (model), `rlinf/data/datasets/recap/gr00t_n1d7_cfg.py`
  (loader), configs under `examples/recap/{cfg,value,process}/config/*_n1d7*.yaml`,
  eval configs/scripts under `evaluations/libero/`.
- Results: `results/n1d7/task3_comparison_table.md`, eval logs, this report.
- wandb (project `rlinf`): value `a2l6ll22`, CFG training `v2uxlvef`.
