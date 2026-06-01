# Methodology Blueprint — HFRVLA

> ★ Ground-truth design document. Update this file before changing research_questions.md / reviewer_defense.md / paper/src/.
> Last updated: 2026-05-16

---

## 1. Research Question

**Core question**
Can a minimal, wrist-camera-only reactive correction module, mounted on top of a frozen Vision-Language-Action (VLA) policy, recover from action-chunk open-loop failures on real robot hardware, without retraining the base policy?

**Hypothesis**
A small (≤ 10 M trainable params) residual policy that observes only the wrist camera, conditioned on the slow VLA's task / phase embeddings, can produce per-step corrections that:
1. Improve task success under externally applied perturbations on real hardware.
2. Run at ≥ 5× the slow VLA's chunk-update rate on the same GPU, restoring closed-loop reactivity inside the open-loop chunk window.
3. Outperform a re-implemented A2C2-style additive correction head by activating a learned confidence gate that suppresses corrections during stable transport and amplifies them near contact.

**Out-of-scope**
- Long-horizon re-planning / adaptive execution horizon (deferred).
- Expected-state / world-model prediction (deferred).
- Force / tactile sensing fusion (kept as future work).
- Multi-VLA portability (SmolVLA only for this paper).

---

## 2. Method Overview

**Core equations**

```
A_t           = SmolVLA(o_t, l, s_t)                        ← frozen, ~chunk rate
z_goal, z_phase = SmolVLA.hook(o_t, l, s_t)                 ← detach()
δa_t, g_t, c_t  = FastReactive(I_wrist_t, s_t, A_t[k], k, z_goal, z_phase)
a_final_t      = SafetyLayer( a_base_t + g_t · clip(δa_t, ±δ_max) )
```

**Key innovation triplet**
1. **G-C gate supervision** with stop-gradient on δa, defined as a prediction-improvement signal rather than a residual-magnitude detector.
2. **Task-conditioned spatial pooling (S-B)** of DINOv3 patch features using `[z_goal, z_phase]` as cross-attention queries.
3. **Auxiliary contact-event head used only during training** as multi-task regularization for the GRU temporal state.

---

## 3. Architecture

### 3.1 System 2 — Frozen SmolVLA (no modification)

| Component | Spec |
|-----------|------|
| Base | `HuggingFaceVLA/smolvla_libero` |
| Forward pass | unchanged |
| Hooks added | 2 detached tensors per chunk call: |
| `z_goal` | output of VLM/text hidden state pooled (current local dim 960) |
| `z_phase` | action-expert intermediate hidden state pooled (current `smolvla_libero` dim 480) |
| Output | action chunk `A_t ∈ R^{B×H×7}` (H = chunk horizon) |

> Implementation: wrap `SmolVLMWithExpertModel.forward` to cache pooled
> `z_goal` / `z_phase` tensors on the same step as `_get_action_chunk()`.

> Contract note: use `HuggingFaceVLA/smolvla_libero` or another checkpoint
> already adapted to LIBERO's `observation.images.image`,
> `observation.images.image2`, `observation.state(8)`, and `action(7)`
> contract. Raw `lerobot/smolvla_base` is only a warm-start checkpoint and must
> not be used directly as HFRVLA's frozen slow planner.

### 3.2 System 1 — Fast Reactive Module (trainable)

**Inputs** (per step `t`, batch dim B implicit):

| Name | Shape | Source | Notes |
|------|-------|--------|-------|
| `wrist_rgb` | `(3, 256, 256)` source / `(196, 384)` cached DINOv3 patches | wrist camera, current frame | RGB only; DINOv3 path resizes to 224 internally |
| `proprio` | `(8,)` | eef position (3) + axis-angle rotation (3) + gripper qpos (2) | LIBERO state |
| `a_base_k` | `(7,)` | `A_t[k]` from System 2 | current chunk step |
| `k_idx_norm` | `(1,)` | `k / H` ∈ [0,1] | progress index |
| `z_goal` | `(960,)` | detached from SmolVLA | task semantic feature |
| `z_phase` | `(480,)` | detached from SmolVLA | action-expert state |

**Components**:

```
1. Visual backbone (frozen):     DINOv3 ViT-S/16 (default) | ConvNeXt-Tiny (ablation)
   → patch features (196 × 384) or (49 × 96)

2. Task-conditioned pool (S-B):  1-layer cross-attention
   Q = MLP([z_goal, z_phase])  →  (1, 256)
   K, V = Linear(patches)       →  (N_patches, 256)
   Output: pooled feature       (256,)

3. State concat:                 [pool_feat ; proprio ; a_base_k ; k_idx_norm ; z_phase]
                                 → Linear → 256-d

4. Temporal (GRU):               1-layer GRU, hidden=256
   carries history within an episode (reset at episode boundary)

5. Three small MLPs:
   - delta_head    : 256 → 64 → 7   (linear output)
   - gate_head     : 256 → 64 → 1   (sigmoid)
   - contact_head  : 256 → 64 → 1   (sigmoid, training-only)
```

**Outputs**:

| Name | Shape | Activation | Use |
|------|-------|------------|-----|
| `delta_a` | `(7,)` | linear | additive residual on `a_base_k` |
| `confidence` | `(,)` | sigmoid → [0,1] | scales δa during merging |
| `contact_event` | `(,)` | sigmoid | training-only auxiliary; NOT used at inference |

**Trainable parameter count target**: < 10 M (excluding frozen backbone).

---

## 4. Training Strategy

### 4.1 Data preparation

For each LIBERO demo (20 Hz, `(o_t, a_t)` pairs):
1. Run frozen SmolVLA once over the trajectory to obtain `A_t` for every chunk boundary.
2. For each step `t` with chunk index `k`, store: `a_base_k`, `a_expert = a_t`, `z_goal`, `z_phase`, `wrist_rgb`, `proprio`, `contact_label` (from MuJoCo).
3. Pre-compute DINOv3 patch features for all wrist frames once (saves time during training).

### 4.2 Loss functions

**L_delta** — residual MSE
```
L_delta = ‖ δa - (a_expert - a_base) ‖²
```

**L_gate** — G-C (prediction-improvement with stop-gradient on δa)
```
δ̄a   = stop_grad(δa)                                    ← detach
g*   = sigmoid( ‖a_expert - a_base‖² - ‖a_expert - (a_base + δ̄a)‖² )
L_gate = BCE(g, g*)
```
Semantics: `g*` is high when the correction strictly improves over `a_base`, low when correction does not help (or hurts). Stop-gradient on `δa` removes the chicken-and-egg coupling between gate and residual.

**L_contact** — auxiliary (training-only)
```
L_contact = BCE(c, contact_label)
```

**Total loss**
```
L_total = L_delta + λ_g · L_gate + λ_c · L_contact
λ_g = 1.0,  λ_c = 0.1   (auxiliary weight, downweighted)
```

### 4.3 Training schedule

| Stage | Duration | Notes |
|-------|----------|-------|
| **Warmup** | 5 % of total steps | Only L_delta active. Gate / contact heads frozen. |
| **Joint** | 95 % | All three losses active. |
| **No fine-tune of SmolVLA** | — | Base weights frozen end-to-end. |

> Phase 2 (joint fine-tune of SmolVLA) is intentionally deferred. It would add scope and risks distorting SmolVLA's pretrained behavior.

### 4.4 Hyperparameters (initial guess; ablation will refine)

- Optimizer: AdamW, lr = 3e-4, weight_decay = 1e-4
- Batch size: 256 trajectory segments × 8 steps
- δ_max (action clip): per-dim tuned, ≈ 20 % of per-dim action range
- Steps: 100k–200k

### 4.5 Pre-training alignment / contract checks

Before long training, run two distinct checks:

1. **Eval-time alignment:** evaluate the LIBERO-adapted SmolVLA slow planner
   directly, then package HFRVLA with `inference_disable_fast=True` and evaluate
   the zero-fast wrapper with `lerobot-eval`. The two should match within
   small-N variance, because the fast module is bypassed and `select_action()`
   returns the popped `a_base` chunk action.
2. **Train-time contract:** run `scripts/check_hfrvla_training_contract.py` on
   the recorded HFRVLA dataset. It validates LeRobot metadata, sequence
   windowing, preprocessor output, and the zero-fast target
   `target_delta = normalized(action) - a_base`, with
   `action/a_base/target_delta ∈ R^{B×seq_len×7}`.

Passing both checks only proves I/O and training-target alignment. It does not
prove the fast module is trained or that the slow-planner checkpoint is strong;
if the slow planner changes, recollect the dataset so `a_base`, `z_goal`, and
`z_phase` share the same provenance.

---

## 5. Action Merging & Safety Layer

```
def merge(a_base, delta_a, gate, prev_a, dt):
    delta_clip = torch.clamp(delta_a, -δ_max, +δ_max)
    a_cand     = a_base + gate * delta_clip
    a_safe     = enforce_joint_velocity_limits(a_cand, prev_a, dt)
    return a_safe
```

**Safety contract for real hardware**:
- Joint velocity hard cap: `|Δq| / dt ≤ v_max` (manufacturer spec).
- Cartesian velocity soft cap (optional): on EE jacobian.
- Fallback when wrist frame is dropped / out-of-focus: `gate ← 0` for that step (return `a_base`).

---

## 6. Baselines (required for IROS / ICRA submission)

| # | Baseline | Why |
|---|----------|-----|
| B1 | SmolVLA alone (frozen) | Lower bound |
| B2 | SmolVLA + A2C2 reimplementation | Direct competitor; no gate, multi-view |
| B3 | SmolVLA with shortened chunk horizon (forced re-plan more often) | "Why not just call VLA more often" defense |
| B4 | SmolVLA + ResNet-18 residual (no gate, no h_task) | Backbone ablation |
| B5 | HFRVLA without gate (`gate ≡ 1`) | Gate ablation |
| B6 | HFRVLA without contact aux | Contact ablation |
| B7 | HFRVLA with ConvNeXt-Tiny backbone | Backbone ablation |

Headline comparison: B1, B2, B3 vs HFRVLA.

---

## 7. Experiments

### 7.1 Simulation (LIBERO)

| Suite | Tasks | Episodes | Purpose |
|-------|-------|----------|---------|
| LIBERO-Spatial | 10 | 20/task | Main result |
| LIBERO-Goal | 10 | 20/task | Generalization |
| **Perturbation-LIBERO** (custom) | 10 | 20/task | Inject random visual / joint perturbation during chunk |

### 7.2 Real Hardware (SO-100 / SO-101)

**Pre-committed 3 tasks** (must demonstrate all three):

| Task | Phase mix | What it tests |
|------|-----------|----------------|
| **T1: Grasp-and-place** | Transport-dominant | Gate-off behavior; should add little overhead |
| **T2: Aligned peg insertion** | Precision phase | Gate-on with corrections; primary success driver |
| **T3: Push-to-target** | Contact phase | Gate transition; force-free contact handling |

**Headline real-hardware experiment**:
- For each task, run with and without externally applied perturbation (manual push, target shift) mid-execution.
- Report success rate ± std over 20 trials per condition.
- **Headline metric**: success-rate improvement under perturbation, HFRVLA vs SmolVLA-alone vs LoRA-finetuned.

### 7.3 Gate Visualization (required)

For each real-hardware trial:
- Time-series plot: `g_t` over the trajectory, with task phase annotations.
- Expected pattern: low `g` during transport, high `g` near contact / alignment.
- This is the qualitative evidence that the gate is doing meaningful work.

---

## 8. Locked Design Decisions

> Do not change without re-running the design review.

| # | Decision | Rationale |
|---|----------|-----------|
| L1 | SmolVLA frozen throughout. No LoRA in HFRVLA path. | "Plug-in" claim, parameter efficiency baseline |
| L2 | Wrist-camera only for System 1 | Distinctive vs A2C2 (multi-view); deployment simplicity |
| L3 | **DINOv3 ViT-S/16** as primary backbone | Newest SOTA dense self-supervised features (Meta, 2025); ConvNeXt-Tiny as ablation. Patch=16, 14×14=196 patches at 224×224, ~21 M params, frozen. |
| L4 | Task-conditioned cross-attention pool (S-B) | Marginal latency cost, preserves DINO's spatial weighting |
| L5 | Single-layer GRU temporal | O(1) per step; sufficient for short-window reactivity |
| L6 | G-C gate supervision (prediction-improvement, stop-grad) | Avoids chicken-and-egg; teaches "trust", not "magnitude" |
| L7 | Contact event = training-only aux head | Auxiliary regularization; not exposed at inference |
| L8 | No re-planning trigger / adaptive horizon | Out of scope for this paper |
| L9 | "Cerebellum-like" analogy not used in paper text | Stays in talk only |
| L10 | LeRobot-compatible subclass with composed fast module | HFRVLAPolicy subclasses SmolVLAPolicy for plugin compatibility and keeps the fast residual module separate. |

---

## 9. Open Questions (to resolve during implementation)

- [ ] DINOv3 patch features pre-computation: cache on disk vs compute on-the-fly? (depends on storage budget)
- [ ] Episode-boundary GRU reset: how to detect in real-time deployment?
- [ ] λ_g, λ_c weights — grid search or principled?
- [ ] Real-robot demo collection: how many demos per task minimum?
- [ ] Camera latency on SO-100: needs measurement before claiming "fast"
- [ ] Joint vs EE-space δa: stick with joint (current plan) or move to EE?

---

## 10. Sprint Plan

| Sprint | Scope | Status |
|--------|-------|--------|
| **S0** | LeRobot plugin scaffold + SmolVLA hook for z_goal/z_phase | [ ] |
| **S1** | Offline data pipeline: rollout, δa computation, DINOv3 cache | [ ] |
| **S2** | Train HFRVLA on LIBERO-Spatial; ablate B5 (no gate), B6 (no contact) | [ ] |
| **S3** | Add B2 (A2C2 reimpl) and B3 (LoRA) baselines | [ ] |
| **S4** | LIBERO-Goal + Perturbation-LIBERO experiments | [ ] |
| **S5** | Real-robot demo collection on SO-100/SO-101 (T1–T3) | [ ] |
| **S6** | Real-robot perturbation experiments + gate visualization | [ ] |
| **S7** | Paper writing (Tectonic) + ablation table | [ ] |

---

## 11. Risk Register

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Gate collapses to constant (g≈1 or g≈0) | Med | G-C supervision; warmup curriculum on L_delta first |
| DINOv3 features too generic for wrist | Low–Med | ConvNeXt-Tiny ablation backup |
| Shortened-horizon SmolVLA wins on sim | Med | Real-hardware GPU latency makes shortened horizon infeasible; show this with timing data |
| A2C2 reimpl wins | Med-High | This is genuinely possible; if so, paper narrows to "we add real-hardware + gate visualization" |
| Real-robot demo collection time-blow | High | Limit to 3 tasks; reuse SO-100 community demos |
| Scoop by A2C2 follow-up / DuoCore-FS extension | High | Submit by next IROS / ICRA deadline; freeze design now |

---

## 12. References to Update

After every design change, update:
1. `research_questions.md` (Sec 1 above)
2. `reviewer_defense.md` (Sec 6 + 11)
3. `contribution_statement.md` (Sec 2 key-innovation triplet)

Lit review lives in `literature_review.md` (already updated 2026-05-12).

---

*AI disclosure: This blueprint was developed with Claude (architecture review, lit synthesis) and Codex (design debate) assistance. All design decisions reviewed and approved by the author.*
