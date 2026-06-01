# Research Questions — HFRVLA

> Synced with `methodology_blueprint.md` (single source of truth).
> Last updated: 2026-05-14

---

## Primary RQ

**RQ1**: Can a wrist-camera-only confidence-gated residual policy, running at every control step (≥ 5× the chunk-update rate), recover task success under perturbations on real hardware without retraining the slow VLA?

## Secondary RQs

**RQ2**: Does a prediction-improvement-supervised confidence gate (G-C) outperform an additive correction head (A2C2-style) on action-chunk open-loop tasks?

**RQ3**: Does an auxiliary contact-event prediction head improve the gate's phase-awareness (visualized as low gate during transport, high during contact)?

**RQ4**: Is task-conditioned spatial pooling (S-B, cross-attn with z_goal/z_phase) measurably better than global pooling (S-A) for wrist-camera residual control on 6 DoF tabletop manipulation?

---

## Ablation Design

| Ablation ID | Variable | Fixed | Expected Effect | Maps to RQ |
|-------------|----------|-------|-----------------|------------|
| **A-gate** | gate vs `gate ≡ 1` | rest of architecture | gate-on improves perturbation SR by ≥ 5 % | RQ2 |
| **A-contact** | with vs without contact aux | rest of architecture | aux improves gate visualization meaningfulness | RQ3 |
| **A-pool** | S-B vs S-A | rest of architecture | S-B improves SR by 1–3 %; latency cost negligible | RQ4 |
| **A-backbone** | DINOv2-S vs ConvNeXt-Tiny | rest of architecture | DINOv2 wins on perturbation; ConvNeXt cheaper | secondary |
| **A-freq** | step-level vs chunk-level System 1 firing | rest of architecture | step-level beats chunk-end-only by perturbation SR | RQ1 |

---

## Key Metrics

**Primary**:
- Real-robot task success rate (%) on T1–T3 under perturbation
- Trainable-parameter count vs success rate (Pareto)

**Secondary**:
- Gate visualization match to manually annotated task phases (qualitative agreement %)
- Mean inference latency per step (ms) at 4090 / SO-100 deployment GPU
- Recovery time from external perturbation (steps until task success continues)

**Tertiary** (for ablation):
- LIBERO-Spatial / LIBERO-Goal success rate (sim)
- A2C2 reimpl vs HFRVLA head-to-head on identical sim setup

---

## Pre-registered Hypotheses

H1: HFRVLA improves real-robot success rate under perturbation by ≥ 10 percentage points over SmolVLA alone.

H2: HFRVLA's per-step inference latency is ≤ 1/5 of SmolVLA's chunk-update latency, enabling closed-loop control at 5× higher effective frequency.

H3: G-C gate supervision produces gate values that visually correlate (Spearman ρ > 0.5) with annotated task phases.

If H1 fails → reposition paper around "gate visualization + open-loop diagnosis" rather than success-rate gain.
If H2 fails → re-measure on target deployment hardware; if still failing, reposition as "semantic-aware visual servoing" without the speed claim.
If H3 fails → drop "phase-aware gate" claim; report gate as a learned scalar without semantic interpretation.

---

*Last updated: 2026-05-14*
