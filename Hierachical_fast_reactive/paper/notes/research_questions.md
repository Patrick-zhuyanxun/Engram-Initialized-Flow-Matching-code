# Research Questions — HFRVLA

> Synced with the current A2C2-Wrist / Fast Wrist Residual mainline.
> Last updated: 2026-06-06

---

## Primary RQs

**RQ1**: Can a small wrist-camera residual module improve execution of frozen
SmolVLA action chunks without fine-tuning the slow planner?

**RQ2**: Under matched planning/execution protocols, where does wrist residual
correction help, and where does long-horizon chunk staleness still dominate?

**RQ3**: Which residual merge settings (`alpha`, `delta_max`, residual safety
limit) are stable enough for `n_action_steps=50` deployment?

---

## Current Ablation Design

| Ablation ID | Variable | Fixed | Claim tested | Priority |
|-------------|----------|-------|--------------|----------|
| **A-alpha-clip** | alpha x `delta_max` | frozen checkpoint, `n_action_steps=50` | residual magnitude is calibrated, not arbitrary | primary |
| **A-action-step** | execution/replan interval | planning chunk 50 | correction interacts with replan frequency | primary |
| **A-matched-K** | planning and execution horizon | same policy family | long matched chunks expose open-loop drift | primary |
| **A-latent** | keep/remove `z_goal` / `z_phase` | same wrist features | wrist evidence needs global task context | secondary |
| **A-target** | simultaneous vs delayed chunk-age target | same architecture | training must match stale deployment distribution | primary |

Retired from the current paper mainline: learned gate, contact auxiliary head,
GRU-centered recurrent head, and gate-preserve/rate losses. These can remain
engineering history, but they should not anchor the first paper narrative.

---

## Metrics

**Primary**:
- LIBERO-Spatial and LIBERO-Object success rate.
- Matched planning/execution success under fixed seed.
- Per-task success vector to expose spatial/observability failures.

**Secondary**:
- Residual norm and clip fraction.
- Action safety-limiter activation.
- Eval latency and maximum safe `eval.batch_size`.

---

## Simulation First

- LIBERO-Spatial is the active diagnostic suite because it exposes spatial
  misalignment and long-chunk drift.
- LIBERO-Object remains useful for combined paper tables, but current
  calibration sweeps should not use Object until Spatial trends are clear.
- The generated-checkpoint spatial 10x10 action-step and matched-chunk sweeps
  are completed at `alpha=0.5`, `delta_max=0.2`, and `eval.batch_size=3`;
  use the eval registry and HTML dashboard for numbers rather than promoting
  them directly into manuscript claims.
- A2C2-style comparison should be written carefully: HFRVLA is wrist-centric
  but still uses slow-planner context, not a pure wrist-only policy.

---

## Hypotheses

**H1**: HFRVLA improves matched short-execution settings over frozen SmolVLA
when both policies use the same planning/execution override.

**H2**: Long matched chunks remain difficult unless the training target includes
stale base actions generated from older observations.

**H3**: Alpha/clip calibration is a first-order deployment variable; the same
residual head can help or hurt depending on effective residual cap.

---

## Fallback Positioning

If long-horizon improvements remain weak, position the paper as a rigorous
diagnostic and reproducible LeRobot-native artifact for frozen VLA chunk
correction, with delayed-target training as the next method contribution.
