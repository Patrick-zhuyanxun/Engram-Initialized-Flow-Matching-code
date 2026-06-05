# Contribution Statement — HFRVLA

> Last updated: 2026-06-05

## One-Sentence Summary

HFRVLA studies whether a small wrist-camera residual module can correct frozen
SmolVLA action chunks at execution time, improving closed-loop LIBERO
manipulation without fine-tuning the slow VLA planner.

## Current Method Position

The current paper direction is the A2C2-Wrist baseline, not the older gated
HFRVLA path:

- Slow planner: frozen `HuggingFaceVLA/smolvla_libero`.
- Fast path: wrist DINO patches plus robot/slow-planner context.
- Merge rule: `a_final = a_base + alpha * clip(delta_a)`.
- Output: full 7D residual correction.
- Current objective: raw residual MSE against `action[t] - a_base[t]`.
- Temporal input: previous/current frames through training-time `seq_len=2`;
  fast-cache remains frame-level.
- Retired from the current mainline: GRU, gate, contact auxiliary loss, and
  gated preserve/rate objectives.

## Main Contributions To Argue

1. **Wrist-camera fast correction on frozen VLA chunks.** HFRVLA keeps the
   SmolVLA planner frozen and trains only a small per-step correction module.
   This isolates whether fast wrist evidence can repair chunk execution errors
   without changing the slow planner.

2. **Simple residual merge without a gate.** The current system intentionally
   uses `a_base + alpha * clip(delta_a)` instead of a learned gate. This makes
   the first paper claim easier to interpret: any improvement comes from the
   learned residual and its wrist/slow-context conditioning, not from a separate
   intervention policy.

3. **Clear planning-vs-execution evaluation protocol.** The eval registry
   separates planning horizon (`planning_chunk_size`) from execution/replan
   interval (`execution_chunk_size`, `replan_interval_steps`). This is important
   because default SmolVLA every-step replanning is not the matched baseline for
   long action-chunk execution.

4. **Reproducible LeRobot-native artifact.** HFRVLA is implemented as a LeRobot
   policy plugin, records a custom LeRobotDataset v3 with cached SmolVLA/DINO
   features, and tracks formal results in `experiments/eval_registry/`.

## Current Evidence Snapshot

All numbers below are from `experiments/eval_registry/eval_results_master.csv`,
seed 42, spatial + object combined, 100 episodes per policy/setting.

| Sweep | Policy | Planning | Execution / replan | Combined |
|---|---|---:|---:|---:|
| action-step | HFRVLA 30k `alpha=0.5` | 50 | 2 | 85.0% |
| action-step | SmolVLA | 50 | 2 | 82.0% |
| action-step | HFRVLA 30k `alpha=0.5` | 50 | 8 | 81.0% |
| action-step | SmolVLA | 50 | 8 | 77.0% |
| matched chunk | HFRVLA 30k `alpha=0.5` | 8 | 8 | 83.0% |
| matched chunk | SmolVLA | 4 | 4 | 79.0% |
| matched chunk | HFRVLA 30k `alpha=0.5` | 50 | 50 | 46.0% |
| matched chunk | SmolVLA | 50 | 50 | 43.0% |

## Active June 2026 Diagnostic

The current foreground research question is whether the residual merge can be
calibrated for `n_action_steps=50` without changing the large framework. The
active LIBERO-Spatial sweep uses 50 episodes per setting, eval batch size 3,
alpha in `{0.25, 0.5, 0.75, 1.0}`, and `delta_max` in
`{0.05, 0.1, 0.15, 0.18, 0.2, 0.22, 0.25, 0.3, 999}`. Early completed rows
show `alpha=0.25` improves only when the clip reaches `0.2`; the `0.22` row is
currently running:

| Alpha | `delta_max` | Spatial success |
|---:|---:|---:|
| 0.25 | 0.05 | 23/50 = 46.0% |
| 0.25 | 0.10 | 23/50 = 46.0% |
| 0.25 | 0.15 | 24/50 = 48.0% |
| 0.25 | 0.18 | 24/50 = 48.0% |
| 0.25 | 0.20 | 27/50 = 54.0% |

Treat these rows as calibration diagnostics, not final paper evidence, until
the expanded sweep completes.

## Positioning vs. Closest Prior Work

| Prior | Their position | Current HFRVLA differentiation |
|---|---|---|
| A2C2-style action chunk correction | Correct frozen VLA chunks with an auxiliary correction policy. | HFRVLA tests a wrist-camera correction path with a LeRobot-native frozen SmolVLA setup and explicit planning/execution chunk sweeps. |
| RDP / fast-slow tactile or phase-aware systems | Use additional modalities or phase/contact structure for fast correction. | Current HFRVLA intentionally avoids tactile/contact labels and asks how far wrist vision plus slow-planner context can go. |
| Default SmolVLA replanning | Replans frequently from the full VLA. | HFRVLA targets the chunk-execution regime where running the full VLA every control step is expensive or undesirable. |

## Wording Guardrails

- It is accurate to say the **per-step visual correction signal** is wrist-camera
  based.
- Do not claim the entire policy is wrist-only: `a_base`, `z_goal`, `z_phase`,
  and robot state remain part of the correction context.
- Do not compare HFRVLA `n_action_steps=8` directly against default SmolVLA
  `n_action_steps=1` as the main matched baseline.
- Keep table numbers tied to the eval registry, not hand-copied terminal logs.

## Limitations To Address

- Current evidence is LIBERO simulation, not real hardware.
- Improvements depend on chunk execution settings; long matched chunks remain
  difficult for both HFRVLA and SmolVLA.
- The current residual is alpha-sensitive, so calibration and training-time
  deployment-scale alignment remain open experimental questions.
- Wrist-only visual evidence may be insufficient for tasks requiring broader
  scene context.
