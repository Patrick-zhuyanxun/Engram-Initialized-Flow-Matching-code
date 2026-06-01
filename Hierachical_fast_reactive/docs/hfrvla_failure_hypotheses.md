# HFRVLA Current Failure Hypotheses

Date: 2026-05-21

This note summarizes the current suspected problems in the HFRVLA
implementation and the evidence used to form those hypotheses. It is written
as a discussion artifact, not as a final diagnosis.

## Context

HFRVLA wraps a frozen SmolVLA planner with a trainable fast reactive residual:

```text
a_final = SafetyLayer(a_base + gate * clip(delta_a))
```

The latest trained run inspected here is:

```text
checkpoints/hfrvla_conservative_seq4
```

The run completed normally at 30,000 steps. The final W&B summary was:

```text
train/loss        0.4379
train/delta       0.0162
train/final       0.3469
train/preserve    0.00058
train/gate        0.0554   # BCE loss, not mean gate
train/gate_prior  0.9530   # mean gate
train/grad_norm   0.4640
```

The absence of NaNs or runtime failures suggests this is a behavioral/design
problem, not a broken training run.

## Eval Evidence

### Task 0, Multi-Seed

`libero_spatial`, `task_id=0`, 20 episodes per seed, seeds 42/43/44:

| Variant | Successes | Success Rate |
|---|---:|---:|
| `zero_fast` | 33/60 | 55.0% |
| `trained` (`delta_max=0.2`) | 38/60 | 63.3% |
| `delta_0.05` | 39/60 | 65.0% |

Interpretation: for task 0, the fast module can provide useful corrections.
This argues against the module being entirely untrained or disconnected.

### All LIBERO Spatial Tasks

`libero_spatial`, all 10 task ids, 5 episodes per task, seed 42, `batch_size=1`
to avoid excessive EGL contexts:

| Variant | Successes | Success Rate |
|---|---:|---:|
| `zero_fast` | 23/50 | 46.0% |
| `trained` (`delta_max=0.2`) | 12/50 | 24.0% |
| `delta_0.05` | 19/50 | 38.0% |

Per-task successes:

| Task | `zero_fast` | `trained` | `delta_0.05` |
|---:|---:|---:|---:|
| 0 | 3/5 | 2/5 | 3/5 |
| 1 | 2/5 | 3/5 | 2/5 |
| 2 | 0/5 | 0/5 | 0/5 |
| 3 | 3/5 | 1/5 | 2/5 |
| 4 | 2/5 | 1/5 | 4/5 |
| 5 | 0/5 | 0/5 | 0/5 |
| 6 | 3/5 | 1/5 | 2/5 |
| 7 | 3/5 | 4/5 | 2/5 |
| 8 | 3/5 | 0/5 | 2/5 |
| 9 | 4/5 | 0/5 | 2/5 |

Interpretation: task 0 improvement does not generalize to the full
`libero_spatial` distribution. The fast residual helps some tasks, but hurts
several tasks where the frozen base is already competent.

## Current Implementation Facts

The current loss in
`policy/lerobot_policy_hfrvla/src/lerobot_policy_hfrvla/modeling_hfrvla.py`
uses:

```text
target_delta = clip(a_expert - a_base)
L_delta      = MSE(delta_a, target_delta)
a_final      = a_base + gate * clip(delta_a)
L_final      = MSE(a_final, a_expert)
L_preserve   = relu(err_final - err_before).mean()
L_gate       = BCE(gate_logit, improvement_target)
L_gate_prior = mean(gate)
```

The gate target is computed offline from action-space error:

```text
improvement = ||a_expert - a_base||^2 - ||a_expert - (a_base + clip(delta))||^2
gate_target = improvement > margin
```

At inference the same fast module is applied online to the SmolVLA action chunk,
using wrist image, proprioception, `a_base`, chunk index, `z_goal`, and
`z_phase`.

## Hypothesis 1: The Gate Is Too Easy To Open

Evidence:

- Final `train/gate_prior` is about `0.953`, which is the logged mean gate.
- This means the gate is open most of the time despite the explicit gate prior.
- `L_gate` is low (`0.0554`), so the network is confident about the current gate
  labels. That does not imply the labels are behaviorally correct.
- All-task eval shows the trained policy can be much worse than `zero_fast`
  (`24%` vs `46%`), which is consistent with corrections being applied in states
  where the base action should be preserved.

Reasoning:

The gate target is derived from one-step action MSE against expert actions. It
does not directly know whether a correction helps closed-loop rollout success.
If many offline states have a clipped residual that reduces MSE slightly, the
gate learns to open broadly. In closed loop, broad opening can destabilize tasks
where the base policy is already good.

## Hypothesis 2: Action-MSE Improvement Is Not A Reliable Success Criterion

Evidence:

- Task 0 gets better with the fast residual, but full spatial tasks get worse.
- `L_final` and `L_preserve` optimize per-step closeness to expert actions, not
  long-horizon task completion.
- `L_preserve` is almost zero at the end (`0.00058`), yet eval still degrades on
  many tasks. This suggests the preserve penalty is weak or misaligned with the
  rollout failures.

Reasoning:

The current objective asks whether the merged action is closer to the expert
under offline supervision. For visuomotor policies, small per-step action
improvements can still produce worse closed-loop trajectories if they perturb
the base policy out of its familiar state distribution. This is especially
plausible when the residual is applied repeatedly over an action chunk.

## Hypothesis 3: A Global Residual Cap Is Too Crude

Evidence:

- On task 0 seed 42:
  - `trained` with `delta_max=0.2`: 65%
  - `delta_0.05`: 75%
  - `delta_0.1`: 55%
- Across all spatial tasks:
  - `trained` with `delta_max=0.2`: 24%
  - `delta_0.05`: 38%
  - `zero_fast`: 46%

Reasoning:

Reducing `delta_max` from `0.2` to `0.05` mitigates damage but does not solve
the distribution-level problem. This suggests the residual magnitude matters,
but a single global clamp is not enough. Different tasks, phases, and action
dimensions likely need different trust levels.

## Hypothesis 4: The Fast Module May Be Learning Task-Specific Corrections
That Do Not Transfer Across LIBERO Spatial Tasks

Evidence:

- Task 0 multi-seed improves on average.
- All-task eval shows large regressions on tasks 8 and 9:
  - task 8: `zero_fast` 3/5, `trained` 0/5, `delta_0.05` 2/5
  - task 9: `zero_fast` 4/5, `trained` 0/5, `delta_0.05` 2/5
- Some tasks improve or match:
  - task 1: `zero_fast` 2/5, `trained` 3/5
  - task 7: `zero_fast` 3/5, `trained` 4/5
  - task 4: `delta_0.05` 4/5

Reasoning:

The module is not uniformly harmful. It appears to learn useful local
corrections for some task/phase combinations but lacks a reliable criterion for
when those corrections are safe elsewhere.

## Hypothesis 5: Training-Time Offline Signals Do Not Match Inference-Time
Closed-Loop Inputs Closely Enough

Evidence:

- Training uses recorded `a_base`, `z_goal`, `z_phase`, DINO wrist patches, and
  expert action labels.
- Inference recomputes SmolVLA and DINO features online and rolls out from
  states produced by previous corrected actions.
- The earlier diagnosis found that disabling fast (`zero_fast`) matches or beats
  trained fast on the all-task distribution.

Reasoning:

Even if offline cached features are aligned sample-by-sample, the policy being
evaluated changes the future state distribution. The fast module is trained on
expert/base trajectories, not on states caused by its own residuals. This can
create a compounding error mode that is invisible to one-step MSE losses.

## Hypothesis 6: The Current Design May Be Solving The Wrong Problem

Evidence:

- HFRVLA was intended as a fast reactive module, but the current training target
  is mostly "predict expert minus base action."
- The current objective does not explicitly identify reactive events such as
  contact, correction necessity, visual slip, object proximity, or base-policy
  uncertainty.
- The auxiliary contact head exists, but the final run reports `train/contact=0`
  and there is no eval evidence that contact prediction controls fast action
  activation.

Reasoning:

If the fast module is supposed to be a sparse reactive override, supervising it
as a dense residual may be conceptually wrong. It encourages the module to
always improve the base action instead of learning a narrow intervention policy.

## What The Current Evidence Does Not Prove

- It does not prove the architecture is useless. Task 0 and some all-task rows
  show nonzero positive effects.
- It does not prove longer training will solve the issue. The main failure is
  that all-task closed-loop performance drops even after the loss converged
  normally.
- It does not prove `delta_max=0.05` is the correct fix. It is a mitigation, not
  a complete solution, because it still underperforms `zero_fast` on all spatial
  tasks.
- It does not yet isolate whether the largest issue is gate labeling, residual
  scale, feature mismatch, or task imbalance. The current data mainly says the
  deployed residual is not safe enough.

## Suggested Discussion Questions

1. Should the fast module be trained as a sparse intervention policy instead of
   a dense residual regressor?
2. Should gate labels come from a stricter criterion, such as "base is likely to
   fail" or "correction is needed in this phase," instead of one-step action MSE
   improvement?
3. Should the default inference path use a safety fallback such as:
   `a_final = a_base` unless gate confidence is high and residual norm is small?
4. Should residual magnitude be predicted as a calibrated trust scalar rather
   than controlled by one global `delta_max`?
5. Should training include negative examples where the correct behavior is to
   preserve `a_base`, especially on tasks where SmolVLA is already strong?
6. Should eval and training report gate statistics by task, phase, and success
   outcome before making another architecture change?

## Proposed Next Diagnostics Before Rewriting

1. Log per-task gate mean, residual norm, and final action deviation during eval.
2. Compare successful vs failed rollouts for tasks 8 and 9, where `trained`
   collapses from nonzero base success to 0/5.
3. Add an eval variant with a much stricter gate threshold or a forced gate
   scale, without retraining, to see whether harm is mostly over-activation.
4. Add an eval variant that only enables fast in later chunk indices or near
   contact-like states.
5. Inspect offline training labels: distribution of `gate_target`, residual
   norm, and `err_before - err_after` per task.

## Current Best Summary

The current implementation probably learned a real residual, but it did not
learn a reliable decision rule for when the residual should be trusted. The
evidence points less to "not enough training" and more to a mismatch between
the dense offline residual objective and the intended sparse, safe, closed-loop
reactive behavior.
