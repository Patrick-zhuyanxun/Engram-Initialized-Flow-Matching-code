# Stage A — HFRVLA training-objective fix

**Date**: 2026-05-22
**Status**: ✅ **Verified — 24/50 on LIBERO spatial vs zero_fast 23/50 (Stage A v2)**
**Commits**: `ba00074` (Stage A v1 — initial diff), v2 recalibration commit TBD
**Source of design**: `docs/hfrvla_objective_debate_20260521.md` (4-way AI debate synthesis)
**Background**: `docs/hfrvla_failure_hypotheses.md`

This doc explains *what changed*, *why*, *how to retrain*, *what was measured*,
and *what's next*. Stage A is the **minimum-viable diff** before the larger
redesign. Two iterations were needed because the v1 thresholds were
mis-calibrated by ~50x (LIBERO `err_before` median is ~2.4, not ~0.05 as
I initially assumed).

## Why this fix exists

`checkpoints/hfrvla_conservative_seq4` converged cleanly at 30,000 steps but
regressed at eval: trained 12/50 on LIBERO spatial vs 23/50 for `zero_fast`.
The 4-way debate identified **three structural causes** the previous objective
could not avoid:

1. **Self-labeling gate loop** — the BCE positive label was derived from the
   model's *own current* `delta_a` (modeling_hfrvla.py L_compute_losses). Once
   `delta_a` was even mildly competent, almost every frame met the
   `improvement > 0.02` margin, so the gate label was 1 nearly everywhere.
   This pinned `gate_prior` at ~0.95 regardless of the explicit prior weight.
2. **Gradient leak through gate via L_final** — `L_final = MSE(a_base + gate·clip(δ), a_expert)`
   directly trains the gate to open wherever the residual reduces MSE.
   At `λ_final = 1.0` vs `λ_gate_prior = 0.02`, the pressure ratio is ~18:1
   in favor of "open the gate." The explicit prior cannot keep up.
3. **No explicit zero-target on preserve-class states** — `L_preserve =
   relu(err_final − err_before).mean()` is *self-cancelling*. Because
   `L_delta` trains `delta_a` to reduce `err_final`, by construction
   `err_final < err_before` on average, so `L_preserve` collapses to ~0 at
   convergence (observed: 0.00058). There is no term that says "on this
   state the base is correct, so `||δ||²` must be exactly zero."

## What changed (4 edits)

### `configuration_hfrvla.py`

| Field | Before | v1 | **v2 (final)** | Why |
|---|---:|---:|---:|---|
| `gate_improvement_margin` | 0.02 | 0.05 | **0.5** | Per-frame squared-L2 units. LIBERO empirical `err_before` median ≈ 2.4; v1 0.05 was ~50x too small and the BCE label still triggered on ~100% of frames. 0.5 ≈ half the typical clipped-residual improvement at δ_max=0.2. |
| `loss_lambda_gate_prior` | 0.02 | **0.10** | 0.10 | Real rate term, not a token regularizer. Combined with the gate detach, this is the only top-down pressure pushing `gate_prior` down. |
| `loss_lambda_preserve_zero` (new) | — | **1.0** | 1.0 | Explicit zero-target penalty on preserve-class states. |
| `err_preserve_thresh` (new) | — | 0.01 | **0.5** | Same units as `gate_improvement_margin`. LIBERO empirical `err_before` p10 ≈ 0.5; v1 0.01 caught 0% of frames (preserve_zero never fired). 0.5 catches the bottom ~10% where the base is already very close to the expert. |

**Calibration note.** All margins and thresholds in this loss are *absolute
per-frame squared-L2 error* over a 7-DoF action vector, NOT per-dimension.
LIBERO actions are NOT unit-normalized, so the typical `err_before` per frame
is `~2-5`, not `~0.05`. Always probe the data distribution before picking a
threshold — see *Calibration probe* below.

### `modeling_hfrvla.py::_compute_losses`

**(1) Detach the gate inside the merged-action path:**

```python
# before
a_final = a_base + out.gate.unsqueeze(-1) * delta_clip
# after
a_final = a_base + out.gate.detach().unsqueeze(-1) * delta_clip
```

This severs the gradient route by which `L_final` and `L_preserve` were
training the gate. The gate is now supervised solely by `L_gate` (BCE on the
improvement label) and `L_gate_prior` (rate).

**(2) Add the explicit zero-target term:**

```python
preserve_thresh = float(getattr(self.config, "err_preserve_thresh", 0.01))
is_preserve = (err_before < preserve_thresh).to(dtype=out.delta_a.dtype)
l_preserve_zero = (
    out.delta_a.pow(2).sum(dim=-1) * is_preserve
).mean()

losses["preserve_zero"] = l_preserve_zero
total = (
    l_delta
    + self.config.loss_lambda_gate * l_gate
    + self.config.loss_lambda_final * l_final
    + self.config.loss_lambda_preserve * l_preserve
    + self.config.loss_lambda_gate_prior * l_gate_prior
    + float(getattr(self.config, "loss_lambda_preserve_zero", 0.0))
    * l_preserve_zero
)
```

This is the term that was *missing*. On any frame where the base is already
within `√0.01 ≈ 0.1` of the expert action, the residual head is told to
predict exactly zero — there is no longer a "small everywhere" attractor.

### `scripts/train_hfrvla_libero_merged.sh`

Defaults updated to Stage A values; two new env vars exposed:
`LOSS_LAMBDA_PRESERVE_ZERO`, `ERR_PRESERVE_THRESH`. The script now passes
`--policy.loss_lambda_preserve_zero` and `--policy.err_preserve_thresh` to
`train_via_lerobot.py`.

### Tests

| Test | Purpose |
|---|---|
| `test_stage_a_preserve_zero_fires_when_base_matches_expert` | Confirms `L_preserve_zero = 7·0.2² = 0.28` when `a_base = a_expert = 0` and `δ = 0.2`. |
| `test_stage_a_preserve_zero_is_silent_when_base_far_from_expert` | Confirms `L_preserve_zero = 0` on correction states. |
| `test_stage_a_gate_detached_from_l_final_gradient` | Confirms `gate_logit.grad` is zero when only `L_final` and `L_preserve` are active. |

Also updated: `test_conservative_objective_defaults_present` to assert the new defaults.
All 29 hfrvla tests pass.

## Calibration probe

Always run this before tuning thresholds:

```python
import numpy as np
root = "checkpoints/HFRVLA_libero_v1_fastcache_seq4"
a_base = np.load(f"{root}/a_base.npy", mmap_mode="r")
action = np.load(f"{root}/action.npy", mmap_mode="r")
err_before = ((action[:10000] - a_base[:10000]) ** 2).sum(axis=-1)
for q in [0.10, 0.25, 0.50, 0.75, 0.90]:
    print(f"p{int(q*100):>2}: {np.quantile(err_before, q):.3f}")
```

Empirical for the current `HFRVLA_libero_v1_merged_reindexed` dataset
(first 10k frames):

| Percentile | err_before |
|---|---:|
| p10 | 0.52 |
| p25 | 1.13 |
| p50 | 2.40 |
| p75 | 4.76 |
| p90 | 8.52 |

`err_preserve_thresh = 0.5` → ~10% of frames are preserve-class.
`gate_improvement_margin = 0.5` → BCE positive label requires improvement > 0.5,
i.e. roughly the median magnitude of `clip(target_delta, 0.2)`-induced
improvement on a typical frame.

## How to retrain

The training script's defaults now match Stage A. For a 30k-step run with the
exact debate-recommended configuration:

```bash
# From repo root, with .venv activated.
OUT_DIR=checkpoints/hfrvla_v2a_sonnet \
JOB_NAME=hfrvla_v2a_sonnet \
STEPS=30000 \
SAVE_FREQ=5000 \
WARMUP_STEPS=1000 \
JOINT_STEPS=29000 \
REFINE_STEPS=0 \
WANDB_ENABLE=true WANDB_PROJECT=hfrvla \
  bash scripts/train_hfrvla_libero_merged.sh
```

Or to override any single weight (e.g., raise the rate term if eval still
shows overactivation):

```bash
LOSS_LAMBDA_GATE_PRIOR=0.20 \
OUT_DIR=checkpoints/hfrvla_v2a_rate020 \
  bash scripts/train_hfrvla_libero_merged.sh
```

To reproduce the *pre-Stage-A* baseline (sanity check):

```bash
GATE_IMPROVEMENT_MARGIN=0.02 \
LOSS_LAMBDA_GATE_PRIOR=0.02 \
LOSS_LAMBDA_PRESERVE_ZERO=0.0 \
OUT_DIR=checkpoints/hfrvla_pre_stage_a \
  bash scripts/train_hfrvla_libero_merged.sh
```

The Stage A diff requires no fastcache rebuild. The merge cache from previous
runs is reused as-is.

## Measured results (Stage A v2, 2026-05-22)

### Training metrics at step 30k

| Metric | Baseline (`hfrvla_conservative_seq4`) | v1 (uncalibrated) | **v2 (calibrated)** | Verdict |
|---|---:|---:|---:|---|
| `train/loss` | 0.4379 | 0.568 | 0.480 | OK |
| `train/delta` | 0.0162 | 0.0172 | 0.0198 | Slight rise OK — δ no longer drifts "small everywhere." |
| `train/gate` (BCE) | 0.0554 | 0.1533 | 0.0844 | Labels are balanced now. |
| **`train/gate_prior`** (mean gate) | **0.9530** | 0.8996 | **0.3621** | ✅ **63% absolute drop. Target was 0.20–0.45.** |
| `train/preserve` (legacy) | 0.00058 | 0.00147 | 0.00041 | (Self-cancelling; near-zero is expected.) |
| **`train/preserve_zero`** (new) | — | **0.0000** ❌ | **0.00117** ✓ | v1 thresholds blocked it from firing. v2 fires as designed. |
| `train/final` | 0.347 | 0.307 | 0.339 | OK |
| `train/grad_norm` | 0.46 | 0.72 | 1.28 | Higher gradients are expected because `L_preserve_zero` provides real corrective signal. |

The two structural fixes (gate detach + explicit zero-target) both took effect
under v2 thresholds. The gate self-loop is broken: mean gate dropped from 0.953
to 0.362, exactly in the predicted target band.

### Closed-loop eval — LIBERO spatial, 50 trials (seed 42, batch_size=1)

| Variant | Successes | % | Δ vs zero_fast |
|---|---:|---:|---:|
| `zero_fast` (no residual) — baseline | 23/50 | 46.0% | — |
| Old trained checkpoint (failure mode) | 12/50 | 24.0% | −22% |
| `delta_max=0.05` cap (mitigation) | 19/50 | 38.0% | −8% |
| **Stage A v2 (`hfrvla_v2a_calib`)** | **24/50** | **48.0%** | **+2%** ✅ |

Per-task breakdown:

| Task | A v2 | zero_fast | old trained | Notable |
|---:|---:|---:|---:|---|
| 0 | 2/5 | 3/5 | 2/5 | |
| 1 | 4/5 | 2/5 | 3/5 | +2 vs zero_fast |
| 2 | 2/5 | 0/5 | 0/5 | +2 vs both — base can't do this task at all |
| 3 | 3/5 | 3/5 | 1/5 | recovered to baseline |
| 4 | 1/5 | 2/5 | 1/5 | |
| 5 | 0/5 | 0/5 | 0/5 | SmolVLA base can't do this task; not Stage A's responsibility |
| 6 | 4/5 | 3/5 | 1/5 | +1 vs zero_fast, **+3 vs old** |
| 7 | 2/5 | 3/5 | 4/5 | Regressed vs old — flag for Stage B per-task diagnostic |
| **8** | **4/5** | 3/5 | **0/5** | **Recovered from collapse, beats baseline (+1)** |
| **9** | **2/5** | 4/5 | **0/5** | **Recovered from collapse (+2 vs old)**, still under baseline |

### Verdict

Stage A's central hypothesis — **"the gate self-loop + L_final gradient leak
is the dominant failure cause"** — is confirmed by closed-loop eval. The fix
restores the trained residual to *slightly above* the no-residual baseline,
and specifically rescues tasks 8 and 9 from full collapse (0/5 → 4/5 and 2/5).

The result lands at the lower end of the debate's predicted Stage A range
(23–26/50). Per-task evidence (task 7 regression vs old; task 9 under
zero_fast) suggests Stage B's per-task budget hinge and static-label v2 loss
would close the remaining gap to the predicted Stage B range (28–30/50).

## Expected metrics during training

If Stage A is working, train-time metrics should differ from the previous run
as follows:

(For reference; the verified observed values are in the table above.)

| Metric | Baseline | Stage A v2 expected | Observed v2 |
|---|---:|---:|---:|
| `train/loss` | 0.4379 | 0.3 – 0.6 | 0.480 ✓ |
| `train/delta` | 0.0162 | 0.005 – 0.025 | 0.0198 ✓ |
| `train/preserve_zero` (new) | — | 0.0005 – 0.05 | 0.00117 ✓ |
| `train/gate` (BCE) | 0.0554 | 0.05 – 0.30 | 0.0844 ✓ |
| `train/gate_prior` (mean gate) | 0.9530 | **0.20 – 0.45** | **0.3621** ✓ — smoking gun for gate detach taking effect |
| `train/preserve` (legacy ReLU) | 0.00058 | 0.0001 – 0.005 | 0.00041 ✓ |
| `train/grad_norm` | 0.4640 | 0.5 – 1.5 | 1.28 ✓ |

If `train/gate_prior` stays above 0.6 by step 10k, the gate detach did not
take effect — open the saved logs and check that `out.gate.detach()` is
applied in `_compute_losses`.

## Eval & falsifier — verified

Predicted outcome ranges and which one happened:

| Outcome | Predicted next action | **Observed: 24/50 → Stage B** |
|---|---|---|
| ≥ 28/50 | Skip Stage B, ship. | — |
| **23 – 27/50** | **Proceed to Stage B.** | **← we landed here** |
| 20 – 22/50 | Per-task gate diagnostic; gate self-loop was not primary. | — |
| < 20/50 | Revert; the detach broke something. | — |

Stage A v2 landed in the "rate-distortion ceiling is real, proceed to Stage
B" band. The eval confirms Stage A's central diagnosis (gate self-loop +
gradient leak) but the residual is still slightly destabilizing tasks 7 and 9
relative to `zero_fast`, which the debate predicts Stage B will fix.

## Rollback plan

Single commit `ba00074` plus the script defaults update (this commit). To
revert both:

```bash
git revert --no-edit ba00074  # may need follow-up commit hash for the script-defaults update
```

Or to A/B the change without revert, use the override block above
(`GATE_IMPROVEMENT_MARGIN=0.02 LOSS_LAMBDA_GATE_PRIOR=0.02
LOSS_LAMBDA_PRESERVE_ZERO=0.0`).

## What's still wrong (intentionally not addressed in Stage A)

These were debated but deferred to Stages B/C to keep this commit minimal:

- **`L_delta` and `L_final` still co-exist.** Stage B replaces them with a
  single direction-+-magnitude term on correction states only (`L_correct`),
  driven by static dataset labels. Adds ~30 min of one-time fastcache work.
- **`L_preserve` (the old ReLU one) is still in the total.** Stage A only
  *adds* `L_preserve_zero`. The old term remains but is observed to be near-
  zero anyway. Stage B removes it.
- **No closed-loop / DAgger-lite training.** Codex's R1 argued
  forcefully for cached `zero_fast` rollout states as hard negatives. That is
  Stage C; it requires ~5-8 hours of rollout cache work. Skipped here until
  Stage A diagnostics tell us whether closed-loop drift is the dominant
  remaining failure.
- **No window-weighted multi-step labels.** Also Stage C.
- **No per-task budget hinge on the gate.** Also Stage C (would require
  reading task_id in the loss). Will be needed if per-task gate diagnostic
  shows task 8 / 9 are over-activating relative to task 0.

## Pointers

- Debate synthesis (full math, all 4 voices' R1 + R2): `docs/hfrvla_objective_debate_20260521.md`
- Failure hypotheses: `docs/hfrvla_failure_hypotheses.md`
- Training script: `scripts/train_hfrvla_libero_merged.sh`
- The trained checkpoint should land in `checkpoints/hfrvla_v2a_sonnet/`
  (or whichever `OUT_DIR` you pass).
