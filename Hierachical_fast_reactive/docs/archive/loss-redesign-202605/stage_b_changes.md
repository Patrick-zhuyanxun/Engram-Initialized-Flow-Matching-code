# Stage B — HFRVLA principled 5-term rate-distortion objective

**Date**: 2026-05-22
**Status**: ⚠️ **Parity with Stage A v2 (24/50). Gate is dramatically more selective (0.36 → 0.17), but closed-loop ceiling did not move.**
**Predecessor**: Stage A v2 (`d659a90`)
**Source of design**: `docs/hfrvla_objective_debate_20260521.md` §3b (the five-term consolidated form)

## Why Stage B exists

Stage A v2 was a *patch* on the existing loss: detach gate, add a zero-target term, calibrate two thresholds. It worked — eval 12/50 → 24/50, gate_prior 0.95 → 0.36 — but Sonnet's R1 explicitly called it a hotfix, not the final answer. The debate synthesis predicted Stage B (replacing the loss with the principled 5-term rate-distortion form) at 28-30/50.

Stage B is the principled rewrite. If the prediction were right, Stage A v2 would be obsoleted.

## What changed

### Fastcache schema v2 — static dataset labels

`scripts/build_hfrvla_fastcache.py` now writes two extra columns:

- `y_correct.npy` — `uint8`, shape `[N]`. True iff `err_offline > p80(err_offline)`. ~20% of frames.
- `y_preserve.npy` — `uint8`, shape `[N]`. True iff `err_offline < p50(err_offline)`. ~50% of frames.

`err_offline = ((a_expert - a_base) ** 2).sum(axis=-1)` per frame. Percentiles are computed once over the whole dataset; thresholds are stored in `meta.json` for sanity-check.

LIBERO empirical thresholds (273,465 frames):
- `p80 ≈ 5.51` (correction)
- `p50 ≈ 2.68` (preserve)

`CACHE_SCHEMA_VERSION` bumped from 1 → 2. The dataset class accepts both v1 (Stage A path) and v2 (Stage B path). Loading v1 with `use_stage_b_objective=True` errors with a clear message.

### `fast_cache_dataset.py` — multi-root support

`HFRVLAFastCacheDataset(roots=[...])` virtually concatenates memmaps from multiple cache roots. `__getitem__(idx)` figures out which root the idx falls into. This is the infrastructure Stage C uses to add rollout data, but it's introduced in Stage B for the schema change.

### Five-term Stage B loss in `modeling_hfrvla.py::_compute_losses`

Branch on `self.config.use_stage_b_objective`:

```python
if self.config.use_stage_b_objective:
    u = self._clip_fast_residual(out.delta_a)
    r_t = self._clip_fast_residual(a_expert - a_base)

    y_correct  = batch["observation.extra.y_correct"].to(out.delta_a.dtype)
    y_preserve = batch["observation.extra.y_preserve"].to(out.delta_a.dtype)

    # L_correct: SmoothL1 between δ and r_t, masked to y_correct frames
    correct_per_frame = F.smooth_l1_loss(out.delta_a, r_t, reduction='none').sum(dim=-1)
    n_correct = y_correct.sum().clamp(min=1.0)
    l_correct = (correct_per_frame * y_correct).sum() / n_correct

    # L_preserve_zero: ||δ||² on y_preserve frames
    preserve_per_frame = out.delta_a.pow(2).sum(dim=-1)
    n_preserve = y_preserve.sum().clamp(min=1.0)
    l_preserve_zero = (preserve_per_frame * y_preserve).sum() / n_preserve

    # L_rate: mean gate + batch budget hinge
    g = torch.sigmoid(out.gate_logit)
    l_budget = F.relu(g.mean() - float(self.config.gate_task_budget)) ** 2
    l_rate = g.mean() + l_budget

    # L_gate: focal BCE with pos_weight against static y_correct
    p = torch.sigmoid(out.gate_logit)
    p_t = p * y_correct + (1 - p) * (1 - y_correct)
    alpha = float(self.config.focal_pos_weight) * y_correct + (1 - y_correct)
    gamma = float(self.config.focal_gamma)
    focal_weight = (1 - p_t).clamp(min=1e-6) ** gamma
    log_p_t = torch.log(p_t.clamp(min=1e-6))
    l_gate = -(alpha * focal_weight * log_p_t).mean()

    # L_smooth: temporal smoothness on (g_detached · u)
    gu = g.detach().unsqueeze(-1) * u
    diff = gu[:, 1:] - gu[:, :-1]
    l_smooth = diff.pow(2).sum(dim=-1).mean()

    total = (
        1.0 * l_correct
      + 2.0 * l_preserve_zero
      + 0.5 * l_rate
      + 3.0 * l_gate
      + 0.2 * l_smooth
    )
```

(Plus `L_contact` aux as in Stage A.)

### `configuration_hfrvla.py` — new config fields

```python
use_stage_b_objective: bool = False  # toggle, default off for back-compat
loss_lambda_correct: float = 1.0
loss_lambda_rate: float = 0.5
loss_lambda_smooth: float = 0.2
focal_gamma: float = 2.0
focal_pos_weight: float = 4.0
gate_task_budget: float = 0.25
```

When `use_stage_b_objective=False`, the Stage A v2 path runs unchanged. All Stage A v2 tests still pass.

### `train_hfrvla_libero_merged.sh` — Stage B env vars

```bash
USE_STAGE_B=true \
LOSS_LAMBDA_PRESERVE_ZERO=2.0 \    # scaled up from 1.0 default for Stage B
  bash scripts/train_hfrvla_libero_merged.sh
```

When `USE_STAGE_B=true`, the script also bumps `LOSS_LAMBDA_GATE` from 1.0 → 3.0 (the debate's recommended weight for focal BCE).

### Tests

Added 5 lock-in tests in `tests/test_hfrvla_forward_lerobot_batch.py`:
- `test_stage_b_correct_loss_uses_static_label_only`
- `test_stage_b_preserve_zero_uses_static_label_not_threshold`
- `test_stage_b_focal_bce_pos_weight_dominates_on_positive`
- `test_stage_b_smooth_loss_zero_when_residuals_constant`
- `test_stage_b_falls_back_to_stage_a_when_flag_false`

All 31+ tests pass.

## Measured results

### Train metrics at step 30k

| Metric | Stage A v2 | **Stage B** |
|---|---:|---:|
| `train/loss` | 0.480 | 0.155 |
| `train/correct` (new) | — | 0.054 |
| `train/preserve_zero` | 0.00117 | 0.00021 |
| `train/rate` (new) | — | 0.168 |
| `train/gate` (focal BCE) | 0.0844 | 0.0045 |
| `train/smooth` (new) | — | 0.018 |
| **`train/gate_prior`** (mean gate) | 0.362 | **0.168** |
| `train/grad_norm` | 1.28 | 1.04 |

The gate is dramatically more selective: mean gate dropped from 0.36 to 0.17. The focal BCE collapsed to 0.005 — the gate is *very confidently* calibrated against the static `y_correct` labels.

### Closed-loop eval — LIBERO spatial, 50 trials

| Variant | Successes | % |
|---|---:|---:|
| Stage A v2 | 24/50 | 48.0% |
| **Stage B** | **24/50** | **48.0%** |

Per-task breakdown:

| Task | B | A v2 | zf | Notes |
|---:|---:|---:|---:|---|
| 0 | 4/5 | 2/5 | 3/5 | B improves on A |
| 1 | 3/5 | 4/5 | 2/5 | B slightly worse |
| 2 | 1/5 | 2/5 | 0/5 | |
| 3 | 4/5 | 3/5 | 3/5 | B improves |
| 4 | 0/5 | 1/5 | 2/5 | B regresses below baseline |
| 5 | 0/5 | 0/5 | 0/5 | |
| 6 | 2/5 | 4/5 | 3/5 | **B -2 vs A** — Stage A's helpful corrections suppressed |
| 7 | 3/5 | 2/5 | 3/5 | B improves to baseline |
| 8 | 3/5 | 4/5 | 3/5 | B slightly worse |
| 9 | 4/5 | 2/5 | 4/5 | **B fully recovers task 9 to baseline** |

## Verdict

Stage B's training dynamics are cleaner than A's (focal BCE near 0, gate prior precisely tracking the rate budget). But this does not translate to better closed-loop. The model is more confidently doing the same thing.

Per-task pattern reveals a **redistribution**: B recovers task 9 (+2 vs A) but loses task 6 (-2). Net total is identical.

Interpretation: Stage A's quick patch already extracted most of the available signal from the offline objective. Stage B confirms that **static labels are not the bottleneck** — switching from threshold proxy to dataset percentile gives a cleaner gate but no closed-loop benefit.

This was the debate's first signal that the ceiling was higher up the stack (closed-loop distribution shift, base model capacity, or dataset size) — which Stage C then tested.

## Pointers

- Stage A v2: `docs/stage_a_changes.md`
- Stage C: `docs/stage_c_changes.md`
- All-stages summary: `docs/stages_summary.md`
- Debate synthesis: `docs/hfrvla_objective_debate_20260521.md`
- Eval result JSON: `outputs/eval_hfrvla_v2b_stage_b_spatial/eval_info.json`
