# Async-Timestep Planner-Delay Eval Results

> Generated: 2026-06-08T12:19:03+00:00

Source: `outputs/async_timestep_planner_delay_eval_sweep/results.csv`.

Protocol:

- Suite: `libero_spatial`.
- Delay mode: `async_timestep`.
- `async_request_interval_steps = 8`.
- `planner_delay_steps = 0..4`.
- Policies: `hfrvla`, `hfrvla_disable_fast`.
- Planning chunk size: `50`; execution/replan interval: `16`.
- Episodes: `10` per task x `10` LIBERO-Spatial tasks = `100` episodes per row.
- HFRVLA eval alpha: `0.5`; delta max: `0.2`; fallback: `hold_last`.
- Video rendering: `HFRVLA_EVAL_MAX_VIDEOS=1`.
- Total eval wall-clock reported by rows: `4.02` hours.

## Success Rate

| Delay | HFRVLA | Disable-fast | HFRVLA - disable | HFRVLA change vs d=0 | Disable-fast change vs d=0 |
|---:|---:|---:|---:|---:|---:|
| 0 | 64.0% | 68.0% | -4.0 pp | +0.0 pp | +0.0 pp |
| 1 | 68.0% | 64.0% | +4.0 pp | +4.0 pp | -4.0 pp |
| 2 | 72.0% | 62.0% | +10.0 pp | +8.0 pp | -6.0 pp |
| 3 | 68.0% | 61.0% | +7.0 pp | +4.0 pp | -7.0 pp |
| 4 | 66.0% | 56.0% | +10.0 pp | +2.0 pp | -12.0 pp |

## Aggregate Summary

| Quantity | Value |
|---|---:|
| Mean HFRVLA success, d=0..4 | 67.60% |
| Mean disable-fast success, d=0..4 | 62.20% |
| Mean HFRVLA margin, d=0..4 | +5.40 pp |
| Mean HFRVLA margin, d=1..4 | +7.75 pp |

## Async Timing Checks

| Delay | HFRVLA chunk-start mean | Disable-fast chunk-start mean | HFRVLA dropped-old-queue mean | Disable-fast dropped-old-queue mean |
|---:|---:|---:|---:|---:|
| 0 | 0.000 | 0.000 | 8.000 | 8.000 |
| 1 | 1.000 | 1.000 | 7.958 | 7.964 |
| 2 | 2.000 | 2.000 | 7.919 | 7.921 |
| 3 | 3.000 | 3.000 | 7.890 | 7.894 |
| 4 | 4.000 | 4.000 | 7.850 | 7.860 |

## Interpretation

- The async-timestep implementation behaved as intended: the mean first executed chunk index equals `d` for both policies at every delay.
- HFRVLA did not degrade over `d=0..4`; its success rates were 64%, 68%, 72%, 68%, and 66%.
- Disable-fast degraded from 68% at `d=0` to 56% at `d=4`, a -12 pp drop.
- The HFRVLA minus disable-fast margin changed from -4 pp at `d=0` to +4, +10, +7, and +10 pp at `d=1..4`.
- The strongest defensible claim from this single-seed 100-episode/row run is that current wrist feedback prevents the degradation seen when the same wrapper disables the fast residual under async-timestep planner latency.
- Treat exact percentages as provisional until replicated across additional seeds or task-order variants.

## Artifacts

- Summary CSV: `outputs/async_timestep_planner_delay_eval_sweep/analysis_summary.csv`.
- Main panel PNG/PDF: `outputs/async_timestep_planner_delay_eval_sweep/async_timestep_planner_delay_summary.{png,pdf}`.
- Success curve PNG/PDF: `outputs/async_timestep_planner_delay_eval_sweep/success_vs_delay.{png,pdf}`.
- Drop curve PNG/PDF: `outputs/async_timestep_planner_delay_eval_sweep/drop_from_delay0.{png,pdf}`.
- Margin chart PNG/PDF: `outputs/async_timestep_planner_delay_eval_sweep/hfrvla_margin_vs_delay.{png,pdf}`.
- Timing debug PNG/PDF: `outputs/async_timestep_planner_delay_eval_sweep/async_timing_debug.{png,pdf}`.
- Per-task margin heatmap PNG/PDF: `outputs/async_timestep_planner_delay_eval_sweep/per_task_margin_heatmap.{png,pdf}`.
