# Async-Timestep Delay Experiment Plan - HFRVLA

> Last updated: 2026-06-08

This note defines the planner-delay experiment for the HFRVLA paper. The only
paper-facing protocol is the async-timestep simulation.

## Core Definition

Delay is slow-planner chunk generation latency, not wrist-feedback latency. At
control step `t`, the slow planner observes `o_t` and starts producing the base
chunk `A_t`. The chunk becomes available after `planner_delay_steps = d` control
steps. At `t+d`, the executor immediately replaces the active base-action queue
and starts from `A_t[d]`, not `A_t[0]`.

The fast wrist correction remains local to the executor and runs every control
step with current wrist feedback:

```text
a_final = a_base + alpha * clip(delta_a)
```

## Main Sweep

Use the existing planning/execution distinction:

- `planning_chunk_size = 50`
- `execution_chunk_size = replan_interval_steps = 16`
- `async_request_interval_steps = 8`
- `planner_delay_steps in {0, 1, 2, 3, 4}`
- `async_request_interval_steps + planner_delay_steps <= execution_chunk_size`
- main suite: `libero_spatial`
- seed: `42`
- trial budget: `10 episodes/task`

Main arms:

1. `hfrvla`
2. `hfrvla_disable_fast`

`hfrvla_disable_fast` uses the same wrapper, queueing, delay, fallback, and
post-processing, but sets `--policy.inference_disable_fast=true` so
`select_action()` returns the selected `a_base` directly. This isolates whether
the fast wrist correction path is the source of the robustness.

## Commands

Smoke:

```bash
/home/hucenrotia/Robotic_infra/lerobot/.venv/bin/python scripts/run_planner_delay_eval_sweep.py \
  --planner-delay-steps 0,1,4 \
  --policies hfrvla,hfrvla_disable_fast \
  --planning-chunk-size 50 \
  --n-action-steps 16 \
  --async-request-interval-steps 8 \
  --suites libero_spatial \
  --task-ids '[0]' \
  --n-episodes 1 \
  --eval-batch-size 1 \
  --device cuda \
  --csv outputs/async_timestep_planner_delay_eval_sweep/smoke_results.csv \
  --eval-root outputs/async_timestep_planner_delay_eval_sweep/smoke_evals
```

Main:

```bash
/home/hucenrotia/Robotic_infra/lerobot/.venv/bin/python scripts/run_planner_delay_eval_sweep.py \
  --planner-delay-steps 0,1,2,3,4 \
  --policies hfrvla,hfrvla_disable_fast \
  --planning-chunk-size 50 \
  --n-action-steps 16 \
  --async-request-interval-steps 8 \
  --suites libero_spatial \
  --n-episodes 10 \
  --hfrvla-alpha 0.5 \
  --hfrvla-delta-max 0.2 \
  --eval-batch-size 3 \
  --device cuda
```

## Metrics

Primary:

- `success_rate`
- `change_from_delay0`
- `hfrvla_minus_disable_fast`

Required secondary metrics:

- `planner_delay_steps`
- `async_request_interval_steps`
- `async_request_count`
- `async_activation_count`
- `async_chunk_start_index_mean`
- `async_dropped_old_queue_steps_mean`
- `slow_replan_count`
- `slow_chunk_latency_ms_mean`
- `fast_applied_ratio`
- `fast_latency_ms_mean`
- `delta_norm_mean`
- `delta_clip_fraction_mean`
- `k_mean`
- `eval_s`

## Interpretation Guardrails

- Do not claim HFRVLA solves asynchronous VLA scheduling.
- This is a deterministic control-timestep latency simulation, not LeRobot RTC.
- The correct claim is narrower: HFRVLA tests whether current wrist feedback can
  locally correct a stale frozen VLA base action during delayed chunk execution.
- Real-robot rollouts remain additional validation, not the core evidence.
