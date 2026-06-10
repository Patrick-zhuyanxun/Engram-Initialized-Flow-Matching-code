# Async-Timestep Planner-Delay Change Log

Updated: 2026-06-08

Purpose: make the planner-delay stress test match the intended split-system
setup without changing LeRobot source code. The simulated server side is the
frozen slow planner; the client side keeps HFRVLA wrist residual correction
local at every control step.

## Code Changes

- `policy/lerobot_policy_hfrvla/src/lerobot_policy_hfrvla/configuration_hfrvla.py`
  - Keeps `planner_delay_mode` serialized for auditability, with
    `async_timestep` as the only valid value.
  - Keeps `async_request_interval_steps`.
  - Keeps `planner_delay_trace_max_events`.
  - Validates `async_request_interval_steps + planner_delay_steps <= n_action_steps`
    for planner-delay eval.

- `policy/lerobot_policy_hfrvla/src/lerobot_policy_hfrvla/modeling_hfrvla.py`
  - Uses async-timestep behavior as the only planner-delay runtime:
    - request at control step `t`;
    - slow planner observes `o_t`;
    - chunk `A_t` becomes ready at `t+d`;
    - the active queue is replaced immediately when ready;
    - execution starts from `A_t[d]`, not `A_t[0]`;
    - fast wrist residual still runs locally every `select_action` step.
  - Added debug stats for request, ready, activation, chunk start index, dropped
    old queue steps, and bounded event traces.

- `scripts/run_planner_delay_eval_sweep.py`
  - Runs the async-timestep protocol by default.
  - Default grid is `planner_delay_steps=0..4`.
  - Default arms are `hfrvla` and `hfrvla_disable_fast`.
  - Supports `--async-request-interval-steps`.
  - Writes async metadata and debug counters to the sweep CSV.
  - Names output directories with both delay mode and `N`.

- `scripts/build_eval_results_master.py`
  - Propagates planner-delay mode, async interval, request/activation counts,
    chunk-start index, and dropped-old-queue statistics into the master CSV.

## Tests And Verification

- `tests/test_hfrvla_planner_delay.py`
  - Covers async config validation.
  - Covers activation from `A_t[d]`.
  - Covers a deterministic select-action sequence where `N=3,d=2` activates a
    new chunk at the expected control step.

- `tests/test_run_planner_delay_eval_sweep.py`
  - Checks that async sweep commands pass the new policy config and output
    directory naming.

- `tests/test_build_eval_results_master.py`
  - Checks that async planner-delay fields survive registry rebuild.

Verified commands:

```bash
/home/hucenrotia/Robotic_infra/lerobot/.venv/bin/python -m pytest \
  tests/test_hfrvla_planner_delay.py \
  tests/test_run_planner_delay_eval_sweep.py \
  tests/test_build_eval_results_master.py -q

python3 scripts/build_eval_results_master.py --check

git diff --check
```

## Paper-Facing Documentation

- Added `paper/notes/async_timestep_planner_delay_protocol.md`.
- Added this change log.
- Copied the async-timestep planner-delay summary plot into:
  - `docs/assets/hfrvla-paper/async-timestep-planner-delay-summary.png`
  - `docs/presentations/hfrvla-training-open-slide/assets/hfrvla-paper/async-timestep-planner-delay-summary.png`
- Updated `docs/presentations/hfrvla-training-open-slide/slides/hfrvla-training/index.tsx`
  with:
  - an `async_timestep` protocol slide;
  - an `async_timestep` result slide;
  - updated experiment matrix and registry counts.

## Eval Plan Being Run

Phase 1 only:

```text
suite = libero_spatial
policies = hfrvla,hfrvla_disable_fast
planner_delay_mode = async_timestep
async_request_interval_steps = 8
planner_delay_steps = 0,1,2,3,4
planning_chunk_size = 50
n_action_steps = 16
n_episodes = 10 per LIBERO-Spatial task
alpha = 0.5
delta_max = 0.2
fallback = hold_last
```

This run is the one to use for the final planner-delay claim.
