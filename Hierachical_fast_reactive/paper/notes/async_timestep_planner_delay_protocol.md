# Async Timestep Planner-Delay Protocol

Updated: 2026-06-08

This protocol is the Phase 1 LIBERO simulation for HFRVLA planner latency. It
does not modify LeRobot source code and does not use LeRobot RTC. It implements
the split-system semantics inside the HFRVLA policy plugin:

- The slow planner side observes at discrete control step `t`.
- The slow planner produces a frozen SmolVLA base action chunk `A_t` plus the
  corresponding slow context (`z_goal`, `z_phase`).
- The chunk becomes available after `d` control timesteps, at `t+d`.
- On arrival, the executor immediately replaces the active base-action queue.
- Because `d` control steps have elapsed since observation `o_t`, execution
  starts at `A_t[d:]`, not `A_t[0:]`.
- The fast wrist correction remains local to the executor and runs every control
  step from the current wrist camera and robot state:

```text
a_final = a_base + alpha * clip(delta_a)
```

Main Phase 1 settings:

```text
planner_delay_mode = async_timestep
planning_chunk_size = 50
execution_chunk_size = 16
async_request_interval_steps = 8
planner_delay_steps = 0,1,2,3,4
policies = hfrvla,hfrvla_disable_fast
suite = libero_spatial
```

The constraint for this protocol is:

```text
async_request_interval_steps + planner_delay_steps <= execution_chunk_size
```

For the main settings, `8 + 4 <= 16`, so all delay values remain inside the
planned latency budget.

Debug fields written by the policy include:

- `async_request_step_last`
- `async_observation_step_last`
- `async_ready_step_last`
- `async_activate_step_last`
- `async_chunk_start_index_last`
- `async_chunk_start_index_mean`
- `async_dropped_old_queue_steps_last`
- `async_dropped_old_queue_steps_mean`
- `async_event_trace`

Interpretation: this is a deterministic control-timestep latency simulation.
It is meant to isolate whether current wrist feedback reduces degradation when
slow-planner chunks arrive late. A later deployment version can map the same
semantics onto a LeRobot async-inference split where the remote server runs
only the frozen slow planner and the client keeps the fast wrist correction local.

Related notes:

- Change log: `paper/notes/async_timestep_planner_delay_change_log.md`.
- Async-timestep eval result: `paper/notes/async_timestep_planner_delay_eval_results.md`.
