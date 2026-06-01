# Codex Handoff — HFRVLA training-objective project status (2026-05-25)

This is the complete record of work done on the HFRVLA loss redesign project,
hand-off-ready for Codex.

## Project context

- **Repo**: `/home/hucenrotia/Patrick/VLA_research/Hierachical_fast_reactive`
- **Branch**: `lerobot-native-migration`
- **Main branch**: `main`
- **Venv**: `~/Robotic_infra/lerobot/.venv` (use `source ~/Robotic_infra/lerobot/.venv/bin/activate` or call binaries directly)
- **GPU**: NVIDIA RTX A5000 (24 GB)
- **Dataset**: `checkpoints/HFRVLA_libero_v1_merged_reindexed/` (LeRobotDataset v3, 273,465 frames, 1693 episodes)

## What was the original problem

`checkpoints/hfrvla_conservative_seq4` was trained for 30k steps with the
original loss. It converged cleanly (no NaNs, descending loss) but at
closed-loop eval on LIBERO spatial it scored **12/50 success, vs 23/50 for
`zero_fast` (no residual)**. The trained residual was actively harmful.

Source failure analysis: `docs/hfrvla_failure_hypotheses.md` — 6 hypotheses
(self-labeling gate, action-MSE-not-success, global cap, task-specific drift,
offline/inference mismatch, wrong-problem).

## Phase 1 — multi-AI debate

To diagnose, ran a 4-way structured debate with Claude (Opus), Sonnet, Gemini,
and Codex. Each gave R1 + R2 analyses. Synthesized into one document.

- **Synthesis**: `docs/hfrvla_objective_debate_20260521.md`
- **Per-voice R1/R2 transcripts**: `~/.claude-octopus/debates/20260521-hfrvla-objective-redesign/rounds/r001_{claude,codex,gemini,sonnet}.md` and `r002_*` (4 files each)
- **Debate context.md**: `~/.claude-octopus/debates/20260521-hfrvla-objective-redesign/context.md`

**Convergence**: all 4 voices identified two structural failures:
1. Self-labeling gate loop (BCE target derived from model's own `delta_a` →
   positive feedback to `gate_prior` ≈ 0.95).
2. Gradient leak through gate via `L_final` (the merged-action MSE trains
   the gate to open whenever the residual helps, at weight 1.0 vs the gate
   prior weight of 0.02 = 18:1 pressure ratio in favor of opening).

And one missing term: explicit `||δ||² = 0` zero-target on preserve-class
states (the existing `L_preserve` was self-cancelling).

**Recommended framework**: rate-distortion correction coding. The base action
is a zero-bit codeword; the fast module is a paid channel that must justify
its bits.

## Phase 2 — Stage A (12-line patch, two iterations)

### Stage A v1 (commit `ba00074`)

Implemented the minimum-viable diff suggested by Sonnet's R2:
1. `out.gate.detach()` in the merged-action path → sever the L_final → gate
   gradient route.
2. New `L_preserve_zero = (||δ||² · is_preserve).mean()` term where
   `is_preserve = (err_before < err_preserve_thresh)` at training time.
3. `gate_improvement_margin` 0.02 → 0.05.
4. `loss_lambda_gate_prior` 0.02 → 0.10.

**Result**: did NOT work. `train/preserve_zero=0.0000` throughout training,
`gate_prior` only dropped from 0.95 to 0.90.

**Root cause**: the thresholds (0.01 for preserve, 0.05 for BCE margin) were
**~50x too small**. LIBERO empirical `err_before` median is ~2.4 (in per-frame
squared-L2 units), not ~0.05 as I had assumed. With `thresh=0.01`, 0% of
frames qualified as preserve; the term never fired.

### Stage A v2 (commit `d659a90`) — **production**

Same code shape as v1, but thresholds recalibrated against actual data:
- `err_preserve_thresh`: **0.5** (catches bottom ~10% of frames)
- `gate_improvement_margin`: **0.5** (roughly half typical clipped-residual improvement)

**Result**: ✅ verified on LIBERO spatial:
- `train/gate_prior`: **0.953 → 0.362** (-63% absolute, target was 0.20-0.45)
- `train/preserve_zero`: **0 → 0.00117** (term now firing)
- **LIBERO spatial eval: 12/50 → 24/50** (+12 vs old, +1 vs zero_fast baseline of 23/50)
- Task 8/9 recovered from collapse (0/5 → 4/5 and 2/5)

**Doc**: `docs/stage_a_changes.md` (~14 KB, full v1/v2 train-metric tables, per-task eval breakdown, calibration probe snippet, rollback instructions).

**Checkpoint**: `checkpoints/hfrvla_v2a_calib/` (trained), `checkpoints/hfrvla_v2a_calib_packaged/` (ready for lerobot-eval).

**Eval JSON**: `outputs/eval_hfrvla_v2a_calib_spatial/eval_info.json`.

## Phase 3 — Stage B (principled 5-term RD loss)

Replaced Stage A's patch with the formal 5-term Lagrangian from debate §3b:

```
L_v2 = 1.0·L_correct        # SmoothL1(δ, r_t) on y_correct=1 frames (static)
     + 2.0·L_preserve_zero  # ||δ||² on y_preserve=1 frames (static)
     + 0.5·L_rate           # E[sigmoid(ℓ)] + budget hinge
     + 3.0·L_gate           # FocalBCE(ℓ, y_correct; γ=2, pos_weight=4)
     + 0.2·L_smooth         # ||g_detached·u_t − g_detached·u_{t-1}||²
```

Static labels precomputed into **fastcache schema v2**:
- `y_correct = err_offline > p80`  (~20% of frames)
- `y_preserve = err_offline < p50` (~50% of frames)

Empirical thresholds (273,465 frames): p80 ≈ 5.51, p50 ≈ 2.68.

Config gated by `use_stage_b_objective: bool = False` — when off, Stage A v2
behaviour is preserved.

**Result**: 24/50 LIBERO spatial — **parity with Stage A v2**, but
`train/gate_prior` dropped further from 0.36 → 0.17 (gate is half as open).
Static labels gave cleaner gate calibration but no closed-loop win.

**Doc**: `docs/stage_b_changes.md` (~7.5 KB, train metrics table + per-task
breakdown + verdict).

**Checkpoint**: `checkpoints/hfrvla_v2b_stage_b{,_packaged}/`.

**Eval JSON**: `outputs/eval_hfrvla_v2b_stage_b_spatial/eval_info.json`.

**Fastcache v2** (with static labels): `checkpoints/HFRVLA_libero_v1_fastcache_seq4_v2/` — built by hardlinking the existing v1 fastcache's heavy arrays (`dino_patches.npy` 41 GB) and writing only `y_correct.npy`, `y_preserve.npy`, and a v2 `meta.json`.

## Phase 4 — Stage C (cached zero_fast rollout preserves)

Recorded 100 closed-loop LIBERO rollouts with packaged zero_fast policy.
75 succeeded → 8,242 frames added to fastcache. All labeled
`y_preserve=1, y_correct=0` since by definition the base was competent in
those states.

`HFRVLAFastCacheDataset` was extended to virtually concatenate multiple cache
roots (no physical merge needed). New env var `HFRVLA_FASTCACHE_ROLLOUT_ROOT`
in the training script controls this.

Loss is identical to Stage B with `LOSS_LAMBDA_PRESERVE_ZERO=3.0` (up from 2.0).

**Result**: 23/50 LIBERO spatial — **dropped 1 below Stage B**. But interesting
per-task trade-off:
- Task 4 unlocked 0/5 → 4/5 (other variants 0-1)
- Task 5 unlocked 0/5 → 2/5 (only variant with non-zero)
- Task 8 regressed 3/5 → 1/5 (residual learned to defer too aggressively)
- Task 9 regressed 4/5 → 2/5 (same pattern)

**Doc**: `docs/stage_c_changes.md` (~12 KB).

**Checkpoint**: `checkpoints/hfrvla_v2c_stage_c{,_packaged}/`.

**Eval JSON**: `outputs/eval_hfrvla_v2c_stage_c_spatial/eval_info.json`.

**Rollout dataset**: `checkpoints/HFRVLA_libero_v1_zero_fast_rollouts/` (LeRobotDataset v3).

**Rollout fastcache**: `checkpoints/HFRVLA_libero_v1_fastcache_seq4_v2_rollouts/` (~1.2 GB, mostly dino_patches).

**Zero_fast packaged checkpoint**: `checkpoints/hfrvla_zero_fast_packaged/` (config has `inference_disable_fast=True`).

## Phase 5 — Stages summary

**Doc**: `docs/stages_summary.md` (~8.5 KB) — executive summary across A/B/C.

**Punchline**: all three stages reach the **same 23-24/50 ceiling on LIBERO spatial**. Three independent loss designs, two label sources, two data distributions — the offline-IL ceiling is robust. Stage A v2 is the recommended production setting because it's simplest and equal-best.

| Variant | LIBERO spatial | gate_prior | Recommended? |
|---|---:|---:|:-:|
| Old trained (broken) | 12/50 | 0.953 | ❌ |
| zero_fast baseline | 23/50 | — | reference |
| **Stage A v2** | **24/50** | **0.362** | **✅ production** |
| Stage B | 24/50 | 0.168 | equivalent, more dev cost |
| Stage C | 23/50 | 0.184 | per-task trade-off, not strict win |

## Phase 6 — Multi-suite eval (in progress)

Per user request, evaluating across LIBERO suites beyond spatial to test
generalisation.

**Decision**: per the user, only `libero_goal` is being evaluated. `libero_object`
and `libero_10` skipped because zero_fast hits 0% running success and every
episode runs to 280-step timeout (~33 min per episode at ~7 s/step due to
CPU-bound sim/render). Spatial was fast because most episodes terminated
early; object/10 don't.

**Lesson learned**: `--eval.batch_size > 1` does NOT speed up LIBERO eval —
each parallel env spawns its own EGL context and the CPU/render is the
bottleneck. Stick with `batch_size=1` per the original `docs/hfrvla_failure_hypotheses.md` note.

**Current state** (as of handoff):
- Runner: `/tmp/run_multi_suite_eval.sh` (with `SUITES=(libero_goal)`)
- Process: PID 944735+ → `bash /tmp/run_multi_suite_eval.sh` → `lerobot-eval` (libero_goal × 4 checkpoints sequentially)
- Summary: `outputs/multi_suite_eval/SUMMARY.txt` (one line per START / DONE / FAIL)
- Per-eval logs: `/tmp/multi_eval_libero_goal_{zero_fast,stage_a,stage_b,stage_c}.log`
- Per-eval results: `outputs/multi_suite_eval/libero_goal__{zero_fast,stage_a,stage_b,stage_c}/eval_info.json`
- ETA: depends on per-suite difficulty. If goal is like spatial → ~30 min/ckpt, total ~2 hr. If like object → could be ~5 hr/ckpt.

## Files inventory

### Source changes (all in commits `ba00074`, `d659a90`, `bad0424`)

- `policy/lerobot_policy_hfrvla/src/lerobot_policy_hfrvla/configuration_hfrvla.py` — Stage A v2 thresholds + Stage B fields (`use_stage_b_objective`, `loss_lambda_correct`, `loss_lambda_rate`, `loss_lambda_smooth`, `focal_gamma`, `focal_pos_weight`, `gate_task_budget`)
- `policy/lerobot_policy_hfrvla/src/lerobot_policy_hfrvla/modeling_hfrvla.py` — `_compute_losses` has two branches: Stage A v2 path (default) and Stage B path (when `use_stage_b_objective=True`)
- `policy/lerobot_policy_hfrvla/src/lerobot_policy_hfrvla/fast_cache_dataset.py` — schema v2 (`y_correct`, `y_preserve`), multi-root mode `HFRVLAFastCacheDataset(roots=[...])`, backward-compat with v1 schema
- `scripts/build_hfrvla_fastcache.py` — `--correct-quantile`, `--preserve-quantile`, `--static-y-preserve` flags; writes schema v2
- `scripts/record_zero_fast_rollouts.py` — **NEW** (~500 lines) sim recorder for zero_fast LIBERO rollouts
- `scripts/train_hfrvla_libero_merged.sh` — env vars `USE_STAGE_B`, `LOSS_LAMBDA_*`, `HFRVLA_FASTCACHE_ROLLOUT_ROOT`, `FOCAL_*`, `GATE_TASK_BUDGET`; auto-bumps `LOSS_LAMBDA_GATE` to 3.0 + `LOSS_LAMBDA_PRESERVE_ZERO` to 2.0 when Stage B is on
- `tests/test_hfrvla_config.py` — assert Stage A v2 defaults
- `tests/test_hfrvla_forward_lerobot_batch.py` — 3 Stage A lock-in tests + 5 Stage B lock-in tests
- `tests/test_hfrvla_fast_cache_dataset.py` — multi-root dataset tests
- `tests/test_build_hfrvla_fastcache.py` — static-y-preserve test

**Test status**: 40+ tests pass. Run:
```bash
~/Robotic_infra/lerobot/.venv/bin/python -m pytest \
  tests/test_hfrvla_config.py \
  tests/test_hfrvla_forward_lerobot_batch.py \
  tests/test_hfrvla_fast_cache_dataset.py \
  tests/test_build_hfrvla_fastcache.py -q
```

### Documentation (all under `docs/`)

| File | Purpose | Size |
|---|---|---:|
| `docs/hfrvla_failure_hypotheses.md` | Original failure analysis (start of project) | 10 KB |
| `docs/hfrvla_objective_debate_20260521.md` | 4-way AI debate synthesis | 14 KB |
| `docs/stage_a_changes.md` | Stage A v1+v2 implementation + verified results | 14 KB |
| `docs/stage_b_changes.md` | Stage B (5-term RD loss) implementation + results | 7.5 KB |
| `docs/stage_c_changes.md` | Stage C (rollout preserves) implementation + trade-off | 12 KB |
| `docs/stages_summary.md` | Executive summary across A/B/C | 8.5 KB |
| **`docs/codex_handoff.md`** | **This file** | — |

### Trained checkpoints

| Variant | Trained dir | Packaged dir | Eval result |
|---|---|---|---|
| Old (failure) | `checkpoints/hfrvla_conservative_seq4/` | — | 12/50 (logged in failure_hypotheses.md) |
| zero_fast | — | `checkpoints/hfrvla_zero_fast_packaged/` | 23/50 (logged in failure_hypotheses.md) |
| Stage A v2 | `checkpoints/hfrvla_v2a_calib/` | `checkpoints/hfrvla_v2a_calib_packaged/` | `outputs/eval_hfrvla_v2a_calib_spatial/eval_info.json` → 24/50 |
| Stage B | `checkpoints/hfrvla_v2b_stage_b/` | `checkpoints/hfrvla_v2b_stage_b_packaged/` | `outputs/eval_hfrvla_v2b_stage_b_spatial/eval_info.json` → 24/50 |
| Stage C | `checkpoints/hfrvla_v2c_stage_c/` | `checkpoints/hfrvla_v2c_stage_c_packaged/` | `outputs/eval_hfrvla_v2c_stage_c_spatial/eval_info.json` → 23/50 |
| Stage A v1 (broken thresholds) | `checkpoints/hfrvla_v2a_v1_uncalibrated/` | — | not packaged, kept for rollback |

### Fastcaches

| Path | Size | Content |
|---|---:|---|
| `checkpoints/HFRVLA_libero_v1_fastcache_seq4/` | 41 GB | v1 schema (expert demos) |
| `checkpoints/HFRVLA_libero_v1_fastcache_seq4_v2/` | 41 GB (hardlinked) | v2 schema (expert demos + `y_correct`, `y_preserve`) |
| `checkpoints/HFRVLA_libero_v1_zero_fast_rollouts/` | ~1.5 GB | LeRobotDataset v3 of zero_fast rollouts |
| `checkpoints/HFRVLA_libero_v1_fastcache_seq4_v2_rollouts/` | ~1.2 GB | v2 schema fastcache of rollout dataset (all `y_preserve=1`) |

### Eval results

- **LIBERO spatial** (50 episodes/ckpt, batch_size=1, seed=42): all 4 ckpts done, JSONs under `outputs/eval_hfrvla_v2*_spatial/`.
- **LIBERO goal** (100 episodes/ckpt, batch_size=1, seed=42): IN PROGRESS, running serially zero_fast → stage_a → stage_b → stage_c.
- LIBERO object / libero_10: skipped per user decision (too slow due to long timeouts when base policy can't solve the task).

## Recommended next actions

1. **Wait for libero_goal multi-suite eval to finish.** Watch `outputs/multi_suite_eval/SUMMARY.txt` for `[ALL DONE]`. Then compare per-suite per-checkpoint success rates to see if Stage A v2's +1 vs zero_fast generalizes.

2. **If goal eval also shows ceiling-like behaviour** (all ~variants within ±2 of zero_fast), commit the multi-suite eval results to `docs/stages_summary.md` and close this research line. The 24/50 plateau is robust.

3. **If a variant unexpectedly wins on libero_goal** (e.g. Stage C suddenly +5 vs zero_fast), investigate which design element (rollout preserves, focal BCE, smoothness) produced the gain. Could indicate the closed-loop hypothesis is right but spatial is the wrong test suite.

4. **Outside the current invariant** — the 48% ceiling is likely a SmolVLA-500M base capability problem, not a loss design problem. If continuing:
   - Try a stronger base (OpenVLA-7B, π0).
   - Try online RL fine-tuning (break the "small-fast-module-as-compression" research scope).
   - Try more demo data (273k frames is borderline thin for 10-task spatial).

## Commands reference

### Re-eval spatial on any packaged checkpoint
```bash
cd /home/hucenrotia/Patrick/VLA_research/Hierachical_fast_reactive
mkdir -p $HOME/tmp/hfrvla/{numba,tmp,matplotlib}
NUMBA_CACHE_DIR=$HOME/tmp/hfrvla/numba \
TMPDIR=$HOME/tmp/hfrvla/tmp \
MPLCONFIGDIR=$HOME/tmp/hfrvla/matplotlib \
  ~/Robotic_infra/lerobot/.venv/bin/lerobot-eval \
    --policy.path=checkpoints/hfrvla_v2a_calib_packaged \
    --env.type=libero --env.task=libero_spatial \
    --eval.n_episodes=5 --eval.batch_size=1 \
    --output_dir=outputs/eval_v2a_redo --seed=42
```

### Retrain Stage A v2
```bash
cd /home/hucenrotia/Patrick/VLA_research/Hierachical_fast_reactive
HFRVLA_DATASET_BACKEND=fastcache \
HFRVLA_FASTCACHE_ROOT=checkpoints/HFRVLA_libero_v1_fastcache_seq4 \
SEQ_LEN=4 BATCH_SIZE=512 NUM_WORKERS=8 \
OUT_DIR=checkpoints/hfrvla_v2a_redo JOB_NAME=hfrvla_v2a_redo \
STEPS=30000 SAVE_FREQ=5000 \
WARMUP_STEPS=1000 JOINT_STEPS=29000 REFINE_STEPS=0 \
WANDB_ENABLE=true WANDB_PROJECT=hfrvla \
  bash scripts/train_hfrvla_libero_merged.sh
```

### Retrain Stage B
Add `USE_STAGE_B=true` and use the v2 fastcache:
```bash
HFRVLA_FASTCACHE_ROOT=checkpoints/HFRVLA_libero_v1_fastcache_seq4_v2 \
USE_STAGE_B=true ... bash scripts/train_hfrvla_libero_merged.sh
```

### Retrain Stage C
Add `HFRVLA_FASTCACHE_ROLLOUT_ROOT` on top of Stage B:
```bash
HFRVLA_FASTCACHE_ROOT=checkpoints/HFRVLA_libero_v1_fastcache_seq4_v2 \
HFRVLA_FASTCACHE_ROLLOUT_ROOT=checkpoints/HFRVLA_libero_v1_fastcache_seq4_v2_rollouts \
USE_STAGE_B=true LOSS_LAMBDA_PRESERVE_ZERO=3.0 ... bash scripts/train_hfrvla_libero_merged.sh
```

## Open items / known unknowns

- libero_goal eval is running NOW; result not yet known.
- libero_object and libero_10 results are NOT measured; skipped per user.
- libero_90 (90-task suite, ~6+ hr eval) is NOT measured.
- Per-task gate-mean diagnostic (Codex R1 §3 proposed item) was NOT implemented.
- Per-task budget hinge (Stage B has it batch-level, not per-task) was NOT
  separately verified to be the right form.
- No ablation of "Stage A v2 with detach but WITHOUT preserve_zero" was run.
  Both were applied together. We don't know which one is the dominant fix.

## Git status (uncommitted at handoff)

There are some uncommitted WIP files in the working tree (CLAUDE.md edits,
GEMINI.md, scripts not in the Stage commits, etc.). The Stage A v2 + B + C
work is fully committed in `ba00074`, `d659a90`, `bad0424`. The handoff doc
(this file) is the latest uncommitted addition.

To verify clean state:
```bash
git log --oneline -3   # should show bad0424 / d659a90 / ba00074 at top
git diff --stat HEAD   # shows what's still uncommitted
```
