# Research Findings

## Research Question

Can a small wrist-camera residual module improve frozen SmolVLA action-chunk execution without fine-tuning the slow planner?

## Current Understanding

The strongest current HFRVLA result is not "more autonomy through longer open-loop chunks" by itself. The current evidence says a calibrated wrist-camera residual can improve a frozen SmolVLA planner in a moderate chunk-execution regime, but residual scale and execution horizon matter.

The most interpretable mainline remains the A2C2-wrist residual: frozen `HuggingFaceVLA/smolvla_libero` provides `a_base` and slow-planner context, while the trainable fast path predicts a full 7D residual from wrist DINO features, robot state, base action, chunk index, and latent context. The deployed merge is `a_final = a_base + alpha * clip(delta_a)`.

## Key Results

All metrics below are from `experiments/eval_registry/eval_results_master.csv`, using combined LIBERO Spatial + Object success.

- SmolVLA at planning=50, execution=8: 77.0% over 100 episodes.
- A2C2-wrist 30k at planning=50, execution=8, alpha=0.5: 81.0% over 100 episodes.
- A2C2-wrist 30k at planning=50, execution=8, alpha=0.75, delta_max=0.2: 88.0% over 100 episodes.
- A2C2-wrist 30k at planning=50, execution=8, alpha=1.0, delta_max=0.4: 65.0% over 100 episodes.
- FWR 50k action-step 10x10 at planning=50, execution=8: 75.5% over 200 episodes.
- FWR 50k matched chunk 10x10 peaks at K=1 with 80.5% over 200 episodes and declines for longer matched chunks.

## Patterns and Insights

The residual helps when it is bounded and calibrated. The alpha/clip sweep is the clearest evidence: a moderate effective cap outperforms the SmolVLA baseline, while larger residual scale degrades sharply.

Longer execution intervals remain a problem. Both older A2C2-wrist sweeps and newer FWR sweeps show decline as execution or matched chunk size grows. This suggests the fast wrist module is useful but does not fully solve open-loop action-chunk drift.

The newer FWR rows need interpretation before they become a paper claim. They are valuable because they use larger 10x10 evaluations, but their current best combined values do not exceed the tuned A2C2-wrist alpha/clip result.

The 2026-06-04 training/inference audit points to a concrete mismatch: current fast-wrist training learns a current-frame raw residual (`expert - a_base`), while the long-horizon deployment problem is stale chunk correction under the latest wrist observation. The v3 fast-cache also reconstructs `a_base_chunk` from frame-level `a_base` rather than preserving the original generated slow-planner chunk and its age. This likely weakens long execution-step gains without invalidating the wrist-centric residual framing.

The inference path is mostly aligned when `select_action()` is the execution entry point: it pops one base action per tick and applies fast correction each call. The main runtime risk is a silent fallback to `a_base` when SmolVLA hook caches for slow context are missing; real robot or async-server deployment should log `fast_applied_ratio`, hook-cache misses, correction latency, action age, delta norm, and clip fraction.

## Lessons and Constraints

- Keep SmolVLA frozen unless the research question explicitly changes.
- Use the eval registry as the source of truth for quantitative claims.
- Do not collapse planning horizon and execution/replan interval when comparing policies.
- Treat alpha and residual clipping as deployment-critical parameters, not cosmetic inference settings.
- Do not infer citation-backed positioning from local intuition; run a focused related-work survey before writing paper claims.

## Open Questions

- Why does alpha=0.75 and delta_max=0.2 work best in the current alpha/clip grid?
- Are failures at long chunks caused by base-policy drift, residual overcorrection, wrist observability limits, or dataset mismatch?
- Does FWR need a different deployment scale, training target, or baseline comparison to show its intended benefit?
- Which prior work most directly frames wrist-only visual correction on top of frozen VLA chunks?
- How much of the late-chunk target residual is correctable under the deployed effective cap, especially alpha=0.75 and delta_max=0.2?
- Does recording true generated base chunks plus chunk age close the gap between current FWR training and A2C2-style delayed correction?

## Optimization Trajectory

Current best combined success is 88.0% for A2C2-wrist 30k with alpha=0.75 and delta_max=0.2 at planning=50, execution=8. The baseline for that execution setting is SmolVLA at 77.0%, so the best observed improvement is +11.0 percentage points. The next useful trajectory plot should separate A2C2 alpha/clip calibration from FWR action-step and matched-chunk sweeps.

## 2026-06-04 Audit Recommendation

Do not change the high-level architecture yet. The next implementation should first add runtime instrumentation, then move the dataset/loss toward stale chunk-age correction:

1. Record true generated SmolVLA chunks, chunk start frame/time, chunk step index, and action age.
2. Train samples where `base = generated_chunk[t-k][k]`, `obs = latest_wrist_obs[t]`, and `target = expert_action[t] - base`.
3. Add `age_norm` and sinusoidal chunk-position features in addition to `k_idx_norm`.
4. Replace pure raw residual MSE with a deployment-aligned final-action loss on `a_base + alpha * clip(delta)`, keeping raw residual as an auxiliary term.
5. Keep the novelty wrist-centric: latest high-frequency visual feedback should remain wrist-only, while global/task disambiguation can come from frozen slow-planner context.

Detailed report: `to_human/hfrvla_training_inference_audit_2026-06-04.md`.
