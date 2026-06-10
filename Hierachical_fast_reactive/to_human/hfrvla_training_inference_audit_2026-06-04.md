# HFRVLA Training / Inference Audit

Date: 2026-06-04

## Summary

目前結果比較不像「HFRVLA 題目不成立」，比較像三個實作分佈沒有完全對齊：

1. training target 目前是 current-step raw residual，不是 A2C2-style stale chunk correction。
2. deployment merge 是 `a_base + alpha * clip(delta_a)`，但 fast-wrist loss 沒有直接訓練這個 deployed final action。
3. inference 在 `select_action()` 路徑會 per-control-step correction，但 hook cache miss 會安靜退回 `a_base`；async / server 路徑仍需要加 instrumentation 來確認沒有 bypass。

因此下一輪不建議改掉大框架。最值得做的是把資料、loss、time feature、logging 對齊到「同一個 stale base action 在最新 wrist observation 下要怎麼被修正」。

## Empirical Pattern

資料來源是 `experiments/eval_registry/eval_results_master.csv`。

### Action-step sweep, planning chunk = 50

| execution steps | HFRVLA | SmolVLA | delta |
|---:|---:|---:|---:|
| 2 | 85.0 | 82.0 | +3.0 |
| 4 | 83.0 | 73.0 | +10.0 |
| 8 | 81.0 | 77.0 | +4.0 |
| 16 | 74.0 | 78.0 | -4.0 |
| 32 | 71.0 | 73.0 | -2.0 |

這符合你的觀察：低到中等 execution steps 有改善，長 execution steps 開始掉。

### Matched chunk-size sweep

| K | HFRVLA | SmolVLA | delta |
|---:|---:|---:|---:|
| 2 | 82.0 | 76.0 | +6.0 |
| 4 | 78.0 | 79.0 | -1.0 |
| 8 | 83.0 | 74.0 | +9.0 |
| 16 | 80.0 | 74.0 | +6.0 |
| 32 | 71.0 | 67.0 | +4.0 |
| 50 | 46.0 | 43.0 | +3.0 |

舊 A2C2-wrist 在 long matched chunks 還有小幅正 margin，但 absolute success 明顯崩掉。這代表 residual 有 signal，但長 open-loop chunk 的 base drift / stale observation 問題很大。

### Alpha / clip sensitivity

在 planning=50, execution=8 的 alpha/clip sweep 中：

| alpha | delta_max | combined success |
|---:|---:|---:|
| 0.75 | 0.2 | 88.0 |
| 1.0 | 0.4 | 65.0 |

這是最強的 merge calibration 證據。模型不是完全沒學到東西，而是 residual magnitude 很容易從「幫助」變成「破壞 base policy」。

## Code Audit

### 1. Inference path: `select_action()` does per-tick correction, with one silent fallback

Relevant code:

- `policy/lerobot_policy_hfrvla/src/lerobot_policy_hfrvla/modeling_hfrvla.py:396`
- `policy/lerobot_policy_hfrvla/src/lerobot_policy_hfrvla/modeling_hfrvla.py:406`
- `policy/lerobot_policy_hfrvla/src/lerobot_policy_hfrvla/modeling_hfrvla.py:436`
- `policy/lerobot_policy_hfrvla/src/lerobot_policy_hfrvla/modeling_hfrvla.py:453`

`select_action()` 只有在 queue 空掉時才呼叫 frozen SmolVLA 產生 chunk，之後每次 control step 都 pop 一個 `a_base`，用當前 batch 的 wrist image / state / chunk index 跑 fast head，再 merge 成 `a_final`。

所以如果 eval / deployment 的主路徑確實是 `select_action()`，fast correction 是 per-tick 生效的。

但如果 `_zgoal_cache` 或 `_zphase_cache` 沒有被 hook 填到，程式會直接 `return a_base`。目前這是 silent fallback，對實驗分析不友善。若 async server 或 `predict_action_chunk()` 類路徑沒有經過這段，就可能出現「訓練有 fast head，實際執行沒有 fast correction」。

Immediate diagnostic:

- log `fast_applied`, `fast_skipped_hook_cache_miss`, `k`, `delta_norm`, `delta_clip_fraction`
- eval summary 中列出每個 episode fast-applied ratio
- 對 robot / async server path 加一個 assertion：每個 sent action 都要經過 HFRVLA merge，除非 explicit `inference_disable_fast=True`

### 2. Training target is current raw residual

Relevant code:

- `policy/lerobot_policy_hfrvla/src/lerobot_policy_hfrvla/modeling_hfrvla.py:621`
- `policy/lerobot_policy_hfrvla/src/lerobot_policy_hfrvla/modeling_hfrvla.py:793`
- `tests/test_hfrvla_forward_lerobot_batch.py:674`

`fast_wrist_chunk` training 只取 current window：

```python
proprio=proprio[:, -1]
a_base_k=a_base[:, -1]
k_idx_norm=k_idx_norm[:, -1]
a_base_chunk=a_base_chunk[:, -1]
chunk_step_idx=chunk_step_idx[:, -1]
dino_patches=dino_patches[:, -1]
```

loss 則是：

```python
target_delta = a_expert_curr - a_base_curr
l_delta = mse(delta_a, target_delta)
```

這不是錯，但它不是 A2C2 最關鍵的 delayed/stale chunk correction dataset。deployment 要修的是「由較舊 observation 產生的 chunk 裡第 k 個 base action」，現在訓練比較像「目前 frame 的 base error」。

### 3. Fast-cache v3 reconstructs chunks from frame-level `a_base`

Relevant code:

- `scripts/build_hfrvla_fastcache.py:87`
- `scripts/build_hfrvla_fastcache.py:114`
- `scripts/build_hfrvla_fastcache.py:120`

`a_base_chunk.npy` 是從已經逐 frame 存下來的 `a_base` 重新切成長度 50 的 block：

```python
chunk[:usable] = a_base[chunk_start:chunk_end]
a_base_chunk[frame_idx] = chunk
chunk_step_idx[frame_idx, 0] = local_step
```

這能讓 chunk module 看一段 base action context，但它沒有保存「SmolVLA 在 chunk start 當下生成的原始完整 chunk」以及 chunk generation timestamp / observation age。這會限制長 execution steps 的訓練對齊能力。

Immediate fix direction:

- recorder 直接存 `generated_a_base_chunk`, `chunk_start_frame`, `chunk_obs_time`, `chunk_step_idx`
- dataset builder 不要只用 frame-level `a_base` 重建 chunk
- 之後再擴成 multi-age training samples

### 4. Packaged configs and eval sweeps are mostly sane

Checked packaged configs:

- `checkpoints/hfrvla_a2c2_wrist_seq2_30k_packaged/config.json`
- `checkpoints/hfrvla_fwr_chunk_seq2_b512_50k_packaged_cuda/config.json`

兩者都是 `chunk_size=50`, `n_action_steps=50`, `delta_max=0.2`, `control_dt=0.1`。FWR chunk packaged config 預設 `fast_residual_alpha=1.0`，sweep scripts 會在 eval 時覆寫 alpha / delta_max。

這表示目前 action-step sweep 的設計不是明顯跑錯；問題比較可能在 target/loss/scheduler 對齊。

## Root Cause Ranking

### P0: Add inference instrumentation before changing architecture

目的不是提高成功率，而是把 false negative 排掉。

Add metrics:

- `fast_applied_ratio`
- `hook_cache_miss_count`
- `delta_norm_mean`
- `delta_clip_fraction`
- `k_idx_histogram`
- `base_chunk_age_steps`
- `camera_to_action_latency_steps`
- `correction_latency_ms`

判準：

- 如果 `fast_applied_ratio < 1.0`，先修 runtime path。
- 如果 long K 的 `delta_clip_fraction` 很高，代表 clip budget / target scale 不對。
- 如果 long K 的 target residual 大量超過 clip，代表 base action 已經超出可修範圍。

### P1: Change dataset target to stale chunk-age correction

這是最高優先的 method improvement，且不會降低 novelty。

Current target:

```text
y_t = expert_action[t] - a_base[t]
```

Recommended A2C2-style target:

```text
base = generated_chunk[t-k][k]
obs  = latest_wrist_obs[t]
y    = expert_action[t] - base
FWR(obs, base, k, age, slow_context) -> y
```

其中 `generated_chunk[t-k]` 必須是 slow planner 在過去某個 chunk start 真的生成的完整 chunk，不應該只從 frame-level `a_base` 重新拼。

Novelty 保留方式：

- high-frequency latest observation 仍然只用 wrist-centric signal
- slow planner 仍然 frozen
- global/task context 只從 frozen SmolVLA latent / base chunk 提供
- 不變成 full A2C2 的 top+wrist+language heavy correction head

### P2: Train the deployed merge, not only raw residual

Current fast-wrist objective:

```text
MSE(delta_hat, expert - base)
```

Recommended objective:

```text
delta_exec = alpha_train * clip(delta_hat, delta_max_train)
a_hat = a_base + delta_exec

L = SmoothL1(a_hat, expert)
  + lambda_raw * SmoothL1(delta_hat, expert - a_base)
  + lambda_res * ||delta_exec||^2
  + lambda_smooth * ||delta_exec_t - delta_exec_{t-1}||^2
  + lambda_clip * clip_fraction_penalty
```

這直接對齊 inference 會送出的 action，也能解釋為什麼 alpha/clip sweep 目前這麼敏感。

### P3: Add time / age features

目前有 `k_idx_norm`，但它只表示 chunk position。對 HFRVLA 真正重要的是 action 的 stale age。

Recommended features:

```text
k_norm = k / (H - 1)
sin_cos_k = [sin(2*pi*k/H), cos(2*pi*k/H)]
age_norm = (now_time - chunk_obs_time) / control_dt
latency_norm = measured_camera_to_action_latency / control_dt
remaining_norm = (H - 1 - k) / (H - 1)
```

若先不想動太大，第一版至少加 `sin_cos_k` 和 `age_norm`。

### P4: Keep wrist-centric novelty, but avoid under-conditioning

風險：LIBERO Spatial / Object 有些任務需要 global target relation，wrist image 可能只看到局部接觸附近。

不要把 fast path 改成 full top+wrist high-rate policy。比較乾淨的做法是：

```text
fast latest observation = wrist image + proprio
global context = frozen slow planner latent / base chunk / optional one pooled global token at chunk start
```

這樣 thesis claim 可以是「latest high-frequency correction is wrist-centric」，而不是「整個 policy 只看 wrist」。

### P5: Add target-space ablations only after P1/P2

建議順序：

1. raw normalized 7D residual, current baseline
2. per-dimension scaled residual with final-action loss
3. EE/tool-frame residual + gripper head

tool-frame residual 可能更適合 wrist-centric local correction，但 implementation blast radius 比 P1/P2 大，先不要當第一刀。

### P6: Borrow RTC / DynamicVLA scheduling ideas as infrastructure

不要把 RTC / DynamicVLA 當主方法；它們可以當 runtime scheduler 對齊。

Recommended contract:

```text
slow planner:
  continuously produce base chunks without blocking control loop

fast loop, every tick:
  read latest wrist/state
  choose next non-expired base action
  compute delta from wrist + base action + age/time + slow context
  send merged action
```

如果 base action age 超過 threshold，應該 discard 或 request new chunk，而不是硬修到底。

## Near-Term Experiment Plan

### Experiment A: no-training runtime audit

Add logging only, rerun a small eval subset.

Success criterion:

- `fast_applied_ratio == 1.0`
- hook cache miss is zero
- long-K failures can be annotated by delta norm / clip fraction / action age

### Experiment B: offline correctability by k

Use or extend `scripts/audit_plan50_chunk_correctability.py`.

Report by k:

- `||expert - generated_chunk[k]||`
- fraction correctable within delta_max=0.2 and effective cap alpha*delta_max=0.15
- clip saturation by action dimension
- early-vs-late chunk gap

This tells whether long chunks are uncorrectable under the current residual budget.

### Experiment C: deployment-aligned loss

No dataset change yet. Train same cache, same architecture:

- baseline raw MSE
- final-action SmoothL1 with alpha=0.75, delta_max=0.2
- final-action SmoothL1 + residual/smooth/clip penalties

This isolates loss/merge mismatch.

### Experiment D: generated-chunk stale dataset

Re-record or extend recorder to save true generated chunks and chunk age, then train FWR on stale chunk-age samples.

This is the most likely path to long execution-step improvement.

## Literature Anchors

- A2C2: action chunking hurts reactivity under delay / long horizons; a lightweight correction head runs every control step using latest observation, base action, chunk position, and base policy features.
- RTC: inference-time scheduler for chunk policies; asynchronously generates next chunk while executing current chunk, freezes committed actions and inpaints remaining actions.
- DynamicVLA: frames dynamic manipulation as a perception-execution gap and uses continuous inference / action streaming for temporal alignment.
- RDP: slow-fast policy design for contact-rich manipulation; useful analogy for wrist camera as high-frequency local feedback, but HFRVLA uses visual wrist feedback rather than tactile/force feedback.

## Recommendation

Do not abandon the current framing. The strongest next thesis version is:

```text
HFRVLA = wrist-centric, A2C2-style residual correction for frozen SmolVLA action chunks.
```

The next implementation target should be:

```text
record true generated base chunks
train stale chunk-age residual targets
use deployment-aligned final-action loss
add runtime instrumentation
```

This keeps the novelty concentrated on lightweight wrist-centric high-frequency correction, while fixing the most plausible reasons long execution steps do not currently improve.
