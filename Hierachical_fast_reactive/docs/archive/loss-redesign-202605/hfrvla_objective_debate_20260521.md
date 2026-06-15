# 🐙 HFRVLA 訓練目標重設計 — 四方辯論最終整合

**Date**: 2026-05-22
**Debate ID**: 20260521-hfrvla-objective-redesign
**Mode**: cross-critique, 2 rounds, 4 participants
**Source**: `docs/hfrvla_failure_hypotheses.md`
**Invariant (immovable)**: small fast module performs *compression*

---

## 摘要 (TL;DR)

四方辯論在兩輪後達成 **75% 共識**，其餘 25% 是工程順序而非方向爭議。

**根本失敗原因 (一致)**：
1. **Gate 自標籤迴圈** — `g_target` 由模型自己當下的 `delta_a` 推出 (`modeling_hfrvla.py:528-533`)，一旦 `delta_a` 略具能力，幾乎每個樣本都 improvement > margin → gate label = 1 幾乎到處 → `gate_prior=0.953`。
2. **梯度洩漏** — `L_final = MSE(a_base + gate·clip(δ), a_expert)` 在 line 538 直接讓 gate 接到 L_final 的梯度。即使 BCE label 修好了，gate 仍會被 L_final 訓練成「開到能降低 MSE 的地方」。 (Sonnet R1 發現。)
3. **無對稱保持訊號** — 沒有任何 term 說「base 已經對了 → residual 必須**剛好為 0**」。`L_preserve` 在收斂時 ≈ 0 因為它是自我抵消的 regularizer。

**正確的訓練典範 (一致)**：**Rate-Distortion Correction Coding** —
- `a_base` 是**零位元 codeword** (zero-bit prior)
- 小型 fast module 是**付費通道**，每次開啟支付 rate (bits)
- Distortion 只在 *correction event* 上計算，不是 per-step MSE everywhere

這是 compression invariant 的形式化：fast module 只在「base 會造成 closed-loop distortion」時才支付 rate 把訊息送出去。

---

## 各方主張對照表

| 議題 | 🟠 Sonnet | 🐙 Opus | 🟡 Gemini | 🔴 Codex |
|---|---|---|---|---|
| **Gate 修法** | `gate.detach()` + bump prior to 0.10 | Static `is_correct` 百分位標籤 + L_rate | β·‖g‖₁ rate penalty | Focal BCE + window label y_t + 任務級 budget hinge |
| **Distortion 來源** | 保留現有 L_delta + L_final | 方向監督 SmoothL1 on r_t | 只留 `MSE(a_final, a_expert)` | Window 加權 multi-step MSE on `a_hat_{t:t+H}` |
| **Preserve 訊號** | `err_before<thresh` 即時門檻 (proxy) | Static `is_preserve` 百分位 + noise injection | (無顯式 preserve) | `1-y_t` 來自 cached `zero_fast` rollout 鄰域 |
| **資料端成本** | 0 hr | ~30 min (一次性百分位計算) | 0 hr (但需 β 掃 3-4 runs) | ~5-8 hr (rollout cache + 特徵重算) |
| **新增 loss 項數** | +1 | +3 (改 5 項) | +1 (但砍 3 項) | +5 (共 8 項) |
| **可上線時間** | 1 天 | 1-2 天 | 1-3 天 (β 掃) | 1 週 |

---

## 最大爭議：一個句子

> **「Preserve label 是否非要來自 cached rollout 才有效？」**

- **Codex/Opus 立場**：closed-loop failure 是 *sequence-level* distribution shift，靜態的 per-step label (或 training-time threshold) 不足以覆蓋 task 8/9 的 rollout 分布。
- **Sonnet/Gemini 立場**：等實驗證實 simpler proxy 不夠之前，不該付 cache rebuild 的工程成本。`err_before < thresh` 已經是足夠強的 zero-target 訊號。

辯論的 **裁決**：Sonnet 正確的順序學家洞見 (還沒有 per-task gate diagnostic 之前不該重建 cache) 配上 Codex 正確的長期架構 (windowed labels + rollout negatives 是真正的最終答案) → **分階段執行**。

---

## 最終訓練目標 (formal)

**Frame**: Rate-Distortion Correction Coding with detached gate gradient.

### 符號

- `a_t^b` = frozen SmolVLA action (zero-bit codeword)
- `a_t*` = expert action
- `δ_t` = fast module 預測的 residual
- `u_t = clip(δ_t, δ_max)` = clipped residual
- `ℓ_t` = gate logit; `g_t = σ(ℓ_t)`
- `r_t = clip(a_t* - a_t^b, δ_max)` = oracle residual target
- `â_t = a_t^b + sg(g_t)·u_t` (**stop-gradient on gate** — Sonnet's contribution)
- `y_t ∈ {0, 1}` = static correction label (見下)

### Lagrangian (v2 final form)

$$
\boxed{
\mathcal{L}_{HFRVLA\text{-}v2} \;=\;
\underbrace{\lambda_D \cdot \mathcal{D}(y_t)}_{\text{distortion (only on corrections)}}
\;+\;
\underbrace{\lambda_P \cdot \mathcal{P}(1-y_t)}_{\text{preserve (zero target)}}
\;+\;
\underbrace{\lambda_R \cdot \mathcal{R}(g_t, u_t)}_{\text{rate}}
\;+\;
\underbrace{\lambda_G \cdot \mathcal{G}(\ell_t, y_t)}_{\text{gate calibration}}
\;+\;
\underbrace{\lambda_S \cdot \mathcal{S}(g_t u_t)}_{\text{smoothness}}
}
$$

### 五個 functional 的展開

$$
\mathcal{D}(y_t) = \mathbb{E}\big[\, y_t \cdot \mathrm{SmoothL1}(\delta_t, r_t) \,\big]
\quad \text{(directional+magnitude on correction states)}
$$

$$
\mathcal{P}(1-y_t) = \mathbb{E}\big[\, (1-y_t) \cdot \|g_t \cdot u_t\|_2^2 \,\big]
\quad \text{(residual must contract to 0 on preserve states)}
$$

$$
\mathcal{R}(g_t, u_t) = \mathbb{E}[g_t] \;+\; \tfrac{1}{2}\mathbb{E}\Big[g_t \sum_j \log\!\big(1 + \tfrac{|u_{t,j}|}{0.02}\big)\Big] \;+\; \big[\bar{g} - p_0\big]_+^2
$$

其中 $p_0 = 0.15$，$[\cdot]_+$ 是 hinge。**第一項是 bernoulli rate**, **第二項是 payload rate**, **第三項是 batch-level budget hinge**.

$$
\mathcal{G}(\ell_t, y_t) = \mathrm{FocalBCE}(\ell_t, y_t; \gamma=2, \mathrm{pos\_weight}=4)
$$

$$
\mathcal{S}(g_t u_t) = \mathbb{E}\big[\, \|g_t u_t - g_{t-1} u_{t-1}\|_2^2 \,\big]
\quad \text{(temporal smoothness; prevents single-step shocks)}
$$

### 權重 (建議初始值)

| Symbol | 值 | 對應 R1/R2 來源 |
|---|---:|---|
| $\lambda_D$ | 1.0 | (all) |
| $\lambda_P$ | 2.0 | Opus L_preserve_zero, Codex L_neg |
| $\lambda_R$ | 0.5 | Gemini RD-IB, Codex L_rate+budget |
| $\lambda_G$ | 3.0 | Codex focal BCE |
| $\lambda_S$ | 0.2 | Codex L_smooth |

### Label source `y_t` — 三種強度 (按工程成本由小到大)

**Strength 1 — training-time proxy (Sonnet's contribution, 0 hr cost)**:
```python
err_before = ((a_expert - a_base) ** 2).sum(dim=-1)
y_t = (err_before > 0.03).float()    # tighter than current 0.02 margin
# 1 - y_t implicitly: (err_before < 0.01).float() is the strict preserve mask
```

**Strength 2 — static percentile (Opus's contribution, ~30 min one-time)**:
```python
# Computed once over the merged dataset, stored in fastcache shards.
err_offline = ((a_expert - a_base) ** 2).sum(dim=-1)
y_correct  = err_offline > quantile(err_offline, 0.80)
y_preserve = err_offline < quantile(err_offline, 0.50)
y_t = y_correct.float()              # (1 - y_t) defaults to includes both preserve + ambiguous
```

**Strength 3 — windowed + rollout-aware (Codex's contribution, ~5-8 hr one-time)**:
```python
# H=4, w=[1, 0.7, 0.5, 0.3]
E_b = sum_h w_h * ||a_expert_{t+h} - a_base_{t+h}||^2
E_r = sum_h w_h * ||a_expert_{t+h} - (a_base_{t+h} + r_{t+h})||^2
c_t = contact_label OR gripper_transition OR proximity_phase
y_t = 1 iff (E_b > 0.03) AND (E_r < E_b - 0.02) AND (c_t = 1)
      AND NOT in_neighborhood_of(zero_fast_successful_rollout)
```

### 移除 / 取代的現有 terms

- ❌ 舊 `L_delta = MSE(δ, clip(a_expert - a_base))` everywhere → 由 $\mathcal{D}$ (只在 y_t=1 處) 取代
- ❌ 舊 `L_final = MSE(a_base + gate·clip(δ), a_expert)` → 拆成 $\mathcal{D}$ + $\mathcal{P}$，並且 **gate detach**
- ❌ 舊 `L_preserve = relu(err_final - err_before)` → 由 $\mathcal{P}$ 取代 (顯式 zero-target)
- ❌ 舊 `L_gate` 自標籤 → 由 $\mathcal{G}$ 用靜態 `y_t` 取代
- ❌ 舊 `L_gate_prior = mean(gate)` → 由 $\mathcal{R}$ 取代 (更完整的 rate)
- ✅ 舊 `L_contact` 保留 (aux head)
- ✅ 新增 $\mathcal{S}$ (temporal smoothness)

---

## 三階段執行計畫 (推薦)

### Stage A — Sonnet's "ship tonight" diff (1 天, 0 hr 資料準備)

採用 **Strength 1** 的 `y_t`。在現有 code 上做 12 行修改：

`configuration_hfrvla.py:54-63`：
```python
loss_lambda_gate: float = 1.0            # gate BCE
gate_improvement_margin: float = 0.05    # 0.02 → 0.05 (more selective)
loss_lambda_final: float = 1.0           # legacy, will be replaced by D+P
loss_lambda_preserve: float = 0.5        # legacy
loss_lambda_gate_prior: float = 0.10     # 0.02 → 0.10 (real rate signal)
loss_lambda_preserve_zero: float = 2.0   # NEW: 顯式 zero-target
err_preserve_thresh: float = 0.01        # NEW: preserve proxy threshold
```

`modeling_hfrvla.py:520-548` 關鍵修改：
```python
delta_clip = self._clip_fast_residual(out.delta_a)
# ★ Sonnet 的關鍵 fix: gate 從 L_final 的梯度路徑切斷
a_final = a_base + out.gate.detach().unsqueeze(-1) * delta_clip

# ★ Opus 的 zero-target term，用 Sonnet 的 training-time proxy
err_before = ((a_expert - a_base) ** 2).sum(dim=-1)
is_preserve = (err_before < self.config.err_preserve_thresh).float()
l_preserve_zero = (out.delta_a.pow(2).sum(dim=-1) * is_preserve).mean()

losses["preserve_zero"] = l_preserve_zero
total = (
    l_delta
    + self.config.loss_lambda_gate * l_gate
    + self.config.loss_lambda_final * l_final
    + self.config.loss_lambda_preserve * l_preserve
    + self.config.loss_lambda_gate_prior * l_gate_prior
    + self.config.loss_lambda_preserve_zero * l_preserve_zero  # NEW
)
```

**預期結果**：all-task spatial 從 12/50 回到 **≥ 23/50** (即至少不輸 `zero_fast`)，gate_mean ≤ 0.40。

**Falsifier**：若 < 20/50，代表自標籤迴圈不是主因 — 跳到 Stage C。

### Stage B — 5-term v2 完整實作 (3-5 天)

採用 **Strength 2** 的 `y_t`，實裝 5 個 functional。Fastcache shard 多兩個 boolean column (`y_correct`, `y_preserve`)。重跑 `merge_hfrvla_shards_fast.py` (~30 min)。

**預期結果**：all-task spatial **≥ 28/50**，gate_mean ≤ 0.25，task 8/9 ≥ 2/5 each。

### Stage C — Codex Windowed RD-IB + DAgger-lite (1-2 週)

採用 **Strength 3** 的 `y_t`。錄製 `zero_fast` rollouts 跨 10 個 spatial tasks (5 episodes × 10 tasks × ~150 steps = ~7500 frames)，重算 SmolVLA + DINO 特徵，加入 fastcache。

**預期結果**：all-task spatial **≥ 32/50**，gate_mean ≤ 0.20，task 8/9 ≥ 3/5 each。

---

## Falsifiable Eval (commit to ONE row)

| Stage | Checkpoint | δ_max | Gate threshold | Expected all-task | Falsified below |
|---|---|---:|---:|---:|---:|
| A | `hfrvla_v2a_sonnet` | 0.2 | 0.5 | 23-26/50 | < 20/50 |
| B | `hfrvla_v2b_consolidated` | 0.2 | 0.7 | 28-30/50 | < 23/50 |
| C | `hfrvla_v2c_wrdil` | 0.2 | 0.7 | 30-32/50 | < 27/50 OR gate_mean > 0.30 |

所有 row 都跑 LIBERO spatial, 10 tasks × 5 episodes, seed 42, batch_size=1.

---

## Implementation order

1. **執行 Stage A** (今晚)。同時開始預備 Stage B 的 fastcache column。
2. **觀察 Stage A 結果** (1 天 + 1 天 eval)。記錄 per-task gate mean。
3. **如果 Stage A 達到 23/50**：直接做 Stage B (已預備好 cache)。
4. **如果 Stage A 沒達到 23/50**：先做 per-task gate diagnostic (`docs/hfrvla_failure_hypotheses.md §"Proposed Next Diagnostics"` item 1)，分辨是 gate over-activation 還是 feature mismatch (H5)，再決定走 Stage B 或 H5 修法。
5. **Stage C 只在 Stage B 已奪回 ≥ 23/50 後執行**。先建立基線再投資 rollout cache。

---

## Compression invariant 的形式化保證

**Theorem (informal)**: 上述 Lagrangian 是 information-theoretic rate-distortion problem 的 variational bound：

- $\mathcal{R}$ 是 channel 的 rate (bernoulli prior on gate-on event + per-channel payload cost in log-magnitude)
- $\mathcal{D}$ 是 distortion，只在 correction states (y_t=1) 上算
- $\mathcal{P}$ 是 "default codeword = a_base" 的 cost-of-deviation
- $\mathcal{G}$ 是 calibration：$\ell_t$ 必須是 $y_t$ 的 sufficient statistic
- $\mathcal{S}$ 是 channel 的 temporal coherence (avoiding bursty codes)

當 $\lambda_R$ 升高時，模型會把 *rate* (= 開閘+傳輸的 bits) 降到下界，這正是 compression 的數學定義。

**和 SmolVLA 的關係**：SmolVLA 是 *prior policy* (zero-bit base distribution)。Fast module 是 *correction codec*。整個系統是一個 **mixture-of-experts where the heavy expert is free and the light expert pays rate**。這是 π0-FAST 和 SV-VLA paper 的核心架構決策 — 我們的訓練目標現在和那一族設計收斂了。

---

## 各方最終立場註記

- 🐙 **Opus**：commit to v2 final form (上面 boxed 公式)。Stage 順序 A→B→C。
- 🔴 **Codex**：commit to Windowed RD-IB (Stage C version)。但同意先做 A 作為 baseline。
- 🟡 **Gemini**：commit to RD-IB minimalism。同意 detach gate (Sonnet) 和 preserve_zero (Opus) 是 sufficient additions。
- 🟠 **Sonnet**：commit to 12-line "ship tonight" diff (Stage A)。明確反對 Stage C 在沒有 per-task diagnostic 前執行。

四方 **一致同意 Stage A 的 12 行 diff 應該今晚就跑**，作為所有後續實驗的 baseline。

---

## 附錄: 為什麼這套設計沒有違反「small model = compression」invariant

1. **架構不變**：fast module 仍是 `FastReactiveModule` (GRU + 兩個 linear head)，參數量 < 10M，與 SmolVLA (2B) 維持 hierarchical 關係。
2. **訓練本質不變**：仍是 offline imitation。沒有 RL、沒有 world model、沒有 large planner。
3. **「壓縮什麼」的新答案**：以前壓縮 `a_expert - a_base` 在所有 states (dense)；現在壓縮 **「correction events 的 minimal payload」** (sparse)，default codeword 是 `a_base` 本身。這嚴格符合 rate-distortion 框架。
4. **資訊瓶頸**：透過 $\mathcal{R}$ 明確 penalize $I(\text{features}; \delta) \cdot \mathbb{1}[g=1]$，這正是 InfoBot (Goyal et al.) 的 conditional bottleneck。

---

## Deliverable for the user

`synthesis.md` 本檔。下一步是把 Stage A 的 12 行 diff 實裝到：
- `policy/lerobot_policy_hfrvla/src/lerobot_policy_hfrvla/configuration_hfrvla.py`
- `policy/lerobot_policy_hfrvla/src/lerobot_policy_hfrvla/modeling_hfrvla.py`

跑 30k-step 訓練，eval 一次，記錄 per-task gate mean。然後依結果決定 Stage B 或 H5 修法。
