# EI-FM: Engram-Initialized Flow Matching

## Overview

EI-FM 將 flow matching 的初始噪聲 `x1 = torch.randn(...)` 替換為從 O(1) hash table 檢索的 motor primitive embedding（engram）。基於 X-VLA (LeRobot) 架構實作。

## 目錄結構

```
eifm/                                    # Engram 建構工具
├── config.py                            # 路徑與超參數
├── ngram_extractor.py                   # spaCy verb-phrase N-gram 提取
├── build_engram_table.py                # Engram hash table 建構（raw + projected）
├── engram_analysis.py                   # Engram 品質分析
├── train_eifm.sh                        # 訓練腳本
└── README.md                            # 本文件

lerobot_policy_eifm/                     # LeRobot Policy Plugin
├── pyproject.toml
└── src/lerobot_policy_eifm/
    ├── __init__.py
    ├── configuration_eifm.py            # ★ EIFMConfig（新增 engram 設定）
    ├── modeling_eifm.py                 # ★ EIFMModel + EIFMPolicy（核心修改）
    ├── processor_eifm.py               # 複製自 X-VLA（修改 import）
    ├── soft_transformer.py             # 複製自 X-VLA（未改）
    ├── action_hub.py                   # 複製自 X-VLA（未改）
    ├── configuration_florence2.py      # 複製自 X-VLA（未改）
    ├── modeling_florence2.py           # 複製自 X-VLA（未改）
    └── utils.py                        # 複製自 X-VLA（未改）

checkpoints/eifm-libero/                 # EI-FM Checkpoint（包裝 X-VLA 權重）
├── config.json                          # ★ type="eifm" + engram 設定
├── model.safetensors → X-VLA 權重       # symlink
├── policy_preprocessor.json → X-VLA     # symlink
└── policy_postprocessor.json → X-VLA    # symlink
```

## 相對於 X-VLA 的修改

### 1. `configuration_eifm.py`（新增 4 個 engram 欄位）

```python
# EI-FM 新增設定
engram_path: str | None = None           # Engram table 路徑
engram_p_engram: float = 0.8             # 訓練時使用 engram 的機率
engram_key_mode: str = "multi"           # N-gram key 模式
train_action_projections: bool = False   # 是否訓練 action_encoder/decoder
```

### 2. `modeling_eifm.py`（核心修改 3 處 + freezing 修正）

#### ① `__init__`：載入 engram + 建立 w_engram
```python
# 新增：
self.engram_table: dict[str, Tensor] = data["engram_table"]  # {key: Tensor(1024,)}
self.w_engram = nn.Linear(1024, chunk_size * dim_action)      # 1024 → 600
```

#### ② `forward()`：混合噪聲初始化（訓練時）
```python
# 原始 X-VLA:
action_noisy = torch.randn_like(action) * t + action * (1-t)

# EI-FM 修改:
noise = torch.randn_like(action)
if self.engram_table is not None:
    engram_noise = self._get_engram_noise(batch_size, device, dtype)
    mask = torch.rand(B, 1, 1) < self.config.engram_p_engram  # per-sample Bernoulli
    noise = torch.where(mask, engram_noise, noise)
action_noisy = noise * t + action * (1-t)
```

#### ③ `generate_actions()`：engram 替換初始噪聲（推論時）
```python
# 原始 X-VLA:
x1 = torch.randn(B, chunk_size, action_dim)

# EI-FM 修改:
if self.engram_table is not None:
    x1 = self._get_engram_noise(B, device, dtype)  # 100% engram
else:
    x1 = torch.randn(B, chunk_size, action_dim)    # fallback
```

#### ④ `_get_engram_noise()`：新方法
```python
def _get_engram_noise(self, batch_size, device, dtype):
    # instruction → key → engram_table[key] → w_engram → (B, chunk, action_dim)
    # Cache miss → fallback to randn
```

#### ⑤ `_apply_freezing()`：修正 soft_prompt 匹配 + VLM projection 凍結
- 修正："soft_prompts" → "soft_prompt"（匹配 `soft_prompt_hub`）
- 新增：凍結 `image_projection`, `image_proj_norm`, `image_pos_embed`

### 3. `checkpoints/eifm-libero/config.json`
- `"type": "xvla"` → `"type": "eifm"`
- 新增 `engram_path`, `engram_p_engram`, `engram_key_mode`, `train_action_projections`
- `freeze_vision_encoder: true`, `freeze_language_encoder: true`
- `train_policy_transformer: false`, `train_soft_prompts: true`
- `dtype: "bfloat16"`

## 可訓練參數

| 元件 | 參數量 | 用途 |
|------|--------|------|
| soft_prompt_hub | 983,040 | 30 domains × 32 tokens × 1024 dim |
| w_engram | 615,000 | Engram → action space 投影 |
| **Total** | **1,598,040** | 880M 總參數的 **0.18%** |

## 使用方式

### Step 1: 安裝依賴
```bash
cd /home/hucenrotia/Patrick/VLA_research/lerobot
uv pip install spacy h5py
uv run python -m spacy download en_core_web_sm
uv pip install -e ../lerobot_policy_eifm
```

### Step 2: 下載 X-VLA Checkpoint
```bash
uv run huggingface-cli download lerobot/xvla-libero \
  --local-dir ~/.cache/huggingface/hub/models--lerobot--xvla-libero/snapshots/latest
```
（已下載至 `~/.cache/huggingface/hub/models--lerobot--xvla-libero/`）

### Step 3: 建構 Engram Table

```bash
# Step A: Raw 7D engram（快速驗證）
cd /home/hucenrotia/Patrick/VLA_research/lerobot
uv run python ../eifm/build_engram_table.py --mode raw

# Step B: Projected 1024D engram（需要 checkpoint）
uv run python ../eifm/build_engram_table.py --mode projected \
  --checkpoint ~/.cache/huggingface/hub/models--lerobot--xvla-libero/snapshots/12e8783e996944f5c97e490d37d4c145484ed70a/

# 分析品質
uv run python ../eifm/engram_analysis.py ../eifm/engram_table_projected.pt
```

### Step 4: 訓練

```bash
cd /home/hucenrotia/Patrick/VLA_research/lerobot
bash ../eifm/train_eifm.sh
```

或直接執行：
```bash
export MUJOCO_GL=egl
cd /home/hucenrotia/Patrick/VLA_research/lerobot

uv run lerobot-train \
  --policy.path=/home/hucenrotia/Patrick/VLA_research/checkpoints/eifm-libero \
  --policy.repo_id=local/eifm-libero-spatial \
  --policy.push_to_hub=false \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --env.type=libero \
  --env.task=libero_spatial \
  --output_dir=/home/hucenrotia/Patrick/VLA_research/outputs/eifm_libero_spatial_v1 \
  --steps=30000 \
  --batch_size=4 \
  --eval.batch_size=1 \
  --eval.n_episodes=3 \
  --eval_freq=5000 \
  --save_freq=5000 \
  --save_checkpoint=true \
  --log_freq=50
```

### Step 5: 評估

```bash
uv run lerobot-eval \
  --policy.path=/home/hucenrotia/Patrick/VLA_research/outputs/eifm_libero_spatial_v1/checkpoints/last/pretrained_model \
  --env.type=libero \
  --env.task=libero_spatial \
  --eval.batch_size=1 \
  --eval.n_episodes=10
```

## Engram Table 統計

- **14 unique engram keys**（`pick_up_place`, `put`, `open`, `close`, `push` 等）
- **103/103 instruction coverage** (100%)
- **6000 demos** contributing
- **Dimension**: 1024（centered, 去除 DomainAwareLinear bias dominance）

## 驗證結果

### Loss 收斂（測試訓練 750 steps）

| Step | Loss | Grad Norm | LR |
|------|------|-----------|----|
| 50 | 224.87 | 7.83 | 1.6e-5 |
| 300 | 207.04 | 52.24 | 9.9e-5 |
| 500 | 167.07 | 130.53 | 9.8e-5 |
| 700 | 129.84 | 65.72 | 9.6e-5 |

Loss 從 224 → 130，正常收斂。

## 實驗矩陣（後續）

| Model | S=1 | S=3 | S=5 | S=10 |
|-------|-----|-----|-----|------|
| Vanilla X-VLA | - | - | - | baseline |
| EI-FM (p=0.5) | - | - | - | - |
| EI-FM (p=0.8) | - | - | - | - |
| EI-FM (p=1.0) | - | - | - | - |
| Random engram | - | - | - | - |
| Cache miss forced | - | - | - | - |

## 技術備註

1. **engram_path=None 時行為與 X-VLA 完全相同**
2. **Domain ID = 3**（LIBERO，由 checkpoint preprocessor 設定）
3. **Centering**：Projected engrams 經過 global mean centering，去除 DomainAwareLinear bias dominance
4. **w_engram 隨機初始化**：Xavier normal，從 X-VLA checkpoint 載入時不存在此層
5. **bf16 訓練**：模型約 1.7GB，4 batch 在 A5000 (24GB) 可跑
