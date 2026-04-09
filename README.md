# Engram-Initialized Flow Matching (EI-FM)

Code for the **EI-FM** research project: replacing the Gaussian noise initialization `x1 = randn()` in flow-matching robot policies with motor primitive embeddings retrieved from an O(1) engram hash table.

Paper name: **PP-FM (Prior-Preserved Flow Matching)**

---

## Repository Structure

```
.
├── lerobot_policy_eifm/   — EI-FM policy plugin for LeRobot
│   └── src/lerobot_policy_eifm/
│       ├── configuration_eifm.py   # EIFMConfig
│       ├── modeling_eifm.py        # EIFMPolicy + EIFMModel
│       ├── processor_eifm.py       # Preprocessing steps
│       ├── action_hub.py           # EE6DActionSpace (20D)
│       ├── soft_transformer.py     # SoftPromptedTransformer
│       └── utils.py                # Rotation utilities
└── eifm/                  — Engram table tools + training script
    ├── build_engram_table.py       # Build engram table from LIBERO demos
    ├── ngram_extractor.py          # Verb-phrase N-gram extraction (spaCy)
    ├── engram_analysis.py          # Coverage and quality analysis
    ├── engram_table_projected.pt   # Pre-built engram table (14 keys, dim=1024)
    ├── train_eifm.sh               # Training script
    └── config.py                   # Paths and hyperparameters
```

---

## Prerequisites

### 1. Install LeRobot

This plugin requires [LeRobot](https://github.com/huggingface/lerobot) as the base framework.

```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot && pip install -e ".[dev]"
```

### 2. Install EI-FM Plugin

```bash
cd lerobot && uv pip install -e ../lerobot_policy_eifm
```

### 3. Download X-VLA Base Checkpoint

The EI-FM policy fine-tunes on top of the [X-VLA](https://huggingface.co/lerobot/xvla-libero) checkpoint:

```bash
huggingface-cli download lerobot/xvla-libero --local-dir checkpoints/eifm-libero
```

---

## Training on LIBERO-Spatial

```bash
bash eifm/train_eifm.sh
```

The script automatically detects its location and sets paths accordingly. Edit `train_eifm.sh` to configure output directory, steps, and WandB settings.

---

## Key Design

| Component | Details |
|-----------|---------|
| VLM Backbone | Florence-2 (encoder-only) |
| Action Head | SoftPromptedTransformer (24 layers, 16 heads) |
| Action Space | EE6D — 20D: `[pos(3), rot6d(6), gripper(1)] × 2` |
| Noise Init | Engram hash table lookup → `w_engram` projection |
| Trainable Params | ~1.6M (soft prompts + w_engram) out of 880M total |
| Base Dataset | LIBERO-Spatial (1693 episodes, 10 tasks) |

---

## Engram Table

The pre-built engram table (`eifm/engram_table_projected.pt`) contains 14 keys extracted from LIBERO task instructions (verb-phrase N-grams via spaCy). Each key maps to a 1024-dimensional embedding projected through the X-VLA action encoder.

To rebuild from scratch:
```bash
cd lerobot
uv run python ../eifm/build_engram_table.py
```
