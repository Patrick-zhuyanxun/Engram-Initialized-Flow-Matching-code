# AGENTS.md - Hierachical_fast_reactive

Scope: this file applies to the `Hierachical_fast_reactive/` project tree.
The directory name is intentionally spelled `Hierachical_fast_reactive` in this
workspace.

## Active Project

This project is **HFRVLA: Hierarchical Fast-Reactive VLA**.

Core idea: wrap a frozen SmolVLA as the slow planner and train a small
wrist-camera fast reactive residual module. At inference:

```text
a_final = SafetyLayer(a_base + gate * clip(delta_a))
```

`a_base` is the SmolVLA chunk action. The trainable fast module emits
`delta_a`, `gate`, and an optional training-only `contact_aux` signal.

## Main Source Files

- Policy plugin: `policy/lerobot_policy_hfrvla/`
- Policy config: `policy/lerobot_policy_hfrvla/src/lerobot_policy_hfrvla/configuration_hfrvla.py`
- Policy model: `policy/lerobot_policy_hfrvla/src/lerobot_policy_hfrvla/modeling_hfrvla.py`
- Fast module: `policy/lerobot_policy_hfrvla/src/lerobot_policy_hfrvla/fast_reactive.py`
- Dataset recorder: `scripts/record_hfrvla_libero.py`
- LeRobot train wrapper: `scripts/train_via_lerobot.py`
- Checkpoint packager: `scripts/package_hfrvla_checkpoint.py`
- Training guide: `docs/training.md`
- Detailed context note: `docs/lerobot_hfrvla_context.md`
- Design contract: `paper/notes/implementation_spec.md`

## LeRobot Plugin Contract

HFRVLA follows the official LeRobot "bring your own policies" convention:

- Python package name starts with `lerobot_policy_`: `lerobot_policy_hfrvla`.
- `pyproject.toml` registers the entry point under `lerobot.policies`:
  `hfrvla = "lerobot_policy_hfrvla:HFRVLAPolicy"`.
- `HFRVLAConfig` registers `@PreTrainedConfig.register_subclass("hfrvla")`.
- `HFRVLAPolicy.name == "hfrvla"` and `config_class == HFRVLAConfig`.
- After plugin source changes, reinstall from the LeRobot venv:
  `cd ~/Robotic_infra/lerobot && uv pip install -e ~/Patrick/VLA_research/Hierachical_fast_reactive/policy/lerobot_policy_hfrvla`

## HFRVLA Dataset Contract

Do not treat training as plain raw LIBERO training. HFRVLA first records a
custom, local LeRobotDataset v3 that bakes in SmolVLA and DINOv3 intermediate
features. The first verified full merged dataset is `HFRVLA_libero_v1` under
`checkpoints/HFRVLA_libero_v1_merged_reindexed`.

Expected features:

- `observation.images.image`: LIBERO third-person image, shape `(256, 256, 3)`
- `observation.images.image2`: LIBERO wrist image, shape `(256, 256, 3)`
- `observation.state`: LIBERO state, shape `(8,)`
- `action`: expert action, shape `(7,)`
- `observation.extra.z_goal`: SmolVLA text pool, current local shape `(960,)`
- `observation.extra.z_phase`: SmolVLA expert pool, current local shape `(480,)`
- `observation.extra.a_base`: SmolVLA predicted base action, shape `(7,)`
- `observation.extra.k_idx_norm`: chunk position in `[0, 1]`, shape `(1,)`
- `observation.extra.dino_patches`: DINOv3 wrist patches, shape `(196, 384)`
- `observation.extra.contact_label`: auxiliary target, shape `(1,)`

`HFRVLAConfig.observation_delta_indices` and `action_delta_indices` both return
`list(range(-(seq_len - 1), 1))`, so the LeRobot loader windows all
`observation.*` keys and `action` to the GRU sequence length. Extras are kept
out of `input_features` normalization on purpose.

## Training And Evaluation

Use the LeRobot-native workflow in `docs/training.md`:

1. Record or merge the HFRVLA LeRobotDataset v3.
2. Run `scripts/test_alignment.py` before long training.
3. Train through `scripts/train_hfrvla_libero_merged.sh`, which pins the
   `HuggingFaceVLA/smolvla_libero` architecture and calls
   `scripts/train_via_lerobot.py`.
4. `scripts/train_via_lerobot.py` monkey-patches
   `lerobot-train` to call `policy.set_training_step(step)` for curriculum.
5. Evaluate with `lerobot-eval`.

For this verified dataset, keep these training overrides unless the dataset is
re-recorded with a different slow planner:

```bash
--policy.vlm_model_name=HuggingFaceTB/SmolVLM2-500M-Instruct
--policy.expert_width_multiplier=0.5
--policy.num_vlm_layers=0
--policy.load_vlm_weights=false
```

For restricted or offline runs, prefer explicit local roots and:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HFRVLA_TMP_ROOT=$HOME/tmp/hfrvla
export HF_DATASETS_CACHE=$HFRVLA_TMP_ROOT/hf_datasets
export TMPDIR=$HFRVLA_TMP_ROOT/tmp
```

## Implementation Rules

- Keep SmolVLA frozen unless the user explicitly asks for a different training
  regime. `get_optim_params()` should return trainable fast-module parameters.
- Do not resurrect the legacy `.pt` cache pipeline as the primary training path;
  legacy scripts stay under `scripts/legacy/`.
- Use `LeRobotDatasetMetadata` when only normalization stats are needed; loading
  a full local-only synthetic dataset can trigger unwanted Hub lookups.
- When creating a LeRobotDataset v3 manually, always call `dataset.finalize()`
  before pushing or relying on the dataset.
- Keep `paper/notes/implementation_spec.md` and `docs/training.md` synchronized
  when architecture, training, or dataset assumptions change.

## Official Docs Captured

These notes were initialized on 2026-05-16 from:

- https://huggingface.co/docs/lerobot/en/bring_your_own_policies
- https://huggingface.co/docs/lerobot/en/il_robots
- https://huggingface.co/docs/lerobot/en/lerobot-dataset-v3
