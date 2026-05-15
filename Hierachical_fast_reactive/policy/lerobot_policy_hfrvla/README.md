# lerobot_policy_hfrvla

**Hierarchical Fast-Reactive VLA** — a LeRobot plugin that mounts a small, trainable, wrist-camera-only residual policy on top of a frozen SmolVLA. The fast module fires at every control step and emits `(δa, gate, contact_aux)`; the final action is `SafetyLayer(a_base + g · clip(δa))`.

Design contract: `Hierachical_fast_reactive/paper/notes/implementation_spec.md`.

## Quick install

```bash
cd ~/Robotic_infra/lerobot
uv pip install -e ~/Patrick/VLA_research/Hierachical_fast_reactive/policy/lerobot_policy_hfrvla
```

## DINOv3 weights

Access must be granted by Meta:
https://github.com/facebookresearch/dinov3#downloading-weights

Default model: `facebook/dinov3-vits16-pretrain-lvd1689m` (ViT-S/16, ~21 M params).

## Train and eval

```bash
cd ~/Patrick/VLA_research/Hierachical_fast_reactive

~/Robotic_infra/lerobot/.venv/bin/python scripts/record_hfrvla_libero.py \
    --src-repo-id HuggingFaceVLA/libero \
    --out-repo-id HFRVLA_libero_v1 \
    --out-root checkpoints/HFRVLA_libero_v1 \
    --smolvla lerobot/smolvla_base \
    --dinov3-repo checkpoints/dinov3_src \
    --dinov3-weights checkpoints/Dino_weight/dinov3_vits16_pretrain_lvd1689m-08c60483.pth

~/Robotic_infra/lerobot/.venv/bin/python scripts/train_via_lerobot.py \
    --dataset.repo_id=HFRVLA_libero_v1 \
    --dataset.root=checkpoints/HFRVLA_libero_v1 \
    --policy.type=hfrvla
```

## Registration

`lerobot-train` and `lerobot-eval` recognize `--policy.type=hfrvla` via the entry point in `pyproject.toml`.
