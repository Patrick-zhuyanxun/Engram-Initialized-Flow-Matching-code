# Legacy scripts (pre-lerobot-train migration)

These files were the working pipeline before the 2026-05-15 migration to
`lerobot-train`. They are kept for reference and as a fallback, but are
**no longer wired into the active workflow**.

| File | Original purpose | Replaced by |
|---|---|---|
| `train_hfrvla.py` | Custom curriculum trainer reading `.pt` files. | `scripts/train_via_lerobot.py` + `lerobot-train`. |
| `precompute_libero.py` | Roll SmolVLA + DINOv3 once, save `.pt` per episode. | `scripts/record_hfrvla_libero.py` (writes a LeRobotDataset v3 instead). |
| `precompute_libero_plan.py` | Sharding helpers (`--ep-from`/`--ep-to`/`--skip-existing`). | (not yet re-implemented; single-process recording for now.) |
| `precompute_libero_parallel.sh` | Multi-shard driver. | (same.) |
| `data.py` | `HFRVLADataset` (loaded `.pt` files). | LeRobotDataset's standard dataloader. |

Spec: `docs/superpowers/specs/2026-05-15-hfrvla-lerobot-native-design.md`
Plan: `docs/superpowers/plans/2026-05-15-hfrvla-lerobot-native.md`
