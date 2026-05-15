#!/usr/bin/env python
"""Offline precomputation for HFRVLA training on the LIBERO dataset.

Pipeline:
    1. Load HuggingFaceVLA/libero as a LeRobotDataset.
    2. Instantiate a frozen HFRVLAPolicy from ``lerobot/smolvla_base``.
       The policy's forward hooks (z_goal, z_phase) will fire during chunk
       inference.
    3. For each episode:
         - Reset policy.
         - Step through frames; let SmolVLA generate chunks as needed.
         - For each control step record:
               wrist_rgb (normalized for DINOv3)
               proprio
               a_base   (popped from SmolVLA's queue)
               k_idx_norm
               z_goal   (cached at last chunk boundary)
               z_phase
               a_expert (ground-truth action from the demo)
               contact_label  (heuristic from action-gripper delta; 0 by default)
    4. Save one ``.pt`` per episode into ``--out-dir``.
    5. Optionally run DINOv3 over all wrist frames and stash ``dino_patches``.

Example:
    python scripts/precompute_libero.py \\
        --repo-id HuggingFaceVLA/libero \\
        --out-dir Hierachical_fast_reactive/checkpoints/libero_chunks \\
        --smolvla lerobot/smolvla_base \\
        --dinov3-repo Hierachical_fast_reactive/checkpoints/dinov3_src \\
        --dinov3-weights Hierachical_fast_reactive/checkpoints/Dino_weight/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \\
        --max-episodes 50 \\
        --device cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

POLICY_SRC = Path(__file__).resolve().parents[2] / "policy" / "lerobot_policy_hfrvla" / "src"
if POLICY_SRC.exists():
    sys.path.insert(0, str(POLICY_SRC))

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
from lerobot.policies.utils import populate_queues
from lerobot.utils.constants import ACTION, OBS_STATE

from lerobot_policy_hfrvla.configuration_hfrvla import HFRVLAConfig
from lerobot_policy_hfrvla.modeling_hfrvla import HFRVLAPolicy
from lerobot_policy_hfrvla.processor_hfrvla import normalize_for_dinov3
try:
    from .data import (
        EpisodeRecord,
        precompute_dinov3 as run_precompute_dinov3,
    )
    from .precompute_libero_plan import (
        atomic_save_episode,
        episode_output_path,
        is_existing_episode_usable,
        resolve_episode_indices,
        should_process_episode,
    )
except ImportError:
    from data import (
        EpisodeRecord,
        precompute_dinov3 as run_precompute_dinov3,
    )
    from precompute_libero_plan import (
        atomic_save_episode,
        episode_output_path,
        is_existing_episode_usable,
        resolve_episode_indices,
        should_process_episode,
    )


# ────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", default="HuggingFaceVLA/libero")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--smolvla", default="lerobot/smolvla_base")

    p.add_argument("--dinov3-repo", type=str, required=True,
                   help="Local clone of facebookresearch/dinov3.")
    p.add_argument("--dinov3-weights", type=str, required=True,
                   help="Local .pth weight file (e.g. dinov3_vits16_pretrain_lvd1689m).")
    p.add_argument("--dinov3-arch", default="dinov3_vits16")

    p.add_argument("--wrist-key", default="observation.images.image2",
                   help="LIBERO key for the wrist camera (default: image2).")
    p.add_argument("--state-key", default=OBS_STATE)
    p.add_argument("--action-key", default=ACTION)

    p.add_argument("--max-episodes", type=int, default=None,
                   help="If set, only process the first N episodes (useful for smoke tests).")
    p.add_argument("--ep-from", type=int, default=0,
                   help="First episode index to process, inclusive. Used for sharding.")
    p.add_argument("--ep-to", type=int, default=None,
                   help="Last episode index to process, exclusive. Used for sharding.")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip episode_*.pt files that already exist in --out-dir.")
    p.add_argument("--skip-rollout", action="store_true",
                   help="Do not run SmolVLA rollouts; useful before a DINO-only pass.")
    p.add_argument("--dinov3-only", action="store_true",
                   help="Alias for --skip-rollout with DINOv3 enabled.")
    p.add_argument("--skip-dinov3", action="store_true",
                   help="Skip the DINOv3 feature precomputation step.")
    p.add_argument("--dinov3-batch-size", type=int, default=1024,
                   help="Batch size for the final DINOv3 patch precompute pass.")
    p.add_argument("--overwrite-dinov3", action="store_true",
                   help="Recompute DINOv3 patches even if dino_patches already exist.")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if args.dinov3_only:
        args.skip_rollout = True
        args.skip_dinov3 = False
    if args.skip_rollout and args.skip_dinov3:
        p.error("--skip-rollout and --skip-dinov3 leave no work to do")
    return args


# ────────────────────────────────────────────────────────────────────────
def _to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def precompute_one_episode(
    dataset: LeRobotDataset,
    ep_idx: int,
    policy: HFRVLAPolicy,
    pre_processor,
    wrist_key: str,
    state_key: str,
    action_key: str,
    device: torch.device,
) -> EpisodeRecord:
    """Roll the frozen SmolVLA over one episode and capture per-step records."""
    ep_meta = dataset.meta.episodes[ep_idx]
    ep_from = int(ep_meta["dataset_from_index"])
    ep_to = int(ep_meta["dataset_to_index"])
    ep_task = (
        ep_meta["tasks"][0] if isinstance(ep_meta.get("tasks"), list) and ep_meta["tasks"]
        else None
    )

    policy.reset()
    n_action_steps = policy.config.n_action_steps

    wrist_buf, proprio_buf, a_base_buf = [], [], []
    k_idx_buf, zgoal_buf, zphase_buf = [], [], []
    a_expert_buf, contact_buf = [], []

    cached_zgoal: torch.Tensor | None = None
    cached_zphase: torch.Tensor | None = None
    chunk_consumed = 0

    for t in range(ep_from, ep_to):
        sample = dataset[t]

        # Attach task string for SmolVLA's processor; it expects this under
        # `complementary_data["task"]` and will tokenize it.
        sample_with_task = dict(sample)
        if "task" not in sample_with_task and ep_task is not None:
            sample_with_task["task"] = ep_task
        # SmolVLA processor wraps the dict into an EnvTransition. Older
        # versions accept a raw dict and add the batch dim themselves.
        try:
            batch = pre_processor(sample_with_task)
        except Exception:
            # Fall back to manual batching if the processor signature changed.
            batch = {k: (v.unsqueeze(0) if isinstance(v, torch.Tensor) else v)
                     for k, v in sample_with_task.items()}
        if not isinstance(batch, dict):
            batch = dict(batch)
        batch = _to_device(batch, device)

        # ── 1. Re-plan a chunk if SmolVLA's queue is empty ──
        batch = policy._prepare_batch(batch)
        policy._queues = populate_queues(policy._queues, batch, exclude_keys=[ACTION])
        if len(policy._queues[ACTION]) == 0:
            policy._clear_hook_cache()
            with torch.no_grad():
                actions = policy._get_action_chunk(batch)
            policy._queues[ACTION].extend(actions.transpose(0, 1)[: n_action_steps])
            cached_zgoal = policy._zgoal_cache
            cached_zphase = policy._zphase_cache
            chunk_consumed = 0

        a_base = policy._queues[ACTION].popleft()      # (1, action_dim)
        chunk_consumed += 1
        k = chunk_consumed - 1
        k_norm = k / max(1, n_action_steps - 1)

        # ── 2. Capture per-step tensors ──
        wrist_raw = sample[wrist_key].float()
        if wrist_raw.dim() == 3:
            wrist_raw = wrist_raw.unsqueeze(0)
        # If HWC, permute to CHW.
        if wrist_raw.shape[-1] == 3 and wrist_raw.shape[1] != 3:
            wrist_raw = wrist_raw.permute(0, 3, 1, 2)
        if wrist_raw.max() > 1.5:
            wrist_raw = wrist_raw / 255.0
        wrist_norm = normalize_for_dinov3(wrist_raw).squeeze(0).cpu()

        wrist_buf.append(wrist_norm)
        proprio_buf.append(sample[state_key].cpu())
        a_base_buf.append(a_base.squeeze(0).cpu())
        k_idx_buf.append(k_norm)
        zgoal_buf.append(
            cached_zgoal.squeeze(0).cpu() if cached_zgoal is not None else torch.zeros(1)
        )
        zphase_buf.append(
            cached_zphase.squeeze(0).cpu() if cached_zphase is not None else torch.zeros(1)
        )
        a_expert_buf.append(sample[action_key].cpu())
        # LIBERO does not have explicit contact labels; we leave 0 here and
        # let the auxiliary head learn from action-gripper deltas only.
        contact_buf.append(0.0)

    return EpisodeRecord(
        wrist_rgb=torch.stack(wrist_buf, dim=0),
        proprio=torch.stack(proprio_buf, dim=0),
        a_base=torch.stack(a_base_buf, dim=0),
        k_idx_norm=torch.tensor(k_idx_buf).unsqueeze(-1),
        z_goal=torch.stack(zgoal_buf, dim=0),
        z_phase=torch.stack(zphase_buf, dim=0),
        a_expert=torch.stack(a_expert_buf, dim=0),
        contact_label=torch.tensor(contact_buf),
    )


# ────────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    saved: list[Path] = []
    skipped = 0
    if not args.skip_rollout:
        print(f"[precompute] loading dataset: {args.repo_id}", flush=True)
        dataset = LeRobotDataset(args.repo_id)
        n_total = int(dataset.num_episodes)
        episode_indices = resolve_episode_indices(
            n_total=n_total,
            max_episodes=args.max_episodes,
            ep_from=args.ep_from,
            ep_to=args.ep_to,
        )
        selected_total = min(args.max_episodes, n_total) if args.max_episodes else n_total
        print(
            f"[precompute] dataset has {n_total} episodes; selected {selected_total}; "
            f"processing shard [{args.ep_from}, "
            f"{args.ep_to if args.ep_to is not None else selected_total}) "
            f"({len(episode_indices)} episodes)",
            flush=True,
        )

        print(f"[precompute] building HFRVLAPolicy from {args.smolvla}", flush=True)
        config = HFRVLAConfig.from_smolvla(
            args.smolvla,
            dinov3_local_repo=args.dinov3_repo,
            dinov3_local_weights=args.dinov3_weights,
            dinov3_arch=args.dinov3_arch,
        )

        # Overwrite input/output features to match the LIBERO dataset schema. The
        # pretrained checkpoint was trained on a different camera naming and DOF
        # count; SmolVLA pads state/action internally so dim mismatches are fine,
        # but image keys must match what the policy sees in each batch.
        from lerobot.configs.types import PolicyFeature, FeatureType
        from lerobot.utils.constants import OBS_STATE, ACTION
        libero_feats = {
            "observation.images.image":  PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
            "observation.images.image2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
            OBS_STATE:                   PolicyFeature(type=FeatureType.STATE,  shape=(8,)),
        }
        config.input_features = libero_feats
        config.output_features = {
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
        }

        policy = HFRVLAPolicy.from_pretrained(args.smolvla, config=config)
        policy = policy.to(device).eval()
        print(f"[precompute] policy on {device}; fast module params: "
              f"{policy.fast.count_parameters():,}", flush=True)

        # Build SmolVLA's pre-processor (tokenizes the task string, etc.). We
        # pass the dataset stats from LeRobotDataset.meta when available.
        pre_processor, _ = make_smolvla_pre_post_processors(
            config, dataset_stats=getattr(dataset.meta, "stats", None)
        )

        for ep in episode_indices:
            path = episode_output_path(args.out_dir, ep)
            if args.skip_existing and path.exists() and not is_existing_episode_usable(path):
                print(f"[precompute]   ep {ep:>4d}  REBUILD unreadable -> {path.name}", flush=True)
            elif not should_process_episode(args.out_dir, ep, skip_existing=args.skip_existing):
                skipped += 1
                print(f"[precompute]   ep {ep:>4d}  SKIP existing -> {path.name}", flush=True)
                continue

            rec = precompute_one_episode(
                dataset, ep, policy, pre_processor,
                wrist_key=args.wrist_key,
                state_key=args.state_key,
                action_key=args.action_key,
                device=device,
            )
            path = atomic_save_episode(rec, args.out_dir, ep)
            saved.append(path)
            print(f"[precompute]   ep {ep:>4d}  T={rec.length():>4d}  -> {path.name}", flush=True)

        print(f"[precompute] saved {len(saved)} episodes; skipped {skipped}.", flush=True)
    else:
        print("[precompute] skipping rollout stage.", flush=True)

    # ── DINOv3 patches ──
    if not args.skip_dinov3:
        print("[precompute] running DINOv3 over all wrist frames ...", flush=True)
        run_precompute_dinov3(
            chunks_dir=args.out_dir,
            local_repo=args.dinov3_repo,
            local_weights=args.dinov3_weights,
            arch=args.dinov3_arch,
            batch_size=args.dinov3_batch_size,
            device=str(device),
            skip_existing=not args.overwrite_dinov3,
        )
        print("[precompute] DINOv3 features attached.", flush=True)

    print("[precompute] done.", flush=True)


if __name__ == "__main__":
    main()
