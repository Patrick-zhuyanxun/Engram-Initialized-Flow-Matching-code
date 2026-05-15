#!/usr/bin/env python
"""Train script for HFRVLA — three-stage curriculum.

See implementation_spec.md §9.

Stages:
    0. Warmup        (~1 k steps): L_delta only; gate + contact heads frozen.
    1. Joint         (1 k – 50 k): L_delta + λ_g·L_gate + λ_c·L_contact.
    2. Refine        (50 k – 60 k): same losses; LR ÷ 10.

Usage:
    python scripts/train_hfrvla.py \\
        --data-dir Hierachical_fast_reactive/checkpoints/libero_spatial_chunks \\
        --output-dir Hierachical_fast_reactive/checkpoints/hfrvla_run01 \\
        --smolvla-pretrained lerobot/smolvla_base
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from lerobot_policy_hfrvla.configuration_hfrvla import HFRVLAConfig
from lerobot_policy_hfrvla.modeling_hfrvla import HFRVLAPolicy
from lerobot_policy_hfrvla.data import HFRVLADataset


# ────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True,
                   help="Directory of precomputed episode_*.pt chunks.")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--smolvla-pretrained", type=str, default="lerobot/smolvla_base")

    # DINOv3 — must use local repo + weights (gated HF repo).
    p.add_argument("--dinov3-repo", type=str,
                   default="/home/hucenrotia/Patrick/VLA_research/Hierachical_fast_reactive/checkpoints/dinov3_src")
    p.add_argument("--dinov3-weights", type=str,
                   default="/home/hucenrotia/Patrick/VLA_research/Hierachical_fast_reactive/checkpoints/Dino_weight/dinov3_vits16_pretrain_lvd1689m-08c60483.pth")
    p.add_argument("--dinov3-arch", default="dinov3_vits16")

    # LIBERO feature shape override (matches what precompute_libero.py uses)
    p.add_argument("--libero-mode", action="store_true", default=True,
                   help="Override SmolVLA's input_features to match LIBERO (image/image2, state=8, action=7).")

    p.add_argument("--seq-len", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)

    p.add_argument("--warmup-steps", type=int, default=1_0)
    p.add_argument("--joint-steps", type=int, default=49_0)   # joint ends at 50k total
    p.add_argument("--refine-steps", type=int, default=10_0)  # 60k total
    p.add_argument("--lr-main", type=float, default=3e-4)
    p.add_argument("--lr-refine", type=float, default=3e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--ckpt-every", type=int, default=5_000)

    p.add_argument("--device", type=str, default="cuda")

    # ── Wandb ──
    p.add_argument("--wandb-project", type=str, default="hfrvla")
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb-run-name", type=str, default=None)
    p.add_argument("--wandb-mode", type=str, default="online",
                   choices=["online", "offline", "disabled"])
    p.add_argument("--no-wandb", action="store_true",
                   help="Disable wandb logging entirely (overrides --wandb-mode).")
    return p.parse_args()


# ────────────────────────────────────────────────────────────────────────
def cosine_with_warmup(step: int, warmup: int, total: int, lr_max: float, lr_min: float) -> float:
    if step < warmup:
        return lr_max * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(1.0, max(0.0, progress))
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * progress))


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for g in optimizer.param_groups:
        g["lr"] = lr


def freeze_gate_and_contact(policy: HFRVLAPolicy) -> None:
    for p in policy.fast.gate_head.parameters():
        p.requires_grad = False
    if policy.fast.contact_head is not None:
        for p in policy.fast.contact_head.parameters():
            p.requires_grad = False


def unfreeze_gate_and_contact(policy: HFRVLAPolicy) -> None:
    for p in policy.fast.gate_head.parameters():
        p.requires_grad = True
    if policy.fast.contact_head is not None:
        for p in policy.fast.contact_head.parameters():
            p.requires_grad = True


# ────────────────────────────────────────────────────────────────────────
def init_wandb(args: argparse.Namespace):
    """Return an initialized wandb run, or None if disabled."""
    if args.no_wandb or args.wandb_mode == "disabled":
        return None
    try:
        import wandb
    except ImportError:
        print("[train] wandb not installed; install with `uv pip install wandb` "
              "to enable logging. Continuing without wandb.")
        return None
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        mode=args.wandb_mode,
        dir=str(args.output_dir),
        config={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    )
    return run


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    wandb_run = init_wandb(args)

    # ── Build dataset ──
    dataset = HFRVLADataset(args.data_dir, seq_len=args.seq_len, require_dino_patches=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    loader_iter = iter(loader)
    print(f"[train] dataset windows: {len(dataset)}")

    # ── Build policy ──
    config = HFRVLAConfig.from_smolvla(
        args.smolvla_pretrained,
        dinov3_local_repo=args.dinov3_repo,
        dinov3_local_weights=args.dinov3_weights,
        dinov3_arch=args.dinov3_arch,
    )
    if args.libero_mode:
        from lerobot.configs.types import PolicyFeature, FeatureType
        from lerobot.utils.constants import OBS_STATE, ACTION
        config.input_features = {
            "observation.images.image":  PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
            "observation.images.image2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
            OBS_STATE:                   PolicyFeature(type=FeatureType.STATE,  shape=(8,)),
        }
        config.output_features = {
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
        }

    policy = HFRVLAPolicy.from_pretrained(args.smolvla_pretrained, config=config)
    policy = policy.to(device)
    policy.train()
    # SmolVLA stays in eval to keep BN/LN frozen.
    policy.model.eval()
    print(f"[train] fast module params: {policy.fast.count_parameters():,}")

    # ── Optimizer ──
    optimizer = AdamW(
        policy.get_optim_params(),
        lr=args.lr_main,
        weight_decay=args.weight_decay,
    )

    total_steps = args.warmup_steps + args.joint_steps + args.refine_steps
    refine_start = args.warmup_steps + args.joint_steps

    # ── Stage 0: Warmup (delta only) ──
    freeze_gate_and_contact(policy)
    config_lambda_gate_cache = policy.config.loss_lambda_gate
    config_lambda_contact_cache = policy.config.loss_lambda_contact
    policy.config.loss_lambda_gate = 0.0
    policy.config.loss_lambda_contact = 0.0

    for step in range(total_steps):
        # Transition: end of warmup → unfreeze heads + restore lambdas
        if step == args.warmup_steps:
            unfreeze_gate_and_contact(policy)
            policy.config.loss_lambda_gate = config_lambda_gate_cache
            policy.config.loss_lambda_contact = config_lambda_contact_cache
            # Rebuild optimizer to include freshly-unfrozen heads.
            optimizer = AdamW(
                policy.get_optim_params(),
                lr=args.lr_main,
                weight_decay=args.weight_decay,
            )
            print(f"[train] step {step}: entered Stage 1 (joint)")
            if wandb_run is not None:
                wandb_run.log({"stage": 1}, step=step)

        # Transition: enter refine → lower LR
        if step == refine_start:
            print(f"[train] step {step}: entered Stage 2 (refine, lr×0.1)")
            if wandb_run is not None:
                wandb_run.log({"stage": 2}, step=step)

        # ── Fetch batch ──
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items() if isinstance(v, torch.Tensor)}

        # ── LR schedule ──
        if step < refine_start:
            lr = cosine_with_warmup(
                step,
                warmup=args.warmup_steps,
                total=refine_start,
                lr_max=args.lr_main,
                lr_min=args.lr_main * 0.1,
            )
        else:
            # Cosine decay inside the refine stage from lr_main*0.1 → lr_refine.
            r = (step - refine_start) / max(1, args.refine_steps)
            lr = args.lr_main * 0.1 * (1 - r) + args.lr_refine * r
        set_lr(optimizer, lr)

        # ── Forward / backward ──
        losses = policy(batch)
        loss = losses["loss"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.get_optim_params(), max_norm=1.0)
        optimizer.step()

        if wandb_run is not None:
            log_payload = {"lr": lr, "loss": float(loss.item())}
            for k, v in losses.items():
                if k == "loss":
                    continue
                log_payload[k] = float(v.item())
            wandb_run.log(log_payload, step=step)

        if step % args.log_every == 0:
            msg = f"step {step:>6d} | lr {lr:.2e} | loss {loss.item():.4f}"
            for k, v in losses.items():
                if k == "loss":
                    continue
                msg += f" | {k} {v.item():.4f}"
            print(msg)

        if step > 0 and step % args.ckpt_every == 0:
            ckpt_path = args.output_dir / f"step_{step:06d}.pt"
            torch.save({"fast_state_dict": policy.fast.state_dict(), "step": step}, ckpt_path)
            print(f"[train] saved {ckpt_path}")

    # Final save
    final_path = args.output_dir / "fast_final.pt"
    torch.save({"fast_state_dict": policy.fast.state_dict(), "step": total_steps}, final_path)
    print(f"[train] done; saved {final_path}")
    if wandb_run is not None:
        wandb_run.summary["fast_final_path"] = str(final_path)
        wandb_run.finish()


if __name__ == "__main__":
    main()
