"""
Build the Static Engram hash table from LIBERO demonstrations.

Step A: Raw action engram (mean-pooled 7D actions, no projection).
Step B: Projected engram (DomainAwareLinear → R^1024) — requires X-VLA checkpoint.

Usage:
    # Step A (no checkpoint needed):
    uv run python build_engram_table.py --mode raw

    # Step B (requires checkpoint):
    uv run python build_engram_table.py --mode projected --checkpoint ./checkpoints/xvla-libero
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch

from config import (
    ACTION_ENCODER_INPUT,
    DIM_ACTION,
    DIM_PROPRIO,
    DIM_TIME,
    ENGRAM_KEY_MODE,
    HIDDEN_SIZE,
    LIBERO_DATA_DIR,
    LIBERO_DOMAIN_ID,
    LIBERO_RAW_ACTION_DIM,
    LIBERO_SUITES,
    NUM_DOMAINS,
    OUTPUT_DIR,
)
from ngram_extractor import compute_engram_key, extract_verb_ngrams, load_spacy_model


def _collect_demos(
    suites: list[str],
) -> list[dict]:
    """
    Read all LIBERO HDF5 demos and return a list of
    {instruction, actions: np.ndarray(T,7), suite, task_name, demo_idx}.
    """
    nlp = load_spacy_model()
    records: list[dict] = []

    for suite in suites:
        suite_dir = LIBERO_DATA_DIR / suite
        if not suite_dir.exists():
            print(f"  [SKIP] {suite_dir}")
            continue

        for fpath in sorted(suite_dir.glob("*.hdf5")):
            with h5py.File(fpath, "r") as f:
                problem_info = json.loads(f["data"].attrs["problem_info"])
                instruction = problem_info["language_instruction"]
                ngrams = extract_verb_ngrams(instruction, nlp)
                key = compute_engram_key(ngrams, mode=ENGRAM_KEY_MODE)

                num_demos = int(f["data"].attrs.get("num_demos", 50))
                for di in range(num_demos):
                    demo_key = f"data/demo_{di}"
                    if demo_key not in f:
                        continue
                    actions = f[f"{demo_key}/actions"][:]  # (T, 7)
                    records.append(
                        {
                            "instruction": instruction,
                            "key": key,
                            "actions": actions,
                            "suite": suite,
                            "task": fpath.stem,
                            "demo_idx": di,
                        }
                    )

    print(f"Collected {len(records)} demos from {len(suites)} suites.")
    return records


def build_raw_engram_table(records: list[dict], output_path: Path) -> dict:
    """
    Step A: Mean-pool raw 7D actions per engram key.
    Result: {key: Tensor(7,)}
    """
    accum: dict[str, list[np.ndarray]] = defaultdict(list)
    instruction_key_map: dict[str, str | None] = {}

    for rec in records:
        key = rec["key"]
        if key is None:
            continue
        # Mean-pool trajectory over time → (7,)
        mean_action = rec["actions"].mean(axis=0).astype(np.float32)
        accum[key].append(mean_action)
        instruction_key_map[rec["instruction"]] = key

    engram_table: dict[str, torch.Tensor] = {}
    metadata: dict[str, dict] = {}

    for key, vecs in accum.items():
        stacked = np.stack(vecs)  # (N, 7)
        engram_table[key] = torch.from_numpy(stacked.mean(axis=0))
        metadata[key] = {
            "count": len(vecs),
            "mean_norm": float(np.linalg.norm(stacked.mean(axis=0))),
            "std_norm": float(np.std(np.linalg.norm(stacked, axis=1))),
        }

    result = {
        "engram_table": engram_table,
        "instruction_key_map": instruction_key_map,
        "metadata": metadata,
        "mode": "raw",
        "dim": LIBERO_RAW_ACTION_DIM,
    }
    torch.save(result, output_path)
    print(f"\n[Raw] Saved engram table ({len(engram_table)} keys) to {output_path}")
    return result


def _timestep_embedding(t: float, dim: int, max_period: int = 100) -> torch.Tensor:
    """Sinusoidal timestep embedding (matches X-VLA's implementation)."""
    half = dim // 2
    freqs = torch.exp(-torch.arange(half, dtype=torch.float32) * (np.log(max_period) / half))
    args = torch.tensor([t], dtype=torch.float32) * freqs
    embedding = torch.cat([torch.cos(args), torch.sin(args)])
    if dim % 2 == 1:
        embedding = torch.cat([embedding, torch.zeros(1)])
    return embedding


def build_projected_engram_table(
    records: list[dict],
    checkpoint_path: str,
    output_path: Path,
) -> dict:
    """
    Step B: Project through frozen DomainAwareLinear → R^1024.

    For each demo:
    1. actions (T,7) → mean-pool → (7,) → pad to 20D
    2. proprio = zeros(32)
    3. time_emb = timestep_embedding(t=0, dim=32)
    4. concat → (84,)
    5. DomainAwareLinear(84 → 1024, domain_id=3) → z_i
    """
    from safetensors.torch import load_file

    # ── Load action_encoder weights ─────────────────────────
    print(f"Loading checkpoint from {checkpoint_path} ...")
    ckpt_path = Path(checkpoint_path)

    # Try safetensors first, then .pt
    safetensors_files = list(ckpt_path.glob("*.safetensors"))
    if safetensors_files:
        state_dict = {}
        for sf in safetensors_files:
            state_dict.update(load_file(str(sf)))
    else:
        pt_files = list(ckpt_path.glob("*.pt")) + list(ckpt_path.glob("*.bin"))
        if pt_files:
            state_dict = torch.load(pt_files[0], map_location="cpu")
        else:
            raise FileNotFoundError(f"No checkpoint files in {ckpt_path}")

    # Extract action_encoder keys
    encoder_fc_key = None
    encoder_bias_key = None
    for k in state_dict:
        if "action_encoder" in k and "fc" in k:
            encoder_fc_key = k
        if "action_encoder" in k and "bias" in k:
            encoder_bias_key = k

    if encoder_fc_key is None:
        # Try alternative key patterns
        print("Available keys with 'encoder':")
        for k in state_dict:
            if "encoder" in k.lower():
                print(f"  {k}: {state_dict[k].shape}")
        raise KeyError("Cannot find action_encoder weights in checkpoint")

    print(f"  action_encoder fc: {encoder_fc_key} → {state_dict[encoder_fc_key].shape}")
    print(f"  action_encoder bias: {encoder_bias_key} → {state_dict[encoder_bias_key].shape}")

    # Build standalone DomainAwareLinear
    fc_weight = state_dict[encoder_fc_key]  # (num_domains, hidden_size * input_size)
    bias_weight = state_dict[encoder_bias_key]  # (num_domains, hidden_size)

    # Reshape fc for domain=LIBERO_DOMAIN_ID
    W = fc_weight[LIBERO_DOMAIN_ID].view(ACTION_ENCODER_INPUT, HIDDEN_SIZE)  # (84, 1024)
    b = bias_weight[LIBERO_DOMAIN_ID]  # (1024,)

    # Pre-compute fixed components
    time_emb = _timestep_embedding(0.0, DIM_TIME)  # (32,) — t=0 means clean action
    proprio_zeros = torch.zeros(DIM_PROPRIO)  # (32,)

    # ── Project all demos ───────────────────────────────────
    accum: dict[str, list[torch.Tensor]] = defaultdict(list)
    instruction_key_map: dict[str, str | None] = {}

    for rec in records:
        key = rec["key"]
        if key is None:
            continue

        # Mean-pool actions → (7,) → pad to (20,)
        mean_action = torch.from_numpy(rec["actions"].mean(axis=0).astype(np.float32))
        action_padded = torch.zeros(DIM_ACTION)
        action_padded[: LIBERO_RAW_ACTION_DIM] = mean_action

        # Concat: [action(20), time(32), proprio(32)] → (84,)
        encoder_input = torch.cat([action_padded, time_emb, proprio_zeros])  # (84,)

        # DomainAwareLinear forward: z = input @ W + b
        z_i = encoder_input @ W + b  # (1024,)

        accum[key].append(z_i)
        instruction_key_map[rec["instruction"]] = key

    # ── Aggregate ───────────────────────────────────────────
    raw_engram_table: dict[str, torch.Tensor] = {}
    for key, vecs in accum.items():
        stacked = torch.stack(vecs)  # (N, 1024)
        raw_engram_table[key] = stacked.mean(dim=0)  # (1024,)

    # ── Center: subtract global mean to remove bias dominance ──
    # The DomainAwareLinear bias term dominates the output, making all
    # engrams nearly identical (cosine sim ≈ 1.0). Centering reveals
    # the actual action-dependent variation.
    all_engrams = torch.stack(list(raw_engram_table.values()))  # (K, 1024)
    global_mean = all_engrams.mean(dim=0)  # (1024,)

    engram_table: dict[str, torch.Tensor] = {}
    metadata: dict[str, dict] = {}

    for key, raw_vec in raw_engram_table.items():
        centered = raw_vec - global_mean  # (1024,)
        engram_table[key] = centered
        metadata[key] = {
            "count": len(accum[key]),
            "raw_norm": float(raw_vec.norm().item()),
            "centered_norm": float(centered.norm().item()),
            "std_norm": float(torch.stack(accum[key]).norm(dim=1).std().item()),
        }

    result = {
        "engram_table": engram_table,
        "instruction_key_map": instruction_key_map,
        "metadata": metadata,
        "global_mean": global_mean,  # stored for potential de-centering
        "mode": "projected_centered",
        "dim": HIDDEN_SIZE,
        "domain_id": LIBERO_DOMAIN_ID,
    }
    torch.save(result, output_path)
    print(f"\n[Projected] Saved engram table ({len(engram_table)} keys, dim={HIDDEN_SIZE}) to {output_path}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Build EI-FM Static Engram Table")
    parser.add_argument("--mode", choices=["raw", "projected"], default="raw")
    parser.add_argument("--checkpoint", type=str, default=None, help="X-VLA checkpoint dir")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument(
        "--suites",
        nargs="+",
        default=LIBERO_SUITES,
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = str(OUTPUT_DIR / f"engram_table_{args.mode}.pt")

    print(f"Building {args.mode} engram table ...")
    print(f"  Suites: {args.suites}")
    print(f"  Output: {args.output}")

    records = _collect_demos(args.suites)

    if args.mode == "raw":
        build_raw_engram_table(records, Path(args.output))
    else:
        if args.checkpoint is None:
            print("ERROR: --checkpoint required for projected mode")
            sys.exit(1)
        build_projected_engram_table(records, args.checkpoint, Path(args.output))


if __name__ == "__main__":
    main()
