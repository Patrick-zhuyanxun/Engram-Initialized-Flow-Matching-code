"""
Analyze the quality of the built engram table.

Reports: vocabulary, per-key counts, inter-key cosine similarity,
intra-key variance, and cache coverage.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F


def analyze(engram_path: str) -> None:
    data = torch.load(engram_path, map_location="cpu")
    table: dict[str, torch.Tensor] = data["engram_table"]
    metadata: dict[str, dict] = data["metadata"]
    mode = data.get("mode", "unknown")
    dim = data.get("dim", "?")
    instruction_map: dict[str, str | None] = data.get("instruction_key_map", {})

    print(f"Engram Table Analysis (mode={mode}, dim={dim})")
    print("=" * 70)

    # ── 1. Vocabulary & Counts ─────────────────────────────
    keys = sorted(table.keys())
    print(f"\nTotal engram keys: {len(keys)}")
    # Adapt column names to mode
    norm_key = "centered_norm" if "centered_norm" in next(iter(metadata.values())) else "mean_norm"
    print(f"{'Key':30s} {'Count':>6s} {'Norm':>8s} {'Std Norm':>10s}")
    print("-" * 60)
    for k in keys:
        m = metadata[k]
        norm_val = m.get(norm_key, m.get("mean_norm", 0.0))
        std_val = m.get("std_norm", 0.0)
        print(f"{k:30s} {m['count']:6d} {norm_val:8.4f} {std_val:10.4f}")

    total_demos = sum(m["count"] for m in metadata.values())
    print(f"\nTotal demos contributing: {total_demos}")

    # ── 2. Cache Coverage ──────────────────────────────────
    total_instructions = len(instruction_map)
    hits = sum(1 for v in instruction_map.values() if v is not None)
    print(f"\nInstruction coverage: {hits}/{total_instructions} = {hits / total_instructions * 100:.1f}%")

    # ── 3. Inter-key Cosine Similarity ────────────────────
    if len(keys) > 1:
        vectors = torch.stack([table[k] for k in keys])  # (K, D)
        vectors_norm = F.normalize(vectors, dim=1)
        sim_matrix = vectors_norm @ vectors_norm.T  # (K, K)

        print(f"\nInter-key Cosine Similarity Matrix:")
        # Header
        short_keys = [k[:12] for k in keys]
        header = f"{'':>12s} " + " ".join(f"{sk:>12s}" for sk in short_keys)
        print(header)
        for i, k in enumerate(keys):
            row = f"{short_keys[i]:>12s} "
            for j in range(len(keys)):
                val = sim_matrix[i, j].item()
                if i == j:
                    row += f"{'---':>12s} "
                else:
                    row += f"{val:>12.4f} "
            print(row)

        # Off-diagonal statistics
        mask = ~torch.eye(len(keys), dtype=torch.bool)
        off_diag = sim_matrix[mask]
        print(f"\nOff-diagonal cosine sim: mean={off_diag.mean():.4f}, "
              f"max={off_diag.max():.4f}, min={off_diag.min():.4f}")

        if off_diag.max() > 0.8:
            print("  [WARNING] Some engram pairs have high similarity (>0.8). "
                  "Consider merging or refining keys.")
        elif off_diag.max() < 0.5:
            print("  [GOOD] All engram pairs well-separated (max sim < 0.5).")
        else:
            print("  [OK] Moderate separation between engrams.")

    # ── 4. Engram Norms ───────────────────────────────────
    norms = torch.tensor([table[k].norm().item() for k in keys])
    print(f"\nEngram norms: mean={norms.mean():.4f}, std={norms.std():.4f}, "
          f"min={norms.min():.4f}, max={norms.max():.4f}")

    # Compare to Gaussian noise norm (expected: sqrt(dim))
    expected_gauss = dim ** 0.5 if isinstance(dim, (int, float)) else "N/A"
    print(f"Expected Gaussian noise norm (sqrt(dim)): {expected_gauss}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("engram_path", type=str, nargs="?",
                        default="/home/hucenrotia/Patrick/VLA_research/eifm/engram_table_raw.pt")
    args = parser.parse_args()
    analyze(args.engram_path)
