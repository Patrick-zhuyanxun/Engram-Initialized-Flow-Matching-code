"""Offline decomposition pipeline for HFRVLA training.

Three responsibilities (see implementation_spec.md §7):

1. :func:`precompute_chunks`
   Rolls a frozen SmolVLA over every demo in a LeRobot dataset, captures the
   per-step ``(wrist_rgb, proprio, a_base, k_idx, z_goal, z_phase, a_expert,
   contact_label)`` records and saves one ``.pt`` file per demo.

2. :func:`precompute_dinov3`
   Walks the saved chunks, runs DINOv3 over every wrist frame **once**, and
   stores ``dino_patches`` alongside.

3. :class:`HFRVLADataset`
   ``torch.utils.data.Dataset`` returning ``seq_len`` consecutive steps per
   item for GRU-friendly training.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.data import Dataset

from lerobot_policy_hfrvla.configuration_hfrvla import HFRVLAConfig
from lerobot_policy_hfrvla.dinov3_backbone import DINOv3Backbone
from lerobot_policy_hfrvla.processor_hfrvla import normalize_for_dinov3


# ────────────────────────────────────────────────────────────────────────
# Per-episode record schema (saved as a torch dict of tensors)
# ────────────────────────────────────────────────────────────────────────
RECORD_KEYS = (
    "wrist_rgb",       # (T, 3, 224, 224) — already normalized for DINOv3
    "proprio",         # (T, proprio_dim)
    "a_base",          # (T, action_dim)
    "k_idx_norm",      # (T, 1)
    "z_goal",          # (T, zgoal_dim)
    "z_phase",         # (T, zphase_dim)
    "a_expert",        # (T, action_dim)
    "contact_label",   # (T,)  optional; missing → all zeros
    "dino_patches",    # (T, 196, 384) — added by precompute_dinov3 (optional at first)
)


@dataclass
class EpisodeRecord:
    wrist_rgb: torch.Tensor
    proprio: torch.Tensor
    a_base: torch.Tensor
    k_idx_norm: torch.Tensor
    z_goal: torch.Tensor
    z_phase: torch.Tensor
    a_expert: torch.Tensor
    contact_label: torch.Tensor
    dino_patches: Optional[torch.Tensor] = None

    def length(self) -> int:
        return int(self.proprio.shape[0])

    def save(self, path: Path) -> None:
        data = {
            "wrist_rgb": self.wrist_rgb,
            "proprio": self.proprio,
            "a_base": self.a_base,
            "k_idx_norm": self.k_idx_norm,
            "z_goal": self.z_goal,
            "z_phase": self.z_phase,
            "a_expert": self.a_expert,
            "contact_label": self.contact_label,
        }
        if self.dino_patches is not None:
            data["dino_patches"] = self.dino_patches
        torch.save(data, path)

    @classmethod
    def load(cls, path: Path) -> "EpisodeRecord":
        d = torch.load(path, map_location="cpu", weights_only=True)
        return cls(
            wrist_rgb=d["wrist_rgb"],
            proprio=d["proprio"],
            a_base=d["a_base"],
            k_idx_norm=d["k_idx_norm"],
            z_goal=d["z_goal"],
            z_phase=d["z_phase"],
            a_expert=d["a_expert"],
            contact_label=d["contact_label"],
            dino_patches=d.get("dino_patches"),
        )


def has_complete_dino_patches(rec: EpisodeRecord) -> bool:
    """Return whether a record already has DINO patches for every timestep."""
    return rec.dino_patches is not None and int(rec.dino_patches.shape[0]) == rec.length()


def _atomic_save_record(rec: EpisodeRecord, path: Path) -> None:
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        rec.save(tmp_path)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


# ────────────────────────────────────────────────────────────────────────
# Offline rollout
# ────────────────────────────────────────────────────────────────────────
def precompute_chunks(
    lerobot_dataset,           # lerobot.datasets.LeRobotDataset
    policy,                    # HFRVLAPolicy with frozen SmolVLA + hooks
    output_dir: Path,
    *,
    n_action_steps: int,
    wrist_image_key: str = "observation.images.wrist",
    proprio_key: str = "observation.state",
    action_key: str = "action",
    contact_key: Optional[str] = None,
    device: str = "cuda",
) -> list[Path]:
    """Roll the frozen SmolVLA over every episode and save records to disk.

    Returns the list of saved file paths.

    NOTE: ``policy`` must be a freshly-constructed :class:`HFRVLAPolicy` (or
    its parent SmolVLAPolicy) with the forward hooks already registered.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    policy = policy.to(device).eval()

    saved: list[Path] = []
    n_episodes = lerobot_dataset.num_episodes \
        if hasattr(lerobot_dataset, "num_episodes") else len(lerobot_dataset.episode_data_index["from"])

    for ep_idx in range(n_episodes):
        ep_from = int(lerobot_dataset.episode_data_index["from"][ep_idx])
        ep_to = int(lerobot_dataset.episode_data_index["to"][ep_idx])

        # Reset policy queues and hook caches per episode.
        policy.reset()

        wrist_buf: list[torch.Tensor] = []
        proprio_buf: list[torch.Tensor] = []
        a_base_buf: list[torch.Tensor] = []
        k_idx_buf: list[float] = []
        zgoal_buf: list[torch.Tensor] = []
        zphase_buf: list[torch.Tensor] = []
        a_expert_buf: list[torch.Tensor] = []
        contact_buf: list[float] = []

        # Per-chunk cached embeddings (re-captured at each chunk boundary).
        cached_zgoal: Optional[torch.Tensor] = None
        cached_zphase: Optional[torch.Tensor] = None
        chunk_consumed = 0

        for t in range(ep_from, ep_to):
            sample = lerobot_dataset[t]
            obs = {k: v.unsqueeze(0).to(device) for k, v in sample.items() if isinstance(v, torch.Tensor)}

            # ── 1. Ensure a fresh chunk is available; capture hook embeddings ──
            need_new_chunk = (len(policy._queues.get("action", [])) == 0)
            if need_new_chunk:
                policy._clear_hook_cache()

            # populate queues + (re)compute chunk if empty (mirrors parent select_action)
            from lerobot.policies.utils import populate_queues
            from lerobot.utils.constants import ACTION
            obs = policy._prepare_batch(obs)
            policy._queues = populate_queues(policy._queues, obs, exclude_keys=[ACTION])
            if len(policy._queues[ACTION]) == 0:
                actions = policy._get_action_chunk(obs)
                policy._queues[ACTION].extend(
                    actions.transpose(0, 1)[: n_action_steps]
                )
                cached_zgoal = policy._zgoal_cache
                cached_zphase = policy._zphase_cache
                chunk_consumed = 0

            a_base = policy._queues[ACTION].popleft()
            chunk_consumed += 1
            k = chunk_consumed - 1
            k_norm = k / max(1, n_action_steps - 1)

            # ── 2. Record per-step tensors ──
            wrist_raw = sample[wrist_image_key].to(device).float()
            if wrist_raw.max() > 1.5:
                wrist_raw = wrist_raw / 255.0
            wrist_norm = normalize_for_dinov3(wrist_raw)

            wrist_buf.append(wrist_norm.squeeze(0).cpu())
            proprio_buf.append(sample[proprio_key].cpu())
            a_base_buf.append(a_base.squeeze(0).cpu())
            k_idx_buf.append(k_norm)
            zgoal_buf.append(cached_zgoal.squeeze(0).cpu() if cached_zgoal is not None else torch.zeros(1))
            zphase_buf.append(cached_zphase.squeeze(0).cpu() if cached_zphase is not None else torch.zeros(1))
            a_expert_buf.append(sample[action_key].cpu())
            contact_buf.append(
                float(sample[contact_key]) if (contact_key and contact_key in sample) else 0.0
            )

        record = EpisodeRecord(
            wrist_rgb=torch.stack(wrist_buf, dim=0),
            proprio=torch.stack(proprio_buf, dim=0),
            a_base=torch.stack(a_base_buf, dim=0),
            k_idx_norm=torch.tensor(k_idx_buf).unsqueeze(-1),
            z_goal=torch.stack(zgoal_buf, dim=0),
            z_phase=torch.stack(zphase_buf, dim=0),
            a_expert=torch.stack(a_expert_buf, dim=0),
            contact_label=torch.tensor(contact_buf),
        )
        path = output_dir / f"episode_{ep_idx:05d}.pt"
        record.save(path)
        saved.append(path)
        print(f"[precompute_chunks] saved {path} (T={record.length()})")
    return saved


def precompute_dinov3(
    chunks_dir: Path,
    dinov3_model_id: str = "facebook/dinov3-vits16-pretrain-lvd1689m",
    local_repo: str | None = None,
    local_weights: str | None = None,
    arch: str = "dinov3_vits16",
    batch_size: int = 32,
    device: str = "cuda",
    drop_wrist_rgb: bool = True,
    skip_existing: bool = True,
) -> None:
    """Add ``dino_patches`` to every saved episode record.

    When ``drop_wrist_rgb=True`` (default) the raw ``wrist_rgb`` tensor is
    removed after patches have been computed — saves ~128 MB / 200-frame
    episode at the cost of needing to re-run DINOv3 if the backbone changes.
    """
    backbone = DINOv3Backbone(
        model_id=dinov3_model_id,
        local_repo=local_repo,
        local_weights=local_weights,
        arch=arch,
        frozen=True,
    ).to(device).eval()
    for path in sorted(chunks_dir.glob("episode_*.pt")):
        rec = EpisodeRecord.load(path)
        T = rec.length()
        if skip_existing and has_complete_dino_patches(rec):
            print(
                f"[precompute_dinov3] {path}: SKIP existing dino_patches "
                f"{tuple(rec.dino_patches.shape)}"
            )
            continue
        if int(rec.wrist_rgb.shape[0]) != T:
            raise ValueError(
                f"{path} needs DINOv3 patches but wrist_rgb has shape "
                f"{tuple(rec.wrist_rgb.shape)}. Re-run rollout for this episode "
                "or restore wrist_rgb before recomputing DINOv3."
            )
        patches_chunks: list[torch.Tensor] = []
        with torch.no_grad():
            for i in range(0, T, batch_size):
                imgs = rec.wrist_rgb[i : i + batch_size].to(device)
                p = backbone(imgs).cpu()
                patches_chunks.append(p)
        rec.dino_patches = torch.cat(patches_chunks, dim=0)
        if drop_wrist_rgb:
            # Replace with an empty placeholder so the record loader is happy.
            rec.wrist_rgb = torch.zeros(0)
        _atomic_save_record(rec, path)
        print(
            f"[precompute_dinov3] {path}: dino_patches {tuple(rec.dino_patches.shape)}"
            f"{'  (wrist_rgb dropped)' if drop_wrist_rgb else ''}"
        )


# ────────────────────────────────────────────────────────────────────────
# Dataset
# ────────────────────────────────────────────────────────────────────────
class HFRVLADataset(Dataset):
    """Sequence-sampling dataset for HFRVLA training.

    Each ``__getitem__`` returns a batch of ``seq_len`` consecutive steps as
    a dict of tensors. If ``dino_patches`` are cached, they are returned;
    otherwise the raw ``wrist_rgb`` is returned and the policy's backbone
    will run online (slower).
    """

    def __init__(
        self,
        chunks_dir: Path | str,
        seq_len: int = 8,
        require_dino_patches: bool = True,
    ) -> None:
        self.chunks_dir = Path(chunks_dir)
        self.seq_len = seq_len
        self.require_dino_patches = require_dino_patches
        self.files = sorted(self.chunks_dir.glob("episode_*.pt"))
        if not self.files:
            raise FileNotFoundError(
                f"No episode_*.pt files in {self.chunks_dir}. Run "
                "precompute_chunks() first."
            )
        # Build a flat index of (file_idx, start_step) windows.
        self._index: list[tuple[int, int]] = []
        for fi, f in enumerate(self.files):
            d = torch.load(f, map_location="cpu", weights_only=True)
            T = int(d["proprio"].shape[0])
            if require_dino_patches and "dino_patches" not in d:
                continue
            for start in range(0, max(1, T - seq_len + 1)):
                self._index.append((fi, start))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        fi, start = self._index[idx]
        rec = EpisodeRecord.load(self.files[fi])
        s, e = start, start + self.seq_len
        batch: dict[str, torch.Tensor] = {
            "proprio": rec.proprio[s:e],
            "a_base": rec.a_base[s:e],
            "k_idx_norm": rec.k_idx_norm[s:e],
            "z_goal": rec.z_goal[s:e],
            "z_phase": rec.z_phase[s:e],
            "a_expert": rec.a_expert[s:e],
            "contact_label": rec.contact_label[s:e],
        }
        if rec.dino_patches is not None:
            batch["dino_patches"] = rec.dino_patches[s:e]
        else:
            batch["wrist_rgb"] = rec.wrist_rgb[s:e]
        return batch
