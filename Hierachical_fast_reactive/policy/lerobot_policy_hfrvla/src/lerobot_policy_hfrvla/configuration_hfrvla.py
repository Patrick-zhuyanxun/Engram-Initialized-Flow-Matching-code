"""HFRVLA configuration — extends SmolVLAConfig with Fast Reactive Module knobs.

See implementation_spec.md §3 for the design contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE


@PreTrainedConfig.register_subclass("hfrvla")
@dataclass
class HFRVLAConfig(SmolVLAConfig):
    push_to_hub: bool = False

    # ── Slow VLA control ──
    freeze_smolvla: bool = True

    # ── DINOv3 backbone ──
    # We support two loading paths:
    #   1) HuggingFace gated repo via AutoModel — set `dinov3_model_id`.
    #   2) Local clone of facebookresearch/dinov3 + a .pth weight file
    #      (recommended; bypasses the gated repo) — set `dinov3_local_repo`
    #      AND `dinov3_local_weights`.
    # If `dinov3_local_repo` is set, it takes precedence over `dinov3_model_id`.
    dinov3_model_id: str = "facebook/dinov3-vits16-pretrain-lvd1689m"
    dinov3_local_repo: str | None = "checkpoints/dinov3_src"
    dinov3_local_weights: str | None = (
        "checkpoints/Dino_weight/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
    )
    dinov3_arch: str = "dinov3_vits16"      # torch.hub entry-point name
    dinov3_image_size: int = 224            # → 14×14 = 196 patches at patch=16
    dinov3_feature_dim: int = 384            # ViT-S/16 hidden
    dinov3_num_patches: int = 196            # (224 / 16) ** 2
    dinov3_frozen: bool = True

    # ── Cross-attn pool (Strategy S-B) ──
    pool_query_dim: int = 256
    pool_n_heads: int = 4

    # ── GRU ──
    gru_hidden: int = 256
    gru_layers: int = 1

    # ── Heads ──
    head_hidden: int = 64
    delta_max: float = 0.2                 # per-DoF clamp on normalized residual
    contact_head_enabled: bool = True       # training-only; dropped at inference

    # ── Loss weights ──
    loss_lambda_gate: float = 1.0
    loss_lambda_contact: float = 0.1
    # Conservative deployment-aligned objective. Training should supervise the
    # residual that inference can actually execute: ``gate * clip(delta_a)``.
    loss_delta_target_clip: bool = True
    # Stage A v2 (debate 20260521, recalibrated 2026-05-22): margin is the
    # absolute reduction in per-frame squared L2 error. On LIBERO the empirical
    # ``err_before`` median is ~2.4 with p10≈0.5; the v1 default of 0.05 was
    # ~50x too small and BCE labels were ~100% positive → gate_prior stayed at
    # 0.9. 0.5 corresponds to roughly half the typical clipped-residual
    # improvement at delta_max=0.2.
    gate_improvement_margin: float = 0.5
    loss_lambda_final: float = 1.0
    loss_lambda_preserve: float = 0.5
    # Stage A (debate 20260521): real rate term, no longer a token regularizer.
    loss_lambda_gate_prior: float = 0.10
    # Stage A v2: explicit zero-target penalty on preserve-class states.
    # ``err_before < err_preserve_thresh`` defines the mask. On LIBERO the
    # empirical p10 of err_before is ~0.5 (v1 default 0.01 caught 0% of
    # frames, so L_preserve_zero never fired). 0.5 catches roughly the bottom
    # 10% where the base policy is already very close to the expert.
    loss_lambda_preserve_zero: float = 1.0
    err_preserve_thresh: float = 0.5

    # ── Curriculum (sprint-2 three-stage) ──
    # Stage 0 (Warmup): only L_delta, gate + contact heads frozen.
    # Stage 1 (Joint):  deployment-aligned delta/final/gate losses + contact.
    # Stage 2 (Refine): same losses; LR / 10 (handled by set_training_step).
    curriculum_warmup_steps: int = 1000
    curriculum_joint_steps: int = 49000
    curriculum_refine_steps: int = 10000

    # ── Sequence windowing for GRU ──
    seq_len: int = 8

    # ── Offline fast-only training ──
    # The recorded HFRVLA dataset already stores SmolVLA and DINOv3 features.
    # During training we can therefore skip constructing the frozen slow VLA
    # and DINO backbone, reducing memory pressure before the first step.
    offline_training_mode: bool = False
    offline_zgoal_dim: int = 960
    offline_zphase_dim: int = 480

    # ── Hook target layer names (resolved via named_modules lookup) ──
    # Format: dotted path inside `self.model` (i.e. inside VLAFlowMatching).
    # Default targets the LAST layer of:
    #   - VLM text encoder  → semantic / task feature  (z_goal)
    #   - LM expert         → action-expert hidden     (z_phase)
    # The implementation walks named_modules() and matches a suffix.
    hook_zgoal_layer: str = "vlm_with_expert.vlm.model.text_model.layers"  # uses [-1]
    hook_zphase_layer: str = "vlm_with_expert.lm_expert.layers"             # uses [-1]

    # Projection dims from raw hook tensors → unified 256-d.
    # Hidden sizes are read at runtime from the captured tensors.
    zgoal_proj_dim: int = 256
    zphase_proj_dim: int = 256

    # ── Inference residual safety ──
    # Limit only the fast residual. Do not velocity-clamp the full SmolVLA
    # action chunk against the previous step; LIBERO actions are normalized
    # chunk outputs, and clamping the base action breaks the frozen planner.
    safety_joint_velocity_limit: float = 2.0
    control_dt: float = 0.1                     # 10 Hz (LIBERO HuggingFaceVLA fps)

    # ── Inference debug switches ──
    # When True, select_action short-circuits to ``a_base`` (the popped SmolVLA
    # action) and skips the fast module entirely. Use for I/O alignment tests
    # against the SmolVLA baseline before training the fast module.
    inference_disable_fast: bool = False

    # ── Misc ──
    name: str = "hfrvla"

    # ────────────────────────────────────────────────────────────────────
    def validate_features(self) -> None:
        super().validate_features()
        if self.input_features:
            if self.offline_training_mode:
                self.input_features = {
                    key: feature
                    for key, feature in self.input_features.items()
                    if key == OBS_STATE
                }
            else:
                self.input_features = {
                    key: feature
                    for key, feature in self.input_features.items()
                    if key == OBS_STATE or key.startswith(f"{OBS_IMAGES}.")
                }

    # ────────────────────────────────────────────────────────────────────
    @classmethod
    def from_smolvla(cls, repo_id_or_path: str, **overrides) -> "HFRVLAConfig":
        """Build an HFRVLAConfig by reading a SmolVLA config.json directly.

        Notes:
            We bypass ``SmolVLAConfig.from_pretrained`` because the draccus
            parser fails on the ``type`` field stored in newer checkpoints.
            Instead we load the JSON, filter to known dataclass fields, and
            construct the dataclass manually. The ``input_features`` and
            ``output_features`` dicts (PolicyFeature values) are reconstructed.
        """
        import json
        from pathlib import Path
        from huggingface_hub import hf_hub_download
        from lerobot.configs.types import FeatureType, PolicyFeature

        local = Path(repo_id_or_path)
        if local.is_dir():
            cfg_path = local / "config.json"
        else:
            cfg_path = Path(hf_hub_download(repo_id_or_path, filename="config.json"))
        with open(cfg_path) as f:
            raw = json.load(f)
        raw.pop("type", None)

        # Re-hydrate PolicyFeature dicts.
        def _hydrate_features(d):
            if not d:
                return {}
            out = {}
            for k, v in d.items():
                if isinstance(v, PolicyFeature):
                    out[k] = v
                else:
                    out[k] = PolicyFeature(
                        type=FeatureType(v["type"]) if "type" in v else FeatureType.STATE,
                        shape=tuple(v["shape"]),
                    )
            return out
        raw["input_features"] = _hydrate_features(raw.get("input_features"))
        raw["output_features"] = _hydrate_features(raw.get("output_features"))

        # Drop unknown fields that don't match our dataclass.
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            for k in unknown:
                raw.pop(k, None)
        raw.update(overrides)
        return cls(**raw)

    def curriculum_total_steps(self) -> int:
        return (
            self.curriculum_warmup_steps
            + self.curriculum_joint_steps
            + self.curriculum_refine_steps
        )

    @property
    def observation_delta_indices(self) -> list[int]:
        """Indices passed to lerobot's delta_timestamps mechanism.

        Returns ``[-(seq_len-1), ..., -1, 0]`` so all observation.* keys are
        windowed to ``seq_len`` consecutive steps ending at the current frame.
        """
        return list(range(-(self.seq_len - 1), 1))

    @property
    def action_delta_indices(self) -> list[int]:
        """Window expert actions to the same temporal context as HFRVLA extras."""
        return self.observation_delta_indices
