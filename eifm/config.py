"""Paths and hyperparameters for EI-FM engram construction."""

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────
VLA_ROOT = Path(__file__).parent.parent
LIBERO_DATA_DIR = VLA_ROOT / "LIBERO" / "libero" / "datasets"
LEROBOT_DIR = VLA_ROOT / "lerobot"
XVLA_POLICY_DIR = LEROBOT_DIR / "src" / "lerobot" / "policies" / "xvla"
OUTPUT_DIR = VLA_ROOT / "eifm"

# ── Engram Construction ─────────────────────────────────────
LIBERO_SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_90"]
LIBERO_DOMAIN_ID = 3  # From X-VLA docs: libero = domain 3
ENGRAM_KEY_MODE = "multi"  # "single" (first verb) or "multi" (all verbs joined)

# ── X-VLA Architecture Constants ────────────────────────────
HIDDEN_SIZE = 1024       # Transformer hidden dimension
DIM_ACTION = 20          # ee6d padded action dim
DIM_PROPRIO = 20         # dim_propio in SoftPromptedTransformer (not max_state_dim=32)
DIM_TIME = 32            # timestep embedding dim
NUM_DOMAINS = 30         # number of soft prompt domains
# action_encoder input = DIM_ACTION + DIM_TIME + DIM_PROPRIO = 20+32+20 = 72
ACTION_ENCODER_INPUT = DIM_ACTION + DIM_TIME + DIM_PROPRIO

# ── LIBERO Raw Action ───────────────────────────────────────
LIBERO_RAW_ACTION_DIM = 7  # xyz(3) + euler(3) + gripper(1)
