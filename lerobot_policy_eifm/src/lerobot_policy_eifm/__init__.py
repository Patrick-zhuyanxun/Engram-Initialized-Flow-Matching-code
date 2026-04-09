"""EI-FM: Engram-Initialized Flow Matching policy plugin for LeRobot."""

from .configuration_eifm import EIFMConfig
from .modeling_eifm import EIFMPolicy

# Import libero_action_processors to trigger @ProcessorStepRegistry.register
# for 'libero_action_to_ee6d'.  The xvla_* steps are already registered by the
# built-in lerobot xvla policy package, so processor_eifm.py is intentionally
# NOT imported here to avoid duplicate-registration errors.
from . import libero_action_processors as _libero_action_processors  # noqa: F401

__all__ = [
    "EIFMConfig",
    "EIFMPolicy",
]
