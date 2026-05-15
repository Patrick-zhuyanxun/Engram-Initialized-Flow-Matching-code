"""LeRobot plugin: Hierarchical Fast-Reactive VLA (HFRVLA).

Public exports:
    HFRVLAPolicy   — wraps a frozen SmolVLA with a Fast Reactive Module.
    HFRVLAConfig   — configuration dataclass extending SmolVLAConfig.

Design spec: Hierachical_fast_reactive/paper/notes/implementation_spec.md
"""

from lerobot_policy_hfrvla.configuration_hfrvla import HFRVLAConfig
from lerobot_policy_hfrvla.modeling_hfrvla import HFRVLAPolicy

__all__ = ["HFRVLAConfig", "HFRVLAPolicy"]
__version__ = "0.1.0"
