"""Deprecated compatibility aliases for the Fast Wrist Residual module."""

from __future__ import annotations

from lerobot_policy_hfrvla.fast_wrist_residual import (
    FastWristResidualModule,
    FastWristResidualOutput,
)

A2C2WristCorrectionModule = FastWristResidualModule
A2C2WristOutput = FastWristResidualOutput

__all__ = [
    "A2C2WristCorrectionModule",
    "A2C2WristOutput",
    "FastWristResidualModule",
    "FastWristResidualOutput",
]
