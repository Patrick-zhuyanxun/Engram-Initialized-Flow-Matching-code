from __future__ import annotations

from typing import Any

import torch

from lerobot.policies.xvla.processor_xvla import (
    XVLAAddDomainIdProcessorStep,
    XVLAImageNetNormalizeProcessorStep,
    XVLAImageToFloatProcessorStep,
)
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
    TokenizerProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.utils.constants import (
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

from .configuration_eifm import EIFMConfig
from .libero_action_processors import LiberoActionToEE6DProcessorStep


def make_eifm_pre_post_processors(
    config: EIFMConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Build EI-FM policy processors.

    Unlike the old implementation, this module does not re-register XVLA steps.
    It reuses the upstream XVLA processor steps and only injects the EI-FM-specific
    7D LIBERO -> 20D ee6d action conversion before normalization.
    """

    features = {**config.input_features, **config.output_features}
    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        TokenizerProcessorStep(
            tokenizer_name=config.tokenizer_name,
            max_length=config.tokenizer_max_length,
            padding=config.pad_language_to,
            padding_side=config.tokenizer_padding_side,
        ),
        XVLAImageToFloatProcessorStep(),
        XVLAImageNetNormalizeProcessorStep(),
        # LIBERO uses domain_id=3 in the XVLA checkpoint/action encoder.
        XVLAAddDomainIdProcessorStep(domain_id=3),
        LiberoActionToEE6DProcessorStep(),
        DeviceProcessorStep(device=config.device),
        NormalizerProcessorStep(
            features=features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
    ]
    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )


# Backward-compatible alias in case local code imported the wrong factory name.
make_xvla_pre_post_processors = make_eifm_pre_post_processors
