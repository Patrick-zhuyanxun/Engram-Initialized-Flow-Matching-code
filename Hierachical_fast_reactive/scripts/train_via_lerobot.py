#!/usr/bin/env python
"""Wrapper that injects HFRVLA curriculum into lerobot-train.

Lerobot-train's main loop calls update_policy(...) once per optimizer step.
This wrapper monkey-patches that function so it:

1. Unwraps the policy if accelerator/DDP wrapped it.
2. Invokes policy.set_training_step(step) for curriculum control.
3. Drops optimizer LR once when stage 2 starts.
4. Delegates to the original update_policy.
"""

from __future__ import annotations

import lerobot.scripts.lerobot_train as lt

_STEP = {"value": 0}
_ORIGINAL_UPDATE_POLICY = lt.update_policy


def _unwrap(policy, accelerator=None):
    """Strip accelerator wrapping when possible."""
    if accelerator is not None and hasattr(accelerator, "unwrap_model"):
        return accelerator.unwrap_model(policy, keep_fp32_wrapper=True)
    return getattr(policy, "module", policy)


def _patched_update_policy(train_tracker, policy, batch, optimizer, *args, **kwargs):
    step = _STEP["value"]
    accelerator = kwargs.get("accelerator")
    if accelerator is None and len(args) >= 2:
        accelerator = args[1]

    raw = _unwrap(policy, accelerator)
    if hasattr(raw, "set_training_step"):
        raw.set_training_step(step)
        consume_signal = getattr(raw, "consume_refine_lr_signal", None)
        if consume_signal is not None and consume_signal():
            for group in optimizer.param_groups:
                group["lr"] = group["lr"] * 0.1
            print(f"[curriculum] step {step}: entered Stage 2 — LR × 0.1", flush=True)

    result = _ORIGINAL_UPDATE_POLICY(train_tracker, policy, batch, optimizer, *args, **kwargs)
    _STEP["value"] += 1
    return result


def main() -> None:
    lt.update_policy = _patched_update_policy
    lt.main()


if __name__ == "__main__":
    main()
