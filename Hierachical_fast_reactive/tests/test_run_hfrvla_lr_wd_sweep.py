from __future__ import annotations

import os
import subprocess
from pathlib import Path


def test_lr_wd_sweep_dry_run_emits_complete_grid() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["DRY_RUN"] = "true"

    result = subprocess.run(
        ["bash", "scripts/run_hfrvla_lr_wd_sweep.sh"],
        cwd=repo_root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    command_lines = [
        line for line in result.stdout.splitlines() if line.startswith("+ RUN_NAME=")
    ]
    assert len(command_lines) == 20
    assert "RUN_NAME=hfrvla_lr1e_5_wd0_b512_50k" in command_lines[0]
    assert "LR=1e-5" in command_lines[0]
    assert "WEIGHT_DECAY=0" in command_lines[0]
    assert "WANDB_ENABLE=true" in command_lines[0]
    assert "SAVE_FREQ=25000" in command_lines[0]
    assert "SCHEDULER_DECAY_LR=1e-5" in command_lines[0]
    assert any(
        "RUN_NAME=hfrvla_lr5e_4_wd3e_4_b512_50k" in line
        and "LR=5e-4" in line
        and "WEIGHT_DECAY=3e-4" in line
        and "SCHEDULER_DECAY_LR=5e-4" in line
        for line in command_lines
    )


def test_lr_wd_sweep_dry_run_accepts_parallel_jobs_limit() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["DRY_RUN"] = "true"
    env["PARALLEL_JOBS"] = "2"
    env["MAX_RUNS"] = "2"

    result = subprocess.run(
        ["bash", "scripts/run_hfrvla_lr_wd_sweep.sh"],
        cwd=repo_root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    command_lines = [
        line for line in result.stdout.splitlines() if line.startswith("+ RUN_NAME=")
    ]
    assert len(command_lines) == 2
    assert "parallel_jobs=2" in result.stdout
