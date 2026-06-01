import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/train_smolvla_libero_baseline.sh"


def _base_env(tmp_path: Path, out_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "LERO_DIR": str(tmp_path),
            "LERO_TRAIN": "/bin/true",
            "OUT_DIR": str(out_dir),
            "DATASET_ROOT": str(tmp_path / "dataset"),
            "STEPS": "1",
            "BATCH_SIZE": "1",
            "NUM_WORKERS": "0",
            "EVAL_FREQ": "0",
            "SAVE_FREQ": "1",
            "WANDB_ENABLE": "false",
        }
    )
    return env


def test_wrapper_does_not_precreate_lerobot_output_dir(tmp_path):
    out_dir = tmp_path / "smolvla_libero_slow"

    subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=REPO_ROOT,
        env=_base_env(tmp_path, out_dir),
        check=True,
        capture_output=True,
        text=True,
    )

    assert not out_dir.exists()


def test_wrapper_fails_fast_when_output_dir_already_exists(tmp_path):
    out_dir = tmp_path / "smolvla_libero_slow"
    out_dir.mkdir()

    result = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=REPO_ROOT,
        env=_base_env(tmp_path, out_dir),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "output directory already exists" in result.stderr
