#!/usr/bin/env python3
"""Run HFRVLA eval diagnostics across checkpoints and fast-module ablations.

The script is intentionally a runner around existing project entrypoints:

* ``scripts/package_hfrvla_checkpoint.py`` packages a train checkpoint.
* ``lerobot-eval`` runs LIBERO evaluation.
* Small config-only policy variants isolate whether failures come from the
  fast residual, velocity safety layer, or the base SmolVLA wrapper.

Useful variants:

* ``trained``: packaged checkpoint as-is.
* ``zero_fast``: exact SmolVLA wrapper path; bypasses fast and safety.
* ``delta_0``: keeps the safety layer active but clamps residual magnitude to 0.
* ``delta_0.02``: allows only a tiny residual.
* ``no_safety``: leaves residual active but removes velocity limiting.
* Combinations use ``+``, e.g. ``delta_0.02+no_safety``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = REPO_ROOT / "checkpoints/hfrvla_run_fastcache_seq4_dtypefix"
DEFAULT_DATASET_ROOT = REPO_ROOT / "checkpoints/HFRVLA_libero_v1_merged_reindexed"
DEFAULT_DINO_REPO = REPO_ROOT / "checkpoints/dinov3_src"
DEFAULT_DINO_WEIGHTS = (
    REPO_ROOT
    / "checkpoints/Dino_weight/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
)
DEFAULT_TMP_ROOT = Path.home() / "tmp/hfrvla"
DEFAULT_SMOLVLA = "HuggingFaceVLA/smolvla_libero"


@dataclass(frozen=True)
class EvalMetrics:
    pc_success: float
    avg_sum_reward: float
    avg_max_reward: float
    n_episodes: int
    n_successes: int


@dataclass(frozen=True)
class CheckpointSpec:
    label: str
    pretrained_model: Path


@dataclass
class EvalRecord:
    checkpoint: str
    variant: str
    policy_path: str
    output_dir: str
    status: str
    pc_success: float | None = None
    avg_sum_reward: float | None = None
    avg_max_reward: float | None = None
    n_episodes: int | None = None
    n_successes: int | None = None
    log_path: str | None = None
    error: str | None = None


def summarize_eval_info(info: dict[str, Any]) -> EvalMetrics:
    overall = info.get("overall", {})
    successes: list[bool] = []
    for task in info.get("per_task", []):
        metrics = task.get("metrics", {})
        successes.extend(bool(v) for v in metrics.get("successes", []))

    n_successes = sum(successes)
    n_episodes = int(overall.get("n_episodes") or len(successes))
    if "pc_success" in overall:
        pc_success = float(overall["pc_success"])
    else:
        pc_success = 100.0 * n_successes / n_episodes if n_episodes else 0.0

    return EvalMetrics(
        pc_success=pc_success,
        avg_sum_reward=float(overall.get("avg_sum_reward", 0.0)),
        avg_max_reward=float(overall.get("avg_max_reward", 0.0)),
        n_episodes=n_episodes,
        n_successes=n_successes,
    )


def materialize_policy_variant(
    source_policy: Path,
    dest_policy: Path,
    config_overrides: dict[str, Any],
    *,
    force: bool = False,
) -> None:
    """Create a lightweight policy directory with modified ``config.json``.

    Non-config files are hardlinked when possible so each ablation does not
    duplicate the large ``model.safetensors`` file. The source policy is never
    mutated.
    """
    source_policy = source_policy.resolve()
    if force and dest_policy.exists():
        shutil.rmtree(dest_policy)
    if dest_policy.exists():
        return

    dest_policy.mkdir(parents=True)
    for item in source_policy.rglob("*"):
        rel = item.relative_to(source_policy)
        dest = dest_policy / rel
        if item.is_dir():
            dest.mkdir(parents=True, exist_ok=True)
            continue
        if rel == Path("config.json"):
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(item, dest)
        except OSError:
            shutil.copy2(item, dest)

    config = json.loads((source_policy / "config.json").read_text())
    config.update(config_overrides)
    (dest_policy / "config.json").write_text(json.dumps(config, indent=2) + "\n")


def parse_task_ids(raw: str) -> list[int] | None:
    text = raw.strip()
    if text.lower() in {"all", "none", ""}:
        return None
    if text.startswith("["):
        values = json.loads(text)
        return [int(v) for v in values]
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def parse_variants(raw: str) -> list[str]:
    variants = [part.strip() for part in raw.split(",") if part.strip()]
    if not variants:
        raise ValueError("--variants must contain at least one variant")
    return variants


def variant_overrides(variant: str) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for token in variant.split("+"):
        if token == "trained":
            continue
        if token == "zero_fast":
            overrides["inference_disable_fast"] = True
            continue
        if token == "no_safety":
            overrides["safety_joint_velocity_limit"] = 1.0e9
            continue
        if token.startswith("delta_"):
            overrides["delta_max"] = float(token.removeprefix("delta_"))
            continue
        if token.startswith("safety_"):
            overrides["safety_joint_velocity_limit"] = float(
                token.removeprefix("safety_")
            )
            continue
        raise ValueError(f"Unknown variant token: {token!r}")
    return overrides


def discover_checkpoints(run_dir: Path, selector: str) -> list[CheckpointSpec]:
    ckpt_root = run_dir / "checkpoints"
    if not ckpt_root.exists():
        raise FileNotFoundError(f"checkpoint root not found: {ckpt_root}")

    numeric_dirs = sorted(p for p in ckpt_root.iterdir() if p.name.isdigit())
    if selector.strip().lower() == "all":
        return [
            CheckpointSpec(label=p.name, pretrained_model=p / "pretrained_model")
            for p in numeric_dirs
        ]

    specs: list[CheckpointSpec] = []
    for raw in selector.split(","):
        token = raw.strip()
        if not token:
            continue
        if token == "last":
            specs.append(
                CheckpointSpec(
                    label="last",
                    pretrained_model=ckpt_root / "last/pretrained_model",
                )
            )
            continue
        step = _parse_step_token(token)
        path = ckpt_root / f"{step:06d}" / "pretrained_model"
        specs.append(CheckpointSpec(label=f"{step:06d}", pretrained_model=path))

    for spec in specs:
        if not (spec.pretrained_model / "model.safetensors").exists():
            raise FileNotFoundError(f"missing checkpoint model: {spec.pretrained_model}")
    return specs


def _parse_step_token(token: str) -> int:
    normalized = token.lower()
    if normalized.endswith("k"):
        return int(float(normalized[:-1]) * 1000)
    return int(normalized)


def safe_name(value: str) -> str:
    return (
        value.replace("/", "_")
        .replace("+", "_")
        .replace(".", "p")
        .replace("[", "")
        .replace("]", "")
        .replace(",", "-")
        .replace(" ", "")
    )


def build_eval_env(tmp_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("HFRVLA_TMP_ROOT", str(tmp_root))
    env.setdefault("HF_DATASETS_CACHE", str(tmp_root / "hf_datasets"))
    env.setdefault("TMPDIR", str(tmp_root / "tmp"))
    env.setdefault("TMP", env["TMPDIR"])
    env.setdefault("TEMP", env["TMPDIR"])
    env.setdefault("NUMBA_CACHE_DIR", str(tmp_root / "numba"))
    env.setdefault("MPLCONFIGDIR", str(tmp_root / "matplotlib"))
    for key in ("HF_DATASETS_CACHE", "TMPDIR", "NUMBA_CACHE_DIR", "MPLCONFIGDIR"):
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    return env


def run_command(
    cmd: list[str],
    *,
    env: dict[str, str],
    log_path: Path,
    dry_run: bool,
) -> None:
    print("+ " + " ".join(shlex.quote(part) for part in cmd))
    if dry_run:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"command failed with exit code {proc.returncode}: {log_path}")


def package_checkpoint(args: argparse.Namespace, spec: CheckpointSpec, env: dict[str, str]) -> Path:
    package_dir = args.package_root / f"{args.run_dir.name}_{spec.label}_packaged"
    if (
        not args.force
        and (package_dir / "config.json").exists()
        and (package_dir / "model.safetensors").exists()
    ):
        print(f"[diagnose] reuse package: {package_dir}")
        return package_dir
    if args.force and package_dir.exists() and not args.dry_run:
        shutil.rmtree(package_dir)

    cmd = [
        str(args.python),
        str(REPO_ROOT / "scripts/package_hfrvla_checkpoint.py"),
        "--fast-ckpt",
        str(spec.pretrained_model),
        "--out-dir",
        str(package_dir),
        "--smolvla-pretrained",
        args.smolvla_pretrained,
        "--dinov3-repo",
        str(args.dinov3_repo),
        "--dinov3-weights",
        str(args.dinov3_weights),
        "--dataset-repo-id",
        args.dataset_repo_id,
        "--dataset-root",
        str(args.dataset_root),
    ]
    run_command(
        cmd,
        env=env,
        log_path=args.summary_dir / "logs" / f"package_{spec.label}.log",
        dry_run=args.dry_run,
    )
    return package_dir


def run_eval(
    args: argparse.Namespace,
    *,
    checkpoint: str,
    variant: str,
    policy_path: Path | str,
    env: dict[str, str],
) -> EvalRecord:
    task_ids = parse_task_ids(args.task_ids)
    task_id_name = "all" if task_ids is None else "-".join(str(v) for v in task_ids)
    output_dir = (
        args.eval_root
        / f"{safe_name(checkpoint)}__{safe_name(variant)}__{args.task}"
        / f"task_ids_{task_id_name}__{args.n_episodes}ep__seed{args.seed}"
    )
    log_path = output_dir / "eval.log"
    info_path = output_dir / "eval_info.json"
    if not args.force and info_path.exists():
        metrics = summarize_eval_info(json.loads(info_path.read_text()))
        return EvalRecord(
            checkpoint=checkpoint,
            variant=variant,
            policy_path=str(policy_path),
            output_dir=str(output_dir),
            status="cached",
            log_path=str(log_path),
            **asdict(metrics),
        )

    cmd = [
        str(args.eval_bin),
        f"--policy.path={policy_path}",
        "--env.type=libero",
        f"--env.task={args.task}",
        f"--eval.n_episodes={args.n_episodes}",
        f"--eval.batch_size={args.batch_size}",
        f"--output_dir={output_dir}",
        f"--seed={args.seed}",
    ]
    if task_ids is not None:
        cmd.append(f"--env.task_ids={json.dumps(task_ids)}")

    try:
        run_command(cmd, env=env, log_path=log_path, dry_run=args.dry_run)
        if args.dry_run:
            return EvalRecord(
                checkpoint=checkpoint,
                variant=variant,
                policy_path=str(policy_path),
                output_dir=str(output_dir),
                status="dry-run",
                log_path=str(log_path),
            )
        metrics = summarize_eval_info(json.loads(info_path.read_text()))
        return EvalRecord(
            checkpoint=checkpoint,
            variant=variant,
            policy_path=str(policy_path),
            output_dir=str(output_dir),
            status="ok",
            log_path=str(log_path),
            **asdict(metrics),
        )
    except Exception as exc:
        if not args.keep_going:
            raise
        return EvalRecord(
            checkpoint=checkpoint,
            variant=variant,
            policy_path=str(policy_path),
            output_dir=str(output_dir),
            status="failed",
            log_path=str(log_path),
            error=str(exc),
        )


def write_summary(summary_dir: Path, records: list[EvalRecord]) -> None:
    summary_dir.mkdir(parents=True, exist_ok=True)
    rows = [asdict(record) for record in records]
    (summary_dir / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    if not rows:
        return
    with (summary_dir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    p.add_argument("--steps", default="last", help="Comma list, e.g. 5k,10k,last; or all.")
    p.add_argument(
        "--variants",
        default="trained,zero_fast,delta_0",
        help="Comma list: trained, zero_fast, delta_0, delta_0.02, no_safety, ...",
    )
    p.add_argument("--include-baseline", action="store_true")
    p.add_argument("--smolvla-pretrained", default=DEFAULT_SMOLVLA)
    p.add_argument("--dataset-repo-id", default="HFRVLA_libero_v1")
    p.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    p.add_argument("--dinov3-repo", type=Path, default=DEFAULT_DINO_REPO)
    p.add_argument("--dinov3-weights", type=Path, default=DEFAULT_DINO_WEIGHTS)
    p.add_argument("--task", default="libero_spatial")
    p.add_argument("--task-ids", default="[0]", help="JSON list, comma list, or all.")
    p.add_argument("--n-episodes", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--summary-dir",
        type=Path,
        default=REPO_ROOT / "outputs/hfrvla_eval_diagnostics",
    )
    p.add_argument(
        "--package-root",
        type=Path,
        default=REPO_ROOT / "outputs/hfrvla_eval_diagnostics/packages",
    )
    p.add_argument(
        "--eval-root",
        type=Path,
        default=REPO_ROOT / "outputs/hfrvla_eval_diagnostics/evals",
    )
    p.add_argument(
        "--python",
        type=Path,
        default=Path.home() / "Robotic_infra/lerobot/.venv/bin/python",
    )
    p.add_argument(
        "--eval-bin",
        type=Path,
        default=Path.home() / "Robotic_infra/lerobot/.venv/bin/lerobot-eval",
    )
    p.add_argument("--tmp-root", type=Path, default=DEFAULT_TMP_ROOT)
    p.add_argument("--force", action="store_true", help="Rebuild packages and rerun evals.")
    p.add_argument("--keep-going", action="store_true", help="Record failures and continue.")
    p.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    env = build_eval_env(args.tmp_root)
    args.summary_dir.mkdir(parents=True, exist_ok=True)
    args.package_root.mkdir(parents=True, exist_ok=True)
    args.eval_root.mkdir(parents=True, exist_ok=True)

    records: list[EvalRecord] = []
    if args.include_baseline:
        records.append(
            run_eval(
                args,
                checkpoint="baseline",
                variant="smolvla",
                policy_path=args.smolvla_pretrained,
                env=env,
            )
        )
        write_summary(args.summary_dir, records)

    variants = parse_variants(args.variants)
    for spec in discover_checkpoints(args.run_dir, args.steps):
        package_dir = package_checkpoint(args, spec, env)
        for variant in variants:
            overrides = variant_overrides(variant)
            policy_path: Path | str = package_dir
            if overrides:
                variant_dir = (
                    args.package_root
                    / f"{package_dir.name}__variant_{safe_name(variant)}"
                )
                if not args.dry_run:
                    materialize_policy_variant(
                        package_dir,
                        variant_dir,
                        overrides,
                        force=args.force,
                    )
                policy_path = variant_dir
            records.append(
                run_eval(
                    args,
                    checkpoint=spec.label,
                    variant=variant,
                    policy_path=policy_path,
                    env=env,
                )
            )
            write_summary(args.summary_dir, records)

    print(f"[diagnose] wrote {args.summary_dir / 'summary.json'}")
    print(f"[diagnose] wrote {args.summary_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
