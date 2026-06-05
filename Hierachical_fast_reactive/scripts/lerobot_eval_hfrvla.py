#!/home/hucenrotia/Robotic_infra/lerobot/.venv/bin/python
"""LeRobot eval entry point with configurable rendered-video count.

The upstream ``lerobot-eval`` currently hard-codes ``max_episodes_rendered=10``.
For large LIBERO sweeps this means every 10-episode task writes videos for all
episodes. This wrapper keeps the same LeRobot eval pipeline but lets HFRVLA
sweeps disable video rendering through ``HFRVLA_EVAL_MAX_VIDEOS``.
"""

import json
import logging
import os
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from pprint import pformat

import torch
from termcolor import colored

from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import close_envs
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.scripts.lerobot_eval import eval_policy_all
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import init_logging


def _max_episodes_rendered() -> int:
    raw_value = os.environ.get("HFRVLA_EVAL_MAX_VIDEOS", "0")
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError("HFRVLA_EVAL_MAX_VIDEOS must be an integer") from exc
    if value < 0:
        raise ValueError("HFRVLA_EVAL_MAX_VIDEOS must be >= 0")
    return value


@parser.wrap()
def eval_main(cfg: EvalPipelineConfig) -> None:
    logging.info(pformat(asdict(cfg)))

    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {output_dir}")

    max_videos = _max_episodes_rendered()
    videos_dir = output_dir / "videos" if max_videos > 0 else None
    logging.info("HFRVLA eval max rendered videos per task: %s", max_videos)

    logging.info("Making environment.")
    envs = make_env(
        cfg.env,
        n_envs=cfg.eval.batch_size,
        use_async_envs=cfg.eval.use_async_envs,
        trust_remote_code=cfg.trust_remote_code,
    )

    try:
        logging.info("Making policy.")
        policy = make_policy(
            cfg=cfg.policy,
            env_cfg=cfg.env,
            rename_map=cfg.rename_map,
        )
        policy.eval()

        preprocessor_overrides = {
            "device_processor": {"device": str(policy.config.device)},
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        }
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=cfg.policy,
            pretrained_path=cfg.policy.pretrained_path,
            preprocessor_overrides=preprocessor_overrides,
        )
        env_preprocessor, env_postprocessor = make_env_pre_post_processors(
            env_cfg=cfg.env,
            policy_cfg=cfg.policy,
        )

        autocast_context = torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext()
        with torch.no_grad(), autocast_context:
            info = eval_policy_all(
                envs=envs,
                policy=policy,
                env_preprocessor=env_preprocessor,
                env_postprocessor=env_postprocessor,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                n_episodes=cfg.eval.n_episodes,
                max_episodes_rendered=max_videos,
                videos_dir=videos_dir,
                start_seed=cfg.seed,
                max_parallel_tasks=cfg.env.max_parallel_tasks,
            )
            print("Overall Aggregated Metrics:")
            print(info["overall"])
            for task_group, task_group_info in info.items():
                print(f"\nAggregated Metrics for {task_group}:")
                print(task_group_info)

        with (output_dir / "eval_info.json").open("w") as f:
            json.dump(info, f, indent=2)
    finally:
        close_envs(envs)

    logging.info("End of eval")


def main() -> None:
    init_logging()
    register_third_party_plugins()
    eval_main()


if __name__ == "__main__":
    main()
