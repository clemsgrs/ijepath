# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse

import multiprocessing as mp

import pprint
import os

from src.utils.distributed import init_distributed
from src.config_loading import load_training_config
from src.train import main as app_main

DEFAULT_CONFIG_PATH = "configs/defaults.yaml"

parser = argparse.ArgumentParser()
parser.add_argument(
    '--profile-config', type=str, default=None,
    help='sampling/profile config to merge on top of defaults')
parser.add_argument(
    '--run-config', type=str, default=None,
    help='run-specific config to merge on top of defaults')
parser.add_argument(
    'opts',
    nargs=argparse.REMAINDER,
    default=None,
    help='override config options with dotlist syntax, e.g. data.batch_size_per_gpu=8',
)


def _discover_visible_devices():
    import torch

    if not torch.cuda.is_available():
        return ["cpu"]

    env = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if env:
        visible = [d.strip() for d in env.split(",") if d.strip()]
        if visible:
            return visible

    return [str(i) for i in range(torch.cuda.device_count())]


def process_main(rank, profile_config, run_config, opts, world_size, visible_devices):
    device_token = visible_devices[rank]
    if device_token != "cpu":
        os.environ['CUDA_VISIBLE_DEVICES'] = device_token
    else:
        os.environ.pop('CUDA_VISIBLE_DEVICES', None)

    import logging
    logging.basicConfig()
    logger = logging.getLogger()
    if rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    logger.info(
        "called-params default=%s profile=%s run=%s opts=%s",
        DEFAULT_CONFIG_PATH,
        profile_config,
        run_config,
        opts,
    )

    # -- load script params
    params = load_training_config(
        default_config=DEFAULT_CONFIG_PATH,
        profile_config=profile_config,
        run_config=run_config,
        opts=opts,
    )
    logger.info('loaded params...')
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(params)

    world_size, rank = init_distributed(rank_and_world_size=(rank, world_size))
    logger.info(f'Running... (rank: {rank}/{world_size})')
    app_main(args=params)


if __name__ == '__main__':
    args = parser.parse_args()

    if not (args.profile_config and args.run_config):
        raise SystemExit(
            "Provide --profile-config <...> and --run-config <...>. "
            f"Defaults are always loaded from {DEFAULT_CONFIG_PATH}."
        )

    visible_devices = _discover_visible_devices()
    num_gpus = len(visible_devices)
    mp.set_start_method('spawn')

    for rank in range(num_gpus):
        mp.Process(
            target=process_main,
            args=(
                rank,
                args.profile_config,
                args.run_config,
                args.opts,
                num_gpus,
                visible_devices,
            )
        ).start()
