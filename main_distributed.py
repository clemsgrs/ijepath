# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import logging
import os
import pprint
import sys

import submitit

from src.config_loading import load_training_config
from src.train import main as app_main

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

DEFAULT_CONFIG_PATH = "configs/defaults.yaml"


parser = argparse.ArgumentParser()
parser.add_argument(
    '--folder', type=str,
    help='location to save submitit logs')
parser.add_argument(
    '--batch-launch', action='store_true',
    help='unused legacy flag')
parser.add_argument(
    '--profile-config', type=str, default=None,
    help='sampling/profile config to merge on top of defaults')
parser.add_argument(
    '--run-config', type=str, default=None,
    help='run-specific config to merge on top of defaults')
parser.add_argument(
    '--partition', type=str,
    help='cluster partition to submit jobs on')
parser.add_argument(
    '--nodes', type=int, default=1,
    help='num. nodes to request for job')
parser.add_argument(
    '--tasks-per-node', type=int, default=1,
    help='num. procs to per node')
parser.add_argument(
    '--time', type=int, default=4300,
    help='time in minutes to run job')
parser.add_argument(
    'opts',
    nargs=argparse.REMAINDER,
    default=None,
    help='override config options with dotlist syntax, e.g. data.batch_size_per_gpu=8',
)


class Trainer:

    def __init__(
        self,
        profile_config=None,
        run_config=None,
        opts=None,
        load_model=None,
    ):
        self.profile_config = profile_config
        self.run_config = run_config
        self.opts = list(opts or [])
        self.load_model = load_model

    def __call__(self):
        load_model = self.load_model
        logger.info(
            "called-params default=%s profile=%s run=%s opts=%s",
            DEFAULT_CONFIG_PATH,
            self.profile_config,
            self.run_config,
            self.opts,
        )

        # -- load script params
        params = load_training_config(
            default_config=DEFAULT_CONFIG_PATH,
            profile_config=self.profile_config,
            run_config=self.run_config,
            opts=self.opts,
        )
        logger.info('loaded params...')
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(params)

        resume_preempt = False if load_model is None else load_model
        app_main(args=params, resume_preempt=resume_preempt)

    def checkpoint(self):
        fb_trainer = Trainer(
            self.profile_config,
            self.run_config,
            self.opts,
            True,
        )
        return submitit.helpers.DelayedSubmission(fb_trainer,)


def launch():
    executor = submitit.AutoExecutor(
        folder=os.path.join(args.folder, 'job_%j'),
        slurm_max_num_timeout=20)
    executor.update_parameters(
        slurm_partition=args.partition,
        slurm_mem_per_gpu='55G',
        timeout_min=args.time,
        nodes=args.nodes,
        tasks_per_node=args.tasks_per_node,
        cpus_per_task=10,
        gpus_per_node=args.tasks_per_node)

    if not (args.profile_config and args.run_config):
        raise SystemExit(
            "Provide --profile-config <...> and --run-config <...>. "
            f"Defaults are always loaded from {DEFAULT_CONFIG_PATH}."
        )

    jobs = []
    with executor.batch():
        fb_trainer = Trainer(
            profile_config=args.profile_config,
            run_config=args.run_config,
            opts=args.opts,
        )
        jobs.append(executor.submit(fb_trainer,))

    for job in jobs:
        print(job.job_id)


if __name__ == '__main__':
    args = parser.parse_args()
    launch()
