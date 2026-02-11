# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse

import multiprocessing as mp

import os
import subprocess
import sys
from pathlib import Path

from ijepath.utils.distributed import init_distributed
from ijepath.config_loading import load_training_config
from ijepath.utils.log_utils import setup_logging
from ijepath.train import main as app_main

DEFAULT_CONFIG_PATH = str(Path(__file__).resolve().parent / "configs" / "defaults.yaml")

parser = argparse.ArgumentParser()
parser.add_argument(
    '--manifest-csv', type=str, default=None,
    help='build slide metadata + anchor catalog from this manifest before training.')
parser.add_argument(
    '--profile-config', type=str, default=None,
    help='sampling/profile config to merge on top of defaults')
parser.add_argument(
    '--run-config', type=str, default=None,
    help='run-specific config to merge on top of defaults')
parser.add_argument(
    '--output-folder', type=str, default=None,
    help='Root folder for generated artifacts (indexes/) and training outputs.')
parser.add_argument(
    'opts',
    nargs=argparse.REMAINDER,
    default=None,
    help='override config options with dotlist syntax, e.g. data.batch_size_per_gpu=8',
)


def _has_opt(opts: list[str] | None, prefix: str) -> bool:
    return any(str(x).startswith(prefix) for x in list(opts or []))


def _run_checked(cmd: list[str]) -> None:
    completed = subprocess.run(
        cmd,
        cwd=Path(__file__).resolve().parent,
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise SystemExit(
            "Pipeline stage failed:\n"
            f"cmd={' '.join(cmd)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )


def orchestrate_pipeline_artifacts(
    *,
    profile_config: str,
    manifest_csv: str,
    output_folder: str,
) -> dict[str, str]:
    repo_root = Path(__file__).resolve().parent
    output_root = Path(output_folder).resolve()
    indexes_dir = output_root / "indexes"
    indexes_dir.mkdir(parents=True, exist_ok=True)

    slide_metadata_parquet = indexes_dir / "slide_metadata.parquet"
    slide_report_csv = indexes_dir / "slide_metadata_build_report.csv"
    anchor_catalog_manifest = indexes_dir / "anchor_catalog_manifest.json"

    python_exe = sys.executable
    _run_checked(
        [
            python_exe,
            str(repo_root / "scripts" / "build_slide_metadata_index_from_manifest.py"),
            "--manifest",
            str(Path(manifest_csv).resolve()),
            "--output",
            str(slide_metadata_parquet),
            "--report",
            str(slide_report_csv),
        ]
    )
    _run_checked(
        [
            python_exe,
            str(repo_root / "scripts" / "build_valid_context_anchor_catalog.py"),
            "--slide-index",
            str(slide_metadata_parquet),
            "--profile",
            str(Path(profile_config).resolve()),
            "--output",
            str(anchor_catalog_manifest),
        ]
    )

    return {
        "slide_manifest_csv": str(Path(manifest_csv).resolve()),
        "slide_metadata_parquet": str(slide_metadata_parquet),
        "anchor_catalog_manifest": str(anchor_catalog_manifest),
        "training_output_folder": str(output_root),
    }


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

    logger = setup_logging(level=logging.INFO if rank == 0 else logging.ERROR)

    logger.info(
        f"called-params default={DEFAULT_CONFIG_PATH} "
        f"profile={profile_config} run={run_config} opts={opts}"
    )

    # -- load script params
    params = load_training_config(
        default_config=DEFAULT_CONFIG_PATH,
        profile_config=profile_config,
        run_config=run_config,
        opts=opts,
    )
    logger.info(
        f"loaded layered config (default={DEFAULT_CONFIG_PATH} "
        f"profile={profile_config} run={run_config})"
    )

    world_size, rank = init_distributed(rank_and_world_size=(rank, world_size))
    logger.info(
        f"Distributed context initialized: rank={rank} "
        f"world_size={world_size} device={device_token}"
    )
    app_main(args=params, distributed_state=(world_size, rank))


def launch_worker_processes(
    *,
    profile_config: str,
    run_config: str,
    opts: list[str] | None,
    visible_devices: list[str],
) -> None:
    world_size = len(visible_devices)
    workers: list[tuple[int, str, mp.Process]] = []

    for rank in range(world_size):
        process = mp.Process(
            target=process_main,
            args=(
                rank,
                profile_config,
                run_config,
                opts,
                world_size,
                visible_devices,
            ),
        )
        process.start()
        workers.append((rank, visible_devices[rank], process))

    failed_workers: list[tuple[int, str, int | None]] = []
    for rank, device, process in workers:
        process.join()
        if process.exitcode != 0:
            failed_workers.append((rank, device, process.exitcode))

    if failed_workers:
        failure_summary = ", ".join(
            f"(rank={rank}, device={device}, exitcode={exitcode})"
            for rank, device, exitcode in failed_workers
        )
        raise SystemExit(f"Worker process failure(s): {failure_summary}")


if __name__ == '__main__':
    args = parser.parse_args()

    if not (args.profile_config and args.run_config):
        raise SystemExit(
            "Provide --profile-config <...> and --run-config <...>. "
            f"Defaults are always loaded from {DEFAULT_CONFIG_PATH}."
        )

    effective_opts = list(args.opts or [])
    if args.manifest_csv is not None:
        if args.output_folder is None:
            raise SystemExit("When using --manifest-csv, also provide --output-folder.")
        artifacts = orchestrate_pipeline_artifacts(
            profile_config=args.profile_config,
            manifest_csv=args.manifest_csv,
            output_folder=args.output_folder,
        )
        effective_opts.extend(
            [
                f"data.slide_manifest_csv={artifacts['slide_manifest_csv']}",
                f"data.slide_metadata_parquet={artifacts['slide_metadata_parquet']}",
                f"data.anchor_catalog_manifest={artifacts['anchor_catalog_manifest']}",
            ]
        )
        if not _has_opt(effective_opts, "logging.folder="):
            effective_opts.append(f"logging.folder={artifacts['training_output_folder']}")

    visible_devices = _discover_visible_devices()
    mp.set_start_method('spawn')

    launch_worker_processes(
        profile_config=args.profile_config,
        run_config=args.run_config,
        opts=effective_opts,
        visible_devices=visible_devices,
    )
