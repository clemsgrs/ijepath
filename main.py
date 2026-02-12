# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse

import multiprocessing as mp

import os
import re
import socket
import subprocess
import sys
import time
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


def _parse_master_port(value: str | int, *, source: str) -> int:
    try:
        port = int(str(value).strip())
    except Exception as exc:
        raise ValueError(f"{source} must be an integer, got {value!r}") from exc
    if port < 1 or port > 65535:
        raise ValueError(f"{source} must be in [1, 65535], got {port}")
    return int(port)


def _has_opt(opts: list[str] | None, prefix: str) -> bool:
    return any(str(x).startswith(prefix) for x in list(opts or []))


def _replace_opt(opts: list[str] | None, prefix: str, value: str) -> list[str]:
    out = [str(x) for x in list(opts or []) if not str(x).startswith(prefix)]
    out.append(value)
    return out


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


def _resolve_reserved_tuning_device_token(
    *,
    profile_config: str,
    run_config: str,
    opts: list[str] | None,
    visible_devices: list[str],
) -> str | None:
    params = load_training_config(
        default_config=DEFAULT_CONFIG_PATH,
        profile_config=profile_config,
        run_config=run_config,
        opts=opts,
    )
    tuning_cfg = dict(params.get("tuning", {}) or {})
    if not bool(tuning_cfg.get("enable", False)):
        return None

    execution_cfg = dict(tuning_cfg.get("execution", {}) or {})
    device = str(execution_cfg.get("device", "auto")).strip().lower()
    if device == "auto":
        if len(visible_devices) < 2:
            raise SystemExit(
                "tuning.execution.device=auto requires at least 2 visible GPUs "
                "(one reserved for tuning)"
            )
        token = str(visible_devices[-1])
    else:
        match = re.fullmatch(r"cuda:(\d+)", device)
        if match is None:
            raise SystemExit("tuning.execution.device must be 'auto' or cuda:<id> for dedicated GPU tuning")
        token = match.group(1)
    if token not in visible_devices:
        raise SystemExit(
            "Configured tuning.execution.device is not visible in CUDA_VISIBLE_DEVICES: "
            f"requested={token} visible={visible_devices}"
        )
    return token


def _resolve_master_port(world_size: int) -> int | None:
    if int(world_size) <= 1:
        return None
    env_port = os.environ.get("MASTER_PORT", "").strip()
    if env_port:
        return _parse_master_port(env_port, source="MASTER_PORT")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _resolve_master_addr(world_size: int) -> str | None:
    if int(world_size) <= 1:
        return None
    return "127.0.0.1"


def _join_process(process: mp.Process, timeout: float | None = None) -> None:
    if timeout is None:
        process.join()
        return
    try:
        process.join(timeout=timeout)
    except TypeError:
        process.join()


def _is_process_running(process: mp.Process) -> bool:
    exitcode = getattr(process, "exitcode", None)
    if exitcode is not None:
        return False
    is_alive = getattr(process, "is_alive", None)
    if callable(is_alive):
        try:
            return bool(is_alive())
        except Exception:
            return True
    return True


def _cleanup_remaining_workers(
    workers: list[tuple[int, str, mp.Process]],
    *,
    join_timeout_s: float = 3.0,
) -> tuple[int, int]:
    terminated = 0
    killed = 0

    for _rank, _device, process in workers:
        if not _is_process_running(process):
            continue
        terminate = getattr(process, "terminate", None)
        if callable(terminate):
            try:
                terminate()
                terminated += 1
            except Exception:
                pass

    deadline = time.time() + float(join_timeout_s)
    for _rank, _device, process in workers:
        if not _is_process_running(process):
            continue
        timeout = max(0.0, deadline - time.time())
        try:
            _join_process(process, timeout=timeout)
        except Exception:
            pass

    for _rank, _device, process in workers:
        if not _is_process_running(process):
            continue
        kill = getattr(process, "kill", None)
        if callable(kill):
            try:
                kill()
                killed += 1
            except Exception:
                pass
        try:
            _join_process(process, timeout=0.2)
        except Exception:
            pass

    return terminated, killed


def process_main(
    rank,
    profile_config,
    run_config,
    opts,
    world_size,
    visible_devices,
    master_addr,
    master_port,
    tuning_device_token=None,
):
    device_token = visible_devices[rank]
    rank_opts = list(opts or [])
    if device_token != "cpu":
        if rank == 0 and tuning_device_token is not None:
            if str(tuning_device_token) == str(device_token):
                raise SystemExit("Dedicated tuning GPU conflicts with rank-0 training GPU assignment")
            os.environ['CUDA_VISIBLE_DEVICES'] = f"{device_token},{tuning_device_token}"
            rank_opts = _replace_opt(rank_opts, "tuning.execution.device=", "tuning.execution.device=cuda:1")
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = device_token
    else:
        os.environ.pop('CUDA_VISIBLE_DEVICES', None)
    if master_addr:
        os.environ["MASTER_ADDR"] = str(master_addr)
    if master_port is not None:
        os.environ["MASTER_PORT"] = str(master_port)

    import logging

    logger = setup_logging(level=logging.INFO if rank == 0 else logging.ERROR)

    try:
        logger.info(
            f"called-params default={DEFAULT_CONFIG_PATH} "
            f"profile={profile_config} run={run_config} opts={rank_opts}"
        )

        # -- load script params
        params = load_training_config(
            default_config=DEFAULT_CONFIG_PATH,
            profile_config=profile_config,
            run_config=run_config,
            opts=rank_opts,
        )
        logger.info(
            f"loaded layered config (default={DEFAULT_CONFIG_PATH} "
            f"profile={profile_config} run={run_config})"
        )

        world_size, rank = init_distributed(
            rank_and_world_size=(rank, world_size),
            master_addr=master_addr,
            port=master_port,
        )
        logger.info(
            f"Distributed context initialized: rank={rank} "
            f"world_size={world_size} device={device_token} "
            f"master_addr={master_addr} master_port={master_port}"
        )
        app_main(args=params, distributed_state=(world_size, rank))
    except BaseException:
        logger.exception(
            "Worker rank=%s device=%s crashed with unhandled exception (profile=%s run=%s opts=%s)",
            rank,
            device_token,
            profile_config,
            run_config,
            rank_opts,
        )
        raise


def launch_worker_processes(
    *,
    profile_config: str,
    run_config: str,
    opts: list[str] | None,
    visible_devices: list[str],
    tuning_device_token: str | None = None,
) -> None:
    world_size = len(visible_devices)
    master_addr = _resolve_master_addr(world_size)
    master_port = _resolve_master_port(world_size)
    workers: list[tuple[int, str, mp.Process]] = []
    try:
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
                    master_addr,
                    master_port,
                    tuning_device_token,
                ),
            )
            process.start()
            workers.append((rank, visible_devices[rank], process))

        failed_workers: list[tuple[int, str, int | None]] = []
        pending = list(workers)
        while pending:
            completed: list[tuple[int, str, mp.Process]] = []
            for rank, device, process in pending:
                _join_process(process, timeout=0.2)
                if process.exitcode is None:
                    continue
                completed.append((rank, device, process))
                if process.exitcode != 0:
                    failed_workers.append((rank, device, process.exitcode))

            if completed:
                pending = [w for w in pending if w not in completed]

            if failed_workers:
                terminated, killed = _cleanup_remaining_workers(pending)
                failure_summary = ", ".join(
                    f"(rank={rank}, device={device}, exitcode={exitcode})"
                    for rank, device, exitcode in failed_workers
                )
                raise SystemExit(
                    "Worker process failure(s): "
                    f"{failure_summary}. cleanup(terminated={terminated}, killed={killed})"
                )

            if pending and not completed:
                time.sleep(0.05)
    except KeyboardInterrupt as exc:
        terminated, killed = _cleanup_remaining_workers(workers)
        raise SystemExit(
            "Interrupted while waiting for worker processes. "
            f"cleanup(terminated={terminated}, killed={killed})"
        ) from exc
    except Exception:
        _cleanup_remaining_workers(workers)
        raise


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
    reserved_tuning_device = _resolve_reserved_tuning_device_token(
        profile_config=args.profile_config,
        run_config=args.run_config,
        opts=effective_opts,
        visible_devices=visible_devices,
    )
    training_visible_devices = list(visible_devices)
    if reserved_tuning_device is not None:
        training_visible_devices = [d for d in visible_devices if str(d) != str(reserved_tuning_device)]
        if not training_visible_devices:
            raise SystemExit("No training GPUs remain after reserving dedicated tuning GPU")

    mp.set_start_method('spawn')

    launch_worker_processes(
        profile_config=args.profile_config,
        run_config=args.run_config,
        opts=effective_opts,
        visible_devices=training_visible_devices,
        tuning_device_token=reserved_tuning_device,
    )
