# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os

import torch
import torch.distributed as dist

from logging import getLogger

logger = getLogger()


def _parse_master_port(port: str | int) -> int:
    try:
        parsed = int(str(port).strip())
    except Exception as exc:
        raise ValueError(f"MASTER_PORT must be an integer, got {port!r}") from exc
    if parsed < 1 or parsed > 65535:
        raise ValueError(f"MASTER_PORT must be in [1, 65535], got {parsed}")
    return int(parsed)


def _resolve_rank_and_world_size(rank_and_world_size: tuple[int | None, int | None]) -> tuple[int, int]:
    rank, world_size = rank_and_world_size
    if rank is None or world_size is None:
        slurm_ntasks = os.environ.get("SLURM_NTASKS")
        slurm_procid = os.environ.get("SLURM_PROCID")
        if slurm_ntasks is None or slurm_procid is None:
            logger.info('SLURM vars not set (distributed training not available)')
            return 0, 1
        try:
            world_size = int(slurm_ntasks)
            rank = int(slurm_procid)
        except ValueError as exc:
            raise ValueError(
                f"Invalid SLURM distributed values: SLURM_NTASKS={slurm_ntasks!r} "
                f"SLURM_PROCID={slurm_procid!r}"
            ) from exc
        host = os.environ.get("HOSTNAME", "").strip()
        if host:
            os.environ["MASTER_ADDR"] = host

    rank = int(rank)
    world_size = int(world_size)
    if world_size <= 0:
        raise ValueError(f"world_size must be > 0, got {world_size}")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank must satisfy 0 <= rank < world_size, got rank={rank}, world_size={world_size}")
    return rank, world_size


def init_distributed(
    port=None,
    rank_and_world_size=(None, None),
    master_addr=None,
    fail_if_multi_init_fails=True,
):

    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()

    rank, world_size = _resolve_rank_and_world_size(rank_and_world_size=rank_and_world_size)
    if world_size <= 1:
        return 1, 0

    backend = 'nccl'
    if backend == 'nccl' and not torch.cuda.is_available():
        raise RuntimeError(
            "Distributed launch requested with world_size > 1, but CUDA is unavailable for NCCL backend."
        )

    if master_addr is None:
        master_addr = os.environ.get("MASTER_ADDR", "").strip()
    else:
        master_addr = str(master_addr).strip()
    if not master_addr:
        raise ValueError("MASTER_ADDR is required for multi-process distributed launch (world_size > 1).")

    if port is None:
        env_port = os.environ.get("MASTER_PORT", "").strip()
        if not env_port:
            raise ValueError("MASTER_PORT is required for multi-process distributed launch (world_size > 1).")
        resolved_port = _parse_master_port(env_port)
    else:
        resolved_port = _parse_master_port(port)

    os.environ['MASTER_ADDR'] = str(master_addr)
    os.environ['MASTER_PORT'] = str(resolved_port)
    try:
        torch.distributed.init_process_group(
            backend=backend,
            world_size=world_size,
            rank=rank)
    except Exception as exc:
        error_msg = (
            "Failed to initialize distributed process group: "
            f"rank={rank} world_size={world_size} backend={backend} "
            f"master_addr={master_addr} master_port={resolved_port} error={exc}"
        )
        if fail_if_multi_init_fails:
            raise RuntimeError(error_msg) from exc
        logger.warning("%s. Falling back to single-process because fail_if_multi_init_fails=False.", error_msg)
        return 1, 0

    return dist.get_world_size(), dist.get_rank()


class AllGather(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        if (
            dist.is_available()
            and dist.is_initialized()
            and (dist.get_world_size() > 1)
        ):
            x = x.contiguous()
            outputs = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
            dist.all_gather(outputs, x)
            return torch.cat(outputs, 0)
        return x

    @staticmethod
    def backward(ctx, grads):
        if (
            dist.is_available()
            and dist.is_initialized()
            and (dist.get_world_size() > 1)
        ):
            s = (grads.shape[0] // dist.get_world_size()) * dist.get_rank()
            e = (grads.shape[0] // dist.get_world_size()) * (dist.get_rank() + 1)
            grads = grads.contiguous()
            dist.all_reduce(grads)
            return grads[s:e]
        return grads


class AllReduceSum(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        if (
            dist.is_available()
            and dist.is_initialized()
            and (dist.get_world_size() > 1)
        ):
            x = x.contiguous()
            dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grads):
        return grads


class AllReduce(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        if (
            dist.is_available()
            and dist.is_initialized()
            and (dist.get_world_size() > 1)
        ):
            x = x.contiguous() / dist.get_world_size()
            dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grads):
        return grads
