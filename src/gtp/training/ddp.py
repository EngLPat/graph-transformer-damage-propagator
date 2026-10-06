"""
Distributed Data Parallel (DDP) utilities.

Handles setup and cleanup of PyTorch DDP for multi-GPU training.
"""

import os
from datetime import timedelta

import torch
import torch.distributed as dist

from ..config import DEFAULTS


def setup_ddp() -> tuple:
    """
    Initialize PyTorch DistributedDataParallel environment.
    
    Returns:
        Tuple of (local_rank, rank, world_size)
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank = 0
        world_size = torch.cuda.device_count()
        local_rank = 0

    timeout_minutes = DEFAULTS["ddp_timeout_minutes"]
    timeout_timedelta = timedelta(minutes=timeout_minutes)

    # NCCL backend on GPUs; GLOO on a CPU-only machine, for testing the DDP
    # code path
    cuda_ok = torch.cuda.is_available()
    backend = "nccl" if cuda_ok else "gloo"

    dist.init_process_group(
        backend=backend,
        init_method="env://",
        world_size=world_size,
        rank=rank,
        timeout=timeout_timedelta
    )

    if cuda_ok:
        torch.cuda.set_device(local_rank)

    if rank == 0:
        if cuda_ok:
            print(f"DDP initialized: {world_size} GPUs, NCCL backend, "
                  f"{timeout_minutes}min timeout")
        else:
            print("=" * 72)
            print(f"WARNING: DDP initialized on CPU (GLOO, {world_size} ranks).")
            print("         This path is intended for testing the DDP pipeline;")
            print("         CPU training is much slower than GPU training. On a GPU")
            print("         cluster this indicates a problem with the GPU allocation.")
            print("=" * 72)

    return local_rank, rank, world_size


def cleanup_ddp() -> None:
    """Clean up the PyTorch DDP process group."""
    if dist.is_initialized():
        dist.destroy_process_group()
