"""
GPU memory management utilities.

Provides monitoring and cleanup functions for CUDA devices.
"""

import logging

import torch

from ..config import DEFAULTS


def monitor_gpu_memory() -> None:
    """Monitor and report GPU memory usage across all available devices."""
    if not DEFAULTS["verbose_gpu_memory"]:
        return
    
    logger = logging.getLogger("gtp")
    
    if torch.cuda.is_available():
        logger.debug("\n=== GPU Memory Status ===")
        for i in range(torch.cuda.device_count()):
            memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
            memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            utilization = (memory_allocated / memory_total) * 100

            logger.debug(f"GPU {i}: {memory_allocated:.2f}GB/{memory_total:.2f}GB ({utilization:.1f}% used)")
        logger.debug("=" * 25)


def clear_gpu_memory() -> None:
    """Clear memory on all available GPUs."""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
