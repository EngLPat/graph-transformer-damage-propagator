"""
Optimizer and scheduler construction.
"""

import math
import os
from typing import Tuple, Optional

import torch.optim as optim


def build_optimizer(args, model) -> Tuple[Optional[object], optim.Optimizer]:
    """
    Build optimizer with uniform learning rate for all parameters.
    
    Args:
        args: Training configuration with lr, weight_decay, opt, opt_scheduler, etc.
        model: PyTorch model
    
    Returns:
        Tuple of (scheduler, optimizer).  scheduler may be a
        ``ReduceLROnPlateau`` (which is **not** an ``_LRScheduler`` subclass)
        so callers must check the type before calling ``step()``.
    """
    weight_decay = getattr(args, "weight_decay", 1e-6)
    lr = getattr(args, "lr", getattr(args, "learning_rate", 0.001))
    
    # Only print from rank 0 to avoid duplicate messages in DDP
    rank = int(os.environ.get("RANK", 0))
    if rank == 0:
        print(f"✓ Building optimizer with UNIFORM learning rate: {lr:.6f}")
        print(f"  Weight decay: {weight_decay:.2e}")
    
    opt_name = getattr(args, "opt", getattr(args, "optimizer", "adam"))
    
    if opt_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.95, weight_decay=weight_decay)
    elif opt_name == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == "adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")
    
    scheduler_name = getattr(args, "opt_scheduler", getattr(args, "scheduler", "step"))
    
    if scheduler_name == "none":
        return None, optimizer
    elif scheduler_name == "step":
        step_size = getattr(args, "opt_decay_step", getattr(args, "scheduler_step", 50))
        gamma = getattr(args, "opt_decay_rate", getattr(args, "scheduler_gamma", 0.85))
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_name == "cos":
        t_max = getattr(args, "opt_restart", getattr(args, "scheduler_restart", 0))
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
    elif scheduler_name == "cosine":
        epochs = getattr(args, "epochs", 500)
        
        def lr_lambda(current_step: int) -> float:
            warmup_steps = 20
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, epochs - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif scheduler_name == "plateau":
        patience = getattr(args, "plateau_patience", 20)
        factor = getattr(args, "plateau_factor", 0.5)
        min_lr = getattr(args, "min_lr", 1e-6)
        if rank == 0:
            print(f"  ReduceLROnPlateau: patience={patience}, factor={factor}, min_lr={min_lr:.1e}")
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=patience, factor=factor, min_lr=min_lr
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")
    
    return scheduler, optimizer
