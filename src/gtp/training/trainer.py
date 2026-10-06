"""
Trainer class for GTP.

Unified training interface supporting both single-GPU and DDP multi-GPU training.
"""

import copy
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader
from tqdm import trange

from ..config import DEFAULTS
from ..models import FCLGA_GraphTransformer
from ..utils import monitor_gpu_memory
from .optimizer import build_optimizer


class Trainer:
    """
    Unified trainer for FCLGA_GraphTransformer.
    
    Supports both single-GPU and DDP multi-GPU training with identical API.
    
    Args:
        config: Configuration object
        device: Device to train on
        local_rank: Local GPU rank (for DDP)
        world_size: Number of GPUs (for DDP)
    """

    def __init__(
        self,
        config,
        device: str = "cuda:0",
        local_rank: int = 0,
        world_size: int = 1,
    ):
        self.config = config
        self.device = device
        self.local_rank = local_rank
        self.world_size = world_size
        self.use_ddp = world_size > 1

    def train(
        self,
        train_dataset: List,
        val_dataset: List,
        stats_list: List[torch.Tensor],
        checkpoint_dir: Optional[Path] = None,
        postprocess_dir: Optional[Path] = None,
        resume_from: Optional[str] = None,
    ) -> Tuple[List[float], List[float], List[float], nn.Module, Optional[dict]]:
        """
        Train the model.

        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            stats_list: Normalization statistics
            checkpoint_dir: Directory to save checkpoints
            postprocess_dir: Directory to save plots
            resume_from: Optional path to a checkpoint to resume from (single-GPU
                only). Accepts either a full checkpoint saved as
                last_checkpoint.pt (model, optimizer, scheduler, epoch) or a
                bare state_dict (weights only; optimizer and scheduler start
                fresh).

        Returns:
            Tuple of (val_losses, train_losses, velo_val_losses, best_model, loss_components_history)
        """
        if self.use_ddp:
            return self._train_ddp(
                train_dataset, val_dataset, stats_list,
                checkpoint_dir, postprocess_dir,
                resume_from=resume_from,
            )
        else:
            return self._train_single(
                train_dataset, val_dataset, stats_list,
                checkpoint_dir, postprocess_dir,
                resume_from=resume_from,
            )

    def _train_single(
        self,
        train_dataset: List,
        val_dataset: List,
        stats_list: List[torch.Tensor],
        checkpoint_dir: Optional[Path] = None,
        postprocess_dir: Optional[Path] = None,
        resume_from: Optional[str] = None,
    ) -> Tuple[List[float], List[float], List[float], nn.Module]:
        """Single-GPU training loop."""
        config = self.config
        device = self.device
        
        df = pd.DataFrame(columns=["epoch", "train_loss", "val_loss", "velo_val_loss"])
        
        # Model name for checkpoints
        model_name = (
            f"model_nl{config.num_layers}_bs{config.batch_size}_"
            f"hd{config.hidden_dim}_ep{config.epochs}_wd{config.weight_decay}_"
            f"lr{config.learning_rate}_shuff_True"
        )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
        
        # Unpack normalization stats
        [mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge,
         mean_vec_y_progressive, std_vec_y_progressive,
         mean_vec_y_catastrophic, std_vec_y_catastrophic] = stats_list
        
        mean_vec_x = mean_vec_x.to(device)
        std_vec_x = std_vec_x.to(device)
        mean_vec_edge = mean_vec_edge.to(device)
        std_vec_edge = std_vec_edge.to(device)
        mean_vec_y_progressive = mean_vec_y_progressive.to(device)
        std_vec_y_progressive = std_vec_y_progressive.to(device)
        mean_vec_y_catastrophic = mean_vec_y_catastrophic.to(device)
        std_vec_y_catastrophic = std_vec_y_catastrophic.to(device)
        
        # Create model
        num_node_features = train_dataset[0].x.shape[1]
        num_edge_features = train_dataset[0].edge_attr.shape[1]
        
        model = FCLGA_GraphTransformer(
            num_node_features, num_edge_features,
            config.hidden_dim, 1, config
        )
        model = model.to(device)
                
        scheduler, opt = build_optimizer(config, model)

        monitor_gpu_memory()

        # Training state
        losses = []
        val_losses = []
        velo_val_losses = []
        loss_components_history = {'rate': [], 'dmg_rate': [], 'total': []}
        best_val_loss = float("inf")
        best_model = None
        patience = config.patience
        patience_counter = 0
        start_epoch = 0

        # ── Optional resume from checkpoint (single-GPU only) ────────────
        if resume_from is not None:
            ckpt = torch.load(resume_from, map_location=device, weights_only=False)
            is_rich = isinstance(ckpt, dict) and 'model_state_dict' in ckpt
            if is_rich:
                model.load_state_dict(ckpt['model_state_dict'])
                if 'optimizer_state_dict' in ckpt:
                    opt.load_state_dict(ckpt['optimizer_state_dict'])
                if scheduler is not None and ckpt.get('scheduler_state_dict') is not None:
                    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
                start_epoch = int(ckpt.get('epoch', -1)) + 1
                if 'best_val_loss' in ckpt:
                    best_val_loss = float(ckpt['best_val_loss'])
                best_model = copy.deepcopy(model)
                print(f"  [resume] Rich resume: weights+optimizer+scheduler loaded. "
                      f"start_epoch={start_epoch}, best_val_loss={best_val_loss:.5f}")
            else:
                model.load_state_dict(ckpt)
                best_model = copy.deepcopy(model)
                print(f"  [resume] Weights-only resume from legacy checkpoint: "
                      f"{resume_from}. Optimizer and scheduler start fresh; "
                      f"start_epoch=0.")
            if start_epoch >= config.epochs:
                raise ValueError(
                    f"[resume] start_epoch={start_epoch} >= epochs={config.epochs}. "
                    f"Raise --epochs to continue training."
                )
        
        # Wall-clock training cap (seconds). Set via GTP_MAX_TRAIN_SECONDS env.
        # If elapsed exceeds this at the end of an epoch, break cleanly so the
        # CSV / best-model save paths still run. 0 / unset = disabled.
        _max_train_seconds = float(os.environ.get("GTP_MAX_TRAIN_SECONDS", 0))
        _train_start = time.time()

        for epoch in trange(start_epoch, config.epochs,
                            initial=start_epoch, total=config.epochs,
                            desc="Training", unit="Epochs"):
            model.train()
            total_loss = 0
            num_loops = 0
            epoch_components = {'rate': 0.0, 'dmg_rate': 0.0}

            for batch_idx, batch in enumerate(train_loader):
                batch = batch.to(device)

                # Set accumulated damage from current node features
                batch.accumulated_damage = batch.x[:, -1].clone()

                opt.zero_grad()

                pred_rate, shock_prob = model(
                    batch, mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge
                )

                loss, components = model.loss(
                    (pred_rate, shock_prob), batch,
                    mean_vec_y_progressive, std_vec_y_progressive,
                    mean_vec_y_catastrophic, std_vec_y_catastrophic,
                    epoch,
                    return_components=True
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=DEFAULTS["gradient_clip_max_norm"])
                opt.step()

                total_loss += loss.item()
                for key in epoch_components:
                    epoch_components[key] += components.get(key, 0.0)
                num_loops += 1
            
            if num_loops % DEFAULTS["gpu_cache_clear_freq"] == 0:
                torch.cuda.empty_cache()
            
            train_loss = total_loss / num_loops
            losses.append(train_loss)
            
            # Store epoch-averaged loss components
            for key in loss_components_history:
                if key == 'total':
                    loss_components_history[key].append(train_loss)
                else:
                    loss_components_history[key].append(epoch_components[key] / num_loops)
            
            # Validation
            model.eval()

            val_total_loss = 0
            val_num_loops = 0
            velo_val_rmse_total = 0
            
            with torch.no_grad():
                for val_batch in val_loader:
                    val_batch = val_batch.to(device)
                    val_batch.accumulated_damage = val_batch.x[:, -1].clone()
                    
                    val_pred_rate, val_shock_prob = model(
                        val_batch, mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge
                    )
                    
                    val_loss_single = model.loss(
                        (val_pred_rate, val_shock_prob), val_batch,
                        mean_vec_y_progressive, std_vec_y_progressive,
                        mean_vec_y_catastrophic, std_vec_y_catastrophic,
                        epoch  # use real epoch, not 0
                    )
                    
                    val_total_loss += val_loss_single.item()

                    # Validation RMSE of the rate, excluding shock nodes (rate > 100)
                    with torch.no_grad():
                        _y = val_batch.y.squeeze()
                        _shock_thresh = 100.0
                        _steady = _y <= _shock_thresh
                        if _steady.any():
                            _pred_s = val_pred_rate.squeeze()[_steady]
                            _y_s = _y[_steady]
                            error = (_pred_s - _y_s) ** 2
                            velo_val_rmse_total += torch.sqrt(error.mean()).item()
                        else:
                            velo_val_rmse_total += 0.0
                    val_num_loops += 1

            val_loss = val_total_loss / val_num_loops
            val_losses.append(val_loss)
            velo_val_losses.append(velo_val_rmse_total / val_num_loops)
            
            # Step scheduler AFTER val_loss is known
            if scheduler is not None:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
            
            # Checkpoint selection on the lowest validation loss, with early stopping
            _sel = float(val_loss)
            if _sel < best_val_loss:
                best_val_loss = _sel
                best_model = copy.deepcopy(model)
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

            avg_components = {k: epoch_components[k] / num_loops for k in epoch_components}
            comp_str = " | ".join(f"{k}={v:.4f}" for k, v in avg_components.items() if v > 0)
            print(f"Epoch {epoch:3d}: Train={train_loss:.5f} Val={val_loss:.5f} | {comp_str}")
            
            # Logging
            new_row = pd.DataFrame({
                "epoch": [epoch],
                "train_loss": [float(train_loss)],
                "val_loss": [float(val_loss)],
                "velo_val_loss": [float(velo_val_losses[-1])]
            })
            df = pd.concat([df, new_row], ignore_index=True)
            
            # Checkpointing
            if epoch % DEFAULTS["checkpoint_save_freq"] == 0 and checkpoint_dir:
                checkpoint_path = Path(checkpoint_dir) / f"{model_name}.pt"
                torch.save(best_model.state_dict(), checkpoint_path)

                # Full checkpoint for resuming training
                last_path = Path(checkpoint_dir) / "last_checkpoint.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                    'best_val_loss': best_val_loss,
                }, last_path)
                # GTP_STAMPED_CKPTS=1 also keeps an epoch-stamped copy
                if os.environ.get("GTP_STAMPED_CKPTS") == "1":
                    import shutil
                    shutil.copyfile(last_path,
                                    Path(checkpoint_dir) / f"epoch_{epoch:04d}.pt")
            
            if epoch % DEFAULTS["memory_monitor_freq"] == 0:
                monitor_gpu_memory()

            # Wall-clock cap: break cleanly so the final CSV / checkpoint saves
            # still run. Triggered only if GTP_MAX_TRAIN_SECONDS > 0.
            if _max_train_seconds > 0:
                _elapsed = time.time() - _train_start
                if _elapsed >= _max_train_seconds:
                    print(f"\n[wall-clock] Elapsed {_elapsed:.0f}s >= "
                          f"{_max_train_seconds:.0f}s (GTP_MAX_TRAIN_SECONDS); "
                          f"stopping at epoch {epoch}.")
                    break

        # Save the loss history; after a resume, keep the earlier epochs of the
        # existing CSV and append the rows of this session
        if checkpoint_dir:
            csv_path = Path(checkpoint_dir) / f"{model_name}.csv"
            if resume_from is not None and start_epoch > 0 and csv_path.exists():
                try:
                    prior = pd.read_csv(csv_path)
                    prior = prior[prior["epoch"] < start_epoch]
                    df = pd.concat([prior, df], ignore_index=True)
                except Exception as _e:
                    print(f"  [resume] CSV merge skipped ({_e}); writing this session only.")
            df.to_csv(csv_path, index=False)

        return val_losses, losses, velo_val_losses, best_model, loss_components_history

    def _train_ddp(
        self,
        train_dataset: List,
        val_dataset: List,
        stats_list: List[torch.Tensor],
        checkpoint_dir: Optional[Path] = None,
        postprocess_dir: Optional[Path] = None,
        resume_from: Optional[str] = None,
    ) -> Tuple[List[float], List[float], List[float], nn.Module]:
        """DDP multi-GPU training loop."""
        config = self.config
        device = self.device
        local_rank = self.local_rank
        world_size = self.world_size
        
        df = pd.DataFrame(columns=["epoch", "train_loss", "val_loss", "velo_val_loss"])
        
        model_name = f"model_nl{config.num_layers}_bs{config.batch_size}_hd{config.hidden_dim}_DDP{world_size}GPUs"
        
        # DDP samplers
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=local_rank, shuffle=False)
        
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, sampler=train_sampler, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, sampler=val_sampler, pin_memory=True)
        
        # Unpack stats
        [mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge,
         mean_vec_y_progressive, std_vec_y_progressive,
         mean_vec_y_catastrophic, std_vec_y_catastrophic] = stats_list
        
        mean_vec_x = mean_vec_x.to(device)
        std_vec_x = std_vec_x.to(device)
        mean_vec_edge = mean_vec_edge.to(device)
        std_vec_edge = std_vec_edge.to(device)
        mean_vec_y_progressive = mean_vec_y_progressive.to(device)
        std_vec_y_progressive = std_vec_y_progressive.to(device)
        mean_vec_y_catastrophic = mean_vec_y_catastrophic.to(device)
        std_vec_y_catastrophic = std_vec_y_catastrophic.to(device)
        
        # Create model with DDP
        num_node_features = train_dataset[0].x.shape[1]
        num_edge_features = train_dataset[0].edge_attr.shape[1]
        
        model = FCLGA_GraphTransformer(
            num_node_features, num_edge_features,
            config.hidden_dim, 1, config
        )
        model = model.to(device)

        # ── Resume: load the model weights on every rank before the DDP wrap;
        # optimizer, scheduler and epoch are restored once the optimizer exists
        _resume_ckpt = None
        if resume_from is not None:
            _resume_ckpt = torch.load(resume_from, map_location=device, weights_only=False)
            _is_rich = isinstance(_resume_ckpt, dict) and 'model_state_dict' in _resume_ckpt
            if _is_rich:
                model.load_state_dict(_resume_ckpt['model_state_dict'])
            else:
                model.load_state_dict(_resume_ckpt)
            if local_rank == 0:
                print(f"  [DDP resume] Model weights loaded on all ranks from {resume_from}")

        # CUDA → device_ids=[local_rank]; CPU smoke-test path → no device_ids.
        # find_unused_parameters=True: the two-stage gate is data-dependent
        # (damage path only fires when FI > threshold), so dmg-side params
        # may not receive grad on every iteration.
        if torch.cuda.is_available():
            model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                        find_unused_parameters=True)
        else:
            model = DDP(model, find_unused_parameters=True)

        scheduler, opt = build_optimizer(config, model.module)

        # ── Resume: restore optimizer / scheduler / epoch / best_val ──────
        start_epoch = 0
        if _resume_ckpt is not None and isinstance(_resume_ckpt, dict) and 'model_state_dict' in _resume_ckpt:
            if 'optimizer_state_dict' in _resume_ckpt:
                opt.load_state_dict(_resume_ckpt['optimizer_state_dict'])
            if scheduler is not None and _resume_ckpt.get('scheduler_state_dict') is not None:
                scheduler.load_state_dict(_resume_ckpt['scheduler_state_dict'])
            start_epoch = int(_resume_ckpt.get('epoch', -1)) + 1
            _resumed_best = float(_resume_ckpt.get('best_val_loss', float('inf')))
            if local_rank == 0:
                print(f"  [DDP resume] Rich resume: optimizer+scheduler loaded. "
                      f"start_epoch={start_epoch}, best_val_loss={_resumed_best:.5f}")
        else:
            _resumed_best = float('inf')

        # Training state
        losses = []
        val_losses = []
        velo_val_losses = []
        loss_components_history = {'rate': [], 'dmg_rate': [], 'total': []}
        best_val_loss = _resumed_best
        best_model = None
        if _resume_ckpt is not None:
            best_model = copy.deepcopy(model.module)
        patience_counter = 0
        patience = config.patience
        accumulation_steps = DEFAULTS["gradient_accumulation_steps"]
        
        if local_rank == 0:
            monitor_gpu_memory()
            epoch_progress = trange(start_epoch, config.epochs,
                                    initial=start_epoch, total=config.epochs,
                                    desc="Training", unit="epoch")
        else:
            epoch_progress = range(start_epoch, config.epochs)

        # Wall-clock training cap (seconds). Set via GTP_MAX_TRAIN_SECONDS env.
        # Rank 0 measures elapsed time and broadcasts the stop decision via
        # the existing should_stop tensor. 0 / unset = disabled.
        _max_train_seconds = float(os.environ.get("GTP_MAX_TRAIN_SECONDS", 0))
        _train_start = time.time()

        for epoch in epoch_progress:
            train_sampler.set_epoch(epoch)
            model.train()
            total_loss = 0
            num_loops = 0
            epoch_components = {'rate': 0.0, 'dmg_rate': 0.0}

            opt.zero_grad()

            for batch_idx, batch in enumerate(train_loader):
                batch = batch.to(device, non_blocking=True)
                
                # Set accumulated damage from current node features
                batch.accumulated_damage = batch.x[:, -1].clone()
                
                pred_rate, shock_prob = model(
                    batch, mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge
                )
                
                loss, components = model.module.loss(
                    (pred_rate, shock_prob), batch,
                    mean_vec_y_progressive, std_vec_y_progressive,
                    mean_vec_y_catastrophic, std_vec_y_catastrophic,
                    epoch,
                    return_components=True
                )

                # DDP + grad accumulation: skip gradient sync for all
                # intermediate micro-batches, sync only on the step boundary.
                # Without no_sync(), every backward() triggers DDP reduction,
                # marking shared parameters (e.g. two-stage dmg_output_scale)
                # ready multiple times per optimizer step.
                is_step_boundary = (
                    (batch_idx + 1) % accumulation_steps == 0
                    or (batch_idx + 1) == len(train_loader)
                )
                if is_step_boundary:
                    loss.backward()
                else:
                    with model.no_sync():
                        loss.backward()
                total_loss += loss.item()
                for key in epoch_components:
                    epoch_components[key] += components.get(key, 0.0)
                num_loops += 1

                if is_step_boundary:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=DEFAULTS["gradient_clip_max_norm"])
                    opt.step()
                    opt.zero_grad()
                
                if num_loops % DEFAULTS["gpu_cache_clear_freq"] == 0:
                    torch.cuda.empty_cache()
            
            train_loss = total_loss / max(num_loops, 1)
            
            # All-reduce across GPUs
            train_loss_tensor = torch.tensor(train_loss, device=device)
            dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
            train_loss = train_loss_tensor.item() / world_size
            losses.append(train_loss)
            
            # All-reduce loss components
            for key in epoch_components:
                comp_tensor = torch.tensor(epoch_components[key] / max(num_loops, 1), device=device)
                dist.all_reduce(comp_tensor, op=dist.ReduceOp.SUM)
                loss_components_history[key].append(comp_tensor.item() / world_size)
            loss_components_history['total'].append(train_loss)
            
            # Validation
            model.eval()

            val_total_loss = 0
            val_num_loops = 0
            velo_val_rmse_total = 0
            
            with torch.no_grad():
                for val_batch in val_loader:
                    val_batch = val_batch.to(device, non_blocking=True)
                    val_batch.accumulated_damage = val_batch.x[:, -1].clone()
                    
                    val_pred_rate, val_shock_prob = model(
                        val_batch, mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge
                    )
                    
                    val_loss_single = model.module.loss(
                        (val_pred_rate, val_shock_prob), val_batch,
                        mean_vec_y_progressive, std_vec_y_progressive,
                        mean_vec_y_catastrophic, std_vec_y_catastrophic,
                        epoch  # use real epoch, not 0
                    )
                    val_total_loss += val_loss_single.item()

                    # Validation RMSE of the rate, excluding shock nodes (rate > 100)
                    with torch.no_grad():
                        _y = val_batch.y.squeeze()
                        _shock_thresh = 100.0
                        _steady = _y <= _shock_thresh
                        if _steady.any():
                            _pred_s = val_pred_rate.squeeze()[_steady]
                            _y_s = _y[_steady]
                            error = (_pred_s - _y_s) ** 2
                            velo_val_rmse_total += torch.sqrt(error.mean()).item()
                        else:
                            velo_val_rmse_total += 0.0
                    val_num_loops += 1

            val_loss = val_total_loss / max(val_num_loops, 1)
            
            val_loss_tensor = torch.tensor(val_loss, device=device)
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
            val_loss = val_loss_tensor.item() / world_size
            val_losses.append(val_loss)
            
            velo_val_rmse = velo_val_rmse_total / max(val_num_loops, 1)
            velo_rmse_tensor = torch.tensor(velo_val_rmse, device=device)
            dist.all_reduce(velo_rmse_tensor, op=dist.ReduceOp.SUM)
            velo_val_losses.append(velo_rmse_tensor.item() / world_size)
            
            # Step scheduler AFTER val_loss is known (all-reduced)
            if scheduler is not None:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
            
            # Early stopping check (rank 0 decides, then broadcasts to all)
            should_stop = torch.tensor([0], device=device)
            
            if local_rank == 0:
                _sel = float(val_loss)
                if _sel < best_val_loss:
                    best_val_loss = _sel
                    best_model = copy.deepcopy(model.module)
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch}")
                    should_stop[0] = 1
                else:
                    comp_str = " | ".join(
                        f"{k}={v:.4f}" for k, v in zip(
                            loss_components_history.keys(),
                            [loss_components_history[k][-1] for k in loss_components_history]
                        ) if v > 0 and k != 'total'
                    )
                    print(f"Epoch {epoch:3d}: Train={train_loss:.5f} Val={val_loss:.5f} | {comp_str}")
                
                # Logging (always do this before potential break)
                new_row = pd.DataFrame({
                    "epoch": [epoch],
                    "train_loss": [float(train_loss)],
                    "val_loss": [float(val_loss)],
                    "velo_val_loss": [float(velo_val_losses[-1])]
                })
                df = pd.concat([df, new_row], ignore_index=True)
                
                if epoch % DEFAULTS["checkpoint_save_freq"] == 0 and checkpoint_dir:
                    checkpoint_path = Path(checkpoint_dir) / f"{model_name}.pt"
                    torch.save(best_model.state_dict(), checkpoint_path)

                    # Full checkpoint for resuming training (rank 0), holding
                    # the underlying module's state rather than the DDP wrapper's
                    last_path = Path(checkpoint_dir) / "last_checkpoint.pt"
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': opt.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                        'best_val_loss': best_val_loss,
                    }, last_path)
                    # GTP_STAMPED_CKPTS=1 also keeps an epoch-stamped copy
                    if os.environ.get("GTP_STAMPED_CKPTS") == "1":
                        import shutil
                        shutil.copyfile(last_path,
                                        Path(checkpoint_dir) / f"epoch_{epoch:04d}.pt")

                if epoch % DEFAULTS["memory_monitor_freq"] == 0:
                    monitor_gpu_memory()

                # Wall-clock cap (rank 0 decides, broadcast below).
                if _max_train_seconds > 0:
                    _elapsed = time.time() - _train_start
                    if _elapsed >= _max_train_seconds:
                        print(f"\n[wall-clock] Elapsed {_elapsed:.0f}s >= "
                              f"{_max_train_seconds:.0f}s "
                              f"(GTP_MAX_TRAIN_SECONDS); stopping at epoch {epoch}.")
                        should_stop[0] = 1

            # Broadcast stop decision to all ranks
            dist.broadcast(should_stop, src=0)
            if should_stop[0] == 1:
                break
        
        if local_rank == 0 and checkpoint_dir:
            csv_path = Path(checkpoint_dir) / f"{model_name}.csv"
            df.to_csv(csv_path, index=False)
        
        dist.barrier()
        
        return val_losses, losses, velo_val_losses, best_model, loss_components_history
