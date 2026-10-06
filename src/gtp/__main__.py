#!/usr/bin/env python3
"""
GTP: Graph Transformer Propagator - Command Line Interface.

Entry point for training (single GPU or DDP), testing with autoregressive
rollout, inference benchmarking and hyperparameter optimisation.

Usage:
    # Training with DDP (4 GPUs):
    torchrun --nproc_per_node=4 --standalone -m src.gtp --mode train

    # Single GPU training:
    python -m src.gtp --mode train --epochs 100

    # Testing:
    python -m src.gtp --mode test --checkpoint results/xxx/best_models/model.pt

Authors: Luca Patrignani, Silvestre T. Pinho
Institution: Imperial College London
"""

import os
import sys
import random
import argparse
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist

from .config import Config, DEFAULTS
from .data import load_dataset, get_data_splits, get_stats_improved, analyze_node_features
from .training import Trainer, setup_ddp, cleanup_ddp
from .evaluation import Evaluator
from .utils import setup_logging, create_results_folder, save_hyperparameters

# Suppress matplotlib debug spam
logging.getLogger("matplotlib").setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="GTP: Graph Transformer Propagator for Damage Prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Mode selection
    parser.add_argument(
        "--mode", type=str, default="train",
        choices=["train", "test", "optimize", "benchmark"],
        help="Execution mode: 'train', 'test', 'optimize' (Optuna HPO), or "
             "'benchmark' (inference timing benchmark)"
    )
    
    # Config file
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config file (overrides defaults)"
    )
    
    # Dataset
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Path to dataset file (overrides config)"
    )
    
    # Model architecture
    parser.add_argument(
        "--num-layers", type=int, default=None,
        help=f"Number of message passing layers (default: {DEFAULTS['num_layers']})"
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=None,
        help=f"Hidden dimension (default: {DEFAULTS['hidden_dim']})"
    )
    parser.add_argument(
        "--dropout", type=float, default=None,
        help=f"Dropout rate (default: {DEFAULTS['dropout_rate']})"
    )
    parser.add_argument(
        "--attention-freq", type=int, default=None,
        help=f"Global attention frequency (default: {DEFAULTS['attention_freq']})"
    )
    parser.add_argument(
        "--local-attention-enabled", action=argparse.BooleanOptionalAction, default=None,
        help="Enable local edge attention gating in message passing. "
             "Use --no-local-attention-enabled to disable local attention."
    )
    parser.add_argument(
        "--alpha-global", type=float, default=None,
        help="Local/global mixing coefficient alpha_global: the weight of the "
             "global-attention residual, h <- h + alpha*A (default 0.2). "
             "Recorded in config.yaml and restored in test mode."
    )
    parser.add_argument(
        "--global-attention-enabled", action=argparse.BooleanOptionalAction, default=None,
        help="Enable periodic global attention blocks. "
             "Use --no-global-attention-enabled to disable global attention."
    )
    parser.add_argument(
        "--no-attention", action="store_true",
        help="Ablation shortcut: disable BOTH local edge-attention AND global "
             "transformer attention in one flag. Equivalent to passing "
             "--no-local-attention-enabled --no-global-attention-enabled."
    )
    parser.add_argument(
        "--two-stage", action=argparse.BooleanOptionalAction, default=None,
        help="Enable two-stage head: FI rate + gated damage rate. "
             "Use --no-two-stage to disable."
    )
    parser.add_argument(
        "--time-injection", action=argparse.BooleanOptionalAction, default=None,
        help="Mid-network (t, dt) injection at layer 3 (default ON). Use "
             "--no-time-injection to force attention to carry the loading signal."
    )

    # Training
    parser.add_argument(
        "--epochs", type=int, default=None,
        help=f"Number of training epochs (default: {DEFAULTS['epochs']})"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help=f"Batch size (default: {DEFAULTS['batch_size']})"
    )
    parser.add_argument(
        "--learning-rate", "--lr", type=float, default=None,
        help=f"Learning rate (default: {DEFAULTS['learning_rate']})"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=None,
        help=f"Weight decay (default: {DEFAULTS['weight_decay']})"
    )
    parser.add_argument(
        "--patience", type=int, default=None,
        help=f"Early stopping patience (default: {DEFAULTS['patience']})"
    )
    parser.add_argument(
        "--optimizer", type=str, default=None,
        choices=["adam", "sgd", "rmsprop", "adagrad"],
        help=f"Optimizer (default: {DEFAULTS['optimizer']})"
    )
    parser.add_argument(
        "--scheduler", type=str, default=None,
        choices=["none", "step", "cos", "cosine", "plateau"],
        help=f"LR scheduler (default: {DEFAULTS['scheduler']})"
    )
    parser.add_argument(
        "--plateau-patience", type=int, default=20,
        help="ReduceLROnPlateau patience (epochs without improvement before LR drop)"
    )
    parser.add_argument(
        "--plateau-factor", type=float, default=0.5,
        help="ReduceLROnPlateau multiplicative factor for LR reduction"
    )
    parser.add_argument(
        "--min-lr", type=float, default=1e-6,
        help="Minimum learning rate for ReduceLROnPlateau"
    )
    
    # DDP
    parser.add_argument(
        "--no-ddp", action="store_true",
        help="Disable DDP even if torchrun is used"
    )
    
    # Testing
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to model checkpoint (required for test mode)"
    )
    parser.add_argument(
        "--resume-from", type=str, default=None,
        help="Resume training from a checkpoint (train mode). Accepts a full "
             "checkpoint (last_checkpoint.pt: model, optimizer, scheduler, "
             "epoch) or a weights-only .pt (optimizer and scheduler start fresh)."
    )
    parser.add_argument(
        "--stats", type=str, default=None,
        help="Path to normalization_stats.pt (auto-detected if not provided)"
    )
    
    # Output
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for results"
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0",
        help="Device for training/testing"
    )
    
    # Optuna optimization
    parser.add_argument(
        "--n-trials", type=int, default=50,
        help="Number of Optuna trials (for --mode optimize)"
    )
    parser.add_argument(
        "--timeout", type=int, default=None,
        help="Timeout in seconds for optimization (for --mode optimize)"
    )
    parser.add_argument(
        "--epochs-per-trial", type=int, default=30,
        help="Max epochs per trial (for --mode optimize)"
    )
    parser.add_argument(
        "--skip-final-train", action="store_true",
        help="--mode optimize: exit after the trials instead of launching the "
             "full best-params DDP training. Read best_params from "
             "optuna_results.json and start the real training manually."
    )
    
    # Verbosity
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress most output"
    )
    
    # Run name for results folder
    parser.add_argument(
        "--run-name", type=str, default=None,
        help="Name prefix for results folder"
    )
    
    # Loss weights
    parser.add_argument(
        "--damage-loss-weight", type=float, default=None,
        help=f"Damage rate loss weight (default: {DEFAULTS['damage_loss_weight']})"
    )
    parser.add_argument(
        "--active-node-weight", type=float, default=None,
        help=f"Weight multiplier for active nodes (rate > 0.01) in rate loss. "
             f"Values > 1 upweight nodes that are actually propagating cracks. "
             f"(default: {DEFAULTS['active_node_weight']})"
    )
    parser.add_argument(
        "--rate-cap", type=float, default=None,
        help="Clamp the target rates of the rate loss at this value. 0 keeps the "
             "unclamped rates; a negative value instead masks out the nodes with "
             "rate > 100. Without the flag, rate_cap in defaults.py (10) applies."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Global RNG seed (model initialisation and dropout; also the "
             "batch order on a single GPU). The simulation-level "
             "train/val/test split is pinned to its own seed 42 regardless, "
             "so runs with different --seed values share identical splits."
    )
    # Rollout/Visualization options
    parser.add_argument(
        "--rollout-steps", type=int, default=None,
        help="Number of timesteps for autoregressive rollout (default: all available)"
    )
    parser.add_argument(
        "--rollout-downsample", type=int, default=3,
        help="Store every Nth frame for visualization (default: 3)"
    )
    
    # Attention analysis (test mode)
    parser.add_argument(
        "--attention-analysis", action="store_true",
        help="Save the global received-attention field (attn_field_frame_*.npz) "
             "and per-ply global-attention plots (test mode)."
    )
    parser.add_argument(
        "--plot-frames", type=int, default=None,
        help="Number of evenly-spaced frames for the attention analysis (default: 3). "
             "Set higher to see more temporal snapshots."
    )
    parser.add_argument(
        "--attention-sim", type=int, default=None,
        help="Restrict attention analysis to one simulation id (sample_idx); "
             "--plot-frames then picks frames within that simulation only."
    )
    parser.add_argument(
        "--attention-block", type=int, default=None,
        help="Which global-attention block to analyse (0-based call order). A "
             "block runs every --attention-freq layers, so block 0 runs before "
             "any message passing, block 1 after attention_freq layers, etc. "
             "Default: the last block. Output goes to attention_analysis_blk<N>/. "
             "Uses forward hooks only; the model is not modified."
    )

    args = parser.parse_args()

    # --no-attention is a shortcut that disables BOTH attention paths.
    # It overrides any explicit setting of --local/global-attention-enabled.
    if getattr(args, "no_attention", False):
        args.local_attention_enabled = False
        args.global_attention_enabled = False

    return args


def main():
    """Main entry point."""
    args = parse_args()

    try:
        # Optimize mode runs as a single process (each trial launches its own
        # DDP training), so refuse to run under torchrun
        if args.mode == "optimize":
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            rank = int(os.environ.get("RANK", 0))
            
            if world_size > 1:
                if rank == 0:
                    print("\n" + "=" * 70)
                    print("ERROR: Do not use torchrun for --mode optimize!")
                    print("=" * 70)
                    print("\nOptuna HPO must run as a single process to avoid OOM.")
                    print("After optimization, full DDP training will be launched automatically.")
                    print("\nCorrect usage:")
                    print("  python -m src.gtp --mode optimize --n-trials 30 --epochs-per-trial 20")
                    print("\nThis will:")
                    print("  1. Run Optuna HPO on single GPU")
                    print("  2. Save best hyperparameters")
                    print("  3. Automatically start full DDP training with best params")
                    print("=" * 70 + "\n")
                # All ranks exit
                sys.exit(1)
        
        def _validate_loaded_dataset(dataset, path):
            """Fail fast if loaded dataset lacks critical metadata."""
            d0 = dataset[0]
            required = ['sample_idx', 'mesh_pos', 'y', 'delta_t']
            missing = [a for a in required if getattr(d0, a, None) is None]
            if missing:
                raise ValueError(
                    f"Dataset {path} is missing required attributes on Data "
                    f"objects: {missing}. Re-run preprocessing."
                )
            # Check the simulation count. Single-simulation datasets are valid
            # in test/benchmark mode (e.g. one extrapolation geometry); the
            # guard protects training from a broken sample_idx column.
            sids = set()
            for d in dataset:
                sid = getattr(d, 'sample_idx', None)
                if sid is not None:
                    sids.add(int(sid.item()))
            inference_only = args.mode in ("test", "benchmark")
            if len(sids) < 2 and len(dataset) > 50 and not inference_only:
                raise ValueError(
                    f"Dataset has {len(sids)} unique sample_idx across "
                    f"{len(dataset)} samples. Simulation-level splitting "
                    f"requires >= 2 simulations. Re-run preprocessing."
                )
            print(f"Dataset validation: {len(sids)} sims, "
                  f"{len(dataset)} transitions")

        # Build config from defaults + YAML + CLI
        config = Config.from_args(args)
        
        # Set verbosity
        if args.quiet:
            config.verbose = False
        elif args.verbose:
            config.verbose = True
        
        # Get project paths
        root_dir = os.getcwd()
        
        # Determine dataset path
        if args.dataset:
            data_path = args.dataset
        else:
            data_path = os.path.join(root_dir, "data", "processed", DEFAULTS["dataset_filename"])
            if not os.path.exists(data_path):
                data_path = os.path.join(root_dir, DEFAULTS["dataset_filename"])
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset not found at: {data_path}")
        
        # Get preliminary rank for loading prints
        preliminary_rank = int(os.environ.get("RANK", 0))
        
        # Load dataset (only rank 0 prints)
        dataset = load_dataset(data_path, rank=preliminary_rank)

        # Fail-fast: verify critical metadata exists on loaded dataset
        _validate_loaded_dataset(dataset, data_path)

        mode = args.mode
        
        if mode == "train":
            _run_training(args, config, dataset, root_dir)
        
        elif mode == "test":
            _run_testing(args, config, dataset, root_dir)

        elif mode == "optimize":
            _run_optimization(args, config, dataset, root_dir)

        elif mode == "benchmark":
            _run_benchmark(args, config, dataset, root_dir)

        else:
            raise ValueError(f"Invalid mode: {mode}. Choose 'train', 'test', "
                             f"'optimize', or 'benchmark'.")
    
    except Exception as e:
        print(f"Fatal Error: {e}")
        import traceback
        traceback.print_exc()
        
        if dist.is_initialized():
            cleanup_ddp()
        
        sys.exit(1)


def _run_benchmark(args, config, dataset, root_dir):
    """Run inference timing benchmark and compare against ABAQUS wallclock."""
    import json as _json
    from collections import defaultdict
    
    device = args.device
    
    # ── Resolve checkpoint ───────────────────────────────────────────
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    elif args.output_dir:
        best_dir = Path(args.output_dir) / "best_models"
        candidates = sorted(best_dir.glob("*.pt")) if best_dir.exists() else []
        if not candidates:
            raise FileNotFoundError(f"No checkpoint found in {best_dir}")
        checkpoint_path = candidates[0]
    else:
        raise ValueError("--checkpoint or --output-dir required for benchmark mode")
    
    # ── Resolve stats ────────────────────────────────────────────────
    if args.stats:
        stats_path = Path(args.stats)
    else:
        search_dirs = []
        if args.output_dir:
            search_dirs.append(Path(args.output_dir))
        search_dirs.append(checkpoint_path.parent.parent)
        stats_path = None
        for d in search_dirs:
            cand = d / "normalization_stats.pt"
            if cand.exists():
                stats_path = cand
                break
        if stats_path is None:
            raise FileNotFoundError("normalization_stats.pt not found")
    
    stats_list = torch.load(stats_path, weights_only=False)
    
    # ── Build test split (same logic as _run_testing) ────────────────
    sim_groups = defaultdict(list)
    for i, d in enumerate(dataset):
        sid = int(d.sample_idx.item()) if hasattr(d, 'sample_idx') else 0
        sim_groups[sid].append(i)
    sim_ids = sorted(sim_groups.keys())
    n_sims = len(sim_ids)
    
    if n_sims < 3:
        test_dataset = list(dataset)
    else:
        import random as _rng
        _rng.seed(42)
        shuffled = list(sim_ids)
        _rng.shuffle(shuffled)
        train_ratio = DEFAULTS["train_ratio"]
        val_ratio = DEFAULTS["val_ratio"]
        n_train = max(1, int(n_sims * train_ratio))
        n_val = max(1, int(n_sims * val_ratio))
        if n_train + n_val >= n_sims:
            n_val = max(1, n_sims - n_train - 1)
        test_sims = sorted(shuffled[n_train + n_val:])
        if not test_sims:
            test_sims = [sorted(shuffled[n_train:n_train + n_val])[-1]]
        test_indices = sorted(i for sid in test_sims for i in sim_groups[sid])
        test_dataset = [dataset[i] for i in test_indices]
    
    print(f"Benchmark: {len(test_dataset)} test samples from {n_sims} sims")
    
    # ── Load model ───────────────────────────────────────────────────
    evaluator = Evaluator(config=config, device=device)
    sample0 = test_dataset[0]
    num_node_features = sample0.x.shape[1]
    num_edge_features = sample0.edge_attr.shape[1]
    model = evaluator.load_model(checkpoint_path, num_node_features, num_edge_features)
    
    # ── Benchmark ────────────────────────────────────────────────────
    results = evaluator.benchmark_inference(
        model=model,
        test_dataset=test_dataset,
        stats_list=stats_list,
        n_warmup=min(20, len(test_dataset) // 4),
        n_timed=min(200, len(test_dataset)),
    )

    # Save results
    out_dir = Path(args.output_dir) if args.output_dir else checkpoint_path.parent.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    bench_path = out_dir / "benchmark_results.json"
    with open(bench_path, "w") as f:
        _json.dump(results, f, indent=2)
    print(f"Saved benchmark results to: {bench_path}")


def _run_training(args, config, dataset, root_dir):
    """Run training mode."""
    from .visualization import plot_loss_curves
    
    # Apply --damage-loss-weight CLI override (propagated via env var to model.loss())
    if args.damage_loss_weight is not None:
        os.environ["GTP_DAMAGE_LOSS_WEIGHT"] = str(args.damage_loss_weight)
        print(f"  damage_loss_weight overridden to {args.damage_loss_weight} (via --damage-loss-weight)")
    elif "GTP_DAMAGE_LOSS_WEIGHT" in os.environ:
        print(f"  damage_loss_weight = {os.environ['GTP_DAMAGE_LOSS_WEIGHT']} (via GTP_DAMAGE_LOSS_WEIGHT env var)")
    
    # Apply --active-node-weight CLI override (extra weight on active nodes)
    if args.active_node_weight is not None:
        os.environ["GTP_ACTIVE_NODE_WEIGHT"] = str(args.active_node_weight)
        print(f"  active_node_weight overridden to {args.active_node_weight} (via --active-node-weight)")
    elif "GTP_ACTIVE_NODE_WEIGHT" in os.environ:
        print(f"  active_node_weight = {os.environ['GTP_ACTIVE_NODE_WEIGHT']} (via GTP_ACTIVE_NODE_WEIGHT env var)")
    
    # Apply --rate-cap CLI override
    if args.rate_cap is not None:
        os.environ["GTP_RATE_CAP"] = str(args.rate_cap)
        print(f"  rate_cap overridden to {args.rate_cap} (via --rate-cap)")
    elif "GTP_RATE_CAP" in os.environ:
        print(f"  rate_cap = {os.environ['GTP_RATE_CAP']} (via GTP_RATE_CAP env var)")
    
    # Check for DDP environment FIRST (before creating folders)
    use_ddp = ("RANK" in os.environ and "WORLD_SIZE" in os.environ) and not args.no_ddp
    
    if use_ddp:
        local_rank, rank, world_size = setup_ddp()
        # CUDA → per-rank GPU; no-CUDA → CPU (smoke-test path only).
        device = (f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

        # Only rank 0 creates folders, then broadcast path to all ranks
        if rank == 0:
            # Use provided output_dir or create a new results folder
            if args.output_dir:
                results_folder = Path(args.output_dir)
                results_folder.mkdir(parents=True, exist_ok=True)
                for _sub in ("best_models", "training_results",
                             "test_results",
                             "hyperparameter_optimization_results"):
                    (results_folder / _sub).mkdir(exist_ok=True)
            else:
                results_folder = create_results_folder()
            folder_str = str(results_folder)
        else:
            folder_str = None
        
        # Broadcast folder path from rank 0 to all ranks
        folder_list = [folder_str]
        dist.broadcast_object_list(folder_list, src=0)
        results_folder = Path(folder_list[0])
    else:
        local_rank = 0
        rank = 0
        world_size = 1
        # Use provided output_dir or create a new results folder
        if args.output_dir:
            results_folder = Path(args.output_dir)
            results_folder.mkdir(parents=True, exist_ok=True)
            for _sub in ("best_models", "training_results",
                         "test_results",
                         "hyperparameter_optimization_results"):
                (results_folder / _sub).mkdir(exist_ok=True)
        else:
            results_folder = create_results_folder()
    
    checkpoint_dir = results_folder / "best_models"
    training_results_dir = results_folder / "training_results"
    
    if use_ddp:
        # Setup logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_level = "DEBUG" if config.verbose else "INFO"
        
        if rank == 0:
            log_file = results_folder / f"execution_log_train_{timestamp}.txt"
            logger = setup_logging(log_level=log_level, log_file=str(log_file))
            logger.info(f"Logging to: {log_file}")
            logger.info(f"\n{'='*70}")
            logger.info(f"EXECUTION MODE: TRAIN with DDP ({world_size} GPUs)")
            logger.info(f"{'='*70}\n")
        else:
            setup_logging(log_level="WARNING")
        
        # Reproducibility. --seed sets the initialisation and dropout streams
        # (the DistributedSampler keeps its own fixed seed, so under DDP the
        # batch order does not change); the simulation-level SPLIT stays
        # pinned to 42 inside get_data_splits regardless.
        _seed = getattr(args, 'seed', 42)
        if rank == 0 and _seed != 42:
            logger.info(f"Seed: {_seed} (split remains seed-42)")
        torch.manual_seed(_seed + rank)
        random.seed(_seed + rank)
        np.random.seed(_seed + rank)
        
        if rank == 0:
            analyze_node_features(dataset, rank=rank, verbose=config.verbose)
    else:
        device = args.device if torch.cuda.is_available() else "cpu"
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = results_folder / f"execution_log_train_{timestamp}.txt"
        logger = setup_logging(log_level="INFO", log_file=str(log_file))
        logger.info("EXECUTION MODE: TRAIN (single GPU)")
        
        _seed = getattr(args, 'seed', 42)
        if _seed != 42:
            logger.info(f"Seed: {_seed} (split remains seed-42)")
        torch.manual_seed(_seed)
        random.seed(_seed)
        np.random.seed(_seed)
        
        analyze_node_features(dataset, rank=rank, verbose=config.verbose)
    
    # Get data splits
    train_dataset, val_dataset, test_dataset = get_data_splits(
        dataset,
        train_ratio=DEFAULTS["train_ratio"],
        val_ratio=DEFAULTS["val_ratio"],
        test_ratio=DEFAULTS["test_ratio"],
        interleave_step=DEFAULTS["interleave_step"],
        rank=rank
    )
    
    # Compute normalization stats
    stats_list = get_stats_improved(train_dataset, rank=rank)
    torch.save(stats_list, results_folder / "normalization_stats.pt")

    # Save config and hyperparameters
    config.save(results_folder / "config.yaml")
    save_hyperparameters(results_folder, config, additional_info={
        "world_size": world_size,
        "dataset_size": len(dataset),
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "test_size": len(test_dataset),
    })
    
    # Create trainer
    trainer = Trainer(
        config=config,
        device=device,
        local_rank=local_rank,
        world_size=world_size,
    )

    # Train
    if use_ddp and rank == 0:
        print(f"Starting DDP training with {world_size} GPUs...")
    elif not use_ddp:
        print(f"\nStarting single-GPU training on {device}...")

    resume_from = getattr(args, "resume_from", None)
    if resume_from:
        # DDP resume is now wired up in trainer._train_ddp (model + optimizer
        # + scheduler + epoch + best_val all restored). Single-GPU resume goes
        # through trainer._train_single. The guard against multi-GPU resume
        # has been removed.
        if not Path(resume_from).is_file():
            raise FileNotFoundError(f"--resume-from: {resume_from} does not exist")
        print(f"  [resume] Will resume training from: {resume_from}")

    val_losses, losses, velo_val_losses, best_model, loss_components = trainer.train(
        train_dataset, val_dataset, stats_list,
        checkpoint_dir=checkpoint_dir,
        postprocess_dir=training_results_dir,
        resume_from=resume_from,
    )
    
    # Get best loss metric
    best_loss_label = "best_val_loss"

    # Save training_results.json for Optuna to read
    if (use_ddp and rank == 0) or not use_ddp:
        import json
        training_results = {
            "best_val_loss": min(val_losses) if val_losses else None,
            "best_train_loss": min(losses) if losses else None,
            "checkpoint_metric": best_loss_label,
            "final_val_loss": val_losses[-1] if val_losses else None,
            "final_train_loss": losses[-1] if losses else None,
            "epochs_trained": len(losses),
            "config": {
                "num_layers": config.num_layers,
                "hidden_dim": config.hidden_dim,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
            }
        }
        with open(results_folder / "training_results.json", "w") as f:
            json.dump(training_results, f, indent=2)
    
    # Save plots on main process
    if (use_ddp and rank == 0) or not use_ddp:
        from .visualization import plot_loss_components
        
        # Loss curve plot
        plot_loss_curves(
            train_losses=losses,
            val_losses=val_losses,
            save_path=training_results_dir / "loss_plot.png",
            title="Training Progress"
        )
        print(f"Loss plot saved to: {training_results_dir / 'loss_plot.png'}")
        
        # Loss components plot
        if loss_components:
            plot_loss_components(
                loss_components=loss_components,
                save_path=training_results_dir / "loss_components.png",
                title="Loss Component Breakdown"
            )
            print(f"Loss components saved to: {training_results_dir / 'loss_components.png'}")
        
        # ── One-frame mesh snapshot ─────────────────────────────────────
        # Run a single forward pass on one active-phase sample and save
        # a 2×3 mesh comparison (true/pred damage + rate + error).
        try:
            from .visualization import plot_damage_comparison, safe_to_numpy
            
            # Pick an active-phase sample with nonzero damage.
            # Prefer val set; fall back to train set.
            snapshot_dataset = val_dataset if len(val_dataset) > 0 else train_dataset
            # Scan for a sample whose accumulated damage (last node feature)
            # is nonzero.  Fall back to the midpoint if nothing found.
            snapshot_idx = len(snapshot_dataset) // 2  # safe default
            for _si in range(len(snapshot_dataset)):
                _dmg = snapshot_dataset[_si].x[:, -1]
                if _dmg.max().item() > 1e-6:
                    snapshot_idx = _si
                    break
            snapshot_sample = snapshot_dataset[snapshot_idx]
            if rank == 0:
                print(f"  Mesh snapshot using sample index {snapshot_idx} "
                      f"(max damage = {snapshot_sample.x[:, -1].max().item():.4f})")
            
            # Move stats to device
            [s_mean_x, s_std_x, s_mean_edge, s_std_edge,
             s_mean_yp, s_std_yp, s_mean_yc, s_std_yc] = stats_list
            s_mean_x = s_mean_x.to(device)
            s_std_x = s_std_x.to(device)
            s_mean_edge = s_mean_edge.to(device)
            s_std_edge = s_std_edge.to(device)
            
            best_model.eval()
            snap = snapshot_sample.clone().to(device)
            with torch.no_grad():
                # The model returns (pred_rate, None)
                snap_pred_rate, _ = best_model(
                    snap, s_mean_x, s_std_x, s_mean_edge, s_std_edge
                )
            
            snap_true_rate = snap.y.squeeze()
            # Accumulated damage is the last node feature
            snap_true_damage = snap.x[:, -1]
            # Predicted damage = true_damage + pred_rate * delta_t
            dt = snap.delta_t
            snap_delta_t = float(dt[0].item()) if dt.numel() > 1 else float(dt.item())
            snap_pred_damage = snap_true_damage + snap_pred_rate.squeeze() * snap_delta_t
            snap_true_damage_next = snap_true_damage + snap_true_rate * snap_delta_t
            
            mesh_save = training_results_dir / "training_sample_mesh.png"
            _is_fi = hasattr(snapshot_sample, 'n_ply_nodes') and snapshot_sample.n_ply_nodes is not None
            plot_damage_comparison(
                data=snap,
                pred_damage=safe_to_numpy(snap_pred_damage),
                true_damage=safe_to_numpy(snap_true_damage_next),
                pred_rate=safe_to_numpy(snap_pred_rate),
                true_rate=safe_to_numpy(snap_true_rate),
                save_path=mesh_save,
                timestep=getattr(snapshot_sample, 'sample_idx', None),
                field_name="FI" if _is_fi else "Damage",
            )
            print(f"Training mesh snapshot saved to: {mesh_save}")
        except Exception as e:
            print(f"Warning: could not generate mesh snapshot: {e}")
        
        print(f"\nTraining completed! Results saved in: {results_folder}")
        print(f"  - Best model: {checkpoint_dir}")
        print(f"  - Training results: {training_results_dir}")
        print(f"  - Hyperparameters: {results_folder / 'hyperparameter_optimization_results'}")
    
    if use_ddp:
        cleanup_ddp()


# ─────────────────────────────────────────────────────────────────────────────
# GTP_TEST_SIMS: restrict an already-built test split to named simulations, so
# that one simulation can be rendered without rolling out the whole test set.
# The split itself is unchanged and only its output is filtered, so a retained
# simulation keeps its frames and their order. The metrics of a narrowed run
# cover only those simulations: the run refuses ids missing from the split and
# writes PARTIAL_TEST_SPLIT.json into its output directory.
# ─────────────────────────────────────────────────────────────────────────────
_PARTIAL_SPLIT_SENTINEL = "PARTIAL_TEST_SPLIT.json"


def parse_test_sims_spec(spec):
    """Parse a GTP_TEST_SIMS spec into a sorted list of simulation ids.

    Accepts comma- and/or whitespace-separated integers ("13", "13,22",
    "13 22"). Returns None for an unset/blank spec, meaning "do not narrow".

    Raises ValueError on any token that is not an integer.
    """
    if spec is None:
        return None
    tokens = spec.replace(",", " ").split()
    if not tokens:
        return None
    ids = []
    for tok in tokens:
        try:
            ids.append(int(tok))
        except ValueError:
            raise ValueError(
                f"GTP_TEST_SIMS: {tok!r} is not an integer simulation id "
                f"(got spec {spec!r})")
    return sorted(set(ids))


def sim_id_of(sample):
    """Simulation id of one transition, or None when it carries no sample_idx."""
    idx = getattr(sample, "sample_idx", None)
    if idx is None:
        return None
    return int(idx.item()) if hasattr(idx, "item") else int(idx)


def narrow_test_split_to_sims(test_dataset, keep_ids):
    """Return the sublist of ``test_dataset`` belonging to ``keep_ids``.

    The order is preserved, so each simulation's timesteps stay contiguous and
    in time order. Raises ValueError if any requested id is absent from the
    split.
    """
    if not keep_ids:
        return list(test_dataset)
    wanted = set(keep_ids)
    present = {sim_id_of(d) for d in test_dataset}
    present.discard(None)
    missing = sorted(wanted - present)
    if missing:
        raise ValueError(
            f"GTP_TEST_SIMS: simulation(s) {missing} are not in the test "
            f"split. The split holds {sorted(present)}. Refusing to run: a "
            f"partial match would silently roll out the wrong simulations.")
    return [d for d in test_dataset if sim_id_of(d) in wanted]


def _write_partial_split_sentinel(output_dir, keep_ids, n_kept, n_total):
    """Record on disk that this directory holds metrics of a partial test split."""
    import json as _json
    payload = {
        "partial_test_split": True,
        "simulations_evaluated": list(keep_ids),
        "frames_evaluated": int(n_kept),
        "frames_in_full_test_split": int(n_total),
        "written_by": "GTP_TEST_SIMS",
        "warning": (
            "Every metric in this directory was computed over the listed "
            "simulations only; it is not a test-set score."),
    }
    path = Path(output_dir) / _PARTIAL_SPLIT_SENTINEL
    path.write_text(_json.dumps(payload, indent=2))
    return path


def _run_testing(args, config, dataset, root_dir):
    """Run testing mode with comprehensive visualization."""
    mode_label = "TEST"
    output_subfolder = "test_results"
    
    # ── Resolve checkpoint path ──────────────────────────────────────────
    # Priority: --checkpoint > auto-detect from --output-dir/best_models/
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    elif args.output_dir:
        best_dir = Path(args.output_dir) / "best_models"
        # Find any .pt in best_models (training checkpoint)
        candidates = sorted(best_dir.glob("*.pt")) if best_dir.exists() else []
        if candidates:
            checkpoint_path = candidates[0]
            print(f"Auto-detected training checkpoint: {checkpoint_path}")
        else:
            raise FileNotFoundError(
                f"No checkpoint found in {best_dir}. "
                f"Use --checkpoint to specify one explicitly."
            )
    else:
        raise ValueError("--checkpoint or --output-dir is required for test mode")
    
    # Auto-detect stats file
    if args.stats:
        stats_path = Path(args.stats)
    else:
        # Try --output-dir first (the gtp_run_XXX folder), then checkpoint parent
        search_dirs = []
        if args.output_dir:
            search_dirs.append(Path(args.output_dir))
        search_dirs.append(checkpoint_path.parent.parent)
        stats_path = None
        for d in search_dirs:
            candidate = d / "normalization_stats.pt"
            if candidate.exists():
                stats_path = candidate
                break
        if stats_path is None:
            raise FileNotFoundError(
                f"normalization_stats.pt not found in {search_dirs}. "
                f"Please specify --stats explicitly."
            )
    
    print(f"Loading normalization stats from: {stats_path}")
    stats_list = torch.load(stats_path, weights_only=False)

    # ── Load training config.yaml to match model architecture ────────
    # When testing, the model must be built with the SAME config used
    # during training (dropout, attention_freq, etc.).  The evaluator
    # overrides hidden_dim/num_layers from the checkpoint, but
    # other params must come from the saved config, not from defaults.
    config_yaml = None
    if args.output_dir:
        config_yaml = Path(args.output_dir) / "config.yaml"
    if config_yaml is None or not config_yaml.exists():
        config_yaml = checkpoint_path.parent.parent / "config.yaml"
    if config_yaml.exists():
        from .config import Config as _Cfg
        saved_config = _Cfg.from_yaml(str(config_yaml))
        # Merge saved config into current config (saved values take
        # precedence for model-architecture keys; CLI overrides still
        # win because they were applied after defaults).
        arch_keys = [
            "dropout_rate", "attention_freq", "attention_dropout",
            "use_unified_decoder",
            "local_attention_enabled", "global_attention_enabled",
            # not stored in the weights, so it must be restored from config.yaml
            "alpha_global",
            "two_stage_enabled",
            "time_injection_enabled",
            "rate_cap", "num_layers", "hidden_dim",
        ]
        for key in arch_keys:
            saved_val = getattr(saved_config, key, None)
            if saved_val is not None:
                setattr(config, key, saved_val)
        print(f"✓ Loaded training config from: {config_yaml}")

    # ── Simulation-level test split ──────────────────────────────
    # Group transitions by sample_idx so that each simulation's
    # timesteps stay together and in order (needed for rollout).
    from collections import defaultdict
    sim_groups = defaultdict(list)
    for i, d in enumerate(dataset):
        sid = int(d.sample_idx.item()) if hasattr(d, 'sample_idx') else 0
        sim_groups[sid].append(i)

    sim_ids = sorted(sim_groups.keys())
    n_sims = len(sim_ids)

    if n_sims < 3:
        # Not enough sims for a proper split → use all data
        test_dataset = list(dataset)
        print(f"Test dataset: {len(test_dataset)} samples "
              f"(only {n_sims} sim(s), using all)")
    else:
        # Use SAME seed+shuffle as get_data_splits() in loader.py
        # so that the test set here matches the one used during training.
        import random as _rng
        _rng.seed(42)
        shuffled = list(sim_ids)
        _rng.shuffle(shuffled)

        train_ratio = DEFAULTS["train_ratio"]
        val_ratio = DEFAULTS["val_ratio"]
        n_train = max(1, int(n_sims * train_ratio))
        n_val = max(1, int(n_sims * val_ratio))
        if n_train + n_val >= n_sims:
            n_val = max(1, n_sims - n_train - 1)
        test_sims = sorted(shuffled[n_train + n_val:])
        if not test_sims:
            test_sims = [sorted(shuffled[n_train:n_train + n_val])[-1]]

        test_indices = sorted(
            i for sid in test_sims for i in sim_groups[sid]
        )
        test_dataset = [dataset[i] for i in test_indices]
        print(f"Test dataset: {len(test_dataset)} samples "
              f"from sim(s) {test_sims} "
              f"(out of {n_sims} simulations)")

    # ── Optional narrowing of the split (GTP_TEST_SIMS, see above) ──
    _keep_sims = parse_test_sims_spec(os.environ.get("GTP_TEST_SIMS"))
    _n_full_test = len(test_dataset)
    if _keep_sims:
        test_dataset = narrow_test_split_to_sims(test_dataset, _keep_sims)
        print("\n" + "!" * 70)
        print(f"  PARTIAL TEST SPLIT: GTP_TEST_SIMS={_keep_sims}")
        print(f"  Rollout narrowed: {_n_full_test} -> {len(test_dataset)} frames.")
        print("  Metrics written by this run cover only those simulations;")
        print("  they are not a test-set score.")
        print("!" * 70 + "\n")

    # Output directory - uses different subfolder for post-test
    if args.output_dir:
        output_dir = Path(args.output_dir) / output_subfolder
    else:
        output_dir = checkpoint_path.parent.parent / output_subfolder
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"{mode_label} results will be saved to: {output_dir}")

    # Stamp partial runs so the directory is self-describing. A narrowed run
    # writes its metrics under the SAME filenames a full run uses, so without
    # this stamp nothing on disk distinguishes a sim-013-only R2 from a
    # test-set R2. See the GTP_TEST_SIMS block above.
    if _keep_sims:
        _s = _write_partial_split_sentinel(
            output_dir, _keep_sims, len(test_dataset), _n_full_test)
        print(f"  [GTP_TEST_SIMS] provenance stamped: {_s}")
    
    # Run evaluation
    print(f"\n{'='*70}")
    print(f"RUNNING {mode_label} MODE")
    print(f"{'='*70}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {args.device}")
    print(f"{'='*70}\n")
    
    evaluator = Evaluator(config=config, device=args.device)
    
    # Get rollout parameters from CLI
    rollout_steps = getattr(args, 'rollout_steps', None)
    rollout_downsample = getattr(args, 'rollout_downsample', 3)
    
    print(f"Rollout settings: steps={rollout_steps or 'all'}, downsample={rollout_downsample}")
    
    # Get predictions and targets for visualization
    metrics, predictions, targets, rollout_results = evaluator.evaluate_with_visualization(
        checkpoint_path=checkpoint_path,
        test_dataset=test_dataset,
        stats_list=stats_list,
        output_dir=output_dir,
        batch_size=1,
        num_rollout_samples=rollout_steps,
        rollout_downsample=rollout_downsample,
    )
    
    # Generate plots
    print("\n[Generating Plots]")

    # Detect field type: FI pipeline has n_ply_nodes attribute
    _is_fi_pipeline = hasattr(test_dataset[0], 'n_ply_nodes') and test_dataset[0].n_ply_nodes is not None
    _field_name = "FI" if _is_fi_pipeline else "Damage"
    print(f"  Field type: {_field_name}")

    # ── Global-attention analysis (received-attention field) ─────────
    if getattr(args, 'attention_analysis', False):
        print("\n[Attention Analysis]")
        from .visualization import plot_global_attention_analysis
        try:
            attn_block = getattr(args, 'attention_block', None)
            attn_dir = output_dir / (
                "attention_analysis" if attn_block is None
                else f"attention_analysis_blk{attn_block}")
            attn_dir.mkdir(parents=True, exist_ok=True)

            num_node_features = test_dataset[0].x.shape[1]
            num_edge_features = test_dataset[0].edge_attr.shape[1]
            attn_model = evaluator.load_model(
                checkpoint_path, num_node_features, num_edge_features
            )
            attn_model.eval()
            attn_model.enable_attention_analysis(True)

            # ── Per-block global-attention capture (--attention-block) ──
            # The global-attention module is shared by all blocks, so only the
            # last block's weights remain in `_attention_weights`. Forward hooks
            # snapshot the weights after each call, and the chosen block's
            # snapshot is put back after the forward pass. The hooks are
            # installed only when --attention-block is set.
            _block_caps = []
            if attn_block is not None:
                def _pre_clear(_m, _inp):
                    attn_model.global_attention._attention_weights = None
                    _block_caps.clear()

                def _snap(_m, _inp, _out):
                    w = _m._attention_weights
                    _block_caps.append(
                        None if w is None else {k: v.clone() for k, v in w.items()})

                attn_model.register_forward_pre_hook(_pre_clear)
                attn_model.global_attention.register_forward_hook(_snap)
                print(f"  Per-block capture ON: analysing global-attention block "
                      f"{attn_block} (0=before any message passing)")

            # Frame pool: whole test set, or one representative simulation
            attn_sim = getattr(args, 'attention_sim', None)
            if attn_sim is not None:
                attn_pool = [
                    i for i, d in enumerate(test_dataset)
                    if hasattr(d, 'sample_idx')
                    and int(d.sample_idx.item()) == attn_sim
                ]
                if attn_pool:
                    print(f"  Attention analysis restricted to sim {attn_sim} "
                          f"({len(attn_pool)} frames)")
                else:
                    print(f"  WARNING: sim {attn_sim} not in the test split; "
                          "using the full test set")
                    attn_pool = list(range(len(test_dataset)))
            else:
                attn_pool = list(range(len(test_dataset)))

            # Determine how many frames to plot
            n_frames = getattr(args, 'plot_frames', None) or 3
            n_frames = min(n_frames, len(attn_pool))
            if n_frames <= 1:
                attn_indices = [attn_pool[0]]
            elif n_frames == 2:
                attn_indices = [attn_pool[0], attn_pool[-1]]
            else:
                step = max(1, (len(attn_pool) - 1) / (n_frames - 1))
                attn_indices = sorted(set(
                    attn_pool[int(round(i * step))] for i in range(n_frames)))

            print(f"  Plotting {len(attn_indices)} frames: {attn_indices}")

            for idx in attn_indices:
                sample = test_dataset[idx]
                sample_dev = sample.to(args.device)

                mean_x, std_x, mean_e, std_e = [s.to(args.device) for s in stats_list[:4]]
                with torch.no_grad():
                    _ = attn_model(sample_dev, mean_x, std_x, mean_e, std_e)

                # Select the requested global-attention block (opt-in). The
                # snapshots are in call order; replace `_attention_weights` with
                # the chosen one so get_global_attention_stats reads that block.
                if attn_block is not None:
                    if 0 <= attn_block < len(_block_caps) and _block_caps[attn_block]:
                        attn_model.global_attention._attention_weights = \
                            _block_caps[attn_block]
                    else:
                        print(f"    WARNING: block {attn_block} unavailable "
                              f"({len(_block_caps)} blocks ran); using the last")

                mesh_pos = sample.mesh_pos
                if torch.is_tensor(mesh_pos):
                    mesh_pos_np = mesh_pos.cpu().numpy()
                else:
                    mesh_pos_np = mesh_pos

                # Get current damage from node features (last feature = damage)
                damage = sample.x[:, -1].cpu().numpy()

                # One figure per ply: element-level plies overlap in the XY
                # projection, so each gets its own hex-quad rendering.
                n_plies_viz = len(np.unique(np.round(mesh_pos_np[:, 2], 4)))
                n_plies_viz = max(1, min(n_plies_viz, 16))

                # Global attention: dump the per-node field, plot each ply
                try:
                    ga_stats = attn_model.get_global_attention_stats(sample_dev.mesh_pos)
                    # Additive dump of per-node attention scalars so the field can
                    # be rendered tikz-native (same mesh recipe as FI/damage).
                    try:
                        np.savez(
                            attn_dir / f"attn_field_frame_{idx:04d}.npz",
                            received_attn=ga_stats['received_attn'].cpu().numpy(),
                            received_max=ga_stats['received_max'].cpu().numpy(),
                            entropy=ga_stats['entropy'].cpu().numpy(),
                            long_range_ratio=ga_stats['long_range_ratio'].cpu().numpy(),
                            mesh_pos=mesh_pos_np,
                            damage=damage,
                            frame_idx=int(idx),
                        )
                    except Exception as _dump_err:
                        print(f"    attn npz dump skipped ({_dump_err})")
                    for p in range(n_plies_viz):
                        suffix = f"_ply{p}" if n_plies_viz > 1 else ""
                        plot_global_attention_analysis(
                            mesh_pos=mesh_pos_np,
                            entropy=ga_stats['entropy'].cpu().numpy(),
                            max_weight=ga_stats['max_weight'].cpu().numpy(),
                            attn_reach=ga_stats['attn_reach'].cpu().numpy(),
                            long_range_ratio=ga_stats['long_range_ratio'].cpu().numpy(),
                            damage=damage,
                            received_attn=ga_stats['received_attn'].cpu().numpy(),
                            received_max=ga_stats['received_max'].cpu().numpy(),
                            received_entropy=ga_stats['received_entropy'].cpu().numpy(),
                            received_reach=ga_stats['received_reach'].cpu().numpy(),
                            save_path=attn_dir / f"global_attention_frame_{idx:04d}{suffix}.png",
                            title=f"Global Attention Analysis, frame {idx}, ply {p + 1}",
                            data_template=sample,
                            ply=p,
                        )
                    print(f"  Frame {idx}: global attention analysis saved "
                          f"({n_plies_viz} plies)")
                except Exception as ga_err:
                    print(f"  Frame {idx}: global attention analysis skipped ({ga_err})")

            attn_model.enable_attention_analysis(False)
            del attn_model
            torch.cuda.empty_cache()
            print(f"  Attention analysis plots saved to: {attn_dir}")
        except Exception as e:
            import traceback
            print(f"  Warning: could not generate attention analysis plots: {e}")
            traceback.print_exc()
    
    # ── Persist rollout arrays for offline rescoring (tools/rescore_r2.py) ──
    # Wrapped in try/except so a save failure never blocks test metrics.
    if rollout_results is not None and len(rollout_results) > 0:
        try:
            import numpy as _np
            _save_keys = (
                "predicted_damage", "ground_truth_damage",
                "predicted_rates", "ground_truth_rates",
                "predicted_matrix_damage", "ground_truth_matrix_damage",
                "sim_ids_per_frame", "times",
            )
            _payload = {}
            for _k in _save_keys:
                _v = rollout_results.get(_k)
                if _v is None:
                    continue
                _arr = _np.asarray(_v)
                if _arr.dtype == object:
                    # Two legitimate object-array shapes:
                    #   (a) list of ints (sim_ids_per_frame) → cast to int64
                    #   (b) ragged 2D (per-frame arrays of varying width when
                    #       plates have different node counts) → keep as
                    #       object dtype, np.savez handles it via pickling.
                    try:
                        _arr = _np.asarray(_v, dtype=_np.int64)
                    except Exception:
                        # Ragged frame arrays: keep the object dtype
                        pass
                _payload[_k] = _arr
            if _payload:
                _npz = output_dir / "rollout_arrays.npz"
                # Use np.savez (not _compressed) for object arrays so
                # allow_pickle=True works on load.
                _np.savez(_npz, **_payload)
                print(f"  [rollout] saved arrays for rescoring: {_npz} "
                      f"({len(_payload)} fields)")
        except Exception as _re:
            print(f"  [rollout] save skipped: {_re}")

    # Per-simulation VTU export, then per-simulation GIFs. GTP_SKIP_ANIM=1 skips
    # only the GIFs; the VTUs and rollout_arrays.npz are always written.
    if rollout_results is not None and len(rollout_results) > 0:
        animation_dir = output_dir / "damage_evolution"

        # --- Split rollout arrays by simulation ---
        sim_ids = rollout_results.get('sim_ids_per_frame', [])
        if not sim_ids:
            # Fallback: treat all frames as a single simulation
            sim_ids = [0] * len(rollout_results['predicted_damage'])

        # Build a mapping: sim_id -> list of frame indices
        from collections import OrderedDict
        sim_frame_map = OrderedDict()
        for frame_idx, sid in enumerate(sim_ids):
            sim_frame_map.setdefault(sid, []).append(frame_idx)

        # Build sim_id -> first sample in test_dataset for mesh_pos / data_template
        sim_sample_map = {}
        for sample in test_dataset:
            sid = int(sample.sample_idx.item()) if hasattr(sample, 'sample_idx') else 0
            if sid not in sim_sample_map:
                sim_sample_map[sid] = sample

        from .visualization.animations import create_multi_field_animations
        _is_two_stage = getattr(config, 'two_stage_enabled', False)

        # ── Per-sim Paraview/Blender VTU export ─────────────────────────
        # Runs before the animation block so failures here don't kill viz,
        # and an animation crash later doesn't prevent VTUs from being saved.
        vtk_dir = output_dir / "vtk"
        for sim_id, frame_indices in sim_frame_map.items():
            data_tmpl = sim_sample_map.get(sim_id, test_dataset[0])
            try:
                from .visualization.vtk_export import export_sim_to_vtu
                export_sim_to_vtu(
                    sim_id=sim_id,
                    frame_indices=frame_indices,
                    rollout_results=rollout_results,
                    data_template=data_tmpl,
                    output_dir=vtk_dir / f"sim_{sim_id:03d}",
                    is_two_stage=_is_two_stage,
                )
            except Exception as _vtk_err:
                print(f"  [vtk_export] sim {sim_id} failed: {_vtk_err}")

        if os.environ.get("GTP_SKIP_ANIM"):
            print("\n[Per-Simulation GIFs skipped: GTP_SKIP_ANIM set; VTUs written]")
        else:
            print("\n[Generating Per-Simulation Animations]")
            # GTP_ANIM_SIMS="13" or "13,22" renders GIFs only for those
            # simulations; unset renders all. The VTUs and rollout_arrays.npz
            # are written for every simulation either way.
            _only = os.environ.get("GTP_ANIM_SIMS", "").replace(",", " ").split()
            if _only:
                _keep = {int(t) for t in _only}
                _skipped = [s for s in sim_frame_map if int(s) not in _keep]
                print(f"  [GTP_ANIM_SIMS] rendering only {sorted(_keep)}; "
                      f"skipping {len(_skipped)} other simulation(s)")
                sim_frame_map = OrderedDict(
                    (k, v) for k, v in sim_frame_map.items() if int(k) in _keep)
                if not sim_frame_map:
                    print("  [GTP_ANIM_SIMS] WARNING: no test simulation matched; "
                          "nothing will be animated")
            for sim_id, frame_indices in sim_frame_map.items():
                sim_dir = animation_dir / f"sim_{sim_id:03d}"
                data_tmpl = sim_sample_map.get(sim_id, test_dataset[0])
                print(f"  Simulation {sim_id}: {len(frame_indices)} frames -> {sim_dir}")

                multi_result = create_multi_field_animations(
                    rollout_results=rollout_results,
                    sim_id=sim_id,
                    frame_indices=frame_indices,
                    data_template=data_tmpl,
                    output_dir=sim_dir,
                    fps=5,
                    is_two_stage=_is_two_stage,
                )

                for group_name, gif_path in multi_result.items():
                    print(f"    {group_name}: {gif_path}")
    
    # ── Laminate through-thickness analysis plots ────────────────────
    try:
        from .visualization.laminate_plots import generate_laminate_test_plots
        generate_laminate_test_plots(
            test_dataset=test_dataset,
            predictions=predictions,
            targets=targets,
            rollout_results=rollout_results,
            output_dir=output_dir,
            n_spatial_frames=5,
        )
    except Exception as e:
        import traceback
        print(f"  Warning: laminate analysis failed: {e}")
        traceback.print_exc()

    # Rollout metrics: R² of the failure index and of the damage variable
    from .evaluation.metrics import format_rollout_metrics
    metric_lines = format_rollout_metrics(metrics)
    print(f"\n{'='*70}")
    print("TEST RESULTS (autoregressive rollout)")
    print("=" * 70)
    for line in metric_lines:
        print(f"  {line}")
    print("=" * 70)

    with open(output_dir / "test_metrics.txt", "w") as f:
        f.write("=" * 60 + "\n")
        f.write("TEST RESULTS (autoregressive rollout)\n")
        f.write("=" * 60 + "\n\n")
        for line in metric_lines:
            f.write(line + "\n")
        f.write("\n" + "=" * 60 + "\n")
        f.write(f"\nCheckpoint: {checkpoint_path}\n")
        f.write(f"Stats: {stats_path}\n")
        f.write(f"Test samples: {len(test_dataset)}\n")

    # ── Unified resume-aware loss plot ─────────────────────────────────
    # Auto-invokes tools/plot_resume_history.py for every test. For non-resumed
    # runs it still produces a single-segment loss plot (no boundaries). Wrapped
    # in try/except so plotting never breaks test metric persistence.
    try:
        import importlib.util as _ilu
        _run_dir = output_dir.parent
        _script = Path(root_dir) / "tools" / "plot_resume_history.py"
        if _script.exists():
            _spec = _ilu.spec_from_file_location("_plot_resume_history", str(_script))
            _mod = _ilu.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)
            _epochs, _train, _val, _bounds = _mod.stitch(_run_dir)
            _tm = _mod.load_test_metrics(output_dir)
            _png = _mod.render(_run_dir, _epochs, _train, _val, _bounds, _tm,
                               "loss_plot_resumed", write_pdf=False)
            print(f"  [unified-plot] wrote: {_png}  (resume boundaries: "
                  f"{_bounds if _bounds else 'none'})")
        else:
            print(f"  [unified-plot] skipped: {_script} not found")
    except Exception as _upe:
        print(f"  [unified-plot] skipped: {_upe}")

    print(f"\n{'='*70}")
    print(f"{mode_label} COMPLETED SUCCESSFULLY!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*70}")


def _run_optimization(args, config, dataset, root_dir):
    """Run Optuna hyperparameter optimization, then full training with best params.
    
    Must be called without torchrun (single process); each trial, and the final
    training, is launched as a torchrun subprocess.
    
    Creates a proper run folder structure:
        gptrun_XXX/
            hyperparameter_optimization_results/
                trial_000/, trial_001/, ...
                optuna_results.json
                optimization_plots.png
            best_models/
            plots/
            execution_log_*.txt
    """
    from .optimization import run_optimization
    import subprocess
    import json
    
    # Create a proper run folder (like training does)
    run_folder = create_results_folder(prefix=args.run_name or "optuna_hpo", mode="train")
    hpo_results_dir = run_folder / "hyperparameter_optimization_results"
    # Already created by create_results_folder, but ensure it exists
    hpo_results_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("OPTUNA HYPERPARAMETER OPTIMIZATION")
    print("=" * 70)
    print(f"Run folder: {run_folder}")
    print(f"HPO results: {hpo_results_dir}")
    print("Running HPO trials as DDP subprocesses...")
    
    # Run optimization - each trial runs as DDP subprocess
    # Trial outputs go to hpo_results_dir/trial_XXX/
    # Build extra CLI flags to forward to each trial subprocess
    extra_train_args = []
    if args.dataset:
        extra_train_args.extend(["--dataset", str(args.dataset)])
    if args.config:
        extra_train_args.extend(["--config", str(args.config)])
    # Forward the flags that define the head and the rate cap, so that the
    # trials train the same model as the final training
    if getattr(args, "two_stage", None):
        extra_train_args.append("--two-stage")
    if getattr(args, "rate_cap", None) is not None:
        extra_train_args.extend(["--rate-cap", str(args.rate_cap)])

    results = run_optimization(
        n_trials=args.n_trials,
        timeout=args.timeout,
        epochs_per_trial=args.epochs_per_trial,
        patience=config.patience or 10,
        output_dir=str(hpo_results_dir),
        verbose=not args.quiet,
        extra_train_args=extra_train_args,
    )
    
    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE!")
    print("=" * 70)
    
    best_params = results.get("best_params", {})
    best_value = results.get("best_value")
    n_completed = results.get("n_completed", 0)
    n_trials = results.get("n_trials", args.n_trials)
    n_pruned = results.get("n_pruned", 0)
    
    if not best_params or best_value is None:
        print("\n⚠️  WARNING: No trials completed successfully!")
        print(f"Trials attempted: {n_trials}")
        print(f"Trials failed: {n_trials - n_completed}")
        print(f"Trials pruned (OOM): {n_pruned}")
        print(f"\nCheck trial logs in: {hpo_results_dir}")
        print("\nPossible causes:")
        print("  1. Out of memory - try smaller hidden_dim or batch_size")
        print("  2. Training code errors - check training.log in trial folders")
        print("  3. Dataset issues - verify dataset loads correctly")
        return  # Exit early, don't try to run full training
    
    print("\nBest parameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    print(f"\nBest validation loss: {best_value:.6f}")
    print(f"Trials completed: {n_completed}/{n_trials}")
    print(f"Trials pruned (OOM): {n_pruned}")
    print(f"\nResults saved to: {hpo_results_dir}")

    if getattr(args, "skip_final_train", False):
        print("\n--skip-final-train set: not launching the best-params "
              "training. Read best_params from optuna_results.json and "
              "start the full training manually.")
        return

    # Now launch full DDP training with best parameters
    # Output goes to the SAME run folder
    print("\n" + "=" * 70)
    print("LAUNCHING FULL DDP TRAINING WITH BEST PARAMETERS")
    print("=" * 70)
    
    # Detect number of GPUs
    n_gpus = torch.cuda.device_count()
    print(f"Detected {n_gpus} GPUs for DDP training")
    print(f"Training output: {run_folder}")
    
    # Build torchrun command with best hyperparameters
    cmd = [
        "torchrun",
        f"--nproc_per_node={n_gpus}",
        "--standalone",
        "-m", "src.gtp",
        "--mode", "train",
        "--run-name", "best_params",
        "--output-dir", str(run_folder),  # Use same run folder
    ]
    
    # Forward dataset path (resolved by main())
    if args.dataset:
        cmd.extend(["--dataset", str(args.dataset)])
    
    # Forward config file
    if args.config:
        cmd.extend(["--config", str(args.config)])
    
    # Add best hyperparameters
    if "hidden_dim" in best_params:
        cmd.extend(["--hidden-dim", str(best_params["hidden_dim"])])
    if "num_layers" in best_params:
        cmd.extend(["--num-layers", str(best_params["num_layers"])])
    if "learning_rate" in best_params:
        cmd.extend(["--learning-rate", str(best_params["learning_rate"])])
    if "batch_size" in best_params:
        cmd.extend(["--batch-size", str(best_params["batch_size"])])
    if "dropout_rate" in best_params:
        cmd.extend(["--dropout", str(best_params["dropout_rate"])])
    if "attention_freq" in best_params:
        cmd.extend(["--attention-freq", str(best_params["attention_freq"])])
    if "weight_decay" in best_params:
        cmd.extend(["--weight-decay", str(best_params["weight_decay"])])
    
    # Add epochs and patience
    epochs = args.epochs if args.epochs else DEFAULTS.get("epochs", 200)
    patience = config.patience if config.patience else 30
    cmd.extend(["--epochs", str(epochs)])
    cmd.extend(["--patience", str(patience)])
    
    # Forward scheduler settings
    if args.scheduler:
        cmd.extend(["--scheduler", str(args.scheduler)])
    cmd.extend(["--plateau-patience", str(args.plateau_patience)])
    cmd.extend(["--plateau-factor", str(args.plateau_factor)])
    cmd.extend(["--min-lr", str(args.min_lr)])

    # Save the command for reference
    cmd_file = hpo_results_dir / "best_training_command.sh"
    with open(cmd_file, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# Auto-generated training command with best HPO parameters\n")
        f.write(f"# Best validation loss: {results['best_value']:.6f}\n\n")
        
        # Add loss weight exports
        if "damage_loss_weight" in best_params:
            f.write(f"# damage_loss_weight: {best_params['damage_loss_weight']}\n")
        f.write("\n" + " ".join(cmd) + "\n")
    
    print(f"\nTraining command saved to: {cmd_file}")
    print(f"Command: {' '.join(cmd)}\n")
    
    # Update DEFAULTS for loss weights before spawning
    # (These need to be set in environment or config file for subprocess)
    config_updates = {}
    if "damage_loss_weight" in best_params:
        config_updates["damage_loss_weight"] = best_params["damage_loss_weight"]
    
    # Save config updates to a temp file that training can read
    config_file = hpo_results_dir / "best_config.json"
    with open(config_file, "w") as f:
        json.dump(config_updates, f, indent=2)
    
    # Add config file to command
    # Note: We'll need to implement config loading for loss weights
    # For now, just run the training
    
    print("Starting DDP training subprocess...")
    print("-" * 70)
    
    # Run training
    try:
        subprocess.run(cmd, check=True)
        print("\n" + "=" * 70)
        print("OPTIMIZATION + TRAINING COMPLETE!")
        print(f"All results in: {run_folder}")
        print("=" * 70)
    except subprocess.CalledProcessError as e:
        print(f"\nTraining subprocess failed with exit code {e.returncode}")
        print("You can manually run the saved command:")
        print(f"  bash {cmd_file}")
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
