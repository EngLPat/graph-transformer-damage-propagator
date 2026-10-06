"""
Hyperparameter optimization using Optuna with DDP support.

Each trial runs as a full DDP training (one trial at a time using all GPUs).
This is more robust than running trials in parallel and avoids OOM issues.

Usage:
    python -m src.gtp --mode optimize --n-trials 20 --epochs-per-trial 50 --epochs 200

This will:
1. Run N Optuna trials, each as a subprocess with full DDP training
2. Each trial uses all available GPUs
3. OOM errors are caught and trial is pruned gracefully
4. After optimization, automatically runs full training with best params
"""

import json
import logging
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


def get_optuna_search_space() -> Dict[str, Any]:
    """
    Define the hyperparameter search space.
    
    Returns:
        Dictionary mapping parameter names to their search ranges/choices.
    """
    return {
        # Model architecture
        "hidden_dim": {"type": "categorical", "choices": [64, 96, 128]},
        "num_layers": {"type": "int", "low": 4, "high": 10},
        # Global attention runs at layers i % freq == 0, i.e. ceil(num_layers
        # / freq) passes; freq >= num_layers gives a single pass at layer 0
        "attention_freq": {"type": "int", "low": 2, "high": 10},
        "dropout_rate": {"type": "float", "low": 0.0, "high": 0.3},

        # Training. Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True in
        # the parent shell so that the trial subprocesses inherit it.
        "learning_rate": {"type": "loguniform", "low": 1e-5, "high": 1e-3},
        "weight_decay": {"type": "loguniform", "low": 1e-6, "high": 1e-3},
        "batch_size": {"type": "categorical", "choices": [2, 4]},

        # Loss weights are not tuned; the defaults are used
    }


def sample_hyperparameters(trial) -> Dict[str, Any]:
    """Sample hyperparameters from Optuna trial."""
    search_space = get_optuna_search_space()
    params = {}
    
    for name, spec in search_space.items():
        if spec["type"] == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
        elif spec["type"] == "int":
            params[name] = trial.suggest_int(name, spec["low"], spec["high"])
        elif spec["type"] == "float":
            params[name] = trial.suggest_float(name, spec["low"], spec["high"])
        elif spec["type"] == "loguniform":
            params[name] = trial.suggest_float(name, spec["low"], spec["high"], log=True)
    
    return params


def run_single_trial_ddp(
    params: Dict[str, Any],
    trial_number: int,
    epochs: int,
    patience: int,
    output_dir: Path,
    n_gpus: int = 4,
    extra_train_args: Optional[List[str]] = None,
) -> Tuple[Optional[float], bool]:
    """
    Run a single Optuna trial as a DDP subprocess.
    
    Each trial runs full DDP training with the sampled hyperparameters.
    This ensures memory is fully cleared between trials.
    
    Args:
        params: Hyperparameters for this trial
        trial_number: Trial index
        epochs: Epochs for this trial
        patience: Early stopping patience
        output_dir: Directory to save trial results
        n_gpus: Number of GPUs for DDP
        
    Returns:
        Tuple of (best_val_loss, oom_occurred)
    """
    trial_dir = output_dir / f"trial_{trial_number:03d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories that training expects
    (trial_dir / "best_models").mkdir(exist_ok=True)
    (trial_dir / "plots").mkdir(exist_ok=True)
    (trial_dir / "training_results").mkdir(exist_ok=True)
    
    # Save params to file for the subprocess to read
    params_file = trial_dir / "params.json"
    with open(params_file, "w") as f:
        json.dump(params, f, indent=2)
    
    # Build training command
    cmd = [
        "torchrun",
        f"--nproc_per_node={n_gpus}",
        "--standalone",
        "-m", "src.gtp",
        "--mode", "train",
        "--run-name", f"optuna_trial_{trial_number:03d}",
        "--output-dir", str(trial_dir),
        "--epochs", str(epochs),
        "--patience", str(patience),
        "--quiet",  # Suppress verbose output
    ]
    
    # Add hyperparameters
    if "hidden_dim" in params:
        cmd.extend(["--hidden-dim", str(params["hidden_dim"])])
    if "num_layers" in params:
        cmd.extend(["--num-layers", str(params["num_layers"])])
    if "learning_rate" in params:
        cmd.extend(["--learning-rate", str(params["learning_rate"])])
    if "batch_size" in params:
        cmd.extend(["--batch-size", str(params["batch_size"])])
    if "dropout_rate" in params:
        cmd.extend(["--dropout", str(params["dropout_rate"])])
    if "attention_freq" in params:
        cmd.extend(["--attention-freq", str(params["attention_freq"])])
    if "weight_decay" in params:
        cmd.extend(["--weight-decay", str(params["weight_decay"])])
    
    # Forward extra CLI args (dataset, config, etc.)
    if extra_train_args:
        cmd.extend(extra_train_args)
    
    # Save command for debugging
    cmd_file = trial_dir / "command.sh"
    with open(cmd_file, "w") as f:
        f.write("#!/bin/bash\n")
        f.write(" ".join(cmd) + "\n")
    
    # Set environment variables for loss weights
    env = os.environ.copy()
    if "damage_loss_weight" in params:
        env["GTP_DAMAGE_LOSS_WEIGHT"] = str(params["damage_loss_weight"])
    
    # Run training
    log_file = trial_dir / "training.log"
    oom_occurred = False
    best_val_loss = None
    
    try:
        with open(log_file, "w") as log_f:
            result = subprocess.run(
                cmd,
                env=env,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                # Per-trial wall cap. 2h default suits reduced datasets;
                # full/medium-fidelity HPO must raise it via env, e.g.
                # GTP_TRIAL_TIMEOUT_SECONDS=10800 for 3h trials.
                timeout=int(os.environ.get("GTP_TRIAL_TIMEOUT_SECONDS",
                                           3600 * 2)),
            )
        
        if result.returncode != 0:
            # Check if OOM or other error
            with open(log_file, "r") as f:
                log_content = f.read()
            if "out of memory" in log_content.lower() or ("cuda" in log_content.lower() and "oom" in log_content.lower()):
                oom_occurred = True
                logger.warning(f"Trial {trial_number}: OOM detected")
            else:
                logger.warning(f"Trial {trial_number}: Failed with exit code {result.returncode}")
                # Print last 20 lines of log for debugging
                log_lines = log_content.strip().split('\n')
                if log_lines:
                    print("    Last lines of log:")
                    for line in log_lines[-15:]:
                        print(f"      {line}")
        else:
            # Try to extract best validation loss from results
            results_file = trial_dir / "training_results.json"
            if results_file.exists():
                with open(results_file, "r") as f:
                    trial_results = json.load(f)
                best_val_loss = trial_results.get("best_val_loss")
            else:
                # Try to parse from log file
                with open(log_file, "r") as f:
                    log_content = f.read()
                # Look for "Best Val:" pattern
                import re
                match = re.search(r"Best Val[:\s]+([0-9.]+)", log_content, re.IGNORECASE)
                if match:
                    best_val_loss = float(match.group(1))
                    
    except subprocess.TimeoutExpired:
        logger.warning(f"Trial {trial_number}: Timed out after 2 hours")
        oom_occurred = True
    except Exception as e:
        logger.error(f"Trial {trial_number}: Exception {e}")
        oom_occurred = True
    
    return best_val_loss, oom_occurred


def run_optimization(
    n_trials: int = 20,
    timeout: Optional[int] = None,
    epochs_per_trial: int = 50,
    patience: int = 15,
    output_dir: str = "optuna_results",
    n_gpus: Optional[int] = None,
    verbose: bool = True,
    extra_train_args: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Run Optuna hyperparameter optimization with DDP trials.
    
    Each trial runs as a separate subprocess with full DDP training.
    This is more robust than in-process trials because:
    1. Memory is fully cleared between trials
    2. OOM in one trial doesn't kill the optimization
    3. Each trial uses all available GPUs efficiently
    
    Args:
        n_trials: Number of trials to run
        timeout: Maximum total time in seconds (None = no limit)
        epochs_per_trial: Epochs per trial
        patience: Early stopping patience per trial
        output_dir: Directory to save all results
        n_gpus: Number of GPUs (auto-detected if None)
        verbose: Print progress
        
    Returns:
        Dictionary with best hyperparameters and study info
    """
    try:
        import optuna
        from optuna.pruners import MedianPruner
        from optuna.samplers import TPESampler
    except ImportError:
        raise ImportError("Optuna not installed. Run: pip install optuna")
    
    # Setup output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Detect GPUs
    if n_gpus is None:
        n_gpus = torch.cuda.device_count()
    
    if verbose:
        print(f"\n{'='*70}")
        print("OPTUNA HYPERPARAMETER OPTIMIZATION")
        print(f"{'='*70}")
        print(f"Trials: {n_trials}")
        print(f"Epochs per trial: {epochs_per_trial}")
        print(f"GPUs per trial: {n_gpus} (full DDP)")
        print(f"Output: {output_path}")
        print(f"{'='*70}\n")
    
    # Create Optuna study
    sampler = TPESampler(seed=42)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    
    study = optuna.create_study(
        study_name="gtp_hpo",
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
    )
    
    # Track results
    trial_results = []
    start_time = time.time()
    
    # Run trials
    for trial_idx in range(n_trials):
        # Check timeout
        if timeout and (time.time() - start_time) > timeout:
            print(f"\nTimeout reached after {trial_idx} trials")
            break
        
        # Create trial and sample params
        trial = study.ask()
        params = sample_hyperparameters(trial)
        
        if verbose:
            print(f"\n{'─'*70}")
            print(f"Trial {trial_idx + 1}/{n_trials}")
            print(f"Parameters: {params}")
            print(f"{'─'*70}")
        
        # Run trial as DDP subprocess
        best_val_loss, oom_occurred = run_single_trial_ddp(
            params=params,
            trial_number=trial_idx,
            epochs=epochs_per_trial,
            patience=patience,
            output_dir=output_path,
            n_gpus=n_gpus,
            extra_train_args=extra_train_args,
        )
        
        # Report to Optuna
        if oom_occurred:
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            if verbose:
                print("  → PRUNED (OOM or error)")
        elif best_val_loss is None:
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            if verbose:
                print("  → FAILED (no result)")
        else:
            study.tell(trial, best_val_loss)
            if verbose:
                print(f"  → Validation Loss: {best_val_loss:.6f}")
        
        # Record
        trial_results.append({
            "number": trial_idx,
            "params": params,
            "value": best_val_loss,
            "state": "PRUNED" if oom_occurred else ("FAIL" if best_val_loss is None else "COMPLETE"),
        })
        
        # Clear GPU memory between trials
        torch.cuda.empty_cache()
        time.sleep(2)  # Brief pause to let system settle
    
    # Get best trial
    completed_trials = [t for t in trial_results if t["state"] == "COMPLETE"]
    
    if not completed_trials:
        print("\nWARNING: No trials completed successfully!")
        best_params = {}
        best_value = None
    else:
        best_trial = min(completed_trials, key=lambda t: t["value"])
        best_params = best_trial["params"]
        best_value = best_trial["value"]
        
        if verbose:
            print(f"\n{'='*70}")
            print("OPTIMIZATION COMPLETE")
            print(f"{'='*70}")
            print(f"Best trial: {best_trial['number']}")
            print(f"Best validation loss: {best_value:.6f}")
            print("Best parameters:")
            for k, v in best_params.items():
                print(f"  {k}: {v}")
            print(f"{'='*70}\n")
    
    # Save results
    results = {
        "best_params": best_params,
        "best_value": best_value,
        "n_trials": len(trial_results),
        "n_completed": len([t for t in trial_results if t["state"] == "COMPLETE"]),
        "n_pruned": len([t for t in trial_results if t["state"] == "PRUNED"]),
        "n_failed": len([t for t in trial_results if t["state"] == "FAIL"]),
        "search_space": get_optuna_search_space(),
        "timestamp": datetime.now().isoformat(),
        "all_trials": trial_results,
    }
    
    with open(output_path / "optuna_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Generate plots
    _save_optimization_plots(study, results, output_path)
    
    return results


def _save_optimization_plots(study, results: Dict, output_path: Path):
    """Save optimization visualization plots."""
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Optuna Hyperparameter Optimization Results", fontsize=14, fontweight="bold")
        
        # Plot 1: Optimization history
        ax = axes[0, 0]
        completed = [t for t in results["all_trials"] if t["value"] is not None]
        if completed:
            trial_numbers = [t["number"] for t in completed]
            values = [t["value"] for t in completed]
            ax.plot(trial_numbers, values, "b.-", alpha=0.7, label="Trial value")
            
            # Best so far
            best_so_far = []
            current_best = float("inf")
            for v in values:
                current_best = min(current_best, v)
                best_so_far.append(current_best)
            ax.plot(trial_numbers, best_so_far, "r-", linewidth=2, label="Best so far")
        ax.set_xlabel("Trial")
        ax.set_ylabel("Validation Loss")
        ax.set_title("Optimization History")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Parameter importance
        ax = axes[0, 1]
        try:
            import optuna
            if len([t for t in study.trials if t.value is not None]) >= 5:
                importances = optuna.importance.get_param_importances(study)
                params_sorted = sorted(importances.items(), key=lambda x: x[1], reverse=True)
                param_names = [p[0] for p in params_sorted]
                param_values = [p[1] for p in params_sorted]
                ax.barh(param_names, param_values, color="steelblue")
                ax.set_xlabel("Importance")
                ax.set_title("Parameter Importance")
        except Exception as e:
            ax.text(0.5, 0.5, f"Importance unavailable:\n{e}", ha="center", va="center", transform=ax.transAxes)
        
        # Plot 3: Best parameters
        ax = axes[1, 0]
        ax.axis("off")
        if results["best_params"]:
            text = "Best Parameters:\n" + "-" * 40 + "\n"
            for k, v in results["best_params"].items():
                if isinstance(v, float):
                    text += f"{k}: {v:.6g}\n"
                else:
                    text += f"{k}: {v}\n"
            text += "-" * 40 + f"\nBest validation loss: {results['best_value']:.6f}"
            ax.text(0.1, 0.9, text, transform=ax.transAxes, fontfamily="monospace", fontsize=10, verticalalignment="top")
        ax.set_title("Best Configuration")
        
        # Plot 4: Trial summary
        ax = axes[1, 1]
        states = ["Completed", "Pruned (OOM)", "Failed"]
        counts = [results["n_completed"], results["n_pruned"], results["n_failed"]]
        colors = ["green", "orange", "red"]
        bars = ax.bar(states, counts, color=colors)
        ax.set_ylabel("Count")
        ax.set_title(f"Trial Summary (Total: {results['n_trials']})")
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, str(count), ha="center", va="bottom")
        
        plt.tight_layout()
        plt.savefig(output_path / "optuna_results.pdf", dpi=150, bbox_inches="tight")
        plt.savefig(output_path / "optuna_results.png", dpi=150, bbox_inches="tight")
        plt.close()
        
        logger.info(f"Saved plots to {output_path}")
        
    except Exception as e:
        logger.warning(f"Could not save plots: {e}")


__all__ = ["run_optimization", "get_optuna_search_space"]
