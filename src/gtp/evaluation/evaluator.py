"""
Model evaluation and testing.

Provides the Evaluator class for running inference and computing metrics.
"""

import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from ..config import Config
from ..models import FCLGA_GraphTransformer
from .metrics import rollout_metrics


class Evaluator:
    """
    Model evaluator for running inference and computing metrics.
    
    Args:
        config: Configuration object
        device: Device for inference
    """
    
    def __init__(self, config: Config, device: str = "cuda:0"):
        self.config = config
        self.device = device
    
    def load_model(self, checkpoint_path: Path, num_node_features: int, num_edge_features: int) -> FCLGA_GraphTransformer:
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            num_node_features: Number of input node features
            num_edge_features: Number of input edge features
        
        Returns:
            Loaded model
        """
        checkpoint_path = Path(checkpoint_path)
        
        # Load state dict first to infer architecture
        checkpoint_raw = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        # Handle both raw state_dict and wrapped checkpoint (e.g. last_checkpoint.pt)
        if isinstance(checkpoint_raw, dict) and 'model_state_dict' in checkpoint_raw:
            state_dict = checkpoint_raw['model_state_dict']
            # Extract saved config from checkpoint if available
            ckpt_config = checkpoint_raw.get('config', {})
            if isinstance(ckpt_config, dict):
                for key in ('hidden_dim', 'num_layers', 'num_heads', 'attention_freq'):
                    if key in ckpt_config and ckpt_config[key] is not None:
                        cur = getattr(self.config, key, None)
                        if cur != ckpt_config[key]:
                            print(f"\u26a0\ufe0f  Overriding config.{key}: {cur} \u2192 {ckpt_config[key]} (from checkpoint)")
                            setattr(self.config, key, ckpt_config[key])
        else:
            state_dict = checkpoint_raw
        
        # Infer num_layers from checkpoint keys
        # Processor layers are named "processor.0", "processor.1", etc.
        processor_indices = []
        for key in state_dict.keys():
            if key.startswith("processor.") and "." in key[10:]:
                # Extract the layer index (e.g., "processor.6.attention..." -> 6)
                try:
                    idx = int(key.split(".")[1])
                    processor_indices.append(idx)
                except ValueError:
                    continue
        
        if processor_indices:
            inferred_num_layers = max(processor_indices) + 1
            if inferred_num_layers != self.config.num_layers:
                print(f"⚠️  Checkpoint has {inferred_num_layers} layers, config has {self.config.num_layers}.")
                print(f"   Overriding config.num_layers = {inferred_num_layers}")
                self.config.num_layers = inferred_num_layers
        
        # Infer hidden_dim from checkpoint (node_encoder.0.weight has shape [hidden_dim, input_dim])
        enc_key = "node_encoder.0.weight"
        if enc_key in state_dict:
            inferred_hidden_dim = state_dict[enc_key].shape[0]
            if inferred_hidden_dim != self.config.hidden_dim:
                print(f"\u26a0\ufe0f  Checkpoint has hidden_dim={inferred_hidden_dim}, config has {self.config.hidden_dim}.")
                print(f"   Overriding config.hidden_dim = {inferred_hidden_dim}")
                self.config.hidden_dim = inferred_hidden_dim
            ckpt_input_dim = state_dict[enc_key].shape[1]
            if ckpt_input_dim != num_node_features:
                raise ValueError(
                    f"Checkpoint expects {ckpt_input_dim} node features but "
                    f"the dataset provides {num_node_features}."
                )
        
        # Two-stage damage head: if the checkpoint carries dmg_decoder but
        # the config says no two-stage, the rebuilt model would lack the head
        # and the strict load below would fail. Enable it from the checkpoint
        # instead (without the head, a rollout would feed back a frozen D=0
        # and saturate FI).
        has_dmg_head = any(k.startswith("dmg_decoder.") for k in state_dict.keys())
        if has_dmg_head and not getattr(self.config, 'two_stage_enabled', False):
            print("⚠️  Checkpoint has dmg_decoder but config.two_stage_enabled"
                  "=False; enabling the two-stage head")
            self.config.two_stage_enabled = True

        model = FCLGA_GraphTransformer(
            input_dim_node=num_node_features,
            input_dim_edge=num_edge_features,
            hidden_dim=self.config.hidden_dim,
            output_dim=1,
            args=self.config
        )
        
        # Strict loading: every tensor of the checkpoint must match the rebuilt
        # model exactly, so an architecture mismatch fails loudly instead of
        # running with default weights.
        model.load_state_dict(state_dict, strict=True)

        model.to(self.device)
        model.eval()
        
        print(f"✓ Model loaded from: {checkpoint_path}")
        
        return model
    
    def run_inference(
        self,
        model: FCLGA_GraphTransformer,
        test_loader: DataLoader,
        stats_list: List[torch.Tensor],
        verbose: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run inference on test dataset.
        
        Args:
            model: Trained model
            test_loader: DataLoader for test data
            stats_list: Normalization statistics
            verbose: Show progress bar
        
        Returns:
            Tuple of (predictions, targets, accumulated_damage, delta_ts) as tensors.
        """
        model.eval()
        
        mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge = stats_list[:4]
        mean_vec_x = mean_vec_x.to(self.device)
        std_vec_x = std_vec_x.to(self.device)
        mean_vec_edge = mean_vec_edge.to(self.device)
        std_vec_edge = std_vec_edge.to(self.device)
        
        all_predictions = []
        all_targets = []
        all_damage = []
        all_delta_t = []
        
        iterator = tqdm(test_loader, desc="Running inference") if verbose else test_loader

        with torch.no_grad():
            for batch in iterator:
                batch = batch.to(self.device)

                pred_rate, _ = model(
                    batch, mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge
                )

                all_predictions.append(pred_rate.flatten().cpu())
                all_targets.append(batch.y.flatten().cpu())
                # Collect the damage state for the regime metrics
                if hasattr(batch, 'x') and batch.x.shape[1] > 7:
                    all_damage.append(batch.x[:, -1].cpu())
                if hasattr(batch, 'delta_t'):
                    all_delta_t.append(batch.delta_t.cpu())
        
        predictions = torch.cat(all_predictions, dim=0)
        targets = torch.cat(all_targets, dim=0)
        accumulated_damage = torch.cat(all_damage, dim=0) if all_damage else torch.tensor([])
        delta_ts = torch.cat(all_delta_t, dim=0) if all_delta_t else torch.tensor([])
        
        return predictions, targets, accumulated_damage, delta_ts
    
    def benchmark_inference(
        self,
        model: FCLGA_GraphTransformer,
        test_dataset: List,
        stats_list: List[torch.Tensor],
        n_warmup: int = 10,
        n_timed: int = 100,
    ) -> Dict[str, float]:
        """
        Benchmark single-step inference time (one graph forward pass).
        
        Performs warmup passes (CUDA kernel compilation, caching) then
        times `n_timed` individual forward passes with CUDA synchronization.
        
        Args:
            model: Trained model
            test_dataset: Test dataset
            stats_list: Normalization statistics
            n_warmup: Number of warmup forward passes to skip
            n_timed: Number of timed forward passes
            
        Returns:
            Dictionary with timing statistics
        """
        model.eval()
        device = self.device
        
        mean_vec_x = stats_list[0].to(device)
        std_vec_x = stats_list[1].to(device)
        mean_vec_edge = stats_list[2].to(device)
        std_vec_edge = stats_list[3].to(device)
        
        n_samples = min(n_warmup + n_timed, len(test_dataset))
        n_warmup = min(n_warmup, n_samples // 2)
        n_timed = n_samples - n_warmup
        
        use_cuda = 'cuda' in str(device)
        
        print(f"\n{'='*70}")
        print("INFERENCE TIMING BENCHMARK")
        print(f"{'='*70}")
        print(f"  Device: {device}")
        print(f"  Warmup passes: {n_warmup}")
        print(f"  Timed passes:  {n_timed}")
        print(f"  Nodes/graph:   {test_dataset[0].x.shape[0]}")
        print(f"  Edges/graph:   {test_dataset[0].edge_index.shape[1]}")
        
        # Warmup
        with torch.no_grad():
            for i in range(n_warmup):
                sample = test_dataset[i % len(test_dataset)].clone().to(device)
                _ = model(sample, mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge)
        
        if use_cuda:
            torch.cuda.synchronize()
        
        # Timed passes
        step_times = []
        with torch.no_grad():
            for i in range(n_timed):
                sample = test_dataset[(n_warmup + i) % len(test_dataset)].clone().to(device)
                
                if use_cuda:
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                
                _ = model(sample, mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge)
                
                if use_cuda:
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                
                step_times.append(t1 - t0)
        
        step_times = np.array(step_times)
        
        results = {
            "mean_ms": float(np.mean(step_times) * 1000),
            "median_ms": float(np.median(step_times) * 1000),
            "std_ms": float(np.std(step_times) * 1000),
            "min_ms": float(np.min(step_times) * 1000),
            "max_ms": float(np.max(step_times) * 1000),
            "p95_ms": float(np.percentile(step_times, 95) * 1000),
            "n_timed": n_timed,
            "n_warmup": n_warmup,
            "nodes_per_graph": int(test_dataset[0].x.shape[0]),
            "edges_per_graph": int(test_dataset[0].edge_index.shape[1]),
        }
        
        print(f"\n  Results ({n_timed} forward passes, batch_size=1):")
        print(f"    Mean:   {results['mean_ms']:.3f} ms/step")
        print(f"    Median: {results['median_ms']:.3f} ms/step")
        print(f"    Std:    {results['std_ms']:.3f} ms")
        print(f"    Min:    {results['min_ms']:.3f} ms")
        print(f"    Max:    {results['max_ms']:.3f} ms")
        print(f"    P95:    {results['p95_ms']:.3f} ms")
        print(f"{'='*70}\n")
        
        return results

    def evaluate_with_visualization(
        self,
        checkpoint_path: Path,
        test_dataset: List,
        stats_list: List[torch.Tensor],
        output_dir: Optional[Path] = None,
        batch_size: int = 1,
        num_rollout_samples: int = None,
        rollout_downsample: int = 3,
    ) -> Tuple[List[Dict], torch.Tensor, torch.Tensor, Optional[Dict]]:
        """
        Complete evaluation pipeline with visualization data.
        
        Args:
            checkpoint_path: Path to model checkpoint
            test_dataset: Test dataset
            stats_list: Normalization statistics
            output_dir: Directory to save results
            batch_size: Batch size for testing
            num_rollout_samples: Number of samples for rollout (None = all)
            rollout_downsample: Take every Nth sample for visualization
        
        Returns:
            Tuple of (rollout metrics, predictions, targets, rollout_results);
            the rollout metrics are the rows of ``metrics.rollout_metrics``.
        """
        # Get feature dimensions from dataset
        num_node_features = test_dataset[0].x.shape[1]
        num_edge_features = test_dataset[0].edge_attr.shape[1]
        
        # Load model
        model = self.load_model(checkpoint_path, num_node_features, num_edge_features)
        
        # Create test loader
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        # Run inference
        print("Running inference on test set...")
        predictions, targets, accumulated_damage, delta_ts = self.run_inference(model, test_loader, stats_list)

        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Collect rollout data for visualization (use all timesteps, downsampled)
        print("\nCollecting data for autoregressive rollout visualization...")
        rollout_results = self._collect_rollout_data(
            model, test_dataset, stats_list,
            num_samples=num_rollout_samples,
            downsample=rollout_downsample
        )
        metrics = rollout_metrics(rollout_results) if rollout_results else []

        return metrics, predictions, targets, rollout_results
    
    def _collect_rollout_data(
        self,
        model: FCLGA_GraphTransformer,
        test_dataset: List,
        stats_list: List[torch.Tensor],
        num_samples: int = None,
        downsample: int = 1
    ) -> Optional[Dict]:
        """
        Collect data for rollout visualisation from an autoregressive rollout.

        The model's own predicted damage is fed back at every timestep instead
        of the ground truth. Every timestep is integrated; only every
        downsample-th frame is stored.
        
        Args:
            model: Trained model
            test_dataset: Test dataset (ordered by time)
            stats_list: Normalization statistics
            num_samples: Number of timesteps to process (None = all)
            downsample: Store every Nth sample for visualization
        
        Returns:
            Dictionary with visualization data or None if not enough data
        """
        if len(test_dataset) < 2:
            return None
        
        mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge = stats_list[:4]
        mean_vec_x = mean_vec_x.to(self.device)
        std_vec_x = std_vec_x.to(self.device)
        mean_vec_edge = mean_vec_edge.to(self.device)
        std_vec_edge = std_vec_edge.to(self.device)
        
        # Use all samples if not specified
        total_samples = len(test_dataset)
        if num_samples is None:
            num_samples = total_samples
        num_samples = min(num_samples, total_samples)
        
        predicted_damage_list = []
        ground_truth_damage_list = []
        predicted_rates_list = []
        ground_truth_rates_list = []
        predicted_states_list = []  # failure states
        ground_truth_states_list = []
        times = []
        sim_boundaries = []  # Frame indices where a new simulation starts
        sim_ids_per_frame = []  # Sim ID for each stored frame
        
        model.eval()

        # Imported once at the top of the rollout for the FI clamp below.
        from src.preprocessing.build_cohesive_dataset import FI_CAP

        # Initialize accumulated damage from first sample's initial state
        first_sample = test_dataset[0]
        accumulated_pred_damage = first_sample.x[:, -1].clone().to(self.device)
        last_sim_id = int(first_sample.sample_idx.item()) if hasattr(first_sample, 'sample_idx') else -1

        # Damage feedback: when x[:, -2] is the damage state, the rollout feeds
        # the predicted damage back into it, next to the FI in x[:, -1]. The
        # column is identified by comparing it with the stored damage state on
        # a frame with non-zero damage (at t=0 every candidate column is zero).
        dmg_in_x = False
        if hasattr(first_sample, 'n_ply_nodes'):
            for probe in range(0, min(len(test_dataset), 200), 10):
                s_probe = test_dataset[probe]
                d_now = getattr(s_probe, 'dmg_now', None)
                if d_now is None or s_probe.x.shape[1] < 2:
                    break
                d_now = d_now.squeeze(-1)
                if float(d_now.abs().max()) <= 0.0:
                    continue                      # nothing to discriminate on yet
                dmg_in_x = bool(torch.allclose(
                    s_probe.x[:, -2].float(), d_now.float(), atol=1e-6))
                break
        if dmg_in_x:
            print(f"  Rollout: D-state feedback ENABLED (x width "
                  f"{first_sample.x.shape[1]}; predicted D fed back into "
                  f"x[:, -2])")
        else:
            print(f"  Rollout: D-state feedback DISABLED (x width "
                  f"{first_sample.x.shape[1]}; x[:, -2] is not the damage "
                  f"state, so ground-truth D is consumed at every step)")

        # Two-stage: also track matrix damage (if model has damage head)
        has_two_stage = hasattr(model, 'two_stage_enabled') and model.two_stage_enabled
        if has_two_stage and hasattr(first_sample, 'dmg_now') and first_sample.dmg_now is not None:
            accumulated_pred_dmg = first_sample.dmg_now.squeeze(-1).clone().to(self.device)
        elif dmg_in_x:
            accumulated_pred_dmg = first_sample.x[:, -2].clone().to(self.device)
        else:
            accumulated_pred_dmg = torch.zeros_like(accumulated_pred_damage)
        predicted_dmg_list = []
        ground_truth_dmg_list = []

        print(f"  Rollout: processing {num_samples} timesteps, storing every {downsample}th frame")

        with torch.no_grad():
            for i in range(num_samples):
                sample = test_dataset[i].clone().to(self.device)

                # ── Detect simulation boundary ──────────────────────
                cur_sim_id = int(sample.sample_idx.item()) if hasattr(sample, 'sample_idx') else -1
                if i > 0 and cur_sim_id != last_sim_id:
                    # New simulation: reset accumulated damage to GT
                    accumulated_pred_damage = sample.x[:, -1].clone().to(self.device)
                    if has_two_stage and hasattr(sample, 'dmg_now') and sample.dmg_now is not None:
                        accumulated_pred_dmg = sample.dmg_now.squeeze(-1).clone().to(self.device)
                    elif dmg_in_x:
                        accumulated_pred_dmg = sample.x[:, -2].clone().to(self.device)
                    else:
                        accumulated_pred_dmg = torch.zeros_like(accumulated_pred_damage)
                    stored_frame = len(predicted_damage_list)
                    sim_boundaries.append(stored_frame)
                    print(f"    Sim boundary at step {i}: sim {last_sim_id} → {cur_sim_id}")
                last_sim_id = cur_sim_id

                # Ground truth at this timestep. clone() is required: on CPU,
                # .numpy() shares storage with sample.x, which is overwritten below.
                gt_damage = sample.x[:, -1].clone().cpu().numpy()
                gt_rate = sample.y.flatten().clone().cpu().numpy()
                if hasattr(sample, 'dmg_now') and sample.dmg_now is not None:
                    gt_dmg = sample.dmg_now.squeeze(-1).clone().cpu().numpy()
                elif dmg_in_x:
                    # GT damage lives in the input column; capture BEFORE the
                    # autoregressive overwrite below.
                    gt_dmg = sample.x[:, -2].clone().cpu().numpy()
                else:
                    gt_dmg = None

                # Autoregressive step: feed back the accumulated prediction
                sample.x[:, -1] = accumulated_pred_damage
                if dmg_in_x:
                    sample.x[:, -2] = accumulated_pred_dmg
                
                # Get model prediction
                pred_rate, _ = model(
                    sample, mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge
                )

                # Time step of this sample
                if not hasattr(sample, 'delta_t'):
                    raise ValueError("test sample has no delta_t; rebuild the dataset "
                                     "with build_cohesive_dataset.py")
                delta_t = sample.delta_t.item()

                # Snapshot the predicted rates
                pred_rate_np = pred_rate.flatten().cpu().numpy()
                
                # Store only the downsampled frames
                if i % downsample == 0:
                    predicted_damage_list.append(accumulated_pred_damage.cpu().numpy().copy())
                    ground_truth_damage_list.append(gt_damage)
                    predicted_rates_list.append(pred_rate_np)
                    ground_truth_rates_list.append(gt_rate)

                    # Two-stage: store matrix damage trajectory
                    if has_two_stage:
                        predicted_dmg_list.append(accumulated_pred_dmg.cpu().numpy().copy())
                        if gt_dmg is not None:
                            ground_truth_dmg_list.append(gt_dmg)

                    # Compute failure states
                    failure_threshold = 0.95
                    pred_states = (accumulated_pred_damage.cpu().numpy() > failure_threshold).astype(float)
                    gt_states = (gt_damage > failure_threshold).astype(float)
                    predicted_states_list.append(pred_states)
                    ground_truth_states_list.append(gt_states)

                    sim_ids_per_frame.append(cur_sim_id)

                    if hasattr(sample, 'current_time'):
                        times.append(sample.current_time.item())
                    else:
                        times.append(i * delta_t)

                # Integrate every timestep, stored or not
                damage_increment = pred_rate.squeeze() * delta_t
                accumulated_pred_damage = accumulated_pred_damage + damage_increment
                accumulated_pred_damage = torch.clamp(accumulated_pred_damage, min=0.0, max=FI_CAP)

                # Two-stage: integrate damage rate (monotonic, bounded [0,1])
                if has_two_stage and model._pred_dmg_rate is not None:
                    dmg_increment = model._pred_dmg_rate.squeeze() * delta_t
                    accumulated_pred_dmg = accumulated_pred_dmg + dmg_increment
                    accumulated_pred_dmg = torch.clamp(accumulated_pred_dmg, min=0.0, max=1.0)

        if sim_boundaries:
            print(f"  Simulation boundaries at visualization frames: {sim_boundaries}")
        
        # Frames may have different node counts when the dataset comes from
        # LHS-sampled geometries (each plate has a different mesh size).  In
        # that case np.array(list_of_arrays) raises since numpy can't build a
        # regular 2D array; fall back to a 1D object-dtype array of arrays.
        def _safe_arr(lst):
            if not lst:
                return np.array([])
            try:
                return np.array(lst)
            except ValueError:
                out = np.empty(len(lst), dtype=object)
                for i, x in enumerate(lst):
                    out[i] = np.asarray(x)
                return out

        result = {
            'predicted_damage': _safe_arr(predicted_damage_list),
            'ground_truth_damage': _safe_arr(ground_truth_damage_list),
            'predicted_rates': _safe_arr(predicted_rates_list),
            'ground_truth_rates': _safe_arr(ground_truth_rates_list),
            'predicted_states': _safe_arr(predicted_states_list),
            'ground_truth_states': _safe_arr(ground_truth_states_list),
            'times': np.array(times),
            'sim_boundaries': sim_boundaries,
            'sim_ids_per_frame': sim_ids_per_frame,
        }
        if has_two_stage and predicted_dmg_list:
            result['predicted_matrix_damage'] = _safe_arr(predicted_dmg_list)
            if ground_truth_dmg_list:
                result['ground_truth_matrix_damage'] = _safe_arr(ground_truth_dmg_list)
        return result
