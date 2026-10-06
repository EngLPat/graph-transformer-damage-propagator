"""
FCLGA_GraphTransformer: Main model architecture.

Frequency-Controlled Local Global Attention Graph Transformer
for time-dependent damage prediction in mesh-based physical systems.
"""

import torch
import torch.nn as nn
from torch.nn import Dropout, LayerNorm, Linear, PReLU, Sequential

from ..config import DEFAULTS
from ..data.normalization import normalize
from ..data.losses import huber_log_loss
from .layers import ProcessorLayer, OptimizedGlobalAttention

# Not called directly: importing it makes a missing torch_scatter fail loudly
# here, instead of PyG silently falling back to its native scatter kernels.
import torch_scatter  # noqa: F401


class FCLGA_GraphTransformer(nn.Module):
    """
    FCLGA_GraphTransformer: Frequency-Controlled Local Global Attention Graph Transformer.
    
    For time-dependent damage prediction in mesh-based physical systems.
    Combines local message passing with periodic global attention.
    
    Args:
        input_dim_node: Number of input node features
        input_dim_edge: Number of input edge features
        hidden_dim: Hidden dimension for all layers
        output_dim: Output dimension (typically 1 for damage rate)
        args: Configuration object with hyperparameters
        emb: Whether to return embeddings (unused, for compatibility)
    """
    
    def __init__(
        self, 
        input_dim_node: int, 
        input_dim_edge: int, 
        hidden_dim: int, 
        output_dim: int, 
        args, 
        emb: bool = False
    ):
        super().__init__()
        
        self.use_attention = True
        self.num_layers = args.num_layers
        self.hidden_dim = hidden_dim
        self.dropout_rate = getattr(args, "dropout_rate", DEFAULTS["dropout_rate"])
        self.attention_freq = getattr(args, "attention_freq", DEFAULTS["attention_freq"])
        self.local_attention_enabled = getattr(args, "local_attention_enabled", DEFAULTS["local_attention_enabled"])
        self.global_attention_enabled = getattr(args, "global_attention_enabled", DEFAULTS["global_attention_enabled"])
        # Mid-network time injection (default on)
        self.time_injection_enabled = getattr(
            args, "time_injection_enabled", DEFAULTS["time_injection_enabled"]
        )

        # Node encoder
        self.node_encoder = Sequential(
            Linear(input_dim_node, hidden_dim),
            PReLU(),
            Dropout(self.dropout_rate),
            Linear(hidden_dim, hidden_dim),
            PReLU(),
            Dropout(self.dropout_rate),
            Linear(hidden_dim, hidden_dim),
            LayerNorm(hidden_dim)
        )

        # Edge encoder
        self.edge_encoder = Sequential(
            Linear(input_dim_edge, hidden_dim),
            PReLU(),
            Dropout(self.dropout_rate),
            Linear(hidden_dim, hidden_dim),
            PReLU(),
            Dropout(self.dropout_rate),
            Linear(hidden_dim, hidden_dim),
            LayerNorm(hidden_dim)
        )

        # Time encoder: 2-component [current_time, delta_t]
        self.time_encoder = Sequential(
            Linear(2, hidden_dim // 2),
            PReLU(),
            Linear(hidden_dim // 2, hidden_dim // 4),
            LayerNorm(hidden_dim // 4)
        )
        
        self.time_projection = Linear(
            hidden_dim + hidden_dim // 4,
            hidden_dim
        )

        # Message passing layers
        self.processor = nn.ModuleList()
        assert self.num_layers >= 1, "Number of message passing layers must be >= 1"

        self.global_attention = OptimizedGlobalAttention(
            hidden_dim,
            num_heads=1,
            dropout=self.dropout_rate,
            use_flash_attention=True,
        )

        # Weight of the global-attention residual (local/global mixing)
        _ag = getattr(args, 'alpha_global', None)
        self.alpha_global = (DEFAULTS.get("alpha_global", 0.2)
                             if _ag is None else float(_ag))

        for _ in range(self.num_layers):
            self.processor.append(
                ProcessorLayer(
                    hidden_dim,
                    hidden_dim,
                    use_attention=self.local_attention_enabled,
                )
            )

        # Unified decoder
        decoder_input_dim = hidden_dim

        self.base_decoder = Sequential(
            Linear(decoder_input_dim, hidden_dim),
            PReLU(),
            Dropout(self.dropout_rate),
            Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        scale_input_dim = decoder_input_dim
        self.scale_predictor = Sequential(
            Linear(scale_input_dim, hidden_dim // 2),
            PReLU(),
            Linear(hidden_dim // 2, 1)
        )
        
        # Learnable global output scale
        self.output_scale = nn.Parameter(torch.tensor(4.6))  # exp(4.6) ≈ 100

        # Two-stage head: second decoder for the damage rate
        self.two_stage_enabled = getattr(args, 'two_stage_enabled', DEFAULTS.get("two_stage_enabled", False))
        if self.two_stage_enabled:
            self.dmg_decoder = Sequential(
                Linear(decoder_input_dim, hidden_dim),
                PReLU(),
                Dropout(self.dropout_rate),
                Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
            self.dmg_scale_predictor = Sequential(
                Linear(scale_input_dim, hidden_dim // 2),
                PReLU(),
                Linear(hidden_dim // 2, 1)
            )
            # Damage rates are smaller than FI rates (bounded by [0,1] range),
            # so initialise scale lower: exp(2.3) ≈ 10
            self.dmg_output_scale = nn.Parameter(torch.tensor(2.3))
            n_dmg_params = sum(
                p.numel() for p in list(self.dmg_decoder.parameters())
                + list(self.dmg_scale_predictor.parameters()) + [self.dmg_output_scale]
            )
            print(f"Two-stage damage head: {n_dmg_params} parameters "
                  f"(FI threshold={DEFAULTS.get('dmg_fi_threshold', 1.0)})")
        self._pred_dmg_rate = None  # populated during forward if two_stage_enabled

    def enable_attention_analysis(self, enable: bool = True):
        """Enable/disable per-edge attention weight storage for all processor layers
        and global attention weight storage."""
        for layer in self.processor:
            layer.store_attention = enable
        self.global_attention.store_attention = enable
        if not enable:
            self.global_attention._attention_weights = None

    def get_global_attention_stats(self, mesh_pos: torch.Tensor) -> dict:
        """
        Extract per-node global-attention statistics from stored attention weights.

        Must be called after a forward pass with ``enable_attention_analysis(True)``.

        Row-wise stats (how each node distributes its query):
            'entropy':          [N] attention entropy (higher = more diffuse)
            'max_weight':       [N] maximum outgoing attention weight
            'attn_reach':       [N] effective number of attended nodes (exp entropy)
            'long_range_ratio': [N] fraction of attention on distant nodes

        Column-wise stats (how much each node is attended to by others):
            'received_attn':      [N] total attention received (column sum)
            'received_max':       [N] max attention received from any single node
            'received_entropy':   [N] entropy of the incoming attention distribution
            'received_reach':     [N] effective number of nodes attending here
        """
        attn_dict = self.global_attention._attention_weights
        if attn_dict is None or len(attn_dict) == 0:
            raise RuntimeError("No global attention weights stored. "
                               "Call enable_attention_analysis(True) before forward().")

        N = mesh_pos.size(0)
        device = mesh_pos.device
        entropy = torch.zeros(N, device=device)
        max_weight = torch.zeros(N, device=device)
        attn_reach = torch.zeros(N, device=device)
        long_range_ratio = torch.zeros(N, device=device)
        received_attn = torch.zeros(N, device=device)
        received_max = torch.zeros(N, device=device)
        received_entropy = torch.zeros(N, device=device)
        received_reach = torch.zeros(N, device=device)

        for batch_idx, W in attn_dict.items():
            # W: [N_b, N_b] attention weight matrix (rows sum to 1)
            N_b = W.size(0)

            # ── Row-wise (query / outgoing) ──────────────────────────
            log_W = torch.log(W + 1e-10)
            ent = -(W * log_W).sum(dim=1)  # [N_b]
            entropy[:N_b] = ent
            max_weight[:N_b] = W.max(dim=1).values
            attn_reach[:N_b] = torch.exp(ent)

            # Long-range ratio
            pos_b = mesh_pos[:N_b]
            if N_b <= 5000:
                dists = torch.cdist(pos_b.unsqueeze(0), pos_b.unsqueeze(0)).squeeze(0)
                median_dist = dists.median()
                far_mask = (dists > median_dist).float()
                long_range_ratio[:N_b] = (W * far_mask).sum(dim=1)
            else:
                long_range_ratio[:N_b] = 0.0

            # ── Column-wise (key / incoming: "who is attended to") ──
            # Column sum: total attention each node receives
            col_sum = W.sum(dim=0)  # [N_b]
            received_attn[:N_b] = col_sum

            # Max incoming attention from any single querier
            received_max[:N_b] = W.max(dim=0).values

            # Normalise columns to get incoming probability distribution
            col_safe = col_sum.clamp(min=1e-10)
            W_col_norm = W / col_safe.unsqueeze(0)  # [N_b, N_b], columns sum to 1
            log_Wc = torch.log(W_col_norm + 1e-10)
            ent_col = -(W_col_norm * log_Wc).sum(dim=0)  # [N_b]
            received_entropy[:N_b] = ent_col
            received_reach[:N_b] = torch.exp(ent_col)

        return {
            # Row-wise (outgoing)
            'entropy': entropy,
            'max_weight': max_weight,
            'attn_reach': attn_reach,
            'long_range_ratio': long_range_ratio,
            # Column-wise (incoming)
            'received_attn': received_attn,
            'received_max': received_max,
            'received_entropy': received_entropy,
            'received_reach': received_reach,
        }

    def forward(
        self, 
        data, 
        mean_vec_x: torch.Tensor, 
        std_vec_x: torch.Tensor, 
        mean_vec_edge: torch.Tensor, 
        std_vec_edge: torch.Tensor
    ):
        """
        Forward pass.
        
        Args:
            data: PyTorch Geometric Data object
            mean_vec_x, std_vec_x: Node normalization stats
            mean_vec_edge, std_vec_edge: Edge normalization stats
        
        Returns:
            Tuple of (pred_rate, None)
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # Normalize inputs
        x = normalize(x, mean_vec_x, std_vec_x)
        edge_attr = normalize(edge_attr, mean_vec_edge, std_vec_edge)

        # Encode
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        num_nodes = x.size(0)

        # Extract time features
        if not (hasattr(data, "delta_t") and hasattr(data, "current_time")):
            raise ValueError(
                "Dataset missing required temporal attributes: 'delta_t' and 'current_time'"
            )

        delta_t_scalar = float(data.delta_t.view(-1)[0].item()) if torch.is_tensor(data.delta_t) else float(data.delta_t)
        current_time_scalar = float(data.current_time.view(-1)[0].item()) if torch.is_tensor(data.current_time) else float(data.current_time)

        time_features = torch.tensor(
            [[current_time_scalar, delta_t_scalar]],
            device=x.device, 
            dtype=torch.float32
        )

        time_emb = None
        layer_outputs = [x]

        # Message passing with periodic attention
        for i in range(self.num_layers):
            # Inject time at layer 3 (disabled by --no-time-injection)
            if self.time_injection_enabled and i == 3 and time_emb is None:
                time_emb_single = self.time_encoder(time_features)
                time_emb = time_emb_single.expand(num_nodes, -1)
                x_with_time = torch.cat([x, time_emb], dim=1)
                x = self.time_projection(x_with_time)
            
            # Global attention at specified frequency
            if self.global_attention_enabled and i % self.attention_freq == 0:
                batch = getattr(data, "batch", None)
                global_info = self.global_attention(x, batch=batch)
                x = x + self.alpha_global * global_info
            
            x, edge_attr = self.processor[i](x, edge_index, edge_attr)
            layer_outputs.append(x)

        x_combined = x

        # Base rate prediction
        base_rate = self.base_decoder(x_combined)

        # Per-node scaling
        scale_input = x_combined
        log_scale = self.scale_predictor(scale_input)
        
        # Global output scale + per-node scale
        total_log_scale = self.output_scale + log_scale
        scale = torch.exp(total_log_scale)

        pred_rate = base_rate * scale

        # Two-stage: damage rate prediction from same embeddings
        # Physics constraint: dD/dt = 0 when FI < 1 (no damage before failure initiation)
        self._pred_dmg_rate = None
        if self.two_stage_enabled:
            dmg_base = self.dmg_decoder(x_combined)
            dmg_log_scale = self.dmg_scale_predictor(scale_input)
            dmg_scale = torch.exp(self.dmg_output_scale + dmg_log_scale)
            raw_dmg_rate = dmg_base * dmg_scale
            # Hard gate: zero out damage rate where FI < threshold
            fi_now = data.x[:, -1]  # current FI (last feature)
            fi_thresh = DEFAULTS.get("dmg_fi_threshold", 1.0)
            fi_gate = (fi_now > fi_thresh).float().unsqueeze(-1)  # [N, 1]
            self._pred_dmg_rate = raw_dmg_rate * fi_gate

        # DDP visibility: _pred_dmg_rate is stored as an instance attribute and
        # consumed by loss(); it is NOT in pred_rate's autograd graph. DDP
        # scans the forward outputs to discover which params receive grad; if
        # dmg head params are absent from pred_rate's graph, DDP marks them
        # "ready" once at forward time, then again when the real backward
        # fires through _pred_dmg_rate, raising "marked ready twice".
        # Adding a zero-magnitude term grafts the dmg path into pred_rate's
        # graph (numerically a no-op).
        if self.two_stage_enabled and self._pred_dmg_rate is not None:
            pred_rate = pred_rate + 0.0 * self._pred_dmg_rate.sum()

        return pred_rate, None

    def loss(
        self, 
        pred_tuple, 
        inputs, 
        mean_vec_y_progressive, 
        std_vec_y_progressive,
        mean_vec_y_catastrophic, 
        std_vec_y_catastrophic, 
        epoch: int = 0,
        return_components: bool = False
    ) -> torch.Tensor:
        """
        Compute the log-space Huber loss of the rate heads.
        
        Args:
            pred_tuple: (pred_rate, None), as returned by forward()
            inputs: Input data batch
            mean_vec_y_*: Normalization stats (unused with log-space loss)
            std_vec_y_*: Normalization stats (unused with log-space loss)
            epoch: Current epoch for logging
            return_components: If True, return dict with individual loss components
        
        Returns:
            Scalar loss value, or tuple of (loss, components_dict) if return_components=True
        """
        pred_rate = pred_tuple[0]
            
        actual_rates = inputs.y
        
        if pred_rate.dim() > 1:
            pred_rate = pred_rate.squeeze(-1)
        if actual_rates.dim() > 1:
            actual_rates = actual_rates.squeeze(-1)
        
        # High-rate ("shock") nodes have rate > shock_rate_threshold (100). With
        # a positive rate_cap they are clamped to the cap; with a negative
        # (legacy) cap they are masked out of the rate loss.
        shock_rate_threshold = DEFAULTS.get("shock_rate_threshold", 100.0)
        
        import os
        rate_cap = float(os.environ.get(
            "GTP_RATE_CAP",
            DEFAULTS.get("rate_cap", 0.0)
        ))
        
        if rate_cap > 0:
            # Clamp GT rates at the cap: shock nodes become "predict the cap"
            actual_rates_for_loss = actual_rates.clamp(max=rate_cap)
            steady_mask = torch.ones_like(actual_rates, dtype=torch.bool)  # All nodes included
        elif rate_cap == 0:
            # No cap: include ALL nodes with true rates; log-Huber handles scale
            actual_rates_for_loss = actual_rates
            steady_mask = torch.ones_like(actual_rates, dtype=torch.bool)
        else:
            # Negative rate_cap (legacy): mask out shock nodes entirely
            actual_rates_for_loss = actual_rates
            steady_mask = actual_rates <= shock_rate_threshold
        
        # Interface nodes are excluded from the FI-rate loss unless the sample
        # sets iface_trainable
        ply_mask = None  # will be set if cohesive laminate metadata present
        if hasattr(inputs, 'n_ply_nodes') and hasattr(inputs, 'ptr') and hasattr(inputs, 'batch'):
            node_offsets = inputs.ptr[inputs.batch]
            local_idx = torch.arange(actual_rates.shape[0], device=actual_rates.device) - node_offsets
            n_ply = inputs.n_ply_nodes[inputs.batch]
            ply_mask = local_idx < n_ply

            iface_trainable = getattr(inputs, 'iface_trainable', None)
            if iface_trainable is not None:
                if torch.is_tensor(iface_trainable):
                    iface_trainable_all = bool(iface_trainable.all().item())
                else:
                    iface_trainable_all = bool(iface_trainable)
            else:
                iface_trainable_all = False

            if not iface_trainable_all:
                steady_mask = steady_mask & ply_mask

        # Optional extra weight on active nodes (rate > 0.01)
        active_node_weight = float(os.environ.get(
            "GTP_ACTIVE_NODE_WEIGHT",
            DEFAULTS.get("active_node_weight", 1.0)
        ))
        node_weights = None
        if active_node_weight > 1.0:
            node_weights = torch.ones_like(actual_rates)
            active_mask = (actual_rates > 0.01) & steady_mask
            node_weights[active_mask] = active_node_weight
        
        loss_rate = huber_log_loss(
            pred_rate, actual_rates_for_loss, delta=1.0,
            mask=steady_mask, weights=node_weights,
        )

        n_steady = steady_mask.sum().item()
        n_shock = (~steady_mask).sum().item()
        
        damage_loss_weight = float(os.environ.get("GTP_DAMAGE_LOSS_WEIGHT", DEFAULTS.get("damage_loss_weight", 1.0)))
        weighted_loss_rate = damage_loss_weight * loss_rate
        
        loss_components = {
            'rate': loss_rate.item(),
            'rate_weighted': weighted_loss_rate.item(),
            'n_steady': n_steady,
            'n_shock_masked': n_shock,
            'active_node_weight': active_node_weight,
            'rate_cap': rate_cap,
        }

        total_loss = weighted_loss_rate

        # Two-stage damage-rate loss
        loss_dmg_rate = torch.tensor(0.0, device=loss_rate.device)
        if self.two_stage_enabled and self._pred_dmg_rate is not None and hasattr(inputs, 'y_dmg'):
            pred_dmg = self._pred_dmg_rate.squeeze(-1)
            actual_dmg_rates = inputs.y_dmg.squeeze(-1) if inputs.y_dmg.dim() > 1 else inputs.y_dmg

            # Clamp GT damage rates at rate_cap, as for the FI rates
            if rate_cap > 0:
                actual_dmg_rates = actual_dmg_rates.clamp(max=rate_cap)

            # Only supervise where FI > threshold (damage only evolves past failure initiation)
            fi_now = inputs.x[:, -1]
            fi_thresh = DEFAULTS.get("dmg_fi_threshold", 1.0)
            fi_active = fi_now > fi_thresh

            # Ply nodes: supervise where FI > 1 (matrix damage evolving).
            # Interface nodes, when present, are excluded.
            if ply_mask is not None:
                dmg_mask = fi_active & ply_mask
            else:
                dmg_mask = fi_active

            if dmg_mask.any():
                loss_dmg_rate = huber_log_loss(
                    pred_dmg, actual_dmg_rates, delta=1.0, mask=dmg_mask
                )
                dmg_weight = float(DEFAULTS.get("dmg_loss_weight", 1.0))
                total_loss = total_loss + dmg_weight * loss_dmg_rate

            loss_components['dmg_rate'] = loss_dmg_rate.item()
            loss_components['n_fi_active'] = int(fi_active.sum().item())


        if DEFAULTS.get("early_timestep_weighting_enabled", True):
            if hasattr(inputs, "sample_idx"):
                sample_idx = inputs.sample_idx
                if sample_idx.numel() > 1:
                    timestep = float(sample_idx[0].item())
                else:
                    timestep = float(sample_idx.item())
            else:
                # If no sample_idx, treat as late timestep (no extra weight)
                timestep = float("inf")
            
            weight_max = DEFAULTS.get("early_timestep_weight_max", 3.0)
            weight_min = DEFAULTS.get("early_timestep_weight_min", 1.0)
            decay_rate = DEFAULTS.get("early_timestep_decay_rate", 0.1)
            
            # Exponential decay: weight = min + (max - min) * exp(-decay * timestep)
            # At timestep 0: weight = max
            # As timestep -> inf: weight -> min
            import math
            timestep_weight = weight_min + (weight_max - weight_min) * math.exp(-decay_rate * timestep)
            
            total_loss = total_loss * timestep_weight
            loss_components['timestep_weight'] = timestep_weight
        else:
            loss_components['timestep_weight'] = 1.0
        
        loss_components['total'] = total_loss.item()

        if return_components:
            return total_loss, loss_components
        return total_loss
