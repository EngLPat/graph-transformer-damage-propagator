"""
Core neural network layers for FCLGA_GraphTransformer.

Contains ProcessorLayer for message passing and OptimizedGlobalAttention
for efficient global attention computation.
"""

import torch
import torch.nn as nn
import torch_scatter
from torch.nn import LayerNorm, Linear, PReLU, Sequential
from torch_geometric.nn.conv import MessagePassing

from ..config import DEFAULTS


class ProcessorLayer(MessagePassing):
    """
    Message passing layer for local relational reasoning.
    
    Implements edge-conditioned message passing with attention weighting.
    
    Args:
        in_channels: Input feature dimension
        out_channels: Output feature dimension
    """
    
    def __init__(self, in_channels: int, out_channels: int, use_attention: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.use_attention = use_attention

        # When True, store per-edge attention weights for analysis
        self.store_attention = False
        self._edge_attention = None  # [E, 1] after forward when store_attention=True

        # The gate is built even when use_attention is False, so that every
        # model has the same parameters and draws the same initialisation.
        self.attention = nn.Sequential(
            Linear(3 * in_channels, 1),
            nn.Sigmoid()
        )

        self.edge_mlp = Sequential(
            Linear(3 * in_channels, out_channels),
            PReLU(),
            Linear(out_channels, out_channels),
            LayerNorm(out_channels)
        )

        self.node_mlp = Sequential(
            Linear(2 * in_channels, out_channels),
            PReLU(),
            Linear(out_channels, out_channels),
            LayerNorm(out_channels)
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset learnable parameters."""
        self.edge_mlp[0].reset_parameters()
        self.edge_mlp[2].reset_parameters()
        self.node_mlp[0].reset_parameters()
        self.node_mlp[2].reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: torch.Tensor, size=None):
        """
        Forward pass.
        
        Args:
            x: Node features [N, in_channels]
            edge_index: Edge connectivity [2, E]
            edge_attr: Edge features [E, in_channels]
            size: Optional size tuple
        
        Returns:
            Tuple of (updated_nodes, updated_edges)
        """
        out, updated_edges = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)
        updated_nodes = torch.cat([x, out], dim=1)
        updated_nodes = x + self.node_mlp(updated_nodes)
        return updated_nodes, updated_edges

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, 
                edge_attr: torch.Tensor) -> torch.Tensor:
        """Compute attention-weighted messages along edges.
        
        The attention gate ``self.attention`` takes [h_i, h_j, e_ij]
        and produces a scalar weight in (0, 1) via Sigmoid.
        The message is the edge-MLP output *multiplied* by this weight,
        so the model can learn to amplify or suppress individual edges.
        """
        cat = torch.cat([x_i, x_j, edge_attr], dim=1)  # [E, 3*C]

        # Compute attention weight per edge  -- shape [E, 1]
        if self.use_attention:
            alpha = self.attention(cat)
        else:
            alpha = torch.ones((cat.size(0), 1), device=cat.device, dtype=cat.dtype)
        
        # Optionally store for analysis
        if self.store_attention:
            self._edge_attention = alpha.detach()

        # Edge update (residual)
        updated_edges = self.edge_mlp(cat) + edge_attr

        # Weight the message that will be aggregated at the target node
        weighted_msg = alpha * updated_edges

        # Store updated edges for the return path (unweighted, as
        # they represent the *edge* hidden state for the next layer)
        self._updated_edges = updated_edges

        return weighted_msg

    def aggregate(self, weighted_msgs: torch.Tensor,
                  edge_index: torch.Tensor, dim_size=None):
        """Aggregate attention-weighted messages at nodes."""
        node_dim = 0
        out = torch_scatter.scatter(
            weighted_msgs, edge_index[0, :], 
            dim=node_dim, dim_size=dim_size, reduce="sum"
        )
        return out, self._updated_edges


class OptimizedGlobalAttention(nn.Module):
    """
    Efficient global attention for Graph Neural Networks.
    
    Uses SDPA/FlashAttention when available for memory-efficient computation.
    
    Args:
        hidden_dim: Hidden dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
        use_flash_attention: Whether to use flash attention
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 1,
        dropout: float = None,
        use_flash_attention: bool = True,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = dropout
        self.use_flash_attention = use_flash_attention
        self.memory_efficient_threshold = DEFAULTS["memory_efficient_threshold"]

        # When True, store per-graph attention weights for analysis
        self.store_attention = False
        self._attention_weights = None   # dict: batch_idx -> [N_b, N_b]

        self.qkv_projection = Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.output_projection = Linear(hidden_dim, hidden_dim)
        self.layer_norm = LayerNorm(hidden_dim)
        self.attention_dropout = nn.Dropout(dropout) if dropout else nn.Identity()

        self._check_flash_attention_availability()

    def _check_flash_attention_availability(self) -> None:
        """Check for SDPA/FlashAttention support."""
        try:
            torch.nn.functional.scaled_dot_product_attention
            self.sdpa_available = True
            
            torch_version = torch.__version__.split("+")[0]
            major, minor = map(int, torch_version.split(".")[:2])
            self.flash_attention_supported = (major > 2 or (major == 2 and minor >= 0))
        except AttributeError:
            self.sdpa_available = False
            self.flash_attention_supported = False

    def forward(self, x: torch.Tensor, batch=None) -> torch.Tensor:
        """
        Forward pass with per-graph attention.

        Args:
            x: Node features [N, hidden_dim]
            batch: Batch assignment tensor

        Returns:
            Updated node features [N, hidden_dim]
        """
        if batch is None:
            batch = torch.zeros(x.size(0), device=x.device, dtype=torch.long)

        unique_batches = torch.unique(batch)
        output = torch.zeros_like(x)

        for batch_idx in unique_batches:
            mask = batch == batch_idx
            x_batch = x[mask]
            num_nodes = x_batch.size(0)

            x_norm = self.layer_norm(x_batch)
            qkv = self.qkv_projection(x_norm)
            q, k, v = qkv.chunk(3, dim=-1)

            q = q.unsqueeze(0)
            k = k.unsqueeze(0)
            v = v.unsqueeze(0)

            if self.store_attention:
                # Force manual path to extract attention weights
                scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
                attn_weights = torch.softmax(scores, dim=-1)
                if self._attention_weights is None:
                    self._attention_weights = {}
                self._attention_weights[int(batch_idx)] = attn_weights.squeeze(0).detach()
                attn_weights = self.attention_dropout(attn_weights)
                attn_output = torch.matmul(attn_weights, v).squeeze(0)
            elif self.sdpa_available and self.use_flash_attention:
                attn_output = nn.functional.scaled_dot_product_attention(
                    q, k, v,
                    dropout_p=self.dropout if self.training and self.dropout else 0.0,
                    is_causal=False
                )
                attn_output = attn_output.squeeze(0)
            else:
                if num_nodes > self.memory_efficient_threshold:
                    attn_output = self._chunked_attention(q, k, v)
                else:
                    scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
                    attn_weights = torch.softmax(scores, dim=-1)
                    attn_weights = self.attention_dropout(attn_weights)
                    attn_output = torch.matmul(attn_weights, v).squeeze(0)

            attn_output = self.output_projection(attn_output)
            
            # AMP-safe assignment: ensure dtypes match
            # Under AMP, attn_output may be float16 while output is float32
            if attn_output.dtype != output.dtype:
                attn_output = attn_output.to(output.dtype)
            
            output[mask] = attn_output

        return output

    def _chunked_attention(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor, 
        chunk_size: int = None
    ) -> torch.Tensor:
        """Memory-efficient chunked attention for large graphs."""
        if chunk_size is None:
            chunk_size = DEFAULTS["attention_chunk_size"]
        
        seq_len = q.size(1)
        output = torch.zeros_like(q)
        
        for i in range(0, seq_len, chunk_size):
            end_i = min(i + chunk_size, seq_len)
            q_chunk = q[:, i:end_i]
            scores = torch.matmul(q_chunk, k.transpose(-2, -1)) * self.scale
            attn_weights = torch.softmax(scores, dim=-1)
            attn_weights = self.attention_dropout(attn_weights)
            
            chunk_output = torch.matmul(attn_weights, v)
            # AMP-safe assignment: ensure dtypes match
            if chunk_output.dtype != output.dtype:
                chunk_output = chunk_output.to(output.dtype)
            output[:, i:end_i] = chunk_output
        
        return output.squeeze(0)
