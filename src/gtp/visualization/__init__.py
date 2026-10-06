"""
Visualization utilities for GTP.

Provides shared plotting functions for training and testing visualization.
"""

from .plotting import (
    plot_loss_curves,
    plot_loss_components,
    plot_damage_comparison,
    plot_damage_evolution_frame,
    create_2d_projection,
    create_mesh_projection,
    safe_to_numpy,
    plot_global_attention_analysis,
)
from .animations import (
    create_damage_evolution_gif,
    create_damage_evolution_video,
    create_animation_from_arrays,
)
from .laminate_plots import (
    generate_laminate_test_plots,
    plot_rate_regression,
    plot_damage_regression,
    plot_per_ply_metrics_summary,
)

__all__ = [
    # Plotting
    "plot_loss_curves",
    "plot_loss_components",
    "plot_damage_comparison",
    "plot_damage_evolution_frame",
    "create_2d_projection",
    "create_mesh_projection",
    "safe_to_numpy",
    "plot_global_attention_analysis",
    # Animations
    "create_damage_evolution_gif",
    "create_damage_evolution_video",
    "create_animation_from_arrays",
    # Laminate analysis
    "generate_laminate_test_plots",
    "plot_rate_regression",
    "plot_damage_regression",
    "plot_per_ply_metrics_summary",
]
