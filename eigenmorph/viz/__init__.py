# -*- coding: utf-8 -*-
"""
eigenmorph.viz
==============

Visualization subpackage for cortical eigenvalue features.

Submodules
----------
styles
    Colour palettes, publication-ready matplotlib defaults, and
    feature → colormap mappings.
static
    Matplotlib-based publication-quality plots: surface renders,
    multi-scale profiles, ternary diagrams, comparison heatmaps,
    radar fingerprints, embedding scatter, and composite hero figures.
interactive
    FURY (VTK)-based high-impact visualizations: RGB identity maps,
    feature landscapes, animated scale sweeps, exploded geometric
    communities, and neighbourhood explorers.  Falls back to matplotlib
    when FURY is not installed.
"""

# Always available (matplotlib only)
from .styles import (
    setup_style,
    FEATURE_COLORS,
    FEATURE_CMAPS,
    get_feature_cmap,
)

from .static import (
    plot_surface_feature,
    plot_feature_overview,
    plot_multiscale_profile,
    plot_ternary_features,
    plot_classical_comparison,
    plot_morphological_radar,
    plot_feature_embedding,
    plot_hero_figure,
    plot_parcellation_bars,
    plot_distance_matrix,
)

# Optionally available (requires fury)
try:
    from .interactive import (
        plot_rgb_identity,
        plot_feature_landscape,
        render_scale_sweep,
        plot_exploded_view,
        plot_neighborhood_explorer,
        plot_dual_hemisphere,
    )
    _HAS_INTERACTIVE = True
except ImportError:
    _HAS_INTERACTIVE = False

__all__ = [
    # styles
    "setup_style", "FEATURE_COLORS", "FEATURE_CMAPS", "get_feature_cmap",
    # static
    "plot_surface_feature", "plot_feature_overview",
    "plot_multiscale_profile", "plot_ternary_features",
    "plot_classical_comparison", "plot_morphological_radar",
    "plot_feature_embedding", "plot_hero_figure",
    "plot_parcellation_bars", "plot_distance_matrix",
    # interactive (when available)
    "plot_rgb_identity", "plot_feature_landscape",
    "render_scale_sweep", "plot_exploded_view",
    "plot_neighborhood_explorer", "plot_dual_hemisphere",
]
