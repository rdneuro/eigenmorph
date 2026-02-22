# -*- coding: utf-8 -*-
"""
eigenmorph.viz.styles
=====================

Colour palettes, style presets, and feature → colormap mappings for
consistent, publication-ready figures across the library.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════
#  PUBLICATION STYLE
# ═══════════════════════════════════════════════════════════════════════════

def setup_style(context: str = "paper"):
    """
    Apply publication-ready matplotlib defaults.

    Parameters
    ----------
    context : str
        ``'paper'`` (tight, small fonts) or ``'poster'`` (larger).
    """
    base = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "axes.grid": False,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "figure.constrained_layout.use": True,
    }

    if context == "poster":
        base.update({
            "font.size": 16,
            "axes.titlesize": 20,
            "axes.labelsize": 18,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "figure.dpi": 100,
            "savefig.dpi": 300,
        })
    else:  # paper
        base.update({
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
        })

    plt.rcParams.update(base)


# ═══════════════════════════════════════════════════════════════════════════
#  FEATURE COLOURS
# ═══════════════════════════════════════════════════════════════════════════

# Distinct, colorblind-friendly accent colours per feature
FEATURE_COLORS = {
    "linearity":         "#E63946",  # warm red
    "planarity":         "#457B9D",  # steel blue
    "sphericity":        "#2A9D8F",  # teal
    "omnivariance":      "#E9C46A",  # gold
    "anisotropy":        "#8338EC",  # purple
    "eigenentropy":      "#06D6A0",  # mint
    "surface_variation": "#F77F00",  # orange
    # Extended features
    "shape_index":       "#264653",  # dark teal
    "curvedness":        "#A8DADC",  # light blue
    "verticality":       "#BC6C25",  # brown
    "fractal_dimension": "#606C38",  # olive
    "normal_displacement": "#DDA15E", # sand
}

# Per-feature colormaps for surface rendering
FEATURE_CMAPS = {
    "linearity":         "YlOrRd",
    "planarity":         "YlGnBu",
    "sphericity":        "PuRd",
    "omnivariance":      "viridis",
    "anisotropy":        "magma",
    "eigenentropy":      "cividis",
    "surface_variation": "plasma",
    "shape_index":       "RdBu_r",
    "curvedness":        "inferno",
    "verticality":       "twilight",
    "fractal_dimension": "Spectral_r",
    "normal_displacement": "coolwarm",
}

# Mathematical labels for figure titles (Unicode)
FEATURE_LABELS = {
    "linearity":         "Linearity\n(λ₁−λ₂)/λ₁",
    "planarity":         "Planarity\n(λ₂−λ₃)/λ₁",
    "sphericity":        "Sphericity\nλ₃/λ₁",
    "omnivariance":      "Omnivariance\n(λ₁λ₂λ₃)^⅓",
    "anisotropy":        "Anisotropy\n(λ₁−λ₃)/λ₁",
    "eigenentropy":      "Eigenentropy\n−Σλ̃ ln(λ̃)",
    "surface_variation": "Surf. Variation\nλ₃/Σλ",
}


def get_feature_cmap(name: str) -> str:
    """
    Get the canonical colormap for a feature.

    Falls back to ``'viridis'`` for unknown features.
    """
    return FEATURE_CMAPS.get(name, "viridis")


def get_feature_color(name: str) -> str:
    """Get the canonical accent colour for a feature."""
    return FEATURE_COLORS.get(name, "#333333")
