# -*- coding: utf-8 -*-
"""
eigenmorph.viz.static
=====================

Publication-quality matplotlib visualizations for eigenvalue features.

Every function returns the ``matplotlib.Figure`` it creates and
optionally saves to disk.  All plots use the library style and
feature-colour conventions from ``eigenmorph.viz.styles``.

Functions
---------
plot_surface_feature        Single 3D surface coloured by a scalar.
plot_feature_overview       Seven-panel overview (one per feature).
plot_multiscale_profile     Feature evolution across radii.
plot_ternary_features       L–P–S ternary + entropy vs roughness.
plot_classical_comparison   Correlation heatmap + unique-variance bars.
plot_morphological_radar    Radar fingerprint per region.
plot_feature_embedding      UMAP / t-SNE of feature space.
plot_hero_figure            Multi-panel composite for graphical abstract.
plot_parcellation_bars      Parcellated feature bar chart.
plot_distance_matrix        Region × region morphological distance.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors as mcolors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import Optional, Dict, List, Tuple

from ..core import SurfaceMesh, EigenFeatures, MultiScaleFeatures
from .styles import (
    setup_style, FEATURE_COLORS, FEATURE_CMAPS, FEATURE_LABELS,
    get_feature_cmap, get_feature_color,
)


# ═══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _surface_to_ax(
    ax, mesh, values, cmap="viridis", vmin=None, vmax=None,
    view=(30, -60),
):
    """Render a mesh on a 3D axes coloured by vertex values."""
    vals = np.nan_to_num(values, nan=0.0)
    valid = values[np.isfinite(values)]
    if vmin is None:
        vmin = np.percentile(valid, 5) if len(valid) > 0 else 0
    if vmax is None:
        vmax = np.percentile(valid, 95) if len(valid) > 0 else 1

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)

    face_vals = vals[mesh.faces].mean(axis=1)
    face_rgba = mapper.to_rgba(face_vals)

    polys = Poly3DCollection(
        mesh.vertices[mesh.faces],
        facecolors=face_rgba, edgecolors="none", linewidths=0,
    )
    ax.add_collection3d(polys)

    v = mesh.vertices
    margin = 5
    ax.set_xlim(v[:, 0].min() - margin, v[:, 0].max() + margin)
    ax.set_ylim(v[:, 1].min() - margin, v[:, 1].max() + margin)
    ax.set_zlim(v[:, 2].min() - margin, v[:, 2].max() + margin)
    ax.view_init(elev=view[0], azim=view[1])
    ax.axis("off")

    return mapper


# ═══════════════════════════════════════════════════════════════════════════
#  1. SINGLE SURFACE
# ═══════════════════════════════════════════════════════════════════════════

def plot_surface_feature(
    mesh: SurfaceMesh,
    values: np.ndarray,
    title: str = "",
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    view: Tuple[float, float] = (30, -60),
    figsize: Tuple[int, int] = (8, 6),
    colorbar: bool = True,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    3D cortical surface coloured by a scalar feature.

    Parameters
    ----------
    mesh : SurfaceMesh
    values : (V,) scalar per vertex
    title, cmap, vmin, vmax, view, figsize, colorbar : display opts
    ax : existing 3D axes (optional)
    save_path : str, optional

    Returns
    -------
    matplotlib.Figure
    """
    setup_style()

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    mapper = _surface_to_ax(ax, mesh, values, cmap=cmap,
                             vmin=vmin, vmax=vmax, view=view)
    ax.set_title(title, fontweight="bold")

    if colorbar:
        fig.colorbar(mapper, ax=ax, shrink=0.6, pad=0.05)

    if save_path:
        fig.savefig(save_path)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
#  2. SEVEN-PANEL OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════

def plot_feature_overview(
    mesh: SurfaceMesh,
    features: EigenFeatures,
    figsize: Tuple[int, int] = (24, 9),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Seven-panel overview: one surface per eigenvalue feature.

    Each panel uses the feature's canonical colormap and accent colour
    for the title.
    """
    setup_style()

    fig = plt.figure(figsize=figsize)
    fig.suptitle(
        f"Eigenvalue Features  (r = {features.radius:.0f} mm, "
        f"weighting = {features.weighting})",
        fontsize=14, fontweight="bold", y=0.98,
    )

    names = EigenFeatures.feature_names()

    for idx, name in enumerate(names):
        ax = fig.add_subplot(2, 4, idx + 1, projection="3d")
        vals = getattr(features, name)
        cmap = get_feature_cmap(name)
        _surface_to_ax(ax, mesh, vals, cmap=cmap)
        label = FEATURE_LABELS.get(name, name.replace("_", " ").title())
        ax.set_title(label, fontweight="bold",
                     color=get_feature_color(name), fontsize=10)

    # Hide unused 8th panel
    if len(names) < 8:
        for i in range(len(names), 8):
            fig.add_subplot(2, 4, i + 1).set_visible(False)

    if save_path:
        fig.savefig(save_path)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
#  3. MULTI-SCALE PROFILES
# ═══════════════════════════════════════════════════════════════════════════

def plot_multiscale_profile(
    ms_features: MultiScaleFeatures,
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Feature evolution across spatial scales (mean ± std).
    """
    setup_style()

    names = EigenFeatures.feature_names()
    radii = ms_features.radii

    fig, axes = plt.subplots(2, 4, figsize=figsize)
    axes = axes.ravel()

    for idx, fn in enumerate(names):
        ax = axes[idx]
        means, stds = [], []
        for s in ms_features.scales:
            v = getattr(s, fn)
            v = v[np.isfinite(v)]
            means.append(np.mean(v) if len(v) else 0)
            stds.append(np.std(v) if len(v) else 0)

        means, stds = np.array(means), np.array(stds)
        col = get_feature_color(fn)

        ax.fill_between(radii, means - stds, means + stds,
                        alpha=0.2, color=col)
        ax.plot(radii, means, "o-", color=col, linewidth=2.5,
                markersize=7, markerfacecolor="white",
                markeredgewidth=2.5, markeredgecolor=col)

        ax.set_xlabel("Radius (mm)")
        ax.set_ylabel(fn.replace("_", " ").title())
        ax.set_title(fn.replace("_", " ").title(),
                     fontweight="bold", color=col)
        ax.grid(True, alpha=0.2)

    for i in range(len(names), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Multi-Scale Feature Profiles",
                 fontsize=14, fontweight="bold")
    if save_path:
        fig.savefig(save_path)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
#  4. TERNARY FEATURE SPACE
# ═══════════════════════════════════════════════════════════════════════════

def plot_ternary_features(
    features: EigenFeatures,
    max_points: int = 10_000,
    seed: int = 42,
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Left: L–P–S ternary diagram coloured by eigenentropy.
    Right: eigenentropy vs surface variation coloured by anisotropy.
    """
    setup_style()

    valid = features.valid_mask
    indices = np.where(valid)[0]
    rng = np.random.default_rng(seed)
    if len(indices) > max_points:
        indices = rng.choice(indices, max_points, replace=False)

    L = features.linearity[indices]
    P = features.planarity[indices]
    S = features.sphericity[indices]

    total = L + P + S
    total[total == 0] = 1.0
    Ln, Pn, Sn = L / total, P / total, S / total

    # Ternary → Cartesian
    x = 0.5 * (2 * Pn + Sn)
    y = (np.sqrt(3) / 2) * Sn

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # ── Left: ternary ──
    sc = ax1.scatter(x, y, c=features.eigenentropy[indices],
                     s=1, alpha=0.3, cmap="viridis", rasterized=True)
    fig.colorbar(sc, ax=ax1, label="Eigenentropy")

    # Triangle
    tri_x, tri_y = [0, 1, 0.5, 0], [0, 0, np.sqrt(3)/2, 0]
    ax1.plot(tri_x, tri_y, "k-", linewidth=1.5)
    ax1.text(0, -0.06, "Linearity\n(ridges)", ha="center", fontsize=8,
             color=get_feature_color("linearity"))
    ax1.text(1, -0.06, "Planarity\n(flat)", ha="center", fontsize=8,
             color=get_feature_color("planarity"))
    ax1.text(0.5, np.sqrt(3)/2+0.04, "Sphericity\n(pits)",
             ha="center", fontsize=8,
             color=get_feature_color("sphericity"))
    ax1.set_xlim(-0.1, 1.1)
    ax1.set_ylim(-0.15, 1.05)
    ax1.set_aspect("equal")
    ax1.axis("off")
    ax1.set_title(f"L–P–S Ternary (r={features.radius:.0f} mm)",
                  fontweight="bold")

    # ── Right: entropy vs roughness ──
    sc2 = ax2.scatter(
        features.eigenentropy[indices],
        features.surface_variation[indices],
        c=features.anisotropy[indices],
        s=1, alpha=0.3, cmap="magma", rasterized=True,
    )
    fig.colorbar(sc2, ax=ax2, label="Anisotropy")
    ax2.set_xlabel("Eigenentropy")
    ax2.set_ylabel("Surface Variation")
    ax2.set_title("Complexity vs Roughness", fontweight="bold")
    ax2.grid(True, alpha=0.2)

    if save_path:
        fig.savefig(save_path)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
#  5. CLASSICAL COMPARISON
# ═══════════════════════════════════════════════════════════════════════════

def plot_classical_comparison(
    comparison: Dict,
    figsize: Tuple[int, int] = (16, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Correlation heatmap + unique-variance bar chart.
    """
    setup_style()

    corr = comparison["correlations"]
    eigen_names = comparison["eigen_names"]
    classical_names = comparison["classical_names"]
    uv = comparison["unique_variance"]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=figsize,
        gridspec_kw={"width_ratios": [3, 1]},
    )

    # ── Heatmap ──
    im = ax1.imshow(corr, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax1, label="Pearson r", shrink=0.8)

    ax1.set_xticks(range(len(classical_names)))
    ax1.set_xticklabels(classical_names, rotation=45, ha="right")
    ax1.set_yticks(range(len(eigen_names)))
    ax1.set_yticklabels(eigen_names, fontsize=7)
    ax1.set_title("Eigenvalue vs Classical Correlations", fontweight="bold")

    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            val = corr[i, j]
            col = "white" if abs(val) > 0.5 else "black"
            ax1.text(j, i, f"{val:.2f}", ha="center", va="center",
                     fontsize=6, color=col)

    # ── Unique-variance bars ──
    y_pos = np.arange(len(eigen_names))
    bar_cols = ["#E63946" if u > 0.5 else "#95a5a6" for u in uv]
    ax2.barh(y_pos, uv, color=bar_cols, edgecolor="none")
    ax2.axvline(0.5, color="black", linestyle="--", linewidth=1,
                label="50% threshold")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(eigen_names, fontsize=7)
    ax2.set_xlabel("Unique Variance (1 − R²)")
    ax2.set_title("Novel Information", fontweight="bold")
    ax2.set_xlim(0, 1)
    ax2.legend(fontsize=7)
    ax2.invert_yaxis()

    if save_path:
        fig.savefig(save_path)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
#  6. RADAR FINGERPRINTS
# ═══════════════════════════════════════════════════════════════════════════

def plot_morphological_radar(
    parcellated: Dict,
    parcel_indices: Optional[List[int]] = None,
    parcel_names: Optional[List[str]] = None,
    scale_idx: int = 1,
    n_cols: int = 4,
    figsize_per: Tuple[float, float] = (3.5, 3.5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Radar chart showing the 7-feature fingerprint per brain region.
    """
    setup_style()

    feat_labels = EigenFeatures.feature_names()
    col_names = parcellated.get("_column_names", [])
    names = parcel_names or parcellated.get("_parcel_names", [])

    # Find columns for requested scale
    feat_cols = {}
    for fn in feat_labels:
        matching = [c for c in col_names if c.startswith(fn)]
        idx = min(scale_idx, len(matching) - 1) if matching else -1
        if idx >= 0:
            feat_cols[fn] = matching[idx]

    n_features = len(feat_labels)
    angles = np.linspace(0, 2*np.pi, n_features, endpoint=False).tolist()
    angles += angles[:1]

    sample_col = list(feat_cols.values())[0]
    n_parcels = len(parcellated[sample_col])
    if parcel_indices is None:
        parcel_indices = list(range(min(8, n_parcels)))

    n_plots = len(parcel_indices)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per[0]*n_cols, figsize_per[1]*n_rows),
        subplot_kw=dict(polar=True),
    )
    if n_plots == 1:
        axes = np.array([axes])
    axes_flat = axes.ravel()

    # Global max for normalisation
    gmax = {}
    for fn in feat_labels:
        col = feat_cols[fn]
        gmax[fn] = max(np.nanmax(np.abs(parcellated[col])), 1e-12)

    palette = plt.cm.Set2(np.linspace(0, 1, n_plots))

    for pi, pidx in enumerate(parcel_indices):
        ax = axes_flat[pi]
        vals = [parcellated[feat_cols[fn]][pidx] / gmax[fn]
                for fn in feat_labels]
        vals += vals[:1]

        ax.fill(angles, vals, alpha=0.25, color=palette[pi])
        ax.plot(angles, vals, "o-", linewidth=2, markersize=4,
                color=palette[pi])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([fn[:4] for fn in feat_labels], fontsize=7)
        ax.set_ylim(0, 1.1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels([], fontsize=5)

        label = names[pidx] if pidx < len(names) else f"Parcel {pidx}"
        if isinstance(label, bytes):
            label = label.decode()
        ax.set_title(label, fontsize=9, fontweight="bold", pad=12)

    for i in range(n_plots, len(axes_flat)):
        axes_flat[i].set_visible(False)

    scale_label = list(feat_cols.values())[0].split("_r")[-1]
    fig.suptitle(f"Morphological Fingerprints  ({scale_label})",
                 fontsize=14, fontweight="bold", y=1.02)
    if save_path:
        fig.savefig(save_path)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
#  7. FEATURE EMBEDDING
# ═══════════════════════════════════════════════════════════════════════════

def _lps_to_rgb(L, P, S, gamma=0.7):
    """Linearity→R, Planarity→G, Sphericity→B with gamma correction."""
    L = np.nan_to_num(L, nan=0.0)
    P = np.nan_to_num(P, nan=0.0)
    S = np.nan_to_num(S, nan=0.0)
    total = L + P + S
    total[total < 1e-12] = 1.0
    r = np.power(np.clip(L/total, 0, 1), gamma)
    g = np.power(np.clip(P/total, 0, 1), gamma)
    b = np.power(np.clip(S/total, 0, 1), gamma)
    return np.column_stack([r, g, b])


def plot_feature_embedding(
    ms_features: MultiScaleFeatures,
    vertex_labels: Optional[np.ndarray] = None,
    method: str = "umap",
    max_points: int = 20_000,
    seed: int = 42,
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    UMAP or t-SNE of the multi-scale feature space (28-D → 2-D).

    Left: coloured by labels (or eigenentropy).
    Right: coloured by L–P–S RGB identity.
    """
    setup_style()

    feat_mat = ms_features.as_matrix()
    valid_idx = np.where(~np.any(np.isnan(feat_mat), axis=1))[0]

    rng = np.random.default_rng(seed)
    if len(valid_idx) > max_points:
        sample = rng.choice(valid_idx, max_points, replace=False)
    else:
        sample = valid_idx

    X = feat_mat[sample]
    mu, sigma = X.mean(0), X.std(0)
    sigma[sigma < 1e-12] = 1.0
    X_std = (X - mu) / sigma

    # Embedding
    if method == "umap":
        try:
            import umap
            embedding = umap.UMAP(
                n_components=2, random_state=seed,
                n_neighbors=30, min_dist=0.3,
            ).fit_transform(X_std)
        except ImportError:
            method = "tsne"

    if method == "tsne":
        from sklearn.manifold import TSNE
        embedding = TSNE(
            n_components=2, random_state=seed,
            perplexity=min(30, len(X_std) - 1),
        ).fit_transform(X_std)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Left
    if vertex_labels is not None:
        c = vertex_labels[sample]
        ax1.scatter(embedding[:, 0], embedding[:, 1],
                    c=c, s=1, alpha=0.4, cmap="nipy_spectral",
                    rasterized=True)
        ax1.set_title("Coloured by Parcellation", fontweight="bold")
    else:
        mid = len(ms_features.scales) // 2
        ent = ms_features.scales[mid].eigenentropy[sample]
        sc = ax1.scatter(embedding[:, 0], embedding[:, 1],
                         c=ent, s=1, alpha=0.4, cmap="viridis",
                         rasterized=True)
        fig.colorbar(sc, ax=ax1, label="Eigenentropy", shrink=0.8)
        ax1.set_title("Coloured by Eigenentropy", fontweight="bold")

    ax1.set_xlabel(f"{method.upper()} 1")
    ax1.set_ylabel(f"{method.upper()} 2")

    # Right: RGB
    mid = len(ms_features.scales) // 2
    s = ms_features.scales[mid]
    rgb = _lps_to_rgb(s.linearity[sample], s.planarity[sample],
                       s.sphericity[sample])
    ax2.scatter(embedding[:, 0], embedding[:, 1],
                c=rgb, s=1, alpha=0.4, rasterized=True)
    ax2.set_xlabel(f"{method.upper()} 1")
    ax2.set_ylabel(f"{method.upper()} 2")
    ax2.set_title("L→R  P→G  S→B", fontweight="bold")

    from matplotlib.patches import Patch
    ax2.legend(handles=[
        Patch(facecolor="red", label="Linearity"),
        Patch(facecolor="green", label="Planarity"),
        Patch(facecolor="blue", label="Sphericity"),
    ], fontsize=7, loc="lower right")

    fig.suptitle(
        f"{method.upper()} Embedding  ({len(sample):,} vertices)",
        fontsize=13, fontweight="bold",
    )
    if save_path:
        fig.savefig(save_path)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
#  8. PARCELLATION BAR CHART
# ═══════════════════════════════════════════════════════════════════════════

def plot_parcellation_bars(
    parcellated: Dict,
    feature_name: str = "linearity_r5mm",
    top_n: int = 20,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Horizontal bar chart of a parcellated feature (top-N regions).
    """
    setup_style()

    vals = parcellated[feature_name]
    names = parcellated.get("_parcel_names", [f"P{i}" for i in range(len(vals))])

    order = np.argsort(vals)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=figsize)
    y = np.arange(len(order))

    base_name = feature_name.split("_r")[0]
    col = get_feature_color(base_name)

    ax.barh(y, vals[order], color=col, alpha=0.85, edgecolor="none")
    ax.set_yticks(y)
    ax.set_yticklabels([names[i] if i < len(names) else f"P{i}"
                        for i in order], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(feature_name.replace("_", " ").title())
    ax.set_title(f"Top {top_n} Regions — {feature_name}", fontweight="bold")

    if save_path:
        fig.savefig(save_path)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
#  9. DISTANCE MATRIX
# ═══════════════════════════════════════════════════════════════════════════

def plot_distance_matrix(
    dist_matrix: np.ndarray,
    labels: Optional[List[str]] = None,
    cmap: str = "viridis_r",
    figsize: Tuple[int, int] = (10, 9),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot a region × region morphological distance matrix.
    """
    setup_style()

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(dist_matrix, cmap=cmap, aspect="equal")
    fig.colorbar(im, ax=ax, label="Morphological Distance", shrink=0.8)

    if labels is not None:
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=5)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=5)

    ax.set_title("Morphological Distance Matrix", fontweight="bold")
    if save_path:
        fig.savefig(save_path)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
#  10. HERO COMPOSITE FIGURE
# ═══════════════════════════════════════════════════════════════════════════

def plot_hero_figure(
    mesh: SurfaceMesh,
    features: EigenFeatures,
    ms_features: MultiScaleFeatures,
    classical_metrics: Optional[Dict[str, np.ndarray]] = None,
    comparison: Optional[Dict] = None,
    figsize: Tuple[int, int] = (24, 16),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Multi-panel composite figure for graphical abstract or README.

    Layout (3 rows):
        Row 1: RGB identity | Ternary L-P-S | Scale profiles
        Row 2: Feature surfaces (fine & coarse) | Unique variance
        Row 3: Radar fingerprints (4 regions)
    """
    setup_style()
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3)

    v = mesh.vertices
    margin = 5

    # ── A: RGB Identity ──
    ax_a = fig.add_subplot(gs[0, 0:2], projection="3d")
    rgb = _lps_to_rgb(features.linearity, features.planarity,
                       features.sphericity)
    face_rgba = np.column_stack([
        rgb[mesh.faces].mean(axis=1),
        np.ones(mesh.n_faces)
    ])
    polys = Poly3DCollection(
        v[mesh.faces], facecolors=face_rgba, edgecolors="none", linewidths=0,
    )
    ax_a.add_collection3d(polys)
    ax_a.set_xlim(v[:, 0].min()-margin, v[:, 0].max()+margin)
    ax_a.set_ylim(v[:, 1].min()-margin, v[:, 1].max()+margin)
    ax_a.set_zlim(v[:, 2].min()-margin, v[:, 2].max()+margin)
    ax_a.view_init(30, -60)
    ax_a.axis("off")
    ax_a.set_title("A.  Geometric Identity (L→R  P→G  S→B)",
                    fontweight="bold", fontsize=11)

    # ── B: Ternary ──
    ax_b = fig.add_subplot(gs[0, 2])
    valid_idx = np.where(features.valid_mask)[0]
    rng = np.random.default_rng(42)
    if len(valid_idx) > 10_000:
        valid_idx = rng.choice(valid_idx, 10_000, replace=False)

    L = features.linearity[valid_idx]
    P = features.planarity[valid_idx]
    S = features.sphericity[valid_idx]
    total = L + P + S
    total[total == 0] = 1
    x = 0.5 * (2*P/total + S/total)
    y = (np.sqrt(3)/2) * S/total
    ax_b.scatter(x, y, c=rgb[valid_idx], s=0.5, alpha=0.3, rasterized=True)
    ax_b.plot([0,1,0.5,0], [0,0,np.sqrt(3)/2,0], "k-", lw=1.5)
    ax_b.set_xlim(-0.1, 1.1)
    ax_b.set_ylim(-0.12, 1.0)
    ax_b.set_aspect("equal")
    ax_b.axis("off")
    ax_b.set_title("B.  Ternary Feature Space", fontweight="bold", fontsize=11)

    # ── C: Multi-scale profiles ──
    ax_c = fig.add_subplot(gs[0, 3])
    names = EigenFeatures.feature_names()
    radii = ms_features.radii
    for idx, fn in enumerate(names):
        means = []
        for s in ms_features.scales:
            vv = getattr(s, fn)
            vv = vv[np.isfinite(vv)]
            means.append(np.mean(vv) if len(vv) else 0)
        ax_c.plot(radii, means, "o-", color=get_feature_color(fn),
                  linewidth=1.5, markersize=4, label=fn[:5])
    ax_c.set_xlabel("Radius (mm)", fontsize=9)
    ax_c.set_ylabel("Mean value", fontsize=9)
    ax_c.legend(fontsize=6, ncol=2, loc="upper left")
    ax_c.grid(True, alpha=0.2)
    ax_c.set_title("C.  Multi-scale Profiles", fontweight="bold", fontsize=11)

    # ── D: Unique variance ──
    if comparison is not None:
        ax_d = fig.add_subplot(gs[1, 0:2])
        en = comparison["eigen_names"]
        uv = comparison["unique_variance"]
        yp = np.arange(len(en))
        bc = ["#E63946" if u > 0.5 else "#95a5a6" for u in uv]
        ax_d.barh(yp, uv, color=bc, edgecolor="none", height=0.7)
        ax_d.axvline(0.5, color="k", ls="--", lw=1)
        ax_d.set_yticks(yp)
        ax_d.set_yticklabels(en, fontsize=6)
        ax_d.set_xlabel("Unique Variance")
        ax_d.set_xlim(0, 1)
        ax_d.invert_yaxis()
        ax_d.set_title("D.  Novel Information vs Classical",
                        fontweight="bold", fontsize=11)
    else:
        ax_d = fig.add_subplot(gs[1, 0:2])
        ax_d.text(0.5, 0.5, "(Comparison not provided)",
                  ha="center", va="center", transform=ax_d.transAxes)
        ax_d.set_title("D.  Classical Comparison", fontweight="bold")

    # ── E, F: eigenentropy at finest & coarsest scale ──
    for pi, (s, lab) in enumerate([
        (ms_features.scales[0], f"r={ms_features.radii[0]:.0f}mm"),
        (ms_features.scales[-1], f"r={ms_features.radii[-1]:.0f}mm"),
    ]):
        ax_e = fig.add_subplot(gs[1, 2+pi], projection="3d")
        _surface_to_ax(ax_e, mesh, s.eigenentropy, cmap="viridis")
        letter = "E" if pi == 0 else "F"
        ax_e.set_title(f"{letter}.  Eigenentropy {lab}",
                        fontweight="bold", fontsize=11)

    # ── G: Radar fingerprints ──
    from scipy.spatial import cKDTree
    centroids = np.array([
        mesh.centroid + [30, 0, 0],
        mesh.centroid + [-30, 0, 0],
        mesh.centroid + [0, 30, 0],
        mesh.centroid + [0, -30, 0],
    ])
    tree = cKDTree(mesh.vertices)
    mid = len(ms_features.scales) // 2
    s_mid = ms_features.scales[mid]

    angles = np.linspace(0, 2*np.pi, len(names), endpoint=False).tolist()
    angles += angles[:1]
    quad_names = ["Anterior", "Posterior", "Superior", "Inferior"]

    for ri in range(4):
        ax_r = fig.add_subplot(gs[2, ri], polar=True)
        _, closest = tree.query(centroids[ri])
        nb = tree.query_ball_point(mesh.vertices[closest], 15.0)

        vals_r = []
        for fn in names:
            rv = getattr(s_mid, fn)[nb]
            rv = rv[np.isfinite(rv)]
            vals_r.append(np.mean(rv) if len(rv) else 0)

        mx = [max(abs(vr), 1e-12) for vr in vals_r]
        vals_n = [vr/m for vr, m in zip(vals_r, mx)]
        vals_n += vals_n[:1]

        col = plt.cm.Set1(ri / 4)
        ax_r.fill(angles, vals_n, alpha=0.25, color=col)
        ax_r.plot(angles, vals_n, "o-", lw=2, ms=3, color=col)
        ax_r.set_xticks(angles[:-1])
        ax_r.set_xticklabels([fn[:4] for fn in names], fontsize=7)
        ax_r.set_ylim(0, 1.2)
        ax_r.set_yticks([])
        ax_r.set_title(f"G{ri+1}. {quad_names[ri]}",
                        fontsize=10, fontweight="bold", pad=12)

    fig.suptitle(
        "EigenMorph: Multi-scale Geometric Features for Cortical Morphology",
        fontsize=16, fontweight="bold", y=0.98,
    )
    if save_path:
        fig.savefig(save_path)
    return fig
