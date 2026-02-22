# -*- coding: utf-8 -*-
"""
eigenmorph.features
===================

Extended geometric descriptors beyond the core seven eigenvalue features.

These capture higher-order, orientation-dependent, and multi-scale
properties of the cortical surface that complement the base features.

Functions
---------
compute_shape_index
    Koenderink shape classification (−1 to +1) from eigenvalues.
compute_curvedness
    Overall bending magnitude from eigenvalues.
compute_verticality
    Alignment of the first principal axis with the z-axis.
compute_normal_displacement
    Signed deviation of neighbours along vertex normal.
compute_surface_gradient
    Spatial gradient magnitude of any scalar feature on the mesh.
compute_fractal_dimension
    Multi-scale complexity via log–log slope of omnivariance.
compute_all_extended_features
    Convenience: compute everything and return a dict.
"""

import numpy as np
from scipy.spatial import cKDTree
from typing import Optional, Dict, List

from .core import SurfaceMesh, EigenFeatures, MultiScaleFeatures


# ═══════════════════════════════════════════════════════════════════════════
#  SHAPE INDEX & CURVEDNESS  (Koenderink & van Doorn, 1992)
# ═══════════════════════════════════════════════════════════════════════════

def compute_shape_index(features: EigenFeatures) -> np.ndarray:
    """
    Koenderink shape index from eigenvalue ratios.

    Maps local geometry onto a continuous scale from −1 (cup/pit)
    through 0 (saddle) to +1 (cap/dome), capturing shape *type*
    independently of scale.

    The shape index is derived from λ₂ and λ₃ of the covariance
    matrix, which encode the two smaller principal spreads::

        SI = (2/π) · arctan((λ₂ + λ₃) / (λ₂ − λ₃))

    Parameters
    ----------
    features : EigenFeatures

    Returns
    -------
    shape_index : np.ndarray (V,)
        Values in [−1, +1].  NaN where features are invalid.
    """
    evals = features.eigenvalues  # (V, 3)
    l2, l3 = evals[:, 1], evals[:, 2]
    diff = l2 - l3
    # Avoid division by zero
    safe_diff = np.where(np.abs(diff) < 1e-12, 1e-12, diff)
    si = (2.0 / np.pi) * np.arctan((l2 + l3) / safe_diff)
    si[~features.valid_mask] = np.nan
    return si


def compute_curvedness(features: EigenFeatures) -> np.ndarray:
    """
    Curvedness: overall bending magnitude.

    Measures *how much* the surface is curved, regardless of shape type.
    Computed as the RMS of the two smaller eigenvalues::

        C = sqrt((λ₂² + λ₃²) / 2)

    Parameters
    ----------
    features : EigenFeatures

    Returns
    -------
    curvedness : np.ndarray (V,)
    """
    evals = features.eigenvalues
    l2, l3 = evals[:, 1], evals[:, 2]
    c = np.sqrt((l2 ** 2 + l3 ** 2) / 2.0)
    c[~features.valid_mask] = np.nan
    return c


# ═══════════════════════════════════════════════════════════════════════════
#  VERTICALITY
# ═══════════════════════════════════════════════════════════════════════════

def compute_verticality(features: EigenFeatures) -> np.ndarray:
    """
    Verticality: alignment of the first principal axis with the z-axis.

    In cortical surfaces oriented in standard space (z = dorsal–ventral),
    verticality measures whether the dominant spread of a local
    neighbourhood is oriented vertically (gyral walls, sulcal banks)
    or horizontally (gyral crowns, sulcal fundi).

    ``V = 1 − |e₁ · ẑ|``

    Values near 1 indicate a horizontal principal axis (the
    neighbourhood is spread in the xy-plane, i.e. a flat region);
    values near 0 indicate a vertical principal axis (a sulcal wall).

    Parameters
    ----------
    features : EigenFeatures
        Must have been computed with ``store_eigenvectors=True``.

    Returns
    -------
    verticality : np.ndarray (V,)

    Raises
    ------
    ValueError
        If eigenvectors were not stored.
    """
    if features.eigenvectors is None:
        raise ValueError(
            "Eigenvectors not stored.  Re-compute with "
            "store_eigenvectors=True."
        )
    # First eigenvector (principal spread direction)
    e1 = features.eigenvectors[:, :, 0]  # (V, 3)
    z_hat = np.array([0.0, 0.0, 1.0])
    vert = 1.0 - np.abs(e1 @ z_hat)
    vert[~features.valid_mask] = np.nan
    return vert


# ═══════════════════════════════════════════════════════════════════════════
#  NORMAL DISPLACEMENT
# ═══════════════════════════════════════════════════════════════════════════

def compute_normal_displacement(
    mesh: SurfaceMesh,
    radius: float = 5.0,
) -> np.ndarray:
    """
    Signed displacement of neighbourhood centroid along vertex normal.

    Positive values indicate that the local neighbourhood bulges
    *outward* (convex — gyral crown), negative values indicate
    *inward* curvature (concave — sulcal fundus).

    Provides a geometrically intuitive complement to the covariance-
    based features: it captures *where* the vertex sits relative to
    its neighbours, not *how* they are spread.

    Parameters
    ----------
    mesh : SurfaceMesh
    radius : float
        Neighbourhood radius in mm.

    Returns
    -------
    displacement : np.ndarray (V,)
    """
    tree = cKDTree(mesh.vertices)
    all_nb = tree.query_ball_tree(tree, r=radius)
    disp = np.full(mesh.n_vertices, np.nan)

    for v in range(mesh.n_vertices):
        nb = all_nb[v]
        if len(nb) < 4:
            continue
        centroid = mesh.vertices[nb].mean(axis=0)
        diff = centroid - mesh.vertices[v]
        # Project onto outward normal
        disp[v] = np.dot(diff, mesh.vertex_normals[v])

    return disp


# ═══════════════════════════════════════════════════════════════════════════
#  SURFACE GRADIENT
# ═══════════════════════════════════════════════════════════════════════════

def compute_surface_gradient(
    mesh: SurfaceMesh,
    values: np.ndarray,
) -> np.ndarray:
    """
    Gradient magnitude of a scalar field on the mesh surface.

    Uses a face-based finite-difference scheme: for each face,
    computes the gradient in the tangent plane, then averages
    gradient magnitudes back to vertices (area-weighted).

    Useful for detecting *boundaries* between morphological regions
    (e.g. the rim of a sulcus, where eigenfeatures change rapidly).

    Parameters
    ----------
    mesh : SurfaceMesh
    values : np.ndarray (V,)
        Scalar field (any eigenfeature, thickness, etc.).

    Returns
    -------
    grad_mag : np.ndarray (V,)
        Gradient magnitude per vertex.
    """
    v0 = mesh.vertices[mesh.faces[:, 0]]
    v1 = mesh.vertices[mesh.faces[:, 1]]
    v2 = mesh.vertices[mesh.faces[:, 2]]

    f0 = values[mesh.faces[:, 0]]
    f1 = values[mesh.faces[:, 1]]
    f2 = values[mesh.faces[:, 2]]

    # Edge vectors
    e1 = v1 - v0
    e2 = v2 - v0

    # Face normals for area
    fn = np.cross(e1, e2)
    area2 = np.linalg.norm(fn, axis=1, keepdims=True)
    area2[area2 < 1e-12] = 1e-12

    # Face gradient in tangent plane (rotated edge method)
    n_hat = fn / area2
    # grad f = (1/(2A)) * Σ f_i * (n × e_opposite_i)
    grad = (
        f0[:, None] * np.cross(n_hat, v2 - v1) +
        f1[:, None] * np.cross(n_hat, v0 - v2) +
        f2[:, None] * np.cross(n_hat, v1 - v0)
    ) / area2

    # Face gradient magnitudes
    face_grad_mag = np.linalg.norm(grad, axis=1)

    # Average to vertices
    fa = 0.5 * area2.ravel()  # face areas
    grad_mag = np.zeros(mesh.n_vertices)
    weight_sum = np.zeros(mesh.n_vertices)
    for i in range(3):
        np.add.at(grad_mag, mesh.faces[:, i], face_grad_mag * fa)
        np.add.at(weight_sum, mesh.faces[:, i], fa)

    weight_sum[weight_sum == 0] = 1.0
    grad_mag /= weight_sum

    # NaN where input was NaN
    nan_verts = np.isnan(values)
    grad_mag[nan_verts] = np.nan

    return grad_mag


# ═══════════════════════════════════════════════════════════════════════════
#  FRACTAL DIMENSION (multi-scale complexity)
# ═══════════════════════════════════════════════════════════════════════════

def compute_fractal_dimension(
    ms_features: MultiScaleFeatures,
    feature_name: str = "omnivariance",
) -> np.ndarray:
    """
    Estimate local fractal dimension from multi-scale feature profiles.

    Uses the log–log slope of a scalar feature (typically omnivariance)
    across spatial scales.  Vertices where the feature grows rapidly
    with radius have higher fractal dimension (more complex geometry).

    ``FD_v = slope of log(feature) vs log(radius)``

    Parameters
    ----------
    ms_features : MultiScaleFeatures
    feature_name : str
        Which feature to use.  ``'omnivariance'`` (default) is the
        geometric mean of eigenvalues and scales naturally with radius.

    Returns
    -------
    fractal_dim : np.ndarray (V,)
        Estimated local fractal dimension (slope).  NaN where
        regression is undefined.
    """
    radii = np.array(ms_features.radii)
    log_r = np.log(radii)

    # (V, n_scales)
    feat_across_scales = ms_features.get_feature(feature_name)

    V = feat_across_scales.shape[0]
    fd = np.full(V, np.nan)

    for v in range(V):
        vals = feat_across_scales[v]
        valid = (vals > 0) & np.isfinite(vals)
        if valid.sum() < 2:
            continue
        log_vals = np.log(vals[valid])
        log_radii = log_r[valid]
        # Simple linear regression
        slope, _ = np.polyfit(log_radii, log_vals, 1)
        fd[v] = slope

    return fd


# ═══════════════════════════════════════════════════════════════════════════
#  CONVENIENCE: ALL EXTENDED FEATURES
# ═══════════════════════════════════════════════════════════════════════════

def compute_all_extended_features(
    mesh: SurfaceMesh,
    features: EigenFeatures,
    ms_features: Optional[MultiScaleFeatures] = None,
    compute_gradient_of: Optional[str] = "eigenentropy",
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Compute all available extended features in one call.

    Parameters
    ----------
    mesh : SurfaceMesh
    features : EigenFeatures
        Single-scale features (for shape index, curvedness, etc.).
    ms_features : MultiScaleFeatures, optional
        Required only for fractal dimension.
    compute_gradient_of : str, optional
        Feature to compute surface gradient of.  None to skip.
    verbose : bool

    Returns
    -------
    dict of {name: (V,) array}
    """
    out = {}

    if verbose:
        print("  Computing extended features…")

    out["shape_index"] = compute_shape_index(features)
    out["curvedness"] = compute_curvedness(features)

    if features.eigenvectors is not None:
        out["verticality"] = compute_verticality(features)
        if verbose:
            print("    ✓ verticality")

    out["normal_displacement"] = compute_normal_displacement(
        mesh, radius=features.radius
    )

    if compute_gradient_of is not None:
        vals = getattr(features, compute_gradient_of, None)
        if vals is not None:
            out[f"gradient_{compute_gradient_of}"] = (
                compute_surface_gradient(mesh, vals)
            )

    if ms_features is not None:
        out["fractal_dimension"] = compute_fractal_dimension(ms_features)
        if verbose:
            print("    ✓ fractal dimension")

    if verbose:
        valid_counts = {k: np.isfinite(v).sum() for k, v in out.items()}
        for k, c in valid_counts.items():
            print(f"    {k}: {c:,} valid vertices")

    return out
