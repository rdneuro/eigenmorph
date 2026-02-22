# -*- coding: utf-8 -*-
"""
eigenmorph.utils
================

Utility functions for mesh processing, feature normalisation, and
surface-based data smoothing.
"""

import numpy as np
from scipy.sparse import csr_matrix, eye as speye
from typing import Optional

from .core import SurfaceMesh


# ═══════════════════════════════════════════════════════════════════════════
#  MESH TOPOLOGY
# ═══════════════════════════════════════════════════════════════════════════

def compute_mesh_adjacency(
    mesh: SurfaceMesh,
    weighted: bool = False,
) -> csr_matrix:
    """
    Compute vertex adjacency matrix from mesh faces.

    Parameters
    ----------
    mesh : SurfaceMesh
    weighted : bool
        If True, edge weights are inverse edge lengths (shorter edges
        → stronger connection).  If False, binary {0, 1}.

    Returns
    -------
    adj : scipy.sparse.csr_matrix (V, V)
    """
    V = mesh.n_vertices
    edges = set()
    for f in mesh.faces:
        for i in range(3):
            a, b = int(f[i]), int(f[(i + 1) % 3])
            edges.add((min(a, b), max(a, b)))

    rows, cols, data = [], [], []
    for a, b in edges:
        if weighted:
            d = np.linalg.norm(mesh.vertices[a] - mesh.vertices[b])
            w = 1.0 / max(d, 1e-6)
        else:
            w = 1.0
        rows.extend([a, b])
        cols.extend([b, a])
        data.extend([w, w])

    adj = csr_matrix((data, (rows, cols)), shape=(V, V))
    return adj


def mesh_edges(mesh: SurfaceMesh) -> np.ndarray:
    """
    Extract unique edges from mesh faces.

    Returns
    -------
    edges : (E, 2) int array, sorted (a < b).
    """
    edge_set = set()
    for f in mesh.faces:
        for i in range(3):
            a, b = int(f[i]), int(f[(i + 1) % 3])
            edge_set.add((min(a, b), max(a, b)))
    return np.array(sorted(edge_set), dtype=np.int64)


def mesh_area(mesh: SurfaceMesh) -> float:
    """Total surface area in mm²."""
    return mesh.total_area()


# ═══════════════════════════════════════════════════════════════════════════
#  SURFACE SMOOTHING
# ═══════════════════════════════════════════════════════════════════════════

def smooth_surface_data(
    mesh: SurfaceMesh,
    data: np.ndarray,
    fwhm: float = 5.0,
    n_iterations: Optional[int] = None,
) -> np.ndarray:
    """
    Iterative nearest-neighbour smoothing on the mesh surface.

    Applies heat-kernel-like diffusion using the mesh adjacency to
    smooth vertex-wise data while respecting cortical topology.

    Parameters
    ----------
    mesh : SurfaceMesh
    data : np.ndarray (V,) or (V, K)
        Per-vertex scalar(s) to smooth.
    fwhm : float
        Full-width at half-maximum in mm.
    n_iterations : int, optional
        Number of smoothing iterations.  If None, auto-computed from
        ``fwhm`` and mean edge length.

    Returns
    -------
    smoothed : same shape as data
    """
    adj = compute_mesh_adjacency(mesh, weighted=False)

    # Number of iterations approximation: fwhm² ≈ 8 ln(2) × n × δ²
    if n_iterations is None:
        delta = mesh.mean_edge_length()
        n_iterations = max(1, int(round(
            fwhm ** 2 / (8.0 * np.log(2.0) * delta ** 2)
        )))

    # Build normalised smoothing operator
    degree = np.array(adj.sum(axis=1)).ravel()
    degree[degree == 0] = 1.0
    D_inv = csr_matrix((1.0 / degree, (range(mesh.n_vertices),
                                         range(mesh.n_vertices))),
                        shape=(mesh.n_vertices, mesh.n_vertices))
    smoother = D_inv @ adj

    # Mix with identity (weighted Jacobi)
    alpha = 0.5
    op = alpha * smoother + (1.0 - alpha) * speye(mesh.n_vertices)

    result = data.copy().astype(np.float64)
    is_1d = result.ndim == 1

    if is_1d:
        result = result[:, np.newaxis]

    # Preserve NaN locations
    nan_mask = np.isnan(result)
    result[nan_mask] = 0.0

    for _ in range(n_iterations):
        result = op @ result

    result[nan_mask] = np.nan

    return result.ravel() if is_1d else result


# ═══════════════════════════════════════════════════════════════════════════
#  FEATURE NORMALISATION
# ═══════════════════════════════════════════════════════════════════════════

def normalize_features(
    features: np.ndarray,
    method: str = "zscore",
    axis: int = 0,
) -> np.ndarray:
    """
    Normalise feature arrays.

    Parameters
    ----------
    features : np.ndarray (V, K) or (V,)
    method : str
        ``'zscore'`` — subtract mean, divide by std.
        ``'minmax'`` — scale to [0, 1].
        ``'robust'`` — median/IQR normalisation.
    axis : int
        Axis along which to normalise.

    Returns
    -------
    normalised : same shape
    """
    f = features.astype(np.float64).copy()

    if method == "zscore":
        mu = np.nanmean(f, axis=axis, keepdims=True)
        sigma = np.nanstd(f, axis=axis, keepdims=True)
        sigma[sigma < 1e-12] = 1.0
        return (f - mu) / sigma

    elif method == "minmax":
        fmin = np.nanmin(f, axis=axis, keepdims=True)
        fmax = np.nanmax(f, axis=axis, keepdims=True)
        rng = fmax - fmin
        rng[rng < 1e-12] = 1.0
        return (f - fmin) / rng

    elif method == "robust":
        med = np.nanmedian(f, axis=axis, keepdims=True)
        q25 = np.nanpercentile(f, 25, axis=axis, keepdims=True)
        q75 = np.nanpercentile(f, 75, axis=axis, keepdims=True)
        iqr = q75 - q25
        iqr[iqr < 1e-12] = 1.0
        return (f - med) / iqr

    else:
        raise ValueError(f"Unknown method '{method}'. "
                         f"Use 'zscore', 'minmax', or 'robust'.")
