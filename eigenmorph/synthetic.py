# -*- coding: utf-8 -*-
"""
eigenmorph.synthetic
====================

Synthetic cortical surface generation for testing and demonstration.

Generates icosphere-based cortical surfaces with sinusoidal radial
deformations that mimic gyral and sulcal folds, plus mock classical
morphometrics.  Also provides group-data generation for statistical
testing pipelines.
"""

import numpy as np
from typing import Tuple, Dict, List, Optional

from .core import SurfaceMesh


# ═══════════════════════════════════════════════════════════════════════════
#  ICOSPHERE
# ═══════════════════════════════════════════════════════════════════════════

def _icosphere(subdivisions: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate an icosphere via recursive subdivision.

    Parameters
    ----------
    subdivisions : int
        4 → 2,562 verts · 5 → 10,242 · 6 → 40,962
    """
    t = (1.0 + np.sqrt(5.0)) / 2.0

    verts_list = [
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
    ]
    for i in range(len(verts_list)):
        v = verts_list[i]
        norm = (v[0]**2 + v[1]**2 + v[2]**2) ** 0.5
        verts_list[i] = [v[0]/norm, v[1]/norm, v[2]/norm]

    faces_list = [
        [0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],
        [1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],
        [3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],
        [4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1],
    ]

    for _ in range(subdivisions):
        edge_cache = {}
        new_faces = []

        def get_midpoint(i, j):
            key = (min(i, j), max(i, j))
            if key in edge_cache:
                return edge_cache[key]
            v1 = verts_list[i]
            v2 = verts_list[j]
            mid = [(v1[0]+v2[0])/2, (v1[1]+v2[1])/2, (v1[2]+v2[2])/2]
            norm = (mid[0]**2 + mid[1]**2 + mid[2]**2) ** 0.5
            mid = [mid[0]/norm, mid[1]/norm, mid[2]/norm]
            idx = len(verts_list)
            verts_list.append(mid)
            edge_cache[key] = idx
            return idx

        for tri in faces_list:
            a, b, c = tri
            ab = get_midpoint(a, b)
            bc = get_midpoint(b, c)
            ca = get_midpoint(c, a)
            new_faces.extend([
                [a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca],
            ])
        faces_list = new_faces

    return np.array(verts_list), np.array(faces_list, dtype=int)


# ═══════════════════════════════════════════════════════════════════════════
#  SYNTHETIC CORTEX
# ═══════════════════════════════════════════════════════════════════════════

def generate_synthetic_cortex(
    n_vertices: int = 10_000,
    n_gyri: int = 8,
    seed: int = 42,
) -> Tuple[SurfaceMesh, Dict[str, np.ndarray]]:
    """
    Generate a synthetic cortical surface with gyri and sulci.

    Uses icosphere subdivision for a proper triangle mesh, then deforms
    radially with sinusoidal folds.

    Parameters
    ----------
    n_vertices : int
        Approximate target: ≤3k→sub4(2562), ≤15k→sub5(10242),
        >15k→sub6(40962).
    n_gyri : int
        Number of major gyral folds.
    seed : int

    Returns
    -------
    mesh : SurfaceMesh
    classical : dict
        ``{'thickness': (V,), 'curv': (V,), 'sulc': (V,)}``
    """
    rng = np.random.default_rng(seed)

    if n_vertices <= 3_000:
        subdiv = 4
    elif n_vertices <= 15_000:
        subdiv = 5
    else:
        subdiv = 6

    unit_verts, faces = _icosphere(subdiv)
    V = len(unit_verts)
    base_radius = 80.0

    x, y, z = unit_verts.T
    theta = np.arctan2(y, x)
    phi = np.arccos(np.clip(z, -1, 1))

    # Major gyral folds (spherical-harmonic-like)
    deform = np.zeros(V)
    for _ in range(n_gyri):
        amp = rng.uniform(2.0, 5.0)
        ft, fp = rng.uniform(1, 4), rng.uniform(1, 3)
        phase = rng.uniform(0, 2 * np.pi)
        deform += amp * np.sin(ft * theta + phase) * np.cos(fp * phi)

    # Fine folds
    for _ in range(20):
        amp = rng.uniform(0.5, 1.5)
        freq = rng.uniform(4, 12)
        phase = rng.uniform(0, 2 * np.pi)
        target = theta if rng.random() < 0.5 else phi
        deform += amp * np.sin(freq * target + phase)

    r = base_radius + deform
    vertices = unit_verts * r[:, np.newaxis]

    mesh = SurfaceMesh(
        vertices=vertices, faces=faces,
        hemisphere="lh", surface_type="synthetic",
    )

    # Mock classical morphometrics
    d_std = max(deform.std(), 1e-6)
    thickness = 2.5 + 0.3 * (deform - deform.mean()) / d_std
    thickness += rng.normal(0, 0.1, V)
    thickness = np.clip(thickness, 1.0, 4.5)

    curv = -np.gradient(deform) + rng.normal(0, 0.05, V)
    sulc = -(deform - deform.max()) / max((-deform + deform.max()).max(), 1e-6)
    sulc += rng.normal(0, 0.02, V)

    return mesh, {"thickness": thickness, "curv": curv, "sulc": sulc}


def generate_vertex_parcellation(
    mesh: SurfaceMesh,
    n_parcels: int = 100,
    seed: int = 42,
) -> np.ndarray:
    """
    Mock vertex-to-parcel assignment via k-means.

    For real data use FreeSurfer annotation files instead.

    Returns
    -------
    labels : (V,) int, 1-indexed (0 = unassigned).
    """
    from scipy.cluster.vq import kmeans2
    _, labels = kmeans2(mesh.vertices, n_parcels, minit="points", seed=seed)
    return labels + 1


# ═══════════════════════════════════════════════════════════════════════════
#  GROUP DATA GENERATION (for stats pipeline testing)
# ═══════════════════════════════════════════════════════════════════════════

def generate_group_data(
    n_subjects_per_group: int = 20,
    n_parcels: int = 100,
    n_features: int = 28,
    effect_parcels: Optional[List[int]] = None,
    effect_size: float = 0.8,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate simulated group-level eigenmorph data for testing stats.

    Creates two groups with a planted effect in specified parcels.

    Parameters
    ----------
    n_subjects_per_group : int
    n_parcels : int
    n_features : int
    effect_parcels : list of int, optional
        Parcel indices (0-based) with planted effects.
        Default: 10% of parcels chosen randomly.
    effect_size : float
        Cohen's d of the planted effect.
    seed : int

    Returns
    -------
    group1 : (n1, n_parcels * n_features)
    group2 : (n2, n_parcels * n_features)
    effect_mask : (n_parcels * n_features,) bool
        True where effects were planted.
    """
    rng = np.random.default_rng(seed)
    P = n_parcels * n_features
    n1 = n2 = n_subjects_per_group

    group1 = rng.standard_normal((n1, P))
    group2 = rng.standard_normal((n2, P))

    if effect_parcels is None:
        n_eff = max(1, n_parcels // 10)
        effect_parcels = rng.choice(n_parcels, n_eff, replace=False).tolist()

    effect_mask = np.zeros(P, dtype=bool)
    for p in effect_parcels:
        start = p * n_features
        end = start + n_features
        group2[:, start:end] += effect_size
        effect_mask[start:end] = True

    return group1, group2, effect_mask
