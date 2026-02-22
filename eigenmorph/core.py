# -*- coding: utf-8 -*-
"""
eigenmorph.core
===============

Core eigenvalue geometric feature computation for cortical surfaces.

Treats cortical surface meshes as 3D point clouds and extracts
eigenvalue-based geometric descriptors from local vertex neighborhoods.
The covariance matrix of each vertex's spatial neighborhood is
decomposed into eigenvalues λ₁ ≥ λ₂ ≥ λ₃, yielding seven
complementary geometric features.

Supports three neighbourhood weighting schemes (uniform, Gaussian,
inverse-distance) and optionally stores the full eigenvector frame
for downstream orientation analysis (verticality, principal directions).

Classes
-------
SurfaceMesh
    Container for cortical surface geometry (vertices + faces).
EigenFeatures
    Single-scale eigenvalue features for all vertices.
MultiScaleFeatures
    Multi-scale features across multiple neighbourhood radii.
"""

import numpy as np
from scipy.spatial import cKDTree
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass, field


# ═══════════════════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SurfaceMesh:
    """
    A cortical surface mesh.

    Parameters
    ----------
    vertices : array-like, shape (V, 3)
        Vertex coordinates in mm (e.g. FreeSurfer RAS space).
    faces : array-like, shape (F, 3)
        Triangle face indices (0-indexed).
    vertex_normals : array-like, shape (V, 3), optional
        Per-vertex normals. Computed automatically if absent.
    hemisphere : str
        ``'lh'``, ``'rh'``, or ``'both'``.
    surface_type : str
        ``'white'``, ``'pial'``, ``'inflated'``, ``'sphere'``,
        ``'synthetic'``, etc.
    metadata : dict
        Arbitrary metadata (subject ID, atlas name, …).
    """

    vertices: np.ndarray
    faces: np.ndarray
    vertex_normals: Optional[np.ndarray] = None
    n_vertices: int = 0
    n_faces: int = 0
    hemisphere: str = "unknown"
    surface_type: str = "unknown"
    metadata: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self.vertices = np.asarray(self.vertices, dtype=np.float64)
        self.faces = np.asarray(self.faces, dtype=np.int64)
        self.n_vertices = self.vertices.shape[0]
        self.n_faces = self.faces.shape[0]
        if self.vertex_normals is None:
            self.vertex_normals = self._compute_normals()
        else:
            self.vertex_normals = np.asarray(
                self.vertex_normals, dtype=np.float64
            )

    # ── normals ──────────────────────────────────────────────────────────
    def _compute_normals(self) -> np.ndarray:
        """Area-weighted per-vertex normals from face cross-products."""
        normals = np.zeros_like(self.vertices)
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]
        face_normals = np.cross(v1 - v0, v2 - v0)  # magnitude = 2×area
        for i in range(3):
            np.add.at(normals, self.faces[:, i], face_normals)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return normals / norms

    # ── geometric properties ─────────────────────────────────────────────
    @property
    def centroid(self) -> np.ndarray:
        """(3,) centre of mass."""
        return self.vertices.mean(axis=0)

    @property
    def extent(self) -> np.ndarray:
        """(3,) spatial extent per axis in mm."""
        return np.ptp(self.vertices, axis=0)

    def face_areas(self) -> np.ndarray:
        """(F,) triangle areas in mm²."""
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]
        return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)

    def total_area(self) -> float:
        """Total surface area in mm²."""
        return float(self.face_areas().sum())

    def vertex_areas(self) -> np.ndarray:
        """(V,) area associated with each vertex (⅓ of incident faces)."""
        fa = self.face_areas()
        va = np.zeros(self.n_vertices)
        for i in range(3):
            np.add.at(va, self.faces[:, i], fa / 3.0)
        return va

    def mean_edge_length(self) -> float:
        """Average edge length in mm (proxy for vertex spacing)."""
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]
        e = np.concatenate([
            np.linalg.norm(v1 - v0, axis=1),
            np.linalg.norm(v2 - v1, axis=1),
            np.linalg.norm(v0 - v2, axis=1),
        ])
        return float(e.mean())

    def copy(self) -> "SurfaceMesh":
        """Deep copy."""
        return SurfaceMesh(
            vertices=self.vertices.copy(),
            faces=self.faces.copy(),
            vertex_normals=(self.vertex_normals.copy()
                            if self.vertex_normals is not None else None),
            hemisphere=self.hemisphere,
            surface_type=self.surface_type,
            metadata=dict(self.metadata) if self.metadata else {},
        )


# ─────────────────────────────────────────────────────────────────────────

@dataclass
class EigenFeatures:
    """
    Eigenvalue geometric features at a single spatial scale.

    The seven features partition the local 3D shape space:

    - **linearity** (λ₁−λ₂)/λ₁ — ridge-like (sulcal fundi)
    - **planarity** (λ₂−λ₃)/λ₁ — planar (gyral crowns)
    - **sphericity** λ₃/λ₁ — isotropic (sulcal pits / intersections)
    - **omnivariance** (λ₁λ₂λ₃)^⅓ — spatial dispersion
    - **anisotropy** (λ₁−λ₃)/λ₁ — directional bias
    - **eigenentropy** −Σλ̃ᵢln(λ̃ᵢ) — shape complexity
    - **surface_variation** λ₃/Σλ — local roughness

    Parameters
    ----------
    radius : float
    linearity … surface_variation : (V,) arrays
    eigenvalues : (V, 3) raw λ₁, λ₂, λ₃
    eigenvectors : (V, 3, 3), optional
        Columns [e₁, e₂, e₃] per vertex (principal directions).
    n_neighbors : (V,) count of neighbours per vertex
    weighting : str
        ``'uniform'``, ``'gaussian'``, or ``'distance'``
    """

    radius: float
    linearity: np.ndarray
    planarity: np.ndarray
    sphericity: np.ndarray
    omnivariance: np.ndarray
    anisotropy: np.ndarray
    eigenentropy: np.ndarray
    surface_variation: np.ndarray
    eigenvalues: np.ndarray = field(repr=False)
    eigenvectors: Optional[np.ndarray] = field(default=None, repr=False)
    n_neighbors: Optional[np.ndarray] = field(default=None, repr=False)
    weighting: str = "uniform"

    # ── accessors ────────────────────────────────────────────────────────
    def as_dict(self) -> Dict[str, np.ndarray]:
        """Return all 7 features as ``{name: (V,) array}``."""
        return {n: getattr(self, n) for n in self.feature_names()}

    def as_matrix(self) -> np.ndarray:
        """(V, 7) feature matrix in canonical order."""
        return np.column_stack([getattr(self, n) for n in self.feature_names()])

    @staticmethod
    def feature_names() -> List[str]:
        """Canonical feature name order."""
        return [
            "linearity", "planarity", "sphericity", "omnivariance",
            "anisotropy", "eigenentropy", "surface_variation",
        ]

    @property
    def valid_mask(self) -> np.ndarray:
        """(V,) boolean — True where all features are finite."""
        return ~np.isnan(self.linearity)

    @property
    def n_valid(self) -> int:
        return int(self.valid_mask.sum())

    def summary(self) -> Dict[str, Dict[str, float]]:
        """Per-feature descriptive statistics (valid vertices only)."""
        out = {}
        for name, vals in self.as_dict().items():
            v = vals[self.valid_mask]
            if len(v) == 0:
                out[name] = {k: float("nan")
                             for k in ("mean", "std", "median", "min", "max")}
            else:
                out[name] = {
                    "mean": float(np.mean(v)), "std": float(np.std(v)),
                    "median": float(np.median(v)),
                    "min": float(np.min(v)), "max": float(np.max(v)),
                }
        return out

    def __repr__(self):
        V = len(self.linearity)
        return (f"EigenFeatures(radius={self.radius}, V={V}, "
                f"valid={self.n_valid}, weighting='{self.weighting}')")


# ─────────────────────────────────────────────────────────────────────────

@dataclass
class MultiScaleFeatures:
    """
    Multi-scale eigenvalue features across multiple neighbourhood radii.

    With default 4 radii (3, 5, 10, 20 mm) × 7 features = 28 features
    per vertex.

    Parameters
    ----------
    scales : list of EigenFeatures
    radii : list of float (in mm)
    n_vertices : int
    """

    scales: List[EigenFeatures]
    radii: List[float]
    n_vertices: int

    def as_matrix(self) -> np.ndarray:
        """Concatenated (V, 7×n_scales) feature matrix."""
        return np.hstack([s.as_matrix() for s in self.scales])

    def column_names(self) -> List[str]:
        """Feature names with scale suffix, e.g. ``linearity_r5mm``."""
        names = []
        for s in self.scales:
            for fn in EigenFeatures.feature_names():
                names.append(f"{fn}_r{s.radius:.0f}mm")
        return names

    def get_feature(self, name: str) -> np.ndarray:
        """Single feature across all scales → (V, n_scales)."""
        return np.column_stack([getattr(s, name) for s in self.scales])

    def get_scale(self, radius: float) -> EigenFeatures:
        """Retrieve the EigenFeatures for a specific radius."""
        for s in self.scales:
            if np.isclose(s.radius, radius):
                return s
        avail = [s.radius for s in self.scales]
        raise ValueError(
            f"Radius {radius} not found. Available: {avail}"
        )

    @property
    def valid_mask(self) -> np.ndarray:
        """(V,) True where ALL scales have valid features."""
        mat = self.as_matrix()
        return ~np.any(np.isnan(mat), axis=1)

    def __repr__(self):
        return (f"MultiScaleFeatures(radii={self.radii}, V={self.n_vertices}, "
                f"cols={len(self.column_names())})")


# ═══════════════════════════════════════════════════════════════════════════
#  CORE COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════

def _compute_weighted_cov(
    pts: np.ndarray,
    center: np.ndarray,
    weighting: str,
    radius: float,
) -> np.ndarray:
    """
    Compute 3×3 weighted covariance matrix of a point set.

    Parameters
    ----------
    pts : (N, 3)
    center : (3,) query vertex coordinate
    weighting : 'uniform', 'gaussian', 'distance'
    radius : float  (used for Gaussian sigma)

    Returns
    -------
    cov : (3, 3)
    """
    pts_c = pts - pts.mean(axis=0)
    n = len(pts)

    if weighting == "uniform":
        return (pts_c.T @ pts_c) / n

    elif weighting == "gaussian":
        # Gaussian weights: w_i = exp(-d²/(2σ²)), σ = radius/3
        dists = np.linalg.norm(pts - center, axis=1)
        sigma = radius / 3.0
        weights = np.exp(-0.5 * (dists / sigma) ** 2)
        weights /= weights.sum()
        # Weighted covariance
        weighted_pts = pts_c * np.sqrt(weights[:, np.newaxis])
        return weighted_pts.T @ weighted_pts

    elif weighting == "distance":
        # Inverse-distance weights: w_i = 1/max(d_i, ε)
        dists = np.linalg.norm(pts - center, axis=1)
        weights = 1.0 / np.maximum(dists, 1e-6)
        weights /= weights.sum()
        weighted_pts = pts_c * np.sqrt(weights[:, np.newaxis])
        return weighted_pts.T @ weighted_pts

    else:
        raise ValueError(f"Unknown weighting '{weighting}'. "
                         f"Use 'uniform', 'gaussian', or 'distance'.")


def compute_eigenfeatures(
    mesh: SurfaceMesh,
    radius: float = 5.0,
    min_neighbors: int = 6,
    weighting: str = "uniform",
    store_eigenvectors: bool = False,
    verbose: bool = True,
) -> EigenFeatures:
    """
    Compute eigenvalue geometric features for each vertex.

    For each vertex the algorithm finds all neighbours within *radius*
    mm via a KD-tree, builds the (optionally weighted) 3×3 covariance
    matrix, extracts eigenvalues λ₁ ≥ λ₂ ≥ λ₃, and derives seven
    normalised geometric features.

    Parameters
    ----------
    mesh : SurfaceMesh
    radius : float
        Neighbourhood radius in mm.  Typical values::

            3 mm  — fine cortical folds / sulcal branches
            5 mm  — individual gyri and sulci
            10 mm — regional geometry
            20 mm — lobar-scale shape

    min_neighbors : int
        Minimum neighbours for valid computation. Vertices with fewer
        neighbours receive NaN features.
    weighting : str
        ``'uniform'`` (default), ``'gaussian'``, or ``'distance'``.
        Gaussian uses σ = radius/3; distance uses 1/d weights.
    store_eigenvectors : bool
        If True, store the full (V, 3, 3) eigenvector array for
        downstream orientation analysis (verticality, principal
        directions). Memory-intensive for large meshes.
    verbose : bool
        Print progress and summary statistics.

    Returns
    -------
    EigenFeatures
    """
    V = mesh.n_vertices
    coords = mesh.vertices

    if verbose:
        print(f"  Computing eigenfeatures "
              f"(radius={radius}mm, V={V:,}, weighting={weighting})…")

    # Build KD-tree
    tree = cKDTree(coords)

    # Allocate output arrays
    eigenvalues = np.full((V, 3), np.nan)
    eigenvectors = np.full((V, 3, 3), np.nan) if store_eigenvectors else None
    n_neighbors = np.zeros(V, dtype=np.int32)

    linearity = np.full(V, np.nan)
    planarity = np.full(V, np.nan)
    sphericity = np.full(V, np.nan)
    omnivariance = np.full(V, np.nan)
    anisotropy = np.full(V, np.nan)
    eigenentropy = np.full(V, np.nan)
    surface_variation = np.full(V, np.nan)

    # Batch neighbourhood query
    all_neighbors = tree.query_ball_tree(tree, r=radius)

    for v in range(V):
        neighbors = all_neighbors[v]
        n_neigh = len(neighbors)
        n_neighbors[v] = n_neigh

        if n_neigh < min_neighbors:
            continue

        pts = coords[neighbors]

        # Weighted covariance
        cov = _compute_weighted_cov(pts, coords[v], weighting, radius)

        # Eigendecomposition (eigh returns ascending order)
        evals, evecs = np.linalg.eigh(cov)
        evals = evals[::-1]             # descending: λ₁ ≥ λ₂ ≥ λ₃
        evecs = evecs[:, ::-1]          # match eigenvector order
        evals = np.maximum(evals, 0.0)  # numerical safety

        eigenvalues[v] = evals
        if store_eigenvectors:
            eigenvectors[v] = evecs

        l1, l2, l3 = evals
        if l1 < 1e-12:
            continue

        sum_l = l1 + l2 + l3

        # ── Seven geometric features ──
        linearity[v] = (l1 - l2) / l1
        planarity[v] = (l2 - l3) / l1
        sphericity[v] = l3 / l1
        omnivariance[v] = (l1 * l2 * l3) ** (1.0 / 3.0)
        anisotropy[v] = (l1 - l3) / l1

        # Eigenentropy: Shannon entropy of normalised eigenvalues
        l_norm = evals / sum_l
        l_norm = l_norm[l_norm > 1e-12]
        eigenentropy[v] = -np.sum(l_norm * np.log(l_norm))

        # Surface variation (change of curvature)
        surface_variation[v] = l3 / sum_l

    if verbose:
        valid = ~np.isnan(linearity)
        pct = valid.sum() / V * 100
        print(f"    Valid vertices: {valid.sum():,}/{V:,} ({pct:.1f}%)")
        if valid.any():
            print(f"    Mean neighbours: {n_neighbors[valid].mean():.0f}")
            print(f"    Linearity:  {np.nanmean(linearity):.4f} "
                  f"± {np.nanstd(linearity):.4f}")
            print(f"    Planarity:  {np.nanmean(planarity):.4f} "
                  f"± {np.nanstd(planarity):.4f}")
            print(f"    Sphericity: {np.nanmean(sphericity):.4f} "
                  f"± {np.nanstd(sphericity):.4f}")

    return EigenFeatures(
        radius=radius,
        linearity=linearity,
        planarity=planarity,
        sphericity=sphericity,
        omnivariance=omnivariance,
        anisotropy=anisotropy,
        eigenentropy=eigenentropy,
        surface_variation=surface_variation,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        n_neighbors=n_neighbors,
        weighting=weighting,
    )


def compute_multiscale_eigenfeatures(
    mesh: SurfaceMesh,
    radii: Optional[List[float]] = None,
    min_neighbors: int = 6,
    weighting: str = "uniform",
    store_eigenvectors: bool = False,
    verbose: bool = True,
) -> MultiScaleFeatures:
    """
    Compute eigenvalue features at multiple spatial scales.

    Parameters
    ----------
    mesh : SurfaceMesh
    radii : list of float, optional
        Default ``[3.0, 5.0, 10.0, 20.0]``.
    min_neighbors : int
    weighting : str
    store_eigenvectors : bool
    verbose : bool

    Returns
    -------
    MultiScaleFeatures
        7 features × len(radii) scales per vertex.
    """
    if radii is None:
        radii = [3.0, 5.0, 10.0, 20.0]

    if verbose:
        print(f"\n  Multi-scale eigenfeatures: {len(radii)} scales")
        print(f"  Radii: {radii} mm | weighting: {weighting}")

    scales = []
    for r in sorted(radii):
        feat = compute_eigenfeatures(
            mesh, radius=r, min_neighbors=min_neighbors,
            weighting=weighting,
            store_eigenvectors=store_eigenvectors,
            verbose=verbose,
        )
        scales.append(feat)

    return MultiScaleFeatures(
        scales=scales, radii=sorted(radii), n_vertices=mesh.n_vertices,
    )
