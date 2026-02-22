# -*- coding: utf-8 -*-
"""
eigenmorph.parcellation
=======================

Parcel-level aggregation and comparison with classical morphometrics.

Provides three complementary analyses:

1. **Parcellation** — summarise vertex-wise features per brain region
   (Schaefer, Brainnetome, DK, etc.) for group-level statistics.
2. **Classical comparison** — quantify how much information eigenvalue
   features add beyond FreeSurfer thickness / curvature / sulcal depth.
3. **Morphological distance** — region × region distance matrix in
   eigenvalue feature space, for graph-theoretic or clustering analyses.
"""

import numpy as np
from scipy.stats import pearsonr
from scipy.spatial.distance import pdist, squareform
from typing import Optional, Dict, List, Union

from .core import EigenFeatures, MultiScaleFeatures


# ═══════════════════════════════════════════════════════════════════════════
#  PARCELLATED SUMMARIES
# ═══════════════════════════════════════════════════════════════════════════

def parcellate_features(
    features: Union[EigenFeatures, MultiScaleFeatures],
    vertex_labels: np.ndarray,
    n_parcels: int,
    parcel_names: Optional[List[str]] = None,
    aggregation: str = "mean",
) -> Dict:
    """
    Summarise vertex-wise features per brain parcel.

    Parameters
    ----------
    features : EigenFeatures or MultiScaleFeatures
    vertex_labels : np.ndarray (V,)
        Integer parcel label per vertex (1-indexed; 0 = medial wall).
    n_parcels : int
    parcel_names : list of str, optional
    aggregation : str
        ``'mean'``, ``'median'``, ``'std'``, or ``'all'``
        (returns mean, std, median, p25, p75).

    Returns
    -------
    dict
        ``{feature_name: (n_parcels,) array}`` for simple aggregation,
        or ``{feature_name: {stat: array}}`` for ``'all'``.
        Special keys: ``_parcel_names``, ``_column_names``.

    Raises
    ------
    ValueError
        If ``vertex_labels`` length does not match feature vertex count.
    """
    if isinstance(features, MultiScaleFeatures):
        feat_matrix = features.as_matrix()
        col_names = features.column_names()
        n_verts = features.n_vertices
    else:
        feat_matrix = features.as_matrix()
        col_names = [f"{fn}_r{features.radius:.0f}mm"
                     for fn in EigenFeatures.feature_names()]
        n_verts = feat_matrix.shape[0]

    if len(vertex_labels) != n_verts:
        raise ValueError(
            f"Shape mismatch: vertex_labels has {len(vertex_labels)} entries "
            f"but features have {n_verts} vertices."
        )

    result = {}

    for col_idx, col_name in enumerate(col_names):
        values = feat_matrix[:, col_idx]

        if aggregation == "all":
            stats = {k: np.zeros(n_parcels)
                     for k in ("mean", "std", "median", "p25", "p75")}

            for p in range(n_parcels):
                mask = vertex_labels == (p + 1)
                v = values[mask]
                v = v[~np.isnan(v)]
                if len(v) > 0:
                    stats["mean"][p] = np.mean(v)
                    stats["std"][p] = np.std(v)
                    stats["median"][p] = np.median(v)
                    stats["p25"][p] = np.percentile(v, 25)
                    stats["p75"][p] = np.percentile(v, 75)

            result[col_name] = stats
        else:
            pvals = np.zeros(n_parcels)
            agg_func = {
                "mean": np.mean, "median": np.median, "std": np.std,
            }.get(aggregation, np.mean)

            for p in range(n_parcels):
                mask = vertex_labels == (p + 1)
                v = values[mask]
                v = v[~np.isnan(v)]
                if len(v) > 0:
                    pvals[p] = agg_func(v)
            result[col_name] = pvals

    result["_parcel_names"] = (
        parcel_names or [f"parcel_{i+1}" for i in range(n_parcels)]
    )
    result["_column_names"] = col_names

    return result


# ═══════════════════════════════════════════════════════════════════════════
#  COMPARISON WITH CLASSICAL MORPHOMETRICS
# ═══════════════════════════════════════════════════════════════════════════

def compare_with_classical(
    eigen_features: MultiScaleFeatures,
    classical_metrics: Dict[str, np.ndarray],
    verbose: bool = True,
) -> Dict:
    """
    Compare eigenvalue features with classical FreeSurfer morphometrics.

    Computes pairwise Pearson correlations and estimates per-feature
    *unique variance* — the fraction NOT explained by any linear
    combination of classical metrics.  Features with unique variance
    > 0.5 capture predominantly novel geometric information.

    Parameters
    ----------
    eigen_features : MultiScaleFeatures
    classical_metrics : dict
        ``{name: (V,) array}`` — e.g. ``'thickness'``, ``'curv'``,
        ``'sulc'``, ``'K1'``, ``'K2'``.
    verbose : bool

    Returns
    -------
    dict
        ``'correlations'``   (n_eigen, n_classical) Pearson r
        ``'eigen_names'``    list
        ``'classical_names'``list
        ``'unique_variance'``(n_eigen,) fraction NOT explained
        ``'vif'``            (n_eigen,) variance inflation factors
    """
    eigen_matrix = eigen_features.as_matrix()
    eigen_names = eigen_features.column_names()
    classical_names = list(classical_metrics.keys())
    classical_matrix = np.column_stack(
        [classical_metrics[k] for k in classical_names]
    )

    n_eigen = eigen_matrix.shape[1]
    n_classical = classical_matrix.shape[1]

    # Remove NaN rows
    valid = (~np.any(np.isnan(eigen_matrix), axis=1) &
             ~np.any(np.isnan(classical_matrix), axis=1))
    eigen_v = eigen_matrix[valid]
    classical_v = classical_matrix[valid]

    if eigen_v.shape[0] < 10:
        if verbose:
            print(f"  ⚠ Only {eigen_v.shape[0]} valid vertices — skipping")
        nan_corr = np.full((n_eigen, n_classical), np.nan)
        return {
            "correlations": nan_corr,
            "eigen_names": eigen_names,
            "classical_names": classical_names,
            "unique_variance": np.full(n_eigen, np.nan),
            "vif": np.full(n_eigen, np.nan),
        }

    # Pearson correlation matrix
    correlations = np.zeros((n_eigen, n_classical))
    for i in range(n_eigen):
        for j in range(n_classical):
            correlations[i, j], _ = pearsonr(eigen_v[:, i], classical_v[:, j])

    # Unique variance via OLS: 1 − R² (eigenfeature ~ classical)
    unique_variance = np.zeros(n_eigen)
    vif = np.zeros(n_eigen)

    X = np.column_stack([np.ones(len(classical_v)), classical_v])

    for i in range(n_eigen):
        y = eigen_v[:, i]
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            pred = X @ beta
            ss_res = np.sum((y - pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            unique_variance[i] = 1.0 - r2
            vif[i] = 1.0 / max(1.0 - r2, 1e-12)
        except np.linalg.LinAlgError:
            unique_variance[i] = np.nan
            vif[i] = np.nan

    if verbose:
        print("\n  Eigenvalue features vs classical morphometrics:")
        print(f"  {'Feature':<35s} {'Max|r|':>7s} {'UniqueVar':>10s} "
              f"{'VIF':>6s}")
        print("  " + "─" * 62)
        for i, name in enumerate(eigen_names):
            mc = np.max(np.abs(correlations[i]))
            uv = unique_variance[i]
            vi = vif[i]
            marker = " ★" if uv > 0.5 else ""
            print(f"  {name:<35s} {mc:>7.3f} {uv:>10.3f} "
                  f"{vi:>6.1f}{marker}")
        n_novel = sum(1 for uv in unique_variance if uv > 0.5)
        print(f"\n  ★ = >50% unique variance ({n_novel}/{n_eigen} features)")

    return {
        "correlations": correlations,
        "eigen_names": eigen_names,
        "classical_names": classical_names,
        "unique_variance": unique_variance,
        "vif": vif,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  MORPHOLOGICAL DISTANCE MATRIX
# ═══════════════════════════════════════════════════════════════════════════

def morphological_distance_matrix(
    parcellated: Dict,
    metric: str = "correlation",
    standardize: bool = True,
) -> np.ndarray:
    """
    Compute a region × region distance matrix in eigenvalue feature space.

    This *morphological distance* encodes how structurally similar two
    brain regions are based on their eigenvalue feature profiles, which
    can be used for hierarchical clustering, community detection, or
    comparison with functional connectivity.

    Parameters
    ----------
    parcellated : dict
        Output from ``parcellate_features()``.
    metric : str
        ``'correlation'``, ``'euclidean'``, ``'cosine'``, or any metric
        accepted by ``scipy.spatial.distance.pdist``.
    standardize : bool
        z-score features before computing distances.

    Returns
    -------
    dist_matrix : np.ndarray (n_parcels, n_parcels)
        Symmetric distance matrix.
    """
    col_names = parcellated.get("_column_names", [])
    # Build (n_parcels, n_features) matrix
    cols = []
    for cn in col_names:
        vals = parcellated[cn]
        if isinstance(vals, dict):
            vals = vals.get("mean", np.zeros(1))
        cols.append(vals)

    feat_mat = np.column_stack(cols)  # (n_parcels, n_features)

    if standardize:
        mu = feat_mat.mean(axis=0)
        sigma = feat_mat.std(axis=0)
        sigma[sigma < 1e-12] = 1.0
        feat_mat = (feat_mat - mu) / sigma

    dists = squareform(pdist(feat_mat, metric=metric))
    return dists
