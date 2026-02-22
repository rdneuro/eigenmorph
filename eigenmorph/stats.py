# -*- coding: utf-8 -*-
"""
eigenmorph.stats
================

Vertex-wise and parcel-level statistical analysis for eigenvalue
morphology features.

Provides tools for group comparison, regression, permutation testing,
multiple-comparison correction, and effect size estimation — the
standard statistical workflow for cortical morphometry studies.

All functions operate on (n_subjects, n_features) matrices and return
arrays of the same shape, making them agnostic to whether the input
is vertex-wise or parcel-level.
"""

import numpy as np
from scipy import stats as sp_stats
from typing import Optional, Tuple, Dict


# ═══════════════════════════════════════════════════════════════════════════
#  MULTIPLE COMPARISON CORRECTION
# ═══════════════════════════════════════════════════════════════════════════

def fdr_correction(
    pvals: np.ndarray,
    alpha: float = 0.05,
    method: str = "bh",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Benjamini–Hochberg (or Benjamini–Yekutieli) FDR correction.

    Parameters
    ----------
    pvals : np.ndarray
        Uncorrected p-values (any shape).
    alpha : float
        Target FDR level.
    method : str
        ``'bh'`` (Benjamini–Hochberg) or ``'by'`` (Benjamini–Yekutieli).

    Returns
    -------
    reject : np.ndarray (bool)
        True where null hypothesis is rejected at FDR-corrected level.
    pvals_corrected : np.ndarray
        Adjusted p-values (same shape as input).
    """
    shape = pvals.shape
    p_flat = pvals.ravel().copy()
    n = len(p_flat)

    # Handle NaN
    nan_mask = np.isnan(p_flat)
    p_work = p_flat.copy()
    p_work[nan_mask] = 1.0

    # Sort
    sort_idx = np.argsort(p_work)
    p_sorted = p_work[sort_idx]

    # Correction factor
    ranks = np.arange(1, n + 1)
    if method == "by":
        c_m = np.sum(1.0 / ranks)
    else:
        c_m = 1.0

    # Adjusted p-values
    p_adjusted = np.minimum(1.0, p_sorted * n * c_m / ranks)

    # Enforce monotonicity (from largest to smallest)
    for i in range(n - 2, -1, -1):
        p_adjusted[i] = min(p_adjusted[i], p_adjusted[i + 1])

    # Unsort
    p_corrected = np.empty(n)
    p_corrected[sort_idx] = p_adjusted
    p_corrected[nan_mask] = np.nan

    reject = p_corrected < alpha
    reject[nan_mask] = False

    return reject.reshape(shape), p_corrected.reshape(shape)


# ═══════════════════════════════════════════════════════════════════════════
#  VERTEX-WISE T-TEST
# ═══════════════════════════════════════════════════════════════════════════

def vertex_wise_ttest(
    group1: np.ndarray,
    group2: np.ndarray,
    alpha: float = 0.05,
    fdr: bool = True,
    equal_var: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Vertex-wise (or parcel-wise) two-sample t-test.

    Parameters
    ----------
    group1 : np.ndarray (n1, P)
        Feature values for group 1 (n1 subjects × P features).
    group2 : np.ndarray (n2, P)
        Feature values for group 2.
    alpha : float
        Significance level.
    fdr : bool
        Apply FDR correction.
    equal_var : bool
        Assume equal variance (Student's t) if True; Welch's t if False.

    Returns
    -------
    dict
        ``'t_stat'``         (P,) t-statistics
        ``'p_uncorrected'``  (P,) uncorrected two-tailed p-values
        ``'p_corrected'``    (P,) FDR-corrected p-values (if fdr=True)
        ``'significant'``    (P,) boolean mask
        ``'cohens_d'``       (P,) Cohen's d effect size
        ``'mean_diff'``      (P,) group1.mean − group2.mean
    """
    n1, P = group1.shape
    n2 = group2.shape[0]

    t_stat = np.full(P, np.nan)
    p_vals = np.full(P, np.nan)

    for j in range(P):
        g1 = group1[:, j]
        g2 = group2[:, j]
        # Drop NaN subjects
        g1 = g1[np.isfinite(g1)]
        g2 = g2[np.isfinite(g2)]
        if len(g1) < 2 or len(g2) < 2:
            continue
        t_stat[j], p_vals[j] = sp_stats.ttest_ind(
            g1, g2, equal_var=equal_var
        )

    result = {
        "t_stat": t_stat,
        "p_uncorrected": p_vals,
        "mean_diff": np.nanmean(group1, axis=0) - np.nanmean(group2, axis=0),
    }

    # Effect size (Cohen's d)
    result["cohens_d"] = compute_effect_sizes(group1, group2)

    if fdr:
        sig, p_corr = fdr_correction(p_vals, alpha=alpha)
        result["p_corrected"] = p_corr
        result["significant"] = sig
    else:
        result["p_corrected"] = p_vals
        result["significant"] = p_vals < alpha

    return result


# ═══════════════════════════════════════════════════════════════════════════
#  VERTEX-WISE GLM
# ═══════════════════════════════════════════════════════════════════════════

def vertex_wise_glm(
    features: np.ndarray,
    design_matrix: np.ndarray,
    contrast: Optional[np.ndarray] = None,
    alpha: float = 0.05,
    fdr: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Vertex-wise general linear model (mass-univariate).

    Fits ``Y[:, j] = X @ β[:, j] + ε`` for each feature *j*
    independently, then tests a contrast vector on the coefficients.

    Parameters
    ----------
    features : np.ndarray (N, P)
        N subjects × P features (vertex-wise or parcel-wise).
    design_matrix : np.ndarray (N, K)
        Full design matrix including intercept if desired.
    contrast : np.ndarray (K,), optional
        Contrast vector.  Default tests the *last* regressor.
    alpha : float
    fdr : bool

    Returns
    -------
    dict
        ``'beta'``           (K, P)
        ``'t_stat'``         (P,)
        ``'p_uncorrected'``  (P,)
        ``'p_corrected'``    (P,) if fdr
        ``'significant'``    (P,)
        ``'r_squared'``      (P,)
    """
    N, P = features.shape
    K = design_matrix.shape[1]

    if contrast is None:
        contrast = np.zeros(K)
        contrast[-1] = 1.0

    beta = np.full((K, P), np.nan)
    t_stat = np.full(P, np.nan)
    p_vals = np.full(P, np.nan)
    r_sq = np.full(P, np.nan)

    X = design_matrix
    XtX_inv = None
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(X.T @ X)

    c = contrast

    for j in range(P):
        y = features[:, j]
        valid = np.isfinite(y)
        if valid.sum() < K + 1:
            continue

        y_v = y[valid]
        X_v = X[valid]

        try:
            XtX_inv_v = np.linalg.inv(X_v.T @ X_v)
        except np.linalg.LinAlgError:
            continue

        b = XtX_inv_v @ X_v.T @ y_v
        beta[:, j] = b

        resid = y_v - X_v @ b
        df = len(y_v) - K
        if df < 1:
            continue

        mse = np.sum(resid ** 2) / df
        var_contrast = mse * (c @ XtX_inv_v @ c)
        if var_contrast <= 0:
            continue

        t_stat[j] = (c @ b) / np.sqrt(var_contrast)
        p_vals[j] = 2.0 * sp_stats.t.sf(np.abs(t_stat[j]), df=df)

        ss_tot = np.sum((y_v - y_v.mean()) ** 2)
        r_sq[j] = 1.0 - np.sum(resid ** 2) / ss_tot if ss_tot > 0 else 0.0

    result = {
        "beta": beta,
        "t_stat": t_stat,
        "p_uncorrected": p_vals,
        "r_squared": r_sq,
    }

    if fdr:
        sig, p_corr = fdr_correction(p_vals, alpha=alpha)
        result["p_corrected"] = p_corr
        result["significant"] = sig
    else:
        result["p_corrected"] = p_vals
        result["significant"] = p_vals < alpha

    return result


# ═══════════════════════════════════════════════════════════════════════════
#  PERMUTATION TESTING
# ═══════════════════════════════════════════════════════════════════════════

def permutation_test(
    group1: np.ndarray,
    group2: np.ndarray,
    n_permutations: int = 5000,
    seed: int = 42,
    stat: str = "t",
) -> Dict[str, np.ndarray]:
    """
    Non-parametric permutation test for group comparison.

    Shuffles group labels to build a null distribution of the test
    statistic for each feature independently.

    Parameters
    ----------
    group1 : (n1, P)
    group2 : (n2, P)
    n_permutations : int
    seed : int
    stat : str
        ``'t'`` (t-statistic) or ``'mean_diff'`` (difference in means).

    Returns
    -------
    dict
        ``'observed'``     (P,) observed statistic
        ``'p_perm'``       (P,) permutation p-values (two-tailed)
        ``'null_dist'``    (n_permutations, P) null statistics
    """
    rng = np.random.default_rng(seed)
    combined = np.vstack([group1, group2])
    n1 = group1.shape[0]
    N, P = combined.shape

    def _stat_fn(g1, g2):
        if stat == "mean_diff":
            return np.nanmean(g1, axis=0) - np.nanmean(g2, axis=0)
        else:
            # Welch's t
            m1, m2 = np.nanmean(g1, axis=0), np.nanmean(g2, axis=0)
            v1 = np.nanvar(g1, axis=0, ddof=1)
            v2 = np.nanvar(g2, axis=0, ddof=1)
            se = np.sqrt(v1 / max(g1.shape[0], 1) +
                         v2 / max(g2.shape[0], 1))
            se[se < 1e-12] = 1e-12
            return (m1 - m2) / se

    observed = _stat_fn(group1, group2)

    null_dist = np.zeros((n_permutations, P))
    for perm in range(n_permutations):
        idx = rng.permutation(N)
        g1_perm = combined[idx[:n1]]
        g2_perm = combined[idx[n1:]]
        null_dist[perm] = _stat_fn(g1_perm, g2_perm)

    # Two-tailed p-value
    p_perm = np.mean(
        np.abs(null_dist) >= np.abs(observed)[np.newaxis, :],
        axis=0
    )

    return {
        "observed": observed,
        "p_perm": p_perm,
        "null_dist": null_dist,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  EFFECT SIZES
# ═══════════════════════════════════════════════════════════════════════════

def compute_effect_sizes(
    group1: np.ndarray,
    group2: np.ndarray,
) -> np.ndarray:
    """
    Cohen's d effect size for each feature.

    Uses pooled standard deviation with Bessel correction.

    Parameters
    ----------
    group1 : (n1, P)
    group2 : (n2, P)

    Returns
    -------
    cohens_d : (P,)
    """
    m1 = np.nanmean(group1, axis=0)
    m2 = np.nanmean(group2, axis=0)
    v1 = np.nanvar(group1, axis=0, ddof=1)
    v2 = np.nanvar(group2, axis=0, ddof=1)
    n1 = np.sum(np.isfinite(group1), axis=0)
    n2 = np.sum(np.isfinite(group2), axis=0)

    # Pooled std
    pooled_var = ((n1 - 1) * v1 + (n2 - 1) * v2) / np.maximum(n1 + n2 - 2, 1)
    pooled_std = np.sqrt(pooled_var)
    pooled_std[pooled_std < 1e-12] = 1e-12

    return (m1 - m2) / pooled_std
