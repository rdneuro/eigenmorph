#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EigenMorph — Group-Level Statistics Demo
=========================================

Demonstrates the complete statistical analysis workflow:
simulated two-group comparison with permutation testing, FDR
correction, effect size estimation, and parcellation-level analysis.

Uses entirely synthetic data — no external files required.

Usage:
    python 03_group_statistics.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")

import eigenmorph as em


def main():
    print("=" * 60)
    print("  EigenMorph — Group Statistics Demo")
    print("=" * 60)

    # ── 1. Generate group data ───────────────────────────────────────
    # Simulate 15 subjects per group with a subtle cortical difference
    print("\n1. Generating group data (15 subjects × 2 groups)…")
    n_per_group = 15
    n_vertices = 5_000

    group_a, group_b = [], []

    for i in range(n_per_group):
        # Group A: baseline cortex
        mesh_a, _ = em.generate_synthetic_cortex(
            n_vertices=n_vertices, n_gyri=8, seed=1000 + i,
        )
        ms_a = em.compute_multiscale_eigenfeatures(
            mesh_a, radii=[5.0, 10.0], verbose=False,
        )
        group_a.append(ms_a)

        # Group B: slightly different cortex (more gyri → more complex)
        mesh_b, _ = em.generate_synthetic_cortex(
            n_vertices=n_vertices, n_gyri=12, seed=2000 + i,
        )
        ms_b = em.compute_multiscale_eigenfeatures(
            mesh_b, radii=[5.0, 10.0], verbose=False,
        )
        group_b.append(ms_b)

    print(f"   Group A: {len(group_a)} subjects")
    print(f"   Group B: {len(group_b)} subjects")

    # ── 2. Vertex-wise t-test ────────────────────────────────────────
    print("\n2. Running vertex-wise t-tests…")

    # Extract a single feature for comparison (eigenentropy at 5mm)
    a_matrices = np.array([ms.scales[0].eigenentropy for ms in group_a])
    b_matrices = np.array([ms.scales[0].eigenentropy for ms in group_b])

    t_stats, p_vals = em.vertex_wise_ttest(a_matrices, b_matrices)
    print(f"   t-statistics: mean={np.nanmean(t_stats):.3f}, "
          f"max |t|={np.nanmax(np.abs(t_stats)):.3f}")

    # ── 3. Multiple comparison correction ────────────────────────────
    print("\n3. Applying FDR correction…")
    reject_bh, p_corrected_bh = em.fdr_correction(p_vals, alpha=0.05, method="bh")
    print(f"   BH-FDR significant vertices: {reject_bh.sum():,} / {len(reject_bh):,}")

    # ── 4. Effect sizes ──────────────────────────────────────────────
    print("\n4. Computing effect sizes…")
    effects = em.compute_effect_sizes(a_matrices, b_matrices)
    d = effects["cohens_d"]
    print(f"   Cohen's d: mean={np.nanmean(d):.3f}, "
          f"range=[{np.nanmin(d):.3f}, {np.nanmax(d):.3f}]")

    # ── 5. Permutation test ──────────────────────────────────────────
    print("\n5. Running permutation test (100 permutations)…")
    t_obs, p_perm, null_dist = em.permutation_test(
        a_matrices, b_matrices, n_permutations=100, seed=42,
    )
    sig_perm = (p_perm < 0.05).sum()
    print(f"   Permutation-significant vertices: {sig_perm:,}")

    # ── 6. Parcellated group comparison ──────────────────────────────
    print("\n6. Parcellated comparison…")
    # Create parcellation for the common vertex space
    ref_mesh, _ = em.generate_synthetic_cortex(n_vertices=n_vertices, seed=0)
    labels = em.generate_vertex_parcellation(ref_mesh, n_parcels=20, seed=42)

    # Parcellate both groups
    parc_a = [em.parcellate_features(ms, labels, n_parcels=20) for ms in group_a]
    parc_b = [em.parcellate_features(ms, labels, n_parcels=20) for ms in group_b]

    print(f"   Parcellated into 20 regions")
    print(f"   Features per region: {parc_a[0]['matrix'].shape[1]}")

    # ── 7. Morphological distance ────────────────────────────────────
    print("\n7. Computing mean morphological distance per group…")
    dist_a = em.morphological_distance_matrix(parc_a[0], metric="correlation")
    dist_b = em.morphological_distance_matrix(parc_b[0], metric="correlation")
    print(f"   Mean distance — Group A: {dist_a.mean():.4f}, "
          f"Group B: {dist_b.mean():.4f}")

    print("\n" + "=" * 60)
    print("  Group statistics demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
