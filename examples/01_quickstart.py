#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EigenMorph — Quickstart Demo
=============================

Demonstrates the complete eigenmorph pipeline using a synthetic cortical
surface: feature computation, multi-scale analysis, classical comparison,
and all major visualization types.

No external data required — runs entirely on generated surfaces.

Usage:
    python 01_quickstart.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless rendering (remove for interactive use)

import eigenmorph as em


def main():
    print("=" * 60)
    print("  EigenMorph — Quickstart Demo")
    print("=" * 60)

    # ── 1. Generate synthetic cortex ──
    print("\n1. Generating synthetic cortical surface…")
    mesh, classical = em.generate_synthetic_cortex(
        n_vertices=10_000, n_gyri=8, seed=42,
    )
    print(f"   Mesh: {mesh.n_vertices:,} vertices, {mesh.n_faces:,} faces")
    print(f"   Surface area: {mesh.total_area():,.0f} mm²")
    print(f"   Mean edge length: {mesh.mean_edge_length():.2f} mm")

    # ── 2. Compute eigenvalue features at multiple scales ──
    print("\n2. Computing multi-scale eigenvalue features…")
    ms = em.compute_multiscale_eigenfeatures(
        mesh,
        radii=[3.0, 5.0, 10.0, 20.0],
        weighting="uniform",
        verbose=True,
    )
    print(f"   Total features per vertex: {len(ms.column_names())}")

    # ── 3. Compare with classical FreeSurfer morphometrics ──
    print("\n3. Comparing with classical morphometrics…")
    comp = em.compare_with_classical(ms, classical, verbose=True)

    # ── 4. Extended features ──
    print("\n4. Computing extended features…")
    # Use the 5mm scale for extended features
    feat_5mm = ms.get_scale(5.0)
    # Recompute with eigenvectors for verticality
    feat_5mm_evec = em.compute_eigenfeatures(
        mesh, radius=5.0, store_eigenvectors=True, verbose=False,
    )
    extended = em.compute_all_extended_features(
        mesh, feat_5mm_evec, ms, verbose=True,
    )

    # ── 5. Parcellation ──
    print("\n5. Parcellating features…")
    labels = em.generate_vertex_parcellation(mesh, n_parcels=50, seed=42)
    parc = em.parcellate_features(ms, labels, n_parcels=50, aggregation="mean")
    print(f"   Parcellated {len(parc['_column_names'])} features "
          f"into {50} regions")

    # ── 6. Morphological distance matrix ──
    print("\n6. Computing morphological distance matrix…")
    dist = em.morphological_distance_matrix(parc, metric="correlation")
    print(f"   Distance matrix: {dist.shape}")

    # ── 7. Visualisations ──
    print("\n7. Generating figures…")

    # 7a. Feature overview
    fig = em.viz.plot_feature_overview(mesh, feat_5mm,
                                        save_path="fig1_feature_overview.png")
    print("   ✓ fig1_feature_overview.png")

    # 7b. Multi-scale profiles
    fig = em.viz.plot_multiscale_profile(ms,
                                          save_path="fig2_multiscale_profiles.png")
    print("   ✓ fig2_multiscale_profiles.png")

    # 7c. Ternary feature space
    fig = em.viz.plot_ternary_features(feat_5mm,
                                        save_path="fig3_ternary.png")
    print("   ✓ fig3_ternary.png")

    # 7d. Classical comparison
    fig = em.viz.plot_classical_comparison(comp,
                                            save_path="fig4_comparison.png")
    print("   ✓ fig4_comparison.png")

    # 7e. Hero composite figure
    fig = em.viz.plot_hero_figure(
        mesh, feat_5mm, ms, classical_metrics=classical, comparison=comp,
        save_path="fig5_hero.png",
    )
    print("   ✓ fig5_hero.png")

    # 7f. Distance matrix
    fig = em.viz.plot_distance_matrix(dist, save_path="fig6_distance.png")
    print("   ✓ fig6_distance.png")

    # ── 8. Save/load features ──
    print("\n8. Testing save/load…")
    em.io.save_multiscale("eigenmorph_features.npz", ms)
    ms_loaded = em.io.load_multiscale("eigenmorph_features.npz")
    print(f"   Loaded: {ms_loaded}")
    np.testing.assert_allclose(
        ms.as_matrix(), ms_loaded.as_matrix(), atol=1e-10,
    )
    print("   ✓ Round-trip save/load verified")

    print("\n" + "=" * 60)
    print("  Demo complete!  See generated figures.")
    print("=" * 60)


if __name__ == "__main__":
    main()
