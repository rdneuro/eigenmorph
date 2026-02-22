#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EigenMorph — FreeSurfer Pipeline
=================================

Demonstrates how to use EigenMorph with real FreeSurfer data.

Prerequisites:
    - FreeSurfer subject directory with recon-all output
    - nibabel installed (pip install nibabel)

Adjust SUBJECTS_DIR and SUBJECT below to match your data.

Usage:
    python 02_freesurfer_pipeline.py
"""

import numpy as np
import os

import eigenmorph as em


# ── Configuration ──
SUBJECTS_DIR = os.environ.get("SUBJECTS_DIR", "/path/to/subjects")
SUBJECT = "sub-01"
HEMISPHERE = "lh"

# Radii for multi-scale analysis
RADII = [3.0, 5.0, 10.0, 20.0]


def main():
    subj_dir = os.path.join(SUBJECTS_DIR, SUBJECT)

    # ── 1. Load surface ──
    print("Loading FreeSurfer surface…")
    surf_path = os.path.join(subj_dir, "surf", f"{HEMISPHERE}.white")
    mesh = em.io.load_freesurfer_surface(surf_path)
    print(f"  {mesh.n_vertices:,} vertices, {mesh.n_faces:,} faces")

    # ── 2. Load classical morphometrics ──
    print("Loading classical morphometrics…")
    surf_dir = os.path.join(subj_dir, "surf")
    classical = {}
    for metric in ["thickness", "curv", "sulc"]:
        path = os.path.join(surf_dir, f"{HEMISPHERE}.{metric}")
        if os.path.exists(path):
            classical[metric] = em.io.load_freesurfer_morph(path)
            print(f"  ✓ {metric}: mean = {classical[metric].mean():.3f}")

    # ── 3. Compute eigenvalue features ──
    print(f"\nComputing multi-scale eigenvalue features (radii={RADII})…")
    ms = em.compute_multiscale_eigenfeatures(
        mesh, radii=RADII, weighting="uniform", verbose=True,
    )

    # ── 4. Compare with classical ──
    if classical:
        print("\nComparing with classical morphometrics…")
        comp = em.compare_with_classical(ms, classical, verbose=True)

    # ── 5. Parcellation (Schaefer-200 or DK) ──
    annot_candidates = [
        os.path.join(subj_dir, "label",
                     f"{HEMISPHERE}.Schaefer2018_200Parcels_7Networks_order.annot"),
        os.path.join(subj_dir, "label", f"{HEMISPHERE}.aparc.annot"),
    ]

    for annot_path in annot_candidates:
        if os.path.exists(annot_path):
            print(f"\nLoading parcellation: {os.path.basename(annot_path)}")
            labels, names = em.io.load_freesurfer_annot(annot_path)
            n_parcels = len(np.unique(labels)) - 1  # exclude 0
            print(f"  {n_parcels} parcels")

            parc = em.parcellate_features(
                ms, labels, n_parcels=n_parcels,
                parcel_names=names, aggregation="mean",
            )
            print(f"  Parcellated {len(parc['_column_names'])} features")
            break

    # ── 6. Save results ──
    out_dir = os.path.join(subj_dir, "eigenmorph")
    os.makedirs(out_dir, exist_ok=True)

    npz_path = os.path.join(out_dir, f"{HEMISPHERE}_eigenmorph.npz")
    em.io.save_multiscale(npz_path, ms)
    print(f"\n  Saved features → {npz_path}")

    # ── 7. Figures ──
    print("\nGenerating figures…")
    feat_5mm = ms.get_scale(5.0)

    em.viz.plot_feature_overview(
        mesh, feat_5mm,
        save_path=os.path.join(out_dir, f"{HEMISPHERE}_feature_overview.png"),
    )
    em.viz.plot_multiscale_profile(
        ms, save_path=os.path.join(out_dir, f"{HEMISPHERE}_profiles.png"),
    )
    if classical:
        em.viz.plot_classical_comparison(
            comp, save_path=os.path.join(
                out_dir, f"{HEMISPHERE}_classical_comparison.png"),
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
