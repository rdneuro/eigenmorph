# Examples

This directory contains worked examples that demonstrate the full **EigenMorph** pipeline, from synthetic data generation through multi-scale feature computation, statistical analysis, and publication-quality visualization.

Each script is self-contained and heavily commented — you can read them sequentially as a tutorial or jump straight to the one that matches your use case.

---

## 01 — Quickstart

**`01_quickstart.py`** — The complete pipeline on synthetic data, no external files needed.

Covers every major component of the library in a single run: generating a synthetic cortical surface, computing multi-scale eigenvalue features (4 radii × 7 features = 28 descriptors per vertex), comparing with classical FreeSurfer-style morphometrics, computing extended descriptors (shape index, curvedness, verticality, fractal dimension), parcellating into brain regions, building a morphological distance matrix, and generating six publication-ready figures including the composite hero figure.

```bash
python 01_quickstart.py
```

**What you'll learn:** how the core data structures (`SurfaceMesh`, `EigenFeatures`, `MultiScaleFeatures`) flow through the pipeline, and how the visualization module produces journal-ready plots with minimal configuration.

**Requirements:** core dependencies only (numpy, scipy, matplotlib).

**Output:** six PNG figures (`fig1_feature_overview.png` through `fig6_distance.png`) plus a round-trip save/load test of the NPZ serialization format.

---

## 02 — FreeSurfer Pipeline

**`02_freesurfer_pipeline.py`** — Real-world workflow with FreeSurfer `recon-all` output.

Shows how to load a white surface and classical morphometry overlays (thickness, curvature, sulcal depth) from a FreeSurfer subject directory, compute multi-scale eigenvalue features on the actual cortical geometry, quantify how much novel information eigenvalue features carry beyond the classical metrics, and save everything for downstream group analyses.

```bash
# Set your FreeSurfer subjects directory
export SUBJECTS_DIR=/path/to/subjects
python 02_freesurfer_pipeline.py
```

**What you'll learn:** how to integrate EigenMorph into an existing FreeSurfer-based neuroimaging pipeline, including I/O for surfaces, morphometry overlays, and annotation files.

**Requirements:** `nibabel` (install via `pip install eigenmorph[neuro]`).

**Configuration:** edit `SUBJECTS_DIR`, `SUBJECT`, and `HEMISPHERE` at the top of the script to point to your data.

---

## 03 — Group-Level Statistics

**`03_group_statistics.py`** — Simulated two-group comparison with full statistical machinery.

Generates 15 synthetic subjects per group with a controlled morphological difference (different gyral complexity), then runs the complete statistical workflow: vertex-wise independent-samples t-tests, Benjamini–Hochberg FDR correction, Cohen's d effect size maps, non-parametric permutation testing (100 permutations), and parcellation-level group comparison with morphological distance matrices.

```bash
python 03_group_statistics.py
```

**What you'll learn:** how to set up group-level analyses entirely within EigenMorph, from extracting per-subject feature matrices through multiple-comparison correction. The synthetic-data approach lets you validate your analysis pipeline before applying it to real patient cohorts.

**Requirements:** core dependencies only (numpy, scipy, matplotlib).

**Note:** the permutation test runs 100 permutations by default for speed. For real analyses, increase to ≥5,000 for stable p-value estimates.

---

## Running All Examples

```bash
cd examples/
python 01_quickstart.py           # ~60 seconds
python 03_group_statistics.py     # ~3–5 minutes (30 synthetic subjects)
# python 02_freesurfer_pipeline.py  # requires real FreeSurfer data
```

The first and third examples generate all data on the fly, so they work anywhere with EigenMorph installed. The second requires a FreeSurfer subject directory with completed `recon-all` output.

---

## Typical Workflow

The examples are ordered to reflect a natural progression in a cortical morphometry study. The quickstart introduces the core concepts and data structures on a toy surface. The FreeSurfer pipeline shows how those same concepts apply to real MRI-derived cortical geometry. The group statistics example demonstrates how to take single-subject features and perform the kind of case-control or cohort analyses that appear in clinical neuroimaging papers.

For a real research project you would typically combine elements from all three: load surfaces with the I/O from example 02, compute features as in example 01, and run group comparisons as in example 03.

---

## Tips

**Performance.** For full FreeSurfer hemispheres (~160k vertices), expect roughly 30–90 seconds per scale depending on radius and hardware. The 20 mm radius is the most expensive because each vertex has many more neighbours. If you're iterating on analysis parameters, start with a single scale (e.g., `radii=[5.0]`) before running the full multi-scale sweep.

**Weighting.** The default `"uniform"` weighting treats all neighbours equally. Gaussian weighting (`weighting="gaussian"`) down-weights distant neighbours and often produces smoother feature maps that better reflect local geometry. Inverse-distance weighting (`weighting="distance"`) is a middle ground. All three are demonstrated in the quickstart.

**Eigenvectors.** Set `store_eigenvectors=True` only when you need orientation-dependent features (verticality, principal directions). The eigenvector array is `(V, 3, 3)` and roughly triples memory usage for large meshes.

**Visualization.** All static plots (matplotlib) work out of the box. For the interactive 3D renders (FURY/VTK), install `pip install eigenmorph[interactive]`. If FURY is not available, every interactive function falls back gracefully to a matplotlib `Poly3DCollection` render — lower quality but functional.
