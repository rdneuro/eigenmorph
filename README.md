# 🧠 EigenMorph

**Eigenvalue geometric features for cortical surface morphology analysis**

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-34%20passed-brightgreen.svg)]()

EigenMorph bridges **3D point cloud analysis** with **computational neuroanatomy** by treating cortical surface meshes as spatial point clouds and computing eigenvalue-based geometric descriptors from local vertex neighbourhoods at multiple scales.

For each vertex on the cortical surface, EigenMorph finds all neighbours within a given radius, builds a 3×3 spatial covariance matrix, decomposes it into eigenvalues **λ₁ ≥ λ₂ ≥ λ₃**, and derives seven complementary geometric features that together form a rich morphological fingerprint.

---

## ✨ Features

- **Seven core eigenvalue descriptors** — linearity, planarity, sphericity, omnivariance, anisotropy, eigenentropy, and surface variation
- **Multi-scale analysis** — compute features at fine-to-coarse radii (3–20 mm) to capture geometry from individual sulcal branches to lobar shape
- **Weighted neighbourhood schemes** — uniform, Gaussian, or inverse-distance weighting for the covariance matrix
- **Extended descriptors** — shape index, curvedness, verticality, normal displacement, surface gradient magnitude, and fractal dimension
- **Parcellation tools** — aggregate vertex-wise features into brain regions (Schaefer, DK, Brainnetome, etc.) and compute morphological distance matrices
- **Classical comparison** — quantify how much information eigenvalue features add beyond FreeSurfer thickness / curvature / sulcal depth
- **Group-level statistics** — vertex-wise t-tests, GLM, permutation testing, FDR/Bonferroni correction, and effect sizes (Cohen's d, η²)
- **Full I/O** — loaders for FreeSurfer surfaces, morphometry overlays, annotations, GIFTI, and HDF5/NPZ serialisation
- **Publication-quality visualisation** — surface renders, multi-scale profiles, ternary diagrams, radar fingerprints, hero composite figures, and distance matrices
- **Interactive 3D rendering** — FURY (VTK) powered RGB identity maps, feature landscapes, scale sweeps, exploded views, and neighbourhood explorers (graceful fallback to matplotlib)
- **Synthetic data** — generate realistic cortical surfaces for testing and demonstration without external data

---

## 📐 Mathematical Foundations

For each vertex **v** with spatial neighbourhood **N(v, r)** = { **pᵢ** : ‖**pᵢ** − **v**‖ < r }, the 3×3 covariance matrix is:

```
C = (1/N) Σᵢ (pᵢ − μ)(pᵢ − μ)ᵀ
```

Eigendecomposition yields λ₁ ≥ λ₂ ≥ λ₃ ≥ 0, from which:

| Feature | Formula | Cortical interpretation |
|:---|:---|:---|
| Linearity | (λ₁ − λ₂) / λ₁ | Ridge-like geometry → sulcal fundi |
| Planarity | (λ₂ − λ₃) / λ₁ | Planar geometry → gyral crowns |
| Sphericity | λ₃ / λ₁ | Isotropic geometry → sulcal pits |
| Omnivariance | (λ₁ λ₂ λ₃)^(1/3) | Overall spatial dispersion |
| Anisotropy | (λ₁ − λ₃) / λ₁ | Directional bias (≈ L + P) |
| Eigenentropy | −Σ λ̃ᵢ ln(λ̃ᵢ) | Shape complexity / disorder |
| Surface variation | λ₃ / Σλ | Local roughness / change of curvature |

where λ̃ᵢ = λᵢ / Σλ are normalised eigenvalues.

The **multi-scale** approach computes these at 4 default radii (3, 5, 10, 20 mm), yielding **28 features per vertex** that capture cortical geometry from fine sulcal branches to lobar shape.

---

## 🚀 Installation

```bash
# Core (numpy, scipy, matplotlib)
pip install eigenmorph

# With neuroimaging I/O (nibabel)
pip install eigenmorph[neuro]

# With interactive 3D visualization (FURY)
pip install eigenmorph[interactive]

# Everything
pip install eigenmorph[all]
```

**From source:**

```bash
git clone https://github.com/rdneuro/eigenmorph.git
cd eigenmorph
pip install -e ".[all]"
```

---

## 💡 Quick Start

```python
import eigenmorph as em

# Generate a synthetic cortical surface (no data needed)
mesh, classical = em.generate_synthetic_cortex(n_vertices=10_000, seed=42)

# Compute multi-scale eigenvalue features
ms = em.compute_multiscale_eigenfeatures(
    mesh,
    radii=[3.0, 5.0, 10.0, 20.0],
    weighting="uniform",
)
# → 28 features per vertex (7 features × 4 scales)

# Compare with classical FreeSurfer morphometrics
comp = em.compare_with_classical(ms, classical, verbose=True)

# Publication-quality hero figure
em.viz.plot_hero_figure(mesh, ms.scales[1], ms, comparison=comp,
                        save_path="hero.png")
```

---

## 📖 Usage Examples

### Single-scale analysis

```python
import eigenmorph as em

mesh = em.io.load_freesurfer_surface("subjects/sub-01/surf/lh.white",
                                      hemisphere="lh")
features = em.compute_eigenfeatures(mesh, radius=5.0, weighting="gaussian")

print(features.summary())
em.viz.plot_feature_overview(mesh, features, save_path="overview.png")
```

### Extended descriptors

```python
# Compute core features with eigenvectors (needed for verticality)
feat = em.compute_eigenfeatures(mesh, radius=5.0, store_eigenvectors=True)
ms = em.compute_multiscale_eigenfeatures(mesh)

# All extended features in one call
extended = em.compute_all_extended_features(mesh, feat, ms)
# → dict with shape_index, curvedness, verticality,
#   normal_displacement, surface_gradient, fractal_dimension
```

### Parcellation and group statistics

```python
# Parcellate into brain regions
labels = em.io.load_freesurfer_annot("subjects/sub-01/label/lh.aparc.annot")
parc = em.parcellate_features(ms, labels, n_parcels=34)

# Morphological distance matrix between regions
dist = em.morphological_distance_matrix(parc, metric="correlation")
em.viz.plot_distance_matrix(dist, save_path="morph_distance.png")

# Group-level vertex-wise comparison
group1 = [...]  # list of MultiScaleFeatures
group2 = [...]
t_stats, p_vals = em.vertex_wise_ttest(group1, group2, feature="linearity")
reject, p_corrected = em.fdr_correction(p_vals, alpha=0.05)
```

### Interactive 3D visualization

```python
# RGB identity map (L→Red, P→Green, S→Blue)
em.viz.plot_rgb_identity(mesh, features, size=(1200, 800),
                          save_path="rgb_identity.png")

# Feature landscape (deformed surface encoding a feature as height)
em.viz.plot_feature_landscape(mesh, features.eigenentropy,
                               title="Eigenentropy Landscape")

# Animated scale sweep (fine → coarse as GIF)
em.viz.render_scale_sweep(mesh, ms, save_path="scale_sweep.gif")
```

### Save and load

```python
# Save computed features (compressed NPZ)
em.io.save_multiscale("sub-01_eigenmorph.npz", ms)

# Reload later
ms_loaded = em.io.load_multiscale("sub-01_eigenmorph.npz")
```

---

## 📦 Package Structure

```
eigenmorph/
├── core.py           # SurfaceMesh, EigenFeatures, compute_eigenfeatures()
├── features.py       # Extended descriptors (shape index, fractal dim, …)
├── parcellation.py   # Region aggregation, classical comparison, distance
├── io.py             # FreeSurfer, GIFTI, NPZ loaders/savers
├── stats.py          # Vertex-wise tests, permutation, FDR correction
├── synthetic.py      # Synthetic cortex generation for demos & testing
├── utils.py          # Adjacency, smoothing, normalization, mesh ops
└── viz/
    ├── styles.py     # Colour palettes and publication defaults
    ├── static.py     # Matplotlib: overviews, ternary, radar, hero fig
    └── interactive.py  # FURY (VTK): RGB maps, landscapes, scale sweeps
```

---

## 🧪 Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

---

## 📚 References

The eigenvalue feature framework draws from 3D point cloud analysis in remote sensing and computer vision:

- Weinmann, M., et al. (2015). Semantic point cloud interpretation based on optimal neighborhoods, relevant features and efficient classifiers. *ISPRS J. Photogramm. Remote Sens.*, 105, 286–304.
- West, K.H.P., et al. (2018). Revisiting the eigenvalues: features for 3D point cloud segmentation. *Int. Arch. Photogramm. Remote Sens.*, XLII-2, 1179–1184.
- Demantké, J., et al. (2011). Dimensionality based scale selection in 3D lidar point clouds. *ISPRS XXII*, 97–102.

Neuroimaging foundations:

- Fischl, B. (2012). FreeSurfer. *NeuroImage*, 62(2), 774–781.
- Dale, A.M., Fischl, B., & Sereno, M.I. (1999). Cortical surface-based analysis: I. Segmentation and surface reconstruction. *NeuroImage*, 9, 179–194.

---

## 📄 License

MIT — see [LICENSE](LICENSE).

---

## 🤝 Contributing

Contributions welcome! Please open an issue to discuss your idea before submitting a PR.

```bash
git clone https://github.com/rdneuro/eigenmorph.git
cd eigenmorph
pip install -e ".[dev]"
pytest tests/ -v
```

---

*Developed for research at the Instituto Nacional de Neurociência Translacional (INNT-UFRJ).*
