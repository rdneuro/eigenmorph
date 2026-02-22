# -*- coding: utf-8 -*-
"""
Tests for eigenmorph.core and eigenmorph.features.

Run with:  pytest tests/ -v
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import eigenmorph as em


# ═══════════════════════════════════════════════════════════════════════════
#  FIXTURES
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def synthetic():
    """Generate a synthetic cortex once for all tests."""
    mesh, classical = em.generate_synthetic_cortex(n_vertices=10_000, seed=42)
    return mesh, classical


@pytest.fixture(scope="module")
def features_5mm(synthetic):
    mesh, _ = synthetic
    return em.compute_eigenfeatures(
        mesh, radius=5.0, verbose=False, store_eigenvectors=True,
    )


@pytest.fixture(scope="module")
def ms_features(synthetic):
    mesh, _ = synthetic
    return em.compute_multiscale_eigenfeatures(
        mesh, radii=[3.0, 5.0, 10.0], verbose=False,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  SURFACE MESH
# ═══════════════════════════════════════════════════════════════════════════

class TestSurfaceMesh:

    def test_shape(self, synthetic):
        mesh, _ = synthetic
        assert mesh.vertices.shape[1] == 3
        assert mesh.faces.shape[1] == 3
        assert mesh.n_vertices == mesh.vertices.shape[0]
        assert mesh.n_faces == mesh.faces.shape[0]

    def test_normals_computed(self, synthetic):
        mesh, _ = synthetic
        assert mesh.vertex_normals is not None
        assert mesh.vertex_normals.shape == mesh.vertices.shape
        # Normals should be approximately unit length
        norms = np.linalg.norm(mesh.vertex_normals, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_centroid(self, synthetic):
        mesh, _ = synthetic
        c = mesh.centroid
        assert c.shape == (3,)

    def test_area_positive(self, synthetic):
        mesh, _ = synthetic
        assert mesh.total_area() > 0
        assert np.all(mesh.face_areas() >= 0)

    def test_copy(self, synthetic):
        mesh, _ = synthetic
        m2 = mesh.copy()
        m2.vertices[0] += 100
        assert not np.allclose(mesh.vertices[0], m2.vertices[0])


# ═══════════════════════════════════════════════════════════════════════════
#  EIGENFEATURES
# ═══════════════════════════════════════════════════════════════════════════

class TestEigenFeatures:

    def test_feature_shapes(self, features_5mm, synthetic):
        mesh, _ = synthetic
        V = mesh.n_vertices
        f = features_5mm
        assert f.linearity.shape == (V,)
        assert f.eigenvalues.shape == (V, 3)
        assert f.eigenvectors.shape == (V, 3, 3)

    def test_valid_fraction(self, features_5mm):
        # Most vertices should have valid features with r=5mm
        assert features_5mm.n_valid > features_5mm.linearity.shape[0] * 0.5

    def test_feature_ranges(self, features_5mm):
        f = features_5mm
        valid = f.valid_mask
        # Linearity, planarity, sphericity are in [0, 1]
        assert np.nanmin(f.linearity[valid]) >= -1e-6
        assert np.nanmax(f.linearity[valid]) <= 1.0 + 1e-6
        assert np.nanmin(f.planarity[valid]) >= -1e-6
        assert np.nanmin(f.sphericity[valid]) >= -1e-6

    def test_anisotropy_decomposition(self, features_5mm):
        """linearity + planarity ≈ anisotropy (mathematical identity)."""
        f = features_5mm
        valid = f.valid_mask
        lps = f.linearity[valid] + f.planarity[valid] + f.sphericity[valid]
        # L + P + S should ≈ anisotropy + sphericity = 1
        # More precisely: L + P = A (exact identity from definitions)
        np.testing.assert_allclose(
            f.linearity[valid] + f.planarity[valid],
            f.anisotropy[valid],
            atol=1e-10,
        )

    def test_eigenvalues_descending(self, features_5mm):
        evals = features_5mm.eigenvalues
        valid = features_5mm.valid_mask
        assert np.all(evals[valid, 0] >= evals[valid, 1] - 1e-10)
        assert np.all(evals[valid, 1] >= evals[valid, 2] - 1e-10)

    def test_as_matrix(self, features_5mm):
        mat = features_5mm.as_matrix()
        assert mat.shape[1] == 7

    def test_summary(self, features_5mm):
        s = features_5mm.summary()
        assert "linearity" in s
        assert "mean" in s["linearity"]


# ═══════════════════════════════════════════════════════════════════════════
#  MULTI-SCALE
# ═══════════════════════════════════════════════════════════════════════════

class TestMultiScale:

    def test_n_scales(self, ms_features):
        assert len(ms_features.scales) == 3
        assert len(ms_features.radii) == 3

    def test_matrix_shape(self, ms_features, synthetic):
        mesh, _ = synthetic
        mat = ms_features.as_matrix()
        assert mat.shape == (mesh.n_vertices, 7 * 3)

    def test_column_names(self, ms_features):
        cn = ms_features.column_names()
        assert len(cn) == 21
        assert "linearity_r3mm" in cn

    def test_get_feature(self, ms_features, synthetic):
        mesh, _ = synthetic
        lin = ms_features.get_feature("linearity")
        assert lin.shape == (mesh.n_vertices, 3)

    def test_get_scale(self, ms_features):
        s = ms_features.get_scale(5.0)
        assert s.radius == 5.0


# ═══════════════════════════════════════════════════════════════════════════
#  WEIGHTING SCHEMES
# ═══════════════════════════════════════════════════════════════════════════

class TestWeighting:

    def test_gaussian(self, synthetic):
        mesh, _ = synthetic
        f = em.compute_eigenfeatures(
            mesh, radius=5.0, weighting="gaussian", verbose=False,
        )
        assert f.weighting == "gaussian"
        assert f.n_valid > 0

    def test_distance(self, synthetic):
        mesh, _ = synthetic
        f = em.compute_eigenfeatures(
            mesh, radius=5.0, weighting="distance", verbose=False,
        )
        assert f.weighting == "distance"
        assert f.n_valid > 0


# ═══════════════════════════════════════════════════════════════════════════
#  EXTENDED FEATURES
# ═══════════════════════════════════════════════════════════════════════════

class TestExtendedFeatures:

    def test_shape_index(self, features_5mm):
        si = em.compute_shape_index(features_5mm)
        valid = np.isfinite(si)
        assert valid.sum() > 0
        assert np.all(si[valid] >= -1.0 - 1e-6)
        assert np.all(si[valid] <= 1.0 + 1e-6)

    def test_curvedness(self, features_5mm):
        c = em.compute_curvedness(features_5mm)
        valid = np.isfinite(c)
        assert np.all(c[valid] >= 0)

    def test_verticality(self, features_5mm):
        vert = em.compute_verticality(features_5mm)
        valid = np.isfinite(vert)
        assert valid.sum() > 0
        assert np.all(vert[valid] >= -1e-6)
        assert np.all(vert[valid] <= 1.0 + 1e-6)

    def test_normal_displacement(self, synthetic):
        mesh, _ = synthetic
        nd = em.compute_normal_displacement(mesh, radius=5.0)
        assert np.isfinite(nd).sum() > 0

    def test_surface_gradient(self, synthetic, features_5mm):
        mesh, _ = synthetic
        grad = em.compute_surface_gradient(mesh, features_5mm.eigenentropy)
        assert np.isfinite(grad).sum() > 0
        assert np.all(grad[np.isfinite(grad)] >= 0)

    def test_fractal_dimension(self, ms_features):
        fd = em.compute_fractal_dimension(ms_features)
        assert np.isfinite(fd).sum() > 0

    def test_all_extended(self, synthetic, features_5mm, ms_features):
        mesh, _ = synthetic
        ext = em.compute_all_extended_features(
            mesh, features_5mm, ms_features, verbose=False,
        )
        assert "shape_index" in ext
        assert "fractal_dimension" in ext


# ═══════════════════════════════════════════════════════════════════════════
#  PARCELLATION & COMPARISON
# ═══════════════════════════════════════════════════════════════════════════

class TestParcellation:

    def test_parcellate(self, ms_features, synthetic):
        mesh, _ = synthetic
        labels = em.generate_vertex_parcellation(mesh, n_parcels=10)
        parc = em.parcellate_features(ms_features, labels, n_parcels=10)
        assert "linearity_r3mm" in parc
        assert len(parc["linearity_r3mm"]) == 10

    def test_compare_classical(self, ms_features, synthetic):
        _, classical = synthetic
        comp = em.compare_with_classical(ms_features, classical, verbose=False)
        assert "correlations" in comp
        assert "unique_variance" in comp
        assert comp["correlations"].shape[1] == len(classical)


# ═══════════════════════════════════════════════════════════════════════════
#  STATISTICS
# ═══════════════════════════════════════════════════════════════════════════

class TestStats:

    def test_ttest(self):
        g1, g2, mask = em.generate_group_data(n_subjects_per_group=15,
                                               n_parcels=10, n_features=7,
                                               seed=99)
        result = em.vertex_wise_ttest(g1, g2, fdr=True)
        assert "t_stat" in result
        assert "significant" in result
        assert result["t_stat"].shape[0] == g1.shape[1]

    def test_fdr(self):
        pvals = np.random.uniform(0, 1, 100)
        pvals[:5] = 0.001  # some truly significant
        reject, pcorr = em.fdr_correction(pvals)
        assert reject[:5].all()

    def test_permutation(self):
        rng = np.random.default_rng(42)
        g1 = rng.standard_normal((10, 5))
        g2 = rng.standard_normal((10, 5)) + 2
        res = em.permutation_test(g1, g2, n_permutations=200, seed=42)
        assert "p_perm" in res
        assert res["p_perm"].shape == (5,)


# ═══════════════════════════════════════════════════════════════════════════
#  UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

class TestUtils:

    def test_adjacency(self, synthetic):
        mesh, _ = synthetic
        adj = em.compute_mesh_adjacency(mesh)
        assert adj.shape == (mesh.n_vertices, mesh.n_vertices)
        # Adjacency should be symmetric
        diff = adj - adj.T
        assert abs(diff).max() < 1e-10

    def test_smoothing(self, synthetic, features_5mm):
        mesh, _ = synthetic
        smoothed = em.smooth_surface_data(mesh, features_5mm.linearity,
                                           fwhm=3.0)
        assert smoothed.shape == features_5mm.linearity.shape
        # Smoothing should reduce variance
        v1 = np.nanvar(features_5mm.linearity)
        v2 = np.nanvar(smoothed)
        assert v2 <= v1 + 1e-6

    def test_normalize_zscore(self):
        data = np.random.randn(100, 5)
        normed = em.normalize_features(data, method="zscore")
        np.testing.assert_allclose(normed.mean(0), 0, atol=1e-10)
        np.testing.assert_allclose(normed.std(0), 1, atol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
