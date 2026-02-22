# -*- coding: utf-8 -*-
"""
EigenMorph — Eigenvalue Geometric Features for Cortical Morphology
===================================================================

Bridges point cloud analysis with computational neuroanatomy by treating
cortical surface meshes as 3D point clouds and computing eigenvalue-
based geometric features from local vertex neighbourhoods at multiple
spatial scales.

Quick start::

    import eigenmorph as em

    mesh, classical = em.generate_synthetic_cortex()
    ms = em.compute_multiscale_eigenfeatures(mesh)
    comp = em.compare_with_classical(ms, classical)
    em.viz.plot_hero_figure(mesh, ms.scales[1], ms, comparison=comp)

License: MIT
"""

__version__ = "0.1.0"
__author__ = "rdneuro"

from .core import (
    SurfaceMesh, EigenFeatures, MultiScaleFeatures,
    compute_eigenfeatures, compute_multiscale_eigenfeatures,
)
from .features import (
    compute_shape_index, compute_curvedness, compute_verticality,
    compute_normal_displacement, compute_surface_gradient,
    compute_fractal_dimension, compute_all_extended_features,
)
from .parcellation import (
    parcellate_features, compare_with_classical,
    morphological_distance_matrix,
)
from . import io
from .stats import (
    vertex_wise_ttest, vertex_wise_glm, permutation_test,
    compute_effect_sizes, fdr_correction,
)
from .synthetic import (
    generate_synthetic_cortex, generate_vertex_parcellation,
    generate_group_data,
)
from .utils import (
    normalize_features, smooth_surface_data,
    compute_mesh_adjacency, mesh_area, mesh_edges,
)
from . import viz
