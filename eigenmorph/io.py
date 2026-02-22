# -*- coding: utf-8 -*-
"""
eigenmorph.io
=============

Input/output for cortical surfaces and eigenvalue features.

Supports FreeSurfer geometry, morphometry overlays, annotation files,
GIFTI surfaces, and HDF5 serialisation of computed features.

All loaders return ``SurfaceMesh`` objects or plain NumPy arrays,
keeping the dependency footprint minimal (nibabel is optional but
recommended for neuroimaging data).
"""

import numpy as np
import json
import os
from typing import Optional, Dict, Tuple

from .core import SurfaceMesh, EigenFeatures, MultiScaleFeatures


# ═══════════════════════════════════════════════════════════════════════════
#  FREESURFER
# ═══════════════════════════════════════════════════════════════════════════

def load_freesurfer_surface(
    filepath: str,
    hemisphere: Optional[str] = None,
    surface_type: Optional[str] = None,
) -> SurfaceMesh:
    """
    Load a FreeSurfer surface file (``?h.white``, ``?h.pial``, etc.).

    Parameters
    ----------
    filepath : str
        Path to the surface file.
    hemisphere : str, optional
        Auto-detected from filename if not given.
    surface_type : str, optional
        Auto-detected from filename if not given.

    Returns
    -------
    SurfaceMesh

    Raises
    ------
    ImportError
        If nibabel is not installed.
    """
    try:
        import nibabel.freesurfer as fs
    except ImportError:
        raise ImportError(
            "nibabel is required for FreeSurfer I/O: "
            "pip install nibabel"
        )

    vertices, faces = fs.read_geometry(filepath)

    # Auto-detect hemisphere and surface type from filename
    basename = os.path.basename(filepath)
    if hemisphere is None:
        if basename.startswith("lh"):
            hemisphere = "lh"
        elif basename.startswith("rh"):
            hemisphere = "rh"
        else:
            hemisphere = "unknown"

    if surface_type is None:
        parts = basename.split(".")
        surface_type = parts[-1] if len(parts) > 1 else "unknown"

    return SurfaceMesh(
        vertices=vertices,
        faces=faces,
        hemisphere=hemisphere,
        surface_type=surface_type,
        metadata={"source": filepath},
    )


def load_freesurfer_morph(filepath: str) -> np.ndarray:
    """
    Load a FreeSurfer per-vertex morphometry file.

    Works for ``?h.thickness``, ``?h.curv``, ``?h.sulc``,
    ``?h.area``, ``?h.volume``, etc.

    Parameters
    ----------
    filepath : str

    Returns
    -------
    np.ndarray (V,)
    """
    try:
        import nibabel.freesurfer as fs
    except ImportError:
        raise ImportError("nibabel required: pip install nibabel")
    return fs.read_morph_data(filepath)


def load_freesurfer_annot(
    filepath: str,
) -> Tuple[np.ndarray, list]:
    """
    Load a FreeSurfer annotation (parcellation) file.

    Returns vertex labels as sequential 1-indexed integers
    (0 = unknown / medial wall) and region names.

    Parameters
    ----------
    filepath : str

    Returns
    -------
    labels : np.ndarray (V,)
        1-indexed integer labels (0 = medial wall).
    names : list of str
        Region names in label order.
    """
    try:
        import nibabel.freesurfer as fs
    except ImportError:
        raise ImportError("nibabel required: pip install nibabel")

    raw_labels, ctab, names = fs.read_annot(filepath)

    # Convert colour-table IDs to sequential integers
    unique_ids = np.unique(raw_labels)
    label_map = {old: new for new, old in enumerate(unique_ids)}
    labels = np.array([label_map[l] for l in raw_labels], dtype=np.int32)

    # Decode names
    names_str = [n.decode() if isinstance(n, bytes) else str(n)
                 for n in names]

    return labels, names_str


# ═══════════════════════════════════════════════════════════════════════════
#  GIFTI
# ═══════════════════════════════════════════════════════════════════════════

def load_gifti_surface(filepath: str) -> SurfaceMesh:
    """
    Load a GIFTI surface file (``.surf.gii``).

    Parameters
    ----------
    filepath : str

    Returns
    -------
    SurfaceMesh
    """
    try:
        import nibabel as nib
    except ImportError:
        raise ImportError("nibabel required: pip install nibabel")

    gii = nib.load(filepath)
    vertices = gii.darrays[0].data.astype(np.float64)
    faces = gii.darrays[1].data.astype(np.int64)

    return SurfaceMesh(
        vertices=vertices, faces=faces,
        metadata={"source": filepath, "format": "gifti"},
    )


def load_gifti_data(filepath: str) -> np.ndarray:
    """Load a GIFTI functional/shape data file (``.func.gii``, ``.shape.gii``)."""
    try:
        import nibabel as nib
    except ImportError:
        raise ImportError("nibabel required: pip install nibabel")

    gii = nib.load(filepath)
    return gii.darrays[0].data


# ═══════════════════════════════════════════════════════════════════════════
#  EIGENMORPH SERIALISATION (NPZ)
# ═══════════════════════════════════════════════════════════════════════════

def save_eigenfeatures(
    filepath: str,
    features: EigenFeatures,
    mesh: Optional[SurfaceMesh] = None,
    metadata: Optional[dict] = None,
):
    """
    Save EigenFeatures to a compressed ``.npz`` file.

    Parameters
    ----------
    filepath : str
        Output path (should end in ``.npz``).
    features : EigenFeatures
    mesh : SurfaceMesh, optional
        If given, mesh geometry is saved alongside features.
    metadata : dict, optional
        Additional metadata (serialised as JSON string).
    """
    data = {
        "radius": np.array([features.radius]),
        "weighting": np.array([features.weighting]),
        "eigenvalues": features.eigenvalues,
        "n_neighbors": features.n_neighbors,
    }

    for name in EigenFeatures.feature_names():
        data[name] = getattr(features, name)

    if features.eigenvectors is not None:
        data["eigenvectors"] = features.eigenvectors

    if mesh is not None:
        data["vertices"] = mesh.vertices
        data["faces"] = mesh.faces
        data["hemisphere"] = np.array([mesh.hemisphere])
        data["surface_type"] = np.array([mesh.surface_type])

    if metadata:
        data["metadata_json"] = np.array([json.dumps(metadata)])

    np.savez_compressed(filepath, **data)


def load_eigenfeatures(filepath: str) -> Tuple[EigenFeatures, Optional[SurfaceMesh]]:
    """
    Load EigenFeatures from a ``.npz`` file.

    Parameters
    ----------
    filepath : str

    Returns
    -------
    features : EigenFeatures
    mesh : SurfaceMesh or None
    """
    data = np.load(filepath, allow_pickle=False)

    features = EigenFeatures(
        radius=float(data["radius"][0]),
        linearity=data["linearity"],
        planarity=data["planarity"],
        sphericity=data["sphericity"],
        omnivariance=data["omnivariance"],
        anisotropy=data["anisotropy"],
        eigenentropy=data["eigenentropy"],
        surface_variation=data["surface_variation"],
        eigenvalues=data["eigenvalues"],
        eigenvectors=data.get("eigenvectors"),
        n_neighbors=data["n_neighbors"],
        weighting=str(data["weighting"][0]),
    )

    mesh = None
    if "vertices" in data:
        mesh = SurfaceMesh(
            vertices=data["vertices"],
            faces=data["faces"],
            hemisphere=str(data["hemisphere"][0]),
            surface_type=str(data["surface_type"][0]),
        )

    return features, mesh


def save_multiscale(filepath: str, ms_features: MultiScaleFeatures):
    """
    Save MultiScaleFeatures to a compressed ``.npz`` file.

    Parameters
    ----------
    filepath : str
    ms_features : MultiScaleFeatures
    """
    data = {
        "radii": np.array(ms_features.radii),
        "n_vertices": np.array([ms_features.n_vertices]),
        "n_scales": np.array([len(ms_features.scales)]),
    }

    for i, s in enumerate(ms_features.scales):
        prefix = f"s{i}_"
        data[f"{prefix}radius"] = np.array([s.radius])
        data[f"{prefix}weighting"] = np.array([s.weighting])
        data[f"{prefix}eigenvalues"] = s.eigenvalues
        if s.n_neighbors is not None:
            data[f"{prefix}n_neighbors"] = s.n_neighbors
        for name in EigenFeatures.feature_names():
            data[f"{prefix}{name}"] = getattr(s, name)

    np.savez_compressed(filepath, **data)


def load_multiscale(filepath: str) -> MultiScaleFeatures:
    """
    Load MultiScaleFeatures from a ``.npz`` file.

    Parameters
    ----------
    filepath : str

    Returns
    -------
    MultiScaleFeatures
    """
    data = np.load(filepath, allow_pickle=False)
    n_scales = int(data["n_scales"][0])
    radii = data["radii"].tolist()

    scales = []
    for i in range(n_scales):
        prefix = f"s{i}_"
        feat = EigenFeatures(
            radius=float(data[f"{prefix}radius"][0]),
            linearity=data[f"{prefix}linearity"],
            planarity=data[f"{prefix}planarity"],
            sphericity=data[f"{prefix}sphericity"],
            omnivariance=data[f"{prefix}omnivariance"],
            anisotropy=data[f"{prefix}anisotropy"],
            eigenentropy=data[f"{prefix}eigenentropy"],
            surface_variation=data[f"{prefix}surface_variation"],
            eigenvalues=data[f"{prefix}eigenvalues"],
            n_neighbors=data.get(f"{prefix}n_neighbors"),
            weighting=str(data[f"{prefix}weighting"][0]),
        )
        scales.append(feat)

    return MultiScaleFeatures(
        scales=scales, radii=radii,
        n_vertices=int(data["n_vertices"][0]),
    )
