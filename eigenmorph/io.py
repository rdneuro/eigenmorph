# -*- coding: utf-8 -*-
"""
eigenmorph.io
=============

Input/output for cortical surfaces and eigenvalue features.

Supports FreeSurfer geometry, morphometry overlays, annotation files,
GIFTI surfaces, volumetric NIfTI/MGZ files (via marching cubes
isosurface extraction), and NPZ serialisation of computed features.

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
#  VOLUMETRIC DATA → SURFACE MESH
# ═══════════════════════════════════════════════════════════════════════════

def load_volume_as_mesh(
    filepath: str,
    level: Optional[float] = None,
    smooth_iter: int = 0,
    step_size: int = 1,
    label: Optional[int] = None,
    apply_affine: bool = True,
) -> SurfaceMesh:
    """
    Load a NIfTI / FreeSurfer volume and extract an isosurface mesh.

    Uses the marching cubes algorithm to convert a 3D volume (binary mask,
    probabilistic map, or label segmentation) into a triangulated surface
    mesh in world (RAS) coordinates.  This enables eigenmorph analysis on
    volumetric parcellations (e.g. ``aseg.mgz``, ``aparc+aseg.nii.gz``,
    or any ``.nii.gz`` binary mask).

    Parameters
    ----------
    filepath : str
        Path to any nibabel-readable volume: ``.nii``, ``.nii.gz``,
        ``.mgz``, ``.mgh``.
    level : float, optional
        Isosurface threshold for marching cubes.  If *None*, the
        function uses the midpoint between the volume's nonzero minimum
        and maximum, which works well for binary masks and probability
        maps.  For label volumes, use the ``label`` parameter instead.
    smooth_iter : int
        Number of Laplacian smoothing iterations applied to the mesh
        after extraction.  0 = no smoothing.  A value of 3-10 often
        helps remove staircase artefacts from voxelised surfaces
        without excessive shrinkage.
    step_size : int
        Step size for marching cubes (1 = full resolution, 2 = half,
        etc.).  Larger values speed up extraction but produce coarser
        meshes.
    label : int, optional
        If given, the volume is first binarised to keep only voxels
        equal to this label.  Useful for extracting a single structure
        from a label atlas (e.g. label 17 = left hippocampus in
        FreeSurfer's ``aseg``).
    apply_affine : bool
        If True (default), transform vertices from voxel indices to
        world (scanner RAS) coordinates using the volume's affine
        matrix.  Set to False to keep vertices in voxel space.

    Returns
    -------
    SurfaceMesh
        Triangulated isosurface with vertices in mm (RAS) if
        ``apply_affine=True``, or in voxel coordinates otherwise.

    Raises
    ------
    ImportError
        If nibabel or scikit-image is not installed.
    ValueError
        If the volume is empty after thresholding or label selection.

    Examples
    --------
    Load a binary mask of a lesion and compute eigenfeatures::

        mesh = em.io.load_volume_as_mesh("lesion_mask.nii.gz")
        feats = em.compute_eigenfeatures(mesh, radius=3.0)

    Extract left hippocampus (label 17) from FreeSurfer aseg::

        mesh = em.io.load_volume_as_mesh(
            "aseg.mgz", label=17, smooth_iter=5,
        )

    Load a probabilistic map with a custom threshold::

        mesh = em.io.load_volume_as_mesh(
            "prob_map.nii.gz", level=0.5,
        )
    """
    try:
        import nibabel as nib
    except ImportError:
        raise ImportError(
            "nibabel is required for volume loading: pip install nibabel"
        )
    try:
        from skimage.measure import marching_cubes
    except ImportError:
        raise ImportError(
            "scikit-image is required for volume-to-mesh conversion: "
            "pip install scikit-image"
        )

    # ── Load volume ──────────────────────────────────────────────────────
    img = nib.load(filepath)
    data = np.asarray(img.dataobj, dtype=np.float64)
    affine = img.affine

    # ── Optional label selection ─────────────────────────────────────────
    if label is not None:
        data = (data == label).astype(np.float64)

    # ── Determine isosurface level ───────────────────────────────────────
    nonzero = data[data > 0]
    if nonzero.size == 0:
        tag = f" (label={label})" if label is not None else ""
        raise ValueError(
            f"Volume '{filepath}'{tag} contains no nonzero voxels "
            f"after thresholding."
        )
    if level is None:
        vmin, vmax = float(nonzero.min()), float(nonzero.max())
        if np.isclose(vmin, vmax):
            # Binary mask or constant-valued region: threshold halfway
            # between zero and the constant (e.g. 0.5 for a {0,1} mask)
            level = 0.5 * vmax
        else:
            level = 0.5 * (vmin + vmax)

    # ── Marching cubes ───────────────────────────────────────────────────
    vertices, faces, normals, _ = marching_cubes(
        data, level=level, step_size=step_size,
    )

    # ── Optional Laplacian smoothing ─────────────────────────────────────
    if smooth_iter > 0:
        vertices = _laplacian_smooth(vertices, faces, n_iter=smooth_iter)

    # ── Affine: voxel → world (RAS mm) ──────────────────────────────────
    if apply_affine:
        # Homogeneous coordinates: append ones column, multiply, drop
        ones = np.ones((vertices.shape[0], 1), dtype=np.float64)
        vertices_h = np.hstack([vertices, ones])
        vertices = (affine @ vertices_h.T).T[:, :3]

    # ── Metadata ─────────────────────────────────────────────────────────
    meta = {
        "source": filepath,
        "format": "volume_isosurface",
        "level": float(level),
        "smooth_iter": smooth_iter,
        "step_size": step_size,
        "apply_affine": apply_affine,
        "voxel_size": np.abs(np.diag(affine[:3, :3])).tolist(),
    }
    if label is not None:
        meta["label"] = label

    return SurfaceMesh(
        vertices=vertices,
        faces=faces,
        hemisphere="unknown",
        surface_type="isosurface",
        metadata=meta,
    )


def load_volume_labels_as_meshes(
    filepath: str,
    labels: Optional[list] = None,
    label_names: Optional[Dict[int, str]] = None,
    smooth_iter: int = 3,
    step_size: int = 1,
    min_voxels: int = 50,
    apply_affine: bool = True,
    verbose: bool = True,
) -> Dict[int, SurfaceMesh]:
    """
    Extract one mesh per label from a multi-label segmentation volume.

    Iterates over unique nonzero labels in the volume (or a user-supplied
    subset) and calls :func:`load_volume_as_mesh` for each.  Useful for
    batch-processing all structures in an ``aseg.mgz`` or an atlas
    parcellation.

    Parameters
    ----------
    filepath : str
        Path to label volume (``.nii``, ``.nii.gz``, ``.mgz``).
    labels : list of int, optional
        Specific labels to extract.  If *None*, all unique nonzero
        values in the volume are used.
    label_names : dict {int: str}, optional
        Mapping from label integers to human-readable names, stored in
        mesh metadata.
    smooth_iter : int
        Laplacian smoothing iterations per mesh.
    step_size : int
        Marching cubes step size.
    min_voxels : int
        Skip labels with fewer than this many voxels.
    apply_affine : bool
        Transform vertices to world (RAS) coordinates.
    verbose : bool
        Print progress.

    Returns
    -------
    dict {int: SurfaceMesh}
        Meshes keyed by label integer.

    Examples
    --------
    Extract all structures from FreeSurfer aseg::

        meshes = em.io.load_volume_labels_as_meshes("aseg.mgz")
        for lbl, mesh in meshes.items():
            feats = em.compute_eigenfeatures(mesh, radius=3.0)
    """
    try:
        import nibabel as nib
    except ImportError:
        raise ImportError("nibabel required: pip install nibabel")

    img = nib.load(filepath)
    data = np.asarray(img.dataobj)

    # Discover labels
    if labels is None:
        labels = sorted(int(v) for v in np.unique(data) if v != 0)

    if verbose:
        print(f"  Extracting meshes from {len(labels)} labels in "
              f"'{os.path.basename(filepath)}'")

    meshes = {}
    for lbl in labels:
        n_vox = int((data == lbl).sum())
        if n_vox < min_voxels:
            if verbose:
                name = (label_names or {}).get(lbl, "")
                print(f"    Label {lbl:>5d} {name:>20s}: "
                      f"{n_vox} voxels → skipped (< {min_voxels})")
            continue

        try:
            mesh = load_volume_as_mesh(
                filepath, label=lbl,
                smooth_iter=smooth_iter,
                step_size=step_size,
                apply_affine=apply_affine,
            )
            if label_names and lbl in label_names:
                mesh.metadata["label_name"] = label_names[lbl]

            meshes[lbl] = mesh

            if verbose:
                name = (label_names or {}).get(lbl, "")
                print(f"    Label {lbl:>5d} {name:>20s}: "
                      f"{mesh.n_vertices:>7,} vertices, "
                      f"{mesh.n_faces:>7,} faces")

        except (ValueError, RuntimeError) as e:
            if verbose:
                print(f"    Label {lbl:>5d}: FAILED ({e})")

    return meshes


def _laplacian_smooth(
    vertices: np.ndarray,
    faces: np.ndarray,
    n_iter: int = 3,
    lam: float = 0.5,
) -> np.ndarray:
    """
    Simple Laplacian smoothing on a triangle mesh.

    Each vertex is moved toward the centroid of its immediate neighbors,
    weighted by ``lam`` ∈ (0, 1).  This removes staircase artefacts from
    marching cubes output without introducing external dependencies.

    Parameters
    ----------
    vertices : (V, 3)
    faces : (F, 3)
    n_iter : int
        Number of iterations.
    lam : float
        Smoothing weight per step (0 = no movement, 1 = full snap to
        centroid).

    Returns
    -------
    smoothed : (V, 3)
    """
    from collections import defaultdict

    V = vertices.shape[0]
    adj = defaultdict(set)
    for f in faces:
        for i in range(3):
            for j in range(3):
                if i != j:
                    adj[f[i]].add(f[j])

    smoothed = vertices.copy()
    for _ in range(n_iter):
        new_verts = smoothed.copy()
        for v in range(V):
            nbrs = adj[v]
            if nbrs:
                centroid = smoothed[list(nbrs)].mean(axis=0)
                new_verts[v] = (1 - lam) * smoothed[v] + lam * centroid
        smoothed = new_verts

    return smoothed


def load_surface(filepath: str, **kwargs) -> SurfaceMesh:
    """
    Auto-detect file format and load as SurfaceMesh.

    Dispatches to the appropriate loader based on file extension:

    - ``.nii``, ``.nii.gz``, ``.mgz``, ``.mgh`` → :func:`load_volume_as_mesh`
    - ``.surf.gii`` or ``.gii`` → :func:`load_gifti_surface`
    - everything else → :func:`load_freesurfer_surface`

    Extra keyword arguments are forwarded to the detected loader.

    Parameters
    ----------
    filepath : str
    **kwargs
        Passed to the specific loader (e.g. ``label``, ``smooth_iter``
        for volumes).

    Returns
    -------
    SurfaceMesh
    """
    lower = filepath.lower()
    if lower.endswith((".nii", ".nii.gz", ".mgz", ".mgh")):
        return load_volume_as_mesh(filepath, **kwargs)
    elif lower.endswith(".gii"):
        return load_gifti_surface(filepath, **kwargs)
    else:
        return load_freesurfer_surface(filepath, **kwargs)


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
