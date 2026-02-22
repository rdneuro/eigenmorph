# -*- coding: utf-8 -*-
"""
eigenmorph.viz.interactive
==========================

High-impact 3D visualizations using FURY (VTK) for publication figures
and interactive exploration.

All functions gracefully fall back to matplotlib ``Poly3DCollection``
when FURY is not installed, producing lower-quality but functional
equivalents.

Functions
---------
plot_rgb_identity          L→R, P→G, S→B per-vertex colouring.
plot_feature_landscape     Topographic surface deformed by a scalar.
render_scale_sweep         Animated GIF from fine to coarse scale.
plot_exploded_view         Vertices clustered and spatially separated.
plot_neighborhood_explorer Multi-radius neighbourhood shells.
plot_dual_hemisphere       Side-by-side lh/rh renders.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors as mcolors
from scipy.spatial import cKDTree
from typing import Optional, Dict, List, Tuple

from ..core import SurfaceMesh, EigenFeatures, MultiScaleFeatures
from .styles import setup_style, get_feature_cmap

# ── Optional imports ──
_HAS_FURY = False
try:
    from fury import actor, window
    _HAS_FURY = True
except ImportError:
    pass

_HAS_IMAGEIO = False
try:
    import imageio
    _HAS_IMAGEIO = True
except ImportError:
    pass


def _require_fury(name: str):
    if not _HAS_FURY:
        raise ImportError(
            f"{name} requires FURY (pip install fury). "
            f"Install for interactive 3D; matplotlib fallback used otherwise."
        )


# ═══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _lps_to_rgb(L, P, S, gamma=0.7):
    L = np.nan_to_num(L, nan=0.0)
    P = np.nan_to_num(P, nan=0.0)
    S = np.nan_to_num(S, nan=0.0)
    total = L + P + S
    total[total < 1e-12] = 1.0
    r = np.power(np.clip(L/total, 0, 1), gamma)
    g = np.power(np.clip(P/total, 0, 1), gamma)
    b = np.power(np.clip(S/total, 0, 1), gamma)
    return np.column_stack([r, g, b])


def _mpl_surface(vertices, faces, face_rgba, title, save_path):
    """Matplotlib 3D fallback renderer."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    setup_style()
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    polys = Poly3DCollection(
        vertices[faces], facecolors=face_rgba,
        edgecolors="none", linewidths=0,
    )
    ax.add_collection3d(polys)
    v = vertices
    m = 5
    ax.set_xlim(v[:,0].min()-m, v[:,0].max()+m)
    ax.set_ylim(v[:,1].min()-m, v[:,1].max()+m)
    ax.set_zlim(v[:,2].min()-m, v[:,2].max()+m)
    ax.view_init(30, -60)
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


def _fury_render(mesh, colors_uint8, size, offscreen, save_path, title,
                 camera_pos=(0, -300, 50)):
    """Common FURY surface rendering."""
    surface = actor.surface(
        vertices=mesh.vertices.astype(np.float64),
        faces=mesh.faces,
        colors=colors_uint8,
    )
    scene = window.Scene()
    scene.add(surface)
    scene.set_camera(
        position=camera_pos,
        focal_point=tuple(mesh.centroid),
        view_up=(0, 0, 1),
    )
    scene.background((1, 1, 1))

    if offscreen or save_path:
        return window.snapshot(scene, fname=save_path, size=size,
                               offscreen=True)
    else:
        window.show(scene, size=size, title=title)
        return None


# ═══════════════════════════════════════════════════════════════════════════
#  1. RGB IDENTITY MAP
# ═══════════════════════════════════════════════════════════════════════════

def plot_rgb_identity(
    mesh: SurfaceMesh,
    features: EigenFeatures,
    gamma: float = 0.7,
    size: Tuple[int, int] = (1200, 900),
    offscreen: bool = False,
    save_path: Optional[str] = None,
) -> Optional[np.ndarray]:
    """
    Render the cortical surface with L→Red, P→Green, S→Blue colouring.

    Each vertex receives a unique RGB encoding its geometric identity:
    sulcal fundi glow red, gyral crowns green, sulcal pits blue.

    Parameters
    ----------
    mesh : SurfaceMesh
    features : EigenFeatures
    gamma : float
        Gamma correction (< 1 brightens midtones).
    size : tuple
        Window / image size in pixels.
    offscreen : bool
        Render to array without window.
    save_path : str, optional
        Save rendered image (PNG).

    Returns
    -------
    np.ndarray or None
        RGBA array if offscreen, else None (interactive window).
    """
    rgb = _lps_to_rgb(features.linearity, features.planarity,
                       features.sphericity, gamma)

    if _HAS_FURY:
        rgba = np.column_stack([rgb, np.ones(len(rgb))])
        colors_u8 = (np.clip(rgba, 0, 1) * 255).astype(np.uint8)
        return _fury_render(mesh, colors_u8, size, offscreen, save_path,
                             "EigenMorph RGB Identity")
    else:
        face_rgba = np.column_stack([
            rgb[mesh.faces].mean(axis=1),
            np.ones(mesh.n_faces)
        ])
        fig = _mpl_surface(mesh.vertices, mesh.faces, face_rgba,
                           "Geometric Identity: L→R  P→G  S→B", save_path)
        from matplotlib.patches import Patch
        ax = fig.axes[0]
        ax.legend(handles=[
            Patch(facecolor="red", label="Linearity (sulcal fundi)"),
            Patch(facecolor="green", label="Planarity (gyral crowns)"),
            Patch(facecolor="blue", label="Sphericity (sulcal pits)"),
        ], loc="lower left", fontsize=8)
        return None


# ═══════════════════════════════════════════════════════════════════════════
#  2. FEATURE LANDSCAPE (topographic deformation)
# ═══════════════════════════════════════════════════════════════════════════

def plot_feature_landscape(
    mesh: SurfaceMesh,
    values: np.ndarray,
    title: str = "Eigenentropy Landscape",
    cmap: str = "inferno",
    deformation_scale: float = 5.0,
    size: Tuple[int, int] = (1200, 900),
    offscreen: bool = False,
    save_path: Optional[str] = None,
) -> Optional[np.ndarray]:
    """
    Topographic surface: the cortex is inflated outward in proportion
    to a scalar feature, creating a terrain where peaks = high values.

    Parameters
    ----------
    mesh : SurfaceMesh
    values : (V,) feature to encode.
    title, cmap, deformation_scale : display opts
    size, offscreen, save_path : rendering opts
    """
    vals = np.nan_to_num(values, nan=0.0)
    valid = values[np.isfinite(values)]
    vmin, vmax = np.percentile(valid, [5, 95]) if len(valid) else (0, 1)
    vals_n = np.clip((vals - vmin) / max(vmax - vmin, 1e-12), 0, 1)

    # Deform along normals
    normals = mesh.vertex_normals
    deformed = mesh.vertices + normals * (vals_n * deformation_scale)[:, np.newaxis]

    mapper = cm.ScalarMappable(
        norm=mcolors.Normalize(vmin=vmin, vmax=vmax), cmap=cmap,
    )
    rgba = mapper.to_rgba(vals)

    if _HAS_FURY:
        colors_u8 = (np.clip(rgba, 0, 1) * 255).astype(np.uint8)
        deformed_mesh = mesh.copy()
        deformed_mesh.vertices = deformed.astype(np.float64)
        return _fury_render(deformed_mesh, colors_u8, size, offscreen,
                             save_path, title,
                             camera_pos=(0, -350, 80))
    else:
        face_rgba = rgba[mesh.faces].mean(axis=1)
        return _mpl_surface(deformed, mesh.faces, face_rgba,
                            title, save_path)


# ═══════════════════════════════════════════════════════════════════════════
#  3. ANIMATED SCALE SWEEP (GIF)
# ═══════════════════════════════════════════════════════════════════════════

def render_scale_sweep(
    mesh: SurfaceMesh,
    ms_features: MultiScaleFeatures,
    feature_name: str = "eigenentropy",
    cmap: str = "viridis",
    n_interp_frames: int = 10,
    size: Tuple[int, int] = (800, 600),
    save_path: str = "scale_sweep.gif",
    fps: int = 8,
) -> Optional[str]:
    """
    Animated GIF sweeping a feature from fine to coarse scale.

    Interpolates smoothly between computed scales for a fluid animation
    showing how cortical geometry evolves across spatial scales.

    Parameters
    ----------
    mesh : SurfaceMesh
    ms_features : MultiScaleFeatures
    feature_name : str
    cmap : str
    n_interp_frames : int
        Frames interpolated between each pair of scales.
    size, save_path, fps : output opts

    Returns
    -------
    str or None
        Path to GIF if saved, None on error.
    """
    if not _HAS_IMAGEIO:
        raise ImportError("render_scale_sweep requires imageio.")

    # Gather per-scale data
    scales_data = []
    for s in ms_features.scales:
        scales_data.append(np.nan_to_num(getattr(s, feature_name), nan=0.0))

    # Global colour range
    all_v = np.concatenate(scales_data)
    vmin, vmax = np.percentile(all_v, [2, 98])
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)

    # Build interpolated frames
    frames = []
    for i in range(len(scales_data) - 1):
        for t in np.linspace(0, 1, n_interp_frames, endpoint=False):
            frames.append((1 - t) * scales_data[i] + t * scales_data[i+1])
    frames.append(scales_data[-1])

    radii = ms_features.radii
    images = []

    if _HAS_FURY:
        for idx, fv in enumerate(frames):
            progress = idx / max(len(frames)-1, 1)
            cur_r = np.interp(progress, np.linspace(0, 1, len(radii)), radii)

            rgba = mapper.to_rgba(fv)
            cu8 = (np.clip(rgba, 0, 1) * 255).astype(np.uint8)

            surface = actor.surface(
                vertices=mesh.vertices.astype(np.float64),
                faces=mesh.faces, colors=cu8,
            )
            label = actor.text_3d(
                f"{feature_name}  r = {cur_r:.1f} mm",
                position=tuple(mesh.centroid + [-80, -120, 80]),
                color=(0.1, 0.1, 0.1), font_size=18,
            )
            scene = window.Scene()
            scene.add(surface)
            scene.add(label)
            scene.set_camera(
                position=(0, -300, 50),
                focal_point=tuple(mesh.centroid),
                view_up=(0, 0, 1),
            )
            scene.background((1, 1, 1))
            images.append(window.snapshot(scene, size=size, offscreen=True))
    else:
        # Matplotlib fallback
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        for idx, fv in enumerate(frames):
            progress = idx / max(len(frames)-1, 1)
            cur_r = np.interp(progress, np.linspace(0, 1, len(radii)), radii)

            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection="3d")
            face_rgba = mapper.to_rgba(fv[mesh.faces].mean(axis=1))
            polys = Poly3DCollection(
                mesh.vertices[mesh.faces],
                facecolors=face_rgba, edgecolors="none", linewidths=0,
            )
            ax.add_collection3d(polys)
            v = mesh.vertices
            ax.set_xlim(v[:,0].min()-5, v[:,0].max()+5)
            ax.set_ylim(v[:,1].min()-5, v[:,1].max()+5)
            ax.set_zlim(v[:,2].min()-5, v[:,2].max()+5)
            ax.view_init(30, -60)
            ax.axis("off")
            ax.set_title(f"{feature_name}  r = {cur_r:.1f} mm",
                         fontsize=14, fontweight="bold")
            fig.canvas.draw()
            buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            w, h = fig.canvas.get_width_height()
            images.append(buf.reshape(h, w, 4)[:, :, :3].copy())
            plt.close(fig)

    imageio.mimsave(save_path, images, fps=fps, loop=0)
    return save_path


# ═══════════════════════════════════════════════════════════════════════════
#  4. EXPLODED VIEW
# ═══════════════════════════════════════════════════════════════════════════

def plot_exploded_view(
    mesh: SurfaceMesh,
    ms_features: MultiScaleFeatures,
    n_clusters: int = 6,
    explosion_factor: float = 30.0,
    seed: int = 42,
    size: Tuple[int, int] = (1200, 900),
    offscreen: bool = False,
    save_path: Optional[str] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Cluster vertices by geometric type and separate spatially.

    Runs k-means in the multi-scale feature space, then displaces each
    cluster outward to create an exploded-view render.

    Returns
    -------
    cluster_labels : (V,)
    image : np.ndarray or None
    """
    from scipy.cluster.vq import kmeans2

    feat = np.nan_to_num(ms_features.as_matrix(), nan=0.0)
    mu, sig = feat.mean(0), feat.std(0)
    sig[sig < 1e-12] = 1.0
    feat_std = (feat - mu) / sig

    _, labels = kmeans2(feat_std, n_clusters, minit="++", seed=seed)

    global_c = mesh.centroid
    directions = np.zeros((n_clusters, 3))
    for c in range(n_clusters):
        mask = labels == c
        if mask.sum() > 0:
            d = mesh.vertices[mask].mean(0) - global_c
            n = np.linalg.norm(d)
            directions[c] = d / n if n > 1e-6 else [1, 0, 0]

    displaced = mesh.vertices.copy()
    for c in range(n_clusters):
        displaced[labels == c] += directions[c] * explosion_factor

    palette = plt.cm.Set1(np.linspace(0, 1, n_clusters))[:, :3]
    vcol = palette[labels]

    if _HAS_FURY:
        rgba_u8 = np.column_stack([vcol, np.ones(len(vcol))])
        rgba_u8 = (np.clip(rgba_u8, 0, 1) * 255).astype(np.uint8)

        displaced_mesh = mesh.copy()
        displaced_mesh.vertices = displaced.astype(np.float64)

        img = _fury_render(displaced_mesh, rgba_u8, size, offscreen,
                            save_path, "Exploded Geometric Communities",
                            camera_pos=(0, -400, 100))
        return labels, img
    else:
        face_rgba = np.column_stack([
            vcol[mesh.faces].mean(axis=1),
            np.ones(mesh.n_faces)
        ])
        fig = _mpl_surface(displaced, mesh.faces, face_rgba,
                           f"Exploded View: {n_clusters} Geometric Communities",
                           save_path)
        # Legend
        from matplotlib.patches import Patch
        ax = fig.axes[0]
        ax.legend(handles=[
            Patch(facecolor=palette[c],
                  label=f"Type {c+1} ({(labels==c).sum():,} verts)")
            for c in range(n_clusters)
        ], loc="lower left", fontsize=7)
        return labels, None


# ═══════════════════════════════════════════════════════════════════════════
#  5. NEIGHBOURHOOD EXPLORER
# ═══════════════════════════════════════════════════════════════════════════

def plot_neighborhood_explorer(
    mesh: SurfaceMesh,
    vertex_idx: int,
    radii: Optional[List[float]] = None,
    size: Tuple[int, int] = (1200, 900),
    offscreen: bool = False,
    save_path: Optional[str] = None,
) -> Optional[np.ndarray]:
    """
    Visualise KD-tree neighbourhoods at multiple scales around one vertex.

    The centre vertex is highlighted in red; each radius is rendered as
    a coloured shell of small spheres (FURY) or scatter points (mpl).
    """
    if radii is None:
        radii = [3.0, 5.0, 10.0, 20.0]

    tree = cKDTree(mesh.vertices)
    center = mesh.vertices[vertex_idx]
    palette = plt.cm.viridis(np.linspace(0.2, 0.9, len(radii)))[:, :3]

    if _HAS_FURY:
        scene = window.Scene()

        # Base surface (semi-transparent grey)
        gray = np.full((mesh.n_vertices, 4), [180, 180, 180, 50],
                        dtype=np.uint8)
        scene.add(actor.surface(
            vertices=mesh.vertices.astype(np.float64),
            faces=mesh.faces, colors=gray,
        ))

        # Centre vertex
        scene.add(actor.sphere(
            centers=np.array([center]),
            colors=np.array([[255, 0, 0, 255]], dtype=np.uint8),
            radii=1.5,
        ))

        # Shells
        for ri, r in enumerate(radii):
            nb = tree.query_ball_point(center, r)
            if not nb:
                continue
            pts = mesh.vertices[nb].astype(np.float64)
            col = (palette[ri] * 255).astype(np.uint8)
            col_rgba = np.tile(
                np.append(col, 140).astype(np.uint8), (len(pts), 1)
            )
            scene.add(actor.sphere(centers=pts, colors=col_rgba, radii=0.4))

        scene.set_camera(
            position=tuple(center + [0, -80, 30]),
            focal_point=tuple(center), view_up=(0, 0, 1),
        )
        scene.background((1, 1, 1))

        if offscreen or save_path:
            return window.snapshot(scene, fname=save_path, size=size,
                                   offscreen=True)
        else:
            window.show(scene, size=size, title="Neighbourhood Explorer")
            return None
    else:
        setup_style()
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection="3d")

        for ri, r in enumerate(reversed(radii)):
            nb = tree.query_ball_point(center, r)
            pts = mesh.vertices[nb]
            ax.scatter(pts[:,0], pts[:,1], pts[:,2],
                       s=3, alpha=0.3, color=palette[len(radii)-1-ri],
                       label=f"r={r:.0f}mm ({len(nb)} pts)")

        ax.scatter(*center, s=100, c="red", marker="*", zorder=10,
                   label="Centre vertex")

        mg = max(radii) + 5
        ax.set_xlim(center[0]-mg, center[0]+mg)
        ax.set_ylim(center[1]-mg, center[1]+mg)
        ax.set_zlim(center[2]-mg, center[2]+mg)
        ax.legend(fontsize=7)
        ax.set_title(f"Neighbourhood Explorer — vertex {vertex_idx}",
                     fontweight="bold")
        ax.axis("off")
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path)
        return None


# ═══════════════════════════════════════════════════════════════════════════
#  6. DUAL HEMISPHERE
# ═══════════════════════════════════════════════════════════════════════════

def plot_dual_hemisphere(
    mesh_lh: SurfaceMesh,
    mesh_rh: SurfaceMesh,
    values_lh: np.ndarray,
    values_rh: np.ndarray,
    title: str = "Bilateral View",
    cmap: str = "viridis",
    figsize: Tuple[int, int] = (20, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Side-by-side left and right hemisphere surface renders.

    Falls back to matplotlib always (FURY dual-window is complex).
    """
    setup_style()
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=figsize, subplot_kw={"projection": "3d"},
    )

    for ax, mesh, vals, hemi in [
        (ax1, mesh_lh, values_lh, "Left"),
        (ax2, mesh_rh, values_rh, "Right"),
    ]:
        v_clean = np.nan_to_num(vals, nan=0.0)
        valid = vals[np.isfinite(vals)]
        vmin = np.percentile(valid, 5) if len(valid) else 0
        vmax = np.percentile(valid, 95) if len(valid) else 1

        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
        fv = v_clean[mesh.faces].mean(axis=1)
        fr = mapper.to_rgba(fv)

        polys = Poly3DCollection(
            mesh.vertices[mesh.faces], facecolors=fr,
            edgecolors="none", linewidths=0,
        )
        ax.add_collection3d(polys)
        v = mesh.vertices
        m = 5
        ax.set_xlim(v[:,0].min()-m, v[:,0].max()+m)
        ax.set_ylim(v[:,1].min()-m, v[:,1].max()+m)
        ax.set_zlim(v[:,2].min()-m, v[:,2].max()+m)
        ax.view_init(30, -60 if hemi == "Left" else 60)
        ax.axis("off")
        ax.set_title(f"{hemi} Hemisphere", fontsize=14, fontweight="bold")
        fig.colorbar(mapper, ax=ax, shrink=0.5, pad=0.05)

    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.95)
    if save_path:
        fig.savefig(save_path)
    return fig
