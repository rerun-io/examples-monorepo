"""The map's surface: every keyframe's depth fused into a sparse colour TSDF (``packages/tsdf-rs``), one mesh.

The backend keeps refining keyframe poses, so a volume that only grows goes stale; the logger re-fuses every
keyframe at its latest pose from scratch instead (:func:`fuse_keyframes`), which ``tsdf-rs`` does in well under a
second for a few hundred keyframes at MASt3R's resolution.
"""

from dataclasses import dataclass

import lietorch_mojo as lietorch
import numpy as np
import rerun as rr
import torch
from jaxtyping import Float32, UInt8, UInt32
from numpy import ndarray
from tsdf_rs import TsdfVolume

from mast3r_slam.lietorch_utils import as_SE3


@dataclass(frozen=True, slots=True)
class KeyframeDepth:
    """What a keyframe contributes to the surface; its pose comes separately, since the backend refines it."""

    depth_hw: Float32[ndarray, "h w"]
    """Z of the keyframe's canonical pointmap in its own camera frame, before the Sim3 scale; 0 where the
    confidence is at or below the logger's threshold."""
    k_33: Float32[ndarray, "3 3"]
    """Pinhole intrinsics on the depth raster."""
    rgb_hw3: UInt8[ndarray, "h w 3"]
    """The model's cropped image, on the depth raster."""


@dataclass(frozen=True, slots=True)
class SurfaceSettings:
    """TSDF fusion parameters, in map units (MASt3R's pointmaps are roughly metric)."""

    voxel_size: float = 0.04
    """Voxel edge."""
    truncation_voxels: float = 3.0
    """Signed-distance truncation in voxels (Open3D's ``3 * voxel`` convention)."""


def fuse_keyframes(
    keyframes: list[tuple[KeyframeDepth, Float32[ndarray, "8"]]],
    settings: SurfaceSettings | None = None,
) -> rr.Mesh3D | None:
    """Fuse keyframes, each at its world-from-camera Sim3 (lietorch ``[t, q, s]``), into one front-faced mesh.

    A Sim3 maps a camera point ``X`` to ``s R X + t``: the rigid pose ``(R, t)`` with the depth scaled by ``s``.

    Returns:
        The mesh, rendering only front faces (the side the cameras saw), so walls seen from outside a room do not
        hide it; None if no surface came out.
    """
    if settings is None:
        settings = SurfaceSettings()
    volume: TsdfVolume = TsdfVolume(
        voxel_size=settings.voxel_size, sdf_trunc=settings.truncation_voxels * settings.voxel_size, fuse_color=True
    )
    for keyframe, world_sim3_cam_data in keyframes:
        sim3: lietorch.Sim3 = lietorch.Sim3(torch.from_numpy(world_sim3_cam_data).unsqueeze(0))
        world_T_cam: Float32[ndarray, "4 4"] = as_SE3(sim3).matrix()[0].numpy().astype(np.float32)
        scale: float = float(world_sim3_cam_data[7])
        volume.integrate(
            np.ascontiguousarray(keyframe.depth_hw * scale, dtype=np.float32),
            keyframe.k_33,
            np.linalg.inv(world_T_cam.astype(np.float64)).astype(np.float32),
            np.ascontiguousarray(keyframe.rgb_hw3),
        )
    vertices: Float32[ndarray, "v 3"]
    triangles: UInt32[ndarray, "t 3"]
    colors: UInt8[ndarray, "v 3"] | None
    normals: Float32[ndarray, "v 3"]
    vertices, triangles, colors, normals = volume.extract_mesh()
    if triangles.shape[0] == 0:
        return None
    return rr.Mesh3D(
        vertex_positions=vertices,
        triangle_indices=triangles,
        vertex_colors=colors,
        vertex_normals=normals,
        face_rendering=rr.components.MeshFaceRendering.Front,
    )
