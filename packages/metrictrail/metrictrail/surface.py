"""MetricTrail's default surface: a colour TSDF mesh fused from every tracked frame (``packages/tsdf-rs``).

Each tracked frame's metric depth (ViPE's depth artifact, MoGe-2 metric depth) is integrated into a sparse TSDF with
the frame's final camera pose, and marching cubes extracts one coloured mesh. It is the run's only surface: it
replaced upstream's fused point cloud (``vipe.metrictrail.pointcloud.fuse_point_cloud``: a multi-view consistency filter
and 2 cm voxel means in NumPy) at a fraction of its time.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from jaxtyping import Float, Float32, Float64, UInt8, UInt32
from numpy import ndarray
from simplecv.rerun_live_mesh import MeshArrays
from tsdf_rs import TsdfVolume

"""A ``tsdf_rs`` mesh: vertices (metres), triangles, colours (None without colour fusion), unit normals."""


@dataclass(frozen=True, slots=True)
class TsdfSettings:
    """TSDF fusion parameters."""

    voxel_size_m: float = 0.04
    """Voxel edge, metres: on pool (1,293 frames) 4 cm to 8 m gives 2.9 M triangles in 2.5 s; 2 cm to 15 m, 27 M."""
    truncation_voxels: float = 3.0
    """Signed-distance truncation in voxels (Open3D's ``3 * voxel`` convention)."""
    min_depth_m: float = 0.2
    """Nearest depth integrated, metres."""
    max_depth_m: float = 8.0
    """Farthest depth integrated, metres: MoGe-2's depth error grows with distance, and far surfaces dominate the
    volume (upstream's cloud fuses to 15 m)."""


DEFAULT_TSDF: TsdfSettings = TsdfSettings()
"""The settings of :func:`fuse_tsdf_mesh` unless given."""


@dataclass(frozen=True, slots=True)
class ColoredMesh:
    """A triangle mesh with per-vertex colours and unit normals, in the frame its producer states."""

    vertices_xyz: Float32[ndarray, "v 3"]
    """Vertex positions, metres."""
    triangles: UInt32[ndarray, "t 3"]
    """Vertex indices, counter-clockwise seen from outside."""
    colors_rgb: UInt8[ndarray, "v 3"]
    """Fused vertex colours."""
    normals_xyz: Float32[ndarray, "v 3"]
    """Unit vertex normals (the TSDF gradient), facing the cameras that observed the surface."""


def new_volume(settings: TsdfSettings) -> TsdfVolume:
    """An empty colour-fusing TSDF volume with the voxel size and truncation of ``settings``."""
    return TsdfVolume(voxel_size=settings.voxel_size_m, sdf_trunc=settings.truncation_voxels * settings.voxel_size_m, fuse_color=True)


def integrate_frame(
    volume: TsdfVolume,
    settings: TsdfSettings,
    depth_m: Float32[ndarray, "h w"],
    intrinsics_px: Float[ndarray, "4"],
    world_T_cam: Float[ndarray, "4 4"],
    rgb: UInt8[ndarray, "h w 3"],
) -> bool:
    """Integrate one frame into ``volume`` at its pose; False (nothing integrated) for a non-finite pose or intrinsics.

    Args:
        volume: The volume to update (:func:`new_volume`).
        settings: The depth range integrated (depths outside it, or non-finite, are skipped in Rust).
        depth_m: Metric Z-depth.
        intrinsics_px: ``fx, fy, cx, cy`` on the depth raster.
        world_T_cam: OpenCV camera-to-world.
        rgb: 8-bit RGB on the depth raster.
    """
    fx, fy, cx, cy = (float(value) for value in intrinsics_px)
    if not (np.isfinite(world_T_cam).all() and np.isfinite([fx, fy, cx, cy]).all() and fx > 0):
        return False
    volume.integrate(
        np.ascontiguousarray(depth_m, dtype=np.float32),
        np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32),
        np.linalg.inv(world_T_cam.astype(np.float64)).astype(np.float32),
        np.ascontiguousarray(rgb, dtype=np.uint8),
        min_depth=settings.min_depth_m,
        max_depth=settings.max_depth_m,
    )
    return True


def fuse_tsdf_mesh(
    depth: Float32[ndarray, "n h w"],
    intrinsics: Float64[ndarray, "n 4"],
    world_T_cam: Float64[ndarray, "n 4 4"],
    rgb: Callable[[int], UInt8[ndarray, "h w 3"]],
    settings: TsdfSettings = DEFAULT_TSDF,
) -> ColoredMesh | None:
    """Integrate every frame with a finite pose into one TSDF and extract its mesh; None if no surface came out.

    Args:
        depth: Metric Z-depth per frame; non-finite or out-of-range pixels are not integrated.
        intrinsics: ``fx, fy, cx, cy`` per frame, on the depth raster.
        world_T_cam: OpenCV camera-to-world per frame (non-finite rows are skipped).
        rgb: Frame index -> 8-bit RGB on the depth raster.
        settings: Voxel size, truncation and depth range.
    """
    volume: TsdfVolume = new_volume(settings)
    for index in range(depth.shape[0]):
        integrate_frame(volume, settings, depth[index], intrinsics[index], world_T_cam[index], rgb(index))
    mesh: MeshArrays = volume.extract_mesh()
    vertices_xyz: Float32[ndarray, "v 3"] = mesh[0]
    triangles: UInt32[ndarray, "t 3"] = mesh[1]
    colors_rgb: UInt8[ndarray, "v 3"] | None = mesh[2]
    normals_xyz: Float32[ndarray, "v 3"] = mesh[3]
    if triangles.shape[0] == 0:
        return None
    assert colors_rgb is not None, "new_volume fuses colour"
    return ColoredMesh(vertices_xyz=vertices_xyz, triangles=triangles, colors_rgb=colors_rgb, normals_xyz=normals_xyz)
