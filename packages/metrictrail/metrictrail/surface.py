"""MetricTrail's default surface: a colour TSDF mesh fused from every tracked frame (``packages/tsdf-rs``).

Each tracked frame's metric depth (ViPE's depth artifact, MoGe-2 metric depth) is integrated into a sparse TSDF with
the frame's final camera pose, and marching cubes extracts one coloured mesh. It replaces upstream's fused point cloud
(``vipe.metrictrail.pointcloud.fuse_point_cloud``: a multi-view consistency filter and 2 cm voxel means in NumPy) as
the default, at a fraction of its time; ``--fuse-cloud`` still produces that cloud for comparison.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from jaxtyping import Float32, Float64, UInt8, UInt32
from numpy import ndarray


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
    """Area-weighted unit vertex normals."""


def vertex_normals(vertices_xyz: Float32[ndarray, "v 3"], triangles: UInt32[ndarray, "t 3"]) -> Float32[ndarray, "v 3"]:
    """Area-weighted unit normals per vertex (a vertex in no triangle gets the zero vector)."""
    corners: Float64[ndarray, "t 3 3"] = vertices_xyz.astype(np.float64)[triangles]
    face: Float64[ndarray, "t 3"] = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    count: int = vertices_xyz.shape[0]
    corner_ids: ndarray = triangles.reshape(-1).astype(np.int64)
    summed: Float64[ndarray, "v 3"] = np.stack(
        [np.bincount(corner_ids, weights=np.repeat(face[:, axis], 3), minlength=count) for axis in range(3)], axis=1
    )
    length: Float64[ndarray, "v 1"] = np.linalg.norm(summed, axis=1, keepdims=True)
    return (summed / np.maximum(length, 1e-12)).astype(np.float32)


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
    import tsdf_rs

    volume: Any = tsdf_rs.TsdfVolume(  # pyrefly: ignore[missing-attribute]  # compiled pyo3 module, no stubs
        voxel_size=settings.voxel_size_m,
        sdf_trunc=settings.truncation_voxels * settings.voxel_size_m,
        fuse_color=True,
    )
    for index in range(depth.shape[0]):
        pose: Float64[ndarray, "4 4"] = world_T_cam[index]
        fx, fy, cx, cy = (float(value) for value in intrinsics[index])
        if not (np.isfinite(pose).all() and np.isfinite([fx, fy, cx, cy]).all() and fx > 0):
            continue
        frame: Float32[ndarray, "h w"] = np.asarray(depth[index], dtype=np.float32)
        in_range: ndarray = np.isfinite(frame) & (frame >= settings.min_depth_m) & (frame <= settings.max_depth_m)
        volume.integrate(
            np.where(in_range, frame, 0.0).astype(np.float32),
            np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32),
            np.linalg.inv(pose).astype(np.float32),
            np.ascontiguousarray(rgb(index), dtype=np.uint8),
        )
    vertices, triangles, colors = volume.extract_mesh()
    if triangles.shape[0] == 0:
        return None
    vertices_xyz: Float32[ndarray, "v 3"] = np.asarray(vertices, dtype=np.float32)
    faces: UInt32[ndarray, "t 3"] = np.asarray(triangles, dtype=np.uint32)
    return ColoredMesh(
        vertices_xyz=vertices_xyz, triangles=faces, colors_rgb=np.asarray(colors, dtype=np.uint8), normals_xyz=vertex_normals(vertices_xyz, faces)
    )
