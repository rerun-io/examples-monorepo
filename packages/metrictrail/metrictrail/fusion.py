"""Surface normals for upstream's fused metric cloud, accumulated beside ViPE's voxel fusion.

``vipe.metrictrail.pointcloud.fuse_point_cloud`` fuses positions and colours; its ``on_points`` hook (recipes/vipe-src
patch 0004) hands each frame's kept pixels and world points to :class:`NormalVoxels`, which sums the pixels' normals
(rotated into the world by the camera pose) per voxel with the same quantisation, the same compaction schedule and the
same merge order as ViPE's ``VoxelAccumulator``. Its voxels are therefore ViPE's voxels in ViPE's order;
:meth:`NormalVoxels.finish` checks that against the fused positions before it returns the normals.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from jaxtyping import Bool, Float32, Float64, Int64
from numpy import ndarray
from vipe.metrictrail.pointcloud import FusionSettings, fuse_point_cloud

from metrictrail.artifacts import ColoredCloud

NormalSource = Callable[[int], Float32[ndarray, "h w 3"]]
"""Frame index -> camera-space unit normals on the depth raster."""

VIPE_COMPACT_EVERY: int = 16
"""``VoxelAccumulator.compact_every`` as ``fuse_point_cloud`` builds it (its default)."""


@dataclass
class NormalVoxels:
    """Per-voxel sums of positions and normals, merged exactly as ViPE's ``VoxelAccumulator`` merges its voxels."""

    voxel_size_m: float
    """ViPE's ``FusionSettings.voxel_size_m``."""
    compact_every: int = VIPE_COMPACT_EVERY
    """Pending frames merged at once (ViPE's schedule: the sums, and so the float64 rounding, follow it)."""
    keys: Int64[ndarray, "v 3"] = field(default_factory=lambda: np.empty((0, 3), dtype=np.int64))
    """Voxel indices, lexicographically sorted after each merge."""
    point_sums: Float64[ndarray, "v 3"] = field(default_factory=lambda: np.empty((0, 3), dtype=np.float64))
    """Sum of the positions per voxel (only to check the voxels against ViPE's)."""
    normal_sums: Float64[ndarray, "v 3"] = field(default_factory=lambda: np.empty((0, 3), dtype=np.float64))
    """Sum of the world normals per voxel."""
    counts: Int64[ndarray, "v"] = field(default_factory=lambda: np.empty((0,), dtype=np.int64))
    """Points per voxel."""
    pending: list[tuple[Float32[ndarray, "n 3"], Float32[ndarray, "n 3"]]] = field(default_factory=list)
    """Frames added since the last merge."""

    def add(self, points: Float32[ndarray, "n 3"], normals: Float32[ndarray, "n 3"]) -> None:
        """One frame's world points (as ViPE's accumulator receives them) and their world normals."""
        finite: Bool[ndarray, "n"] = np.isfinite(points).all(axis=1)
        if finite.any():
            self.pending.append((points[finite], normals[finite]))
        if len(self.pending) >= self.compact_every:
            self.compact()

    def compact(self) -> None:
        """Merge the pending frames into the voxel sums."""
        if not self.pending:
            return
        points: Float32[ndarray, "n 3"] = np.concatenate([item[0] for item in self.pending], axis=0)
        normals: Float32[ndarray, "n 3"] = np.concatenate([item[1] for item in self.pending], axis=0)
        new_keys: Int64[ndarray, "n 3"] = np.floor(points.astype(np.float64) / self.voxel_size_m).astype(np.int64)
        keys: Int64[ndarray, "m 3"] = np.concatenate((self.keys, new_keys), axis=0)
        point_sums: Float64[ndarray, "m 3"] = np.concatenate((self.point_sums, points.astype(np.float64)), axis=0)
        normal_sums: Float64[ndarray, "m 3"] = np.concatenate((self.normal_sums, normals.astype(np.float64)), axis=0)
        counts: Int64[ndarray, "m"] = np.concatenate((self.counts, np.ones(len(points), dtype=np.int64)), axis=0)
        order: Int64[ndarray, "m"] = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
        sorted_keys: Int64[ndarray, "m 3"] = keys[order]
        group_start: Bool[ndarray, "m"] = np.empty(len(order), dtype=bool)
        group_start[0] = True
        group_start[1:] = np.any(sorted_keys[1:] != sorted_keys[:-1], axis=1)
        starts: Int64[ndarray, "v"] = np.flatnonzero(group_start)
        self.keys = sorted_keys[starts]
        self.point_sums = np.add.reduceat(point_sums[order], starts, axis=0)
        self.normal_sums = np.add.reduceat(normal_sums[order], starts, axis=0)
        self.counts = np.add.reduceat(counts[order], starts)
        self.pending.clear()

    def finish(self, fused_points: Float32[ndarray, "v 3"]) -> Float32[ndarray, "v 3"]:
        """Unit voxel normals (the normalised sums), aligned with ViPE's fused positions ``fused_points``."""
        self.compact()
        if len(self.counts) == 0:
            raise RuntimeError("no normals were accumulated")
        points: Float32[ndarray, "v 3"] = (self.point_sums / self.counts[:, None]).astype(np.float32)
        if not np.array_equal(points, fused_points):
            raise RuntimeError(f"normal voxels ({len(points)}) do not match ViPE's fused voxels ({len(fused_points)})")
        length: Float64[ndarray, "v 1"] = np.linalg.norm(self.normal_sums, axis=1, keepdims=True)
        return (self.normal_sums / np.maximum(length, 1e-12)).astype(np.float32)


def fuse_world_cloud(
    depth: Float32[ndarray, "f h w"],
    intrinsics: Float64[ndarray, "f 4"] | Float32[ndarray, "f 4"],
    c2w: Float64[ndarray, "f 4 4"] | Float32[ndarray, "f 4 4"],
    rgb: Callable[[int], ndarray],
    settings: FusionSettings,
    normals: NormalSource | None = None,
) -> ColoredCloud:
    """ViPE's fused cloud in the frame of ``c2w`` (``world_frame=True``), each point with its pixels' mean normal.

    Args:
        depth: Per-frame metric z-depth, metres.
        intrinsics: Per-frame ``fx, fy, cx, cy``.
        c2w: Per-frame OpenCV camera-to-world poses.
        rgb: Frame index -> 8-bit RGB on the depth raster.
        settings: ViPE's ``FusionSettings``.
        normals: Frame index -> camera-space unit normals on the depth raster; None fuses no normals.

    Returns:
        Positions and colours from ViPE's ``fuse_point_cloud``, normals from :class:`NormalVoxels` (or None).
    """
    if normals is None:
        plain = fuse_point_cloud(depth, intrinsics, c2w, rgb, settings, world_frame=True)
        return ColoredCloud(points_xyz=plain.points.astype(np.float32), colors_rgb=plain.colors.astype(np.uint8))
    voxels: NormalVoxels = NormalVoxels(float(settings.voxel_size_m))
    source: NormalSource = normals

    def on_points(index: int, rows: Int64[ndarray, "n"], cols: Int64[ndarray, "n"], world: Float32[ndarray, "n 3"]) -> None:
        frame_normals: Float32[ndarray, "h w 3"] = np.asarray(source(index), dtype=np.float32)
        if frame_normals.shape != (*depth[index].shape, 3):
            raise ValueError(f"frame {index}: normals {frame_normals.shape} are not on the depth raster {depth[index].shape}")
        world_normals: Float32[ndarray, "n 3"] = (frame_normals[rows, cols] @ c2w[index][:3, :3].T).astype(np.float32)
        voxels.add(world, world_normals)

    cloud = fuse_point_cloud(depth, intrinsics, c2w, rgb, settings, world_frame=True, on_points=on_points)
    points: Float32[ndarray, "v 3"] = cloud.points.astype(np.float32)
    return ColoredCloud(points_xyz=points, colors_rgb=cloud.colors.astype(np.uint8), normals_xyz=voxels.finish(points))
