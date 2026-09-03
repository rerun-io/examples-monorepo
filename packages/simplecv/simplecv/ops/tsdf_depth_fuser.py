"""Utilities for TSDF fusion of multi-view depth maps.

This module provides a scale-invariant Open3D-based fuser that estimates
reasonable TSDF volume parameters from the scene geometry instead of requiring
metric depths. The implementation mirrors the API of the simplecv fuser but
automatically adapts the voxel size and truncation distance to the observed
point cloud extent.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d
import rerun as rr
from jaxtyping import Float, Float32, UInt8, UInt16
from numpy import ndarray

from simplecv.camera_parameters import PinholeParameters


def log_fused_mesh(
    entity_path: str,
    mesh: o3d.geometry.TriangleMesh,
    *,
    recording: rr.RecordingStream | None = None,
    static: bool = True,
    face_rendering: rr.components.MeshFaceRendering = rr.components.MeshFaceRendering.Front,
) -> None:
    """Log a fused Open3D triangle mesh to Rerun.

    Args:
        entity_path: Rerun entity path for the mesh.
        mesh: Open3D triangle mesh to log.
        recording: Explicit recording stream, or ``None`` for the global stream.
        static: Whether to log timeless data instead of the current time.
        face_rendering: Which wound faces the viewer renders. Front-face-only
            is the room-reconstruction default; pass ``DoubleSided`` to retain
            Rerun's general mesh default.
    """
    rr.log(
        entity_path,
        rr.Mesh3D(
            vertex_positions=np.asarray(mesh.vertices),
            triangle_indices=np.asarray(mesh.triangles),
            vertex_normals=np.asarray(mesh.vertex_normals),
            vertex_colors=np.asarray(mesh.vertex_colors),
            face_rendering=face_rendering,
        ),
        static=static,
        recording=recording,
    )


@dataclass
class DepthFuser:
    """Base configuration shared by TSDF depth fusers."""

    fusion_resolution: float | None = None
    """Optional fixed voxel resolution for the TSDF volume."""
    max_fusion_depth: float | None = None
    """Optional hard depth truncation threshold applied during fusion."""


class Open3DFuser(DepthFuser):
    """Open3D-based TSDF integrator for calibrated depth maps.

    The implementation mirrors the historical simplecv fuser API while
    clarifying the expected metric conventions. Depth frames must be stored in
    millimetres (``depth_scale=1000``) and camera poses must follow the
    ``cam_T_world`` transformation convention (world → camera). Incorrect units
    will silently skew the reconstructed geometry, so double-check sensor
    outputs before fusion.

    Args:
        fusion_resolution: Linear voxel size in metres used for the internal
            :class:`~open3d.pipelines.integration.ScalableTSDFVolume`.
        max_fusion_depth: Maximum metric depth (metres) retained when creating
            Open3D RGB-D frames. Samples beyond the threshold are discarded
            prior to integration.

    Attributes:
        fusion_max_depth: Copy of ``max_fusion_depth`` used during RGB-D
            construction.
        volume: Underlying Open3D TSDF volume that accumulates fused frames.

    Example:
        ```python
        fuser = Open3DFuser(fusion_resolution=0.04, max_fusion_depth=3.0)
        fuser.fuse_frames(depth_mm, intrinsics, cam_T_world, rgb)
        mesh = fuser.get_mesh()
        mesh.compute_vertex_normals()
        ```
    """

    def __init__(
        self,
        fusion_resolution: float = 0.04,
        max_fusion_depth: float = 3.0,
    ):
        super().__init__(
            fusion_resolution,
            max_fusion_depth,
        )

        self.fusion_max_depth = max_fusion_depth

        voxel_size: float = fusion_resolution * 100
        self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=float(voxel_size) / 100,
            sdf_trunc=3 * float(voxel_size) / 100,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )

    def fuse_frames(
        self,
        depth_hw: UInt16[np.ndarray, "h w"],
        K_33: Float[np.ndarray, "3 3"],
        cam_T_world_44: Float[np.ndarray, "4 4"],
        rgb_hw3: UInt8[np.ndarray, "h w 3"],
    ) -> None:
        """Fuse a single RGB-D frame into the TSDF volume.

        Args:
            depth_hw: Depth map encoded in millimetres with shape ``(H, W)``.
                The data are converted to metres internally using a
                ``depth_scale`` of ``1000.0``.
            K_33: Camera intrinsics matrix expressed in pixels.
            cam_T_world_44: Homogeneous transform that maps world coordinates
                into the camera frame (world → camera).
            rgb_hw3: Aligned RGB image in ``uint8`` with shape ``(H, W, 3)``.
        """
        height: int = depth_hw.shape[0]
        width: int = depth_hw.shape[1]

        rgbd: o3d.geometry.RGBDImage = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb_hw3),
            o3d.geometry.Image(depth_hw),
            depth_scale=1000.0,
            depth_trunc=self.fusion_max_depth,
            convert_rgb_to_intensity=False,
        )

        self.volume.integrate(
            rgbd,
            o3d.camera.PinholeCameraIntrinsic(
                width=width,
                height=height,
                fx=K_33[0, 0],
                fy=K_33[1, 1],
                cx=K_33[0, 2],
                cy=K_33[1, 2],
            ),
            cam_T_world_44,
        )

    def export_mesh(self, path: str | Path) -> None:
        """Persist the reconstructed mesh to ``path``."""

        o3d.io.write_triangle_mesh(str(path), self.volume.extract_triangle_mesh())

    def get_mesh(self, export_single_mesh=None, convert_to_trimesh=False) -> o3d.geometry.TriangleMesh:
        # Kept for compatibility with the broader DepthFuser call surface.
        _ = export_single_mesh, convert_to_trimesh
        mesh = self.volume.extract_triangle_mesh()

        return mesh


class Open3DScaleInvariantFuser(DepthFuser):
    """Open3D TSDF fuser that infers scale from the reconstructed scene.

    Args:
        grid_resolution: Number of voxels placed along the longest scene axis
            when estimating the TSDF volume. Increasing the value produces
            smaller voxels (finer fusion) at the cost of more memory and slower
            updates; decreasing it yields coarser fusion but runs faster.

        truncation_ratio: Multiplier converting the voxel length into the TSDF
            truncation distance. Lower ratios tighten the signed-distance band
            and sharpen surfaces, while higher ratios smooth the result and are
            more forgiving to noisy depths.

        depth_percentile: Percentile in ``[0, 1]`` applied to each depth map to
            choose a per-frame depth truncation. Raising it includes more
            distant samples in the fusion volume; lowering it suppresses far
            depths and aggressively discards outliers.

        minimum_extent: Smallest allowed scene diagonal used when the reference
            point cloud is empty or extremely compact. Reducing it enables very
            small volumes for tight scenes; increasing it enforces a larger
            baseline voxel size for numerical stability.

    Depth maps provided to :meth:`fuse_frame` must be metric (metres) and the
    :class:`~simplecv.camera_parameters.PinholeParameters` instance must expose a
    ``cam_T_world`` transform (world → camera). Calling
    :meth:`initialise_from_points` with a representative point cloud before
    fusion ensures the automatically estimated voxel length matches the scene
    scale; skipping it leaves the fuser uninitialised.

    The fuser initialises a :class:`~open3d.pipelines.integration.ScalableTSDFVolume`
    whose voxel size and truncation distance scale with the diagonal of a
    reference point cloud, enabling reliable fusion even when the predicted
    depths are only known up to scale.

    Example:
        ```python
        from monopriors.tsdf_fuser import Open3DScaleInvariantFuser

        fuser = Open3DScaleInvariantFuser(grid_resolution=512)
        fuser.initialise_from_points(point_cloud_xyz)
        for depth, pinhole, rgb in zip(depth_list, pinhole_params, rgb_list, strict=True):
            fuser.fuse_frame(depth_hw=depth, pinhole=pinhole, rgb_hw3=rgb)
        mesh = fuser.get_mesh()
        mesh.compute_vertex_normals()
        ```
    """

    def __init__(
        self,
        *,
        grid_resolution: int = 384,
        truncation_ratio: float = 4.0,
        depth_percentile: float = 0.98,
        minimum_extent: float = 1e-2,
    ) -> None:
        super().__init__()
        self.grid_resolution: int = grid_resolution
        self.truncation_ratio: float = truncation_ratio
        self.depth_percentile: float = depth_percentile
        self.minimum_extent: float = minimum_extent
        self.volume: o3d.pipelines.integration.ScalableTSDFVolume | None = None
        self._voxel_length: float | None = None

    def reset(self) -> None:
        """Reset the internal TSDF volume."""

        self.volume = None
        self._voxel_length = None

    def initialise_from_points(self, points: Float32[ndarray, "num_points 3"]) -> None:
        """Initialise the TSDF volume based on a reference point cloud.

        Args:
            points: Reference 3D points in world coordinates with shape
                ``(num_points, 3)`` and dtype ``float32``. The diagonal of the
                point cloud bounding box drives the voxel length estimation.
        """

        points_np: Float32[ndarray, "num_points 3"] = np.asarray(points, dtype=np.float32)
        if points_np.size == 0:
            scene_extent: float = self.minimum_extent
        else:
            min_bounds: Float32[ndarray, "3"] = np.min(points_np, axis=0)
            max_bounds: Float32[ndarray, "3"] = np.max(points_np, axis=0)
            scene_extent = float(np.linalg.norm(max_bounds - min_bounds))
            if scene_extent < self.minimum_extent:
                scene_extent = self.minimum_extent

        voxel_length: float = scene_extent / float(self.grid_resolution)
        self._voxel_length = voxel_length
        sdf_trunc: float = self.truncation_ratio * voxel_length

        self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=float(voxel_length),
            sdf_trunc=float(sdf_trunc),
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )

    def _infer_depth_truncation(self, depth_hw: Float32[ndarray, "H W"]) -> float:
        """Derive a robust depth truncation threshold for the input frame."""

        positive_depths: Float32[ndarray, "num_valid"] = depth_hw[depth_hw > 0]
        if positive_depths.size == 0:
            if self._voxel_length is None:
                return 1.0
            fallback_truncation: float = 6.0 * self._voxel_length
            return fallback_truncation

        percentile: float = np.clip(self.depth_percentile * 100.0, 0.0, 100.0)
        truncation: float = float(np.percentile(positive_depths, percentile))
        if self.max_fusion_depth is not None:
            truncation = min(truncation, float(self.max_fusion_depth))
        return truncation

    def fuse_frame(
        self,
        *,
        depth_hw: Float32[ndarray, "H W"],
        pinhole: PinholeParameters,
        rgb_hw3: UInt8[ndarray, "H W 3"] | None = None,
    ) -> None:
        """Integrate a depth (and optional RGB) frame into the TSDF volume.

        Args:
            depth_hw: Depth image in meters with shape ``(H, W)`` and dtype
                ``float32``.
            pinhole: Calibrated pinhole parameters that provide camera
                intrinsics and extrinsics.
            rgb_hw3: Optional RGB image aligned with ``depth_hw`` with shape
                ``(H, W, 3)`` and dtype ``uint8``. When omitted, a zero image is
                used so the resulting mesh still contains geometry.
        """

        if self.volume is None:
            raise RuntimeError("The TSDF volume must be initialised before fusing frames.")
        volume: o3d.pipelines.integration.ScalableTSDFVolume = self.volume

        depth_np: Float32[ndarray, "H W"] = np.asarray(depth_hw, dtype=np.float32)
        height: int = int(depth_np.shape[0])
        width: int = int(depth_np.shape[1])

        if rgb_hw3 is None:
            rgb_np: UInt8[ndarray, "H W 3"] = np.zeros((height, width, 3), dtype=np.uint8)
        else:
            rgb_np = np.asarray(rgb_hw3, dtype=np.uint8)

        depth_trunc: float = self._infer_depth_truncation(depth_np)

        rgbd: o3d.geometry.RGBDImage = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb_np),
            o3d.geometry.Image(depth_np),
            depth_scale=1.0,
            depth_trunc=float(depth_trunc),
            convert_rgb_to_intensity=False,
        )

        k_mat: Float32[ndarray, "3 3"] = np.asarray(pinhole.intrinsics.k_matrix, dtype=np.float32)
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=width,
            height=height,
            fx=float(k_mat[0, 0]),
            fy=float(k_mat[1, 1]),
            cx=float(k_mat[0, 2]),
            cy=float(k_mat[1, 2]),
        )

        cam_T_world: Float32[ndarray, "4 4"] = np.asarray(pinhole.extrinsics.cam_T_world, dtype=np.float32)
        extrinsic: ndarray = cam_T_world.astype(np.float64)

        volume.integrate(rgbd, intrinsic, extrinsic)

    def export_mesh(self, path: str | Path) -> None:
        """Write the fused mesh to disk.

        Args:
            path: Target path where the triangle mesh is saved.
        """

        if self.volume is None:
            raise RuntimeError("Cannot export mesh before fusion is performed.")
        volume: o3d.pipelines.integration.ScalableTSDFVolume = self.volume
        mesh: o3d.geometry.TriangleMesh = volume.extract_triangle_mesh()
        o3d.io.write_triangle_mesh(str(path), mesh)

    def get_mesh(self) -> o3d.geometry.TriangleMesh:
        """Extract a triangle mesh from the TSDF volume.

        Returns:
            The :class:`~open3d.geometry.TriangleMesh` reconstructed from the
            fused TSDF volume.
        """

        if self.volume is None:
            raise RuntimeError("Cannot extract mesh before fusion is performed.")
        volume: o3d.pipelines.integration.ScalableTSDFVolume = self.volume
        mesh: o3d.geometry.TriangleMesh = volume.extract_triangle_mesh()
        return mesh
