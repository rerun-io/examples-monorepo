"""Fusion node.

Fuses per-view depth maps into a 3D point cloud and optionally a TSDF mesh.
This is a pure CPU utility with no GPU models — it takes generic depth maps,
pinhole parameters, and RGB images with no coupling to upstream node types.
"""

from dataclasses import dataclass

import numpy as np
import open3d as o3d
from jaxtyping import Float32, UInt8
from numpy import ndarray
from simplecv.camera_parameters import PinholeParameters
from simplecv.ops.pc_utils import estimate_voxel_size
from simplecv.ops.tsdf_depth_fuser import Open3DScaleInvariantFuser


@dataclass
class FusionConfig:
    """Configuration for point cloud and mesh fusion."""

    grid_resolution: int = 512
    """TSDF grid resolution."""
    target_points: int = 150_000
    """Target point count after voxel downsampling."""


@dataclass
class FusionResult:
    """Output of depth fusion: point cloud and mesh."""

    pcd: o3d.geometry.PointCloud
    """Fused and downsampled point cloud."""
    mesh: o3d.geometry.TriangleMesh | None
    """TSDF-fused mesh, or None if fusion fails."""


def run_fusion(
    *,
    depth_list: list[Float32[ndarray, "H W"]],
    pinhole_param_list: list[PinholeParameters],
    rgb_list: list[UInt8[ndarray, "H W 3"]],
    config: FusionConfig | None = None,
) -> FusionResult:
    """Fuse per-view depth maps into a point cloud and TSDF mesh.

    Args:
        depth_list: Per-view depth maps (metric-scale).
        pinhole_param_list: Per-view camera parameters.
        rgb_list: Per-view RGB images for coloring.
        config: Fusion configuration.

    Returns:
        FusionResult with downsampled point cloud and optional mesh.
    """
    from einops import rearrange

    if config is None:
        config = FusionConfig()

    # Build point cloud from depths + cameras
    depth_maps: Float32[ndarray, "b h w 1"] = np.stack(
        [rearrange(d, "h w -> h w 1") for d in depth_list], axis=0
    ).astype(np.float32)

    world_T_cam_b44: Float32[ndarray, "b 4 4"] = np.stack(
        [p.extrinsics.world_T_cam for p in pinhole_param_list], axis=0
    ).astype(np.float32)

    K_b33: Float32[ndarray, "b 3 3"] = np.stack(
        [p.intrinsics.k_matrix for p in pinhole_param_list], axis=0
    ).astype(np.float32)

    from monopriors.depth_utils import multidepth_to_points

    world_points: Float32[ndarray, "b h w 3"] = multidepth_to_points(
        depth_maps=depth_maps, world_T_cam_batch=world_T_cam_b44, K_b33=K_b33
    )

    pointcloud: Float32[ndarray, "num_points 3"] = world_points.reshape(-1, 3)
    rgb_stack: UInt8[ndarray, "num_points 3"] = np.concatenate(
        [rearrange(rgb, "h w c -> (h w) c") for rgb in rgb_list]
    )

    # Create and downsample point cloud
    pcd: o3d.geometry.PointCloud = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud)
    pcd.colors = o3d.utility.Vector3dVector(rgb_stack / 255.0)

    voxel_size: float = estimate_voxel_size(pointcloud.astype(np.float32), target_points=config.target_points)
    pcd_ds: o3d.geometry.PointCloud = pcd.voxel_down_sample(voxel_size)

    # TSDF mesh fusion
    mesh: o3d.geometry.TriangleMesh | None = None
    if depth_list and pinhole_param_list:
        depth_fuser: Open3DScaleInvariantFuser = Open3DScaleInvariantFuser(grid_resolution=config.grid_resolution)
        reference_points: Float32[ndarray, "num_points 3"] = np.asarray(pcd.points, dtype=np.float32)
        depth_fuser.initialise_from_points(reference_points)

        for depth_map, pinhole_param, rgb in zip(depth_list, pinhole_param_list, rgb_list, strict=True):
            depth_fuser.fuse_frame(depth_hw=depth_map, pinhole=pinhole_param, rgb_hw3=rgb)

        mesh = depth_fuser.get_mesh()
        mesh.compute_vertex_normals()

    return FusionResult(pcd=pcd_ds, mesh=mesh)
