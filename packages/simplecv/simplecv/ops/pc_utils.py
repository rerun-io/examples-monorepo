from typing import cast

import numpy as np
import open3d as o3d
from jaxtyping import Float32
from numpy import ndarray
from tqdm import trange
from tqdm.std import tqdm as Tqdm


def estimate_voxel_size(
    points: Float32[ndarray, "N 3"],
    target_points: int = 100_000,
    tolerance: float = 0.25,
    max_iterations: int = 10,
    min_voxel_ratio: float = 0.0001,
    max_voxel_ratio: float = 0.5,
) -> float:
    """
    Use binary search to find optimal voxel size for target point count.

    Args:
        points: Input point cloud points
        target_points: Desired number of points after downsampling
        tolerance: Acceptable relative error (0.25 = within 25% of target)
        max_iterations: Maximum binary search iterations
        min_voxel_ratio: Minimum voxel size as ratio of scene diagonal
        max_voxel_ratio: Maximum voxel size as ratio of scene diagonal

    Returns:
        Voxel size that results in point count within tolerance of target_points
    """
    if len(points) == 0:
        fallback_voxel_size: float = 0.01
        return fallback_voxel_size

    # Calculate scene bounds for voxel size limits
    min_bounds: Float32[ndarray, "3"] = np.min(points, axis=0)
    max_bounds: Float32[ndarray, "3"] = np.max(points, axis=0)
    scene_diagonal: float = float(np.linalg.norm(max_bounds - min_bounds))

    # Set search bounds
    min_voxel_size: float = scene_diagonal * min_voxel_ratio
    max_voxel_size: float = scene_diagonal * max_voxel_ratio

    # Create Open3D point cloud once for reuse
    pcd_temp: o3d.geometry.PointCloud = o3d.geometry.PointCloud()
    pcd_temp.points = o3d.utility.Vector3dVector(points)

    # Binary search for optimal voxel size
    low: float = min_voxel_size
    high: float = max_voxel_size
    best_voxel_size: float = (low + high) / 2

    progress: Tqdm = cast(Tqdm, trange(max_iterations, desc="Estimating voxel size"))
    try:
        for _ in progress:
            current_voxel_size: float = (low + high) / 2

            # Test this voxel size
            pcd_test: o3d.geometry.PointCloud = pcd_temp.voxel_down_sample(current_voxel_size)
            current_points: int = len(pcd_test.points)

            # Calculate relative error
            error: float = abs(current_points - target_points) / target_points

            # update progress bar postfix
            postfix: dict[str, float | int | str] = {
                "voxel_size": float(current_voxel_size),
                "points": current_points,
                "error": float(error),
            }
            progress.set_postfix(postfix)

            # Check if we're within tolerance
            within_tolerance: bool = error <= tolerance

            if within_tolerance:
                best_voxel_size = current_voxel_size
                progress.write(f"  - ✓ Found optimal voxel size: {best_voxel_size:.6f}")
                break

            # Update search bounds
            if current_points > target_points:
                # Too many points, need larger voxel size
                low = current_voxel_size
            else:
                # Too few points, need smaller voxel size
                high = current_voxel_size

            best_voxel_size = current_voxel_size
    finally:
        progress.close()

    return float(best_voxel_size)
