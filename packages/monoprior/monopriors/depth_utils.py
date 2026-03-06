import numpy as np
from einops import rearrange
from jaxtyping import Bool, Float32, Int64
from numpy import ndarray


def estimate_intrinsics(H: int, W: int, fov: float = 55.0) -> Float32[ndarray, "3 3"]:
    """
    Intrinsics for a pinhole camera model from image dimensions.
    Assume fov of 55 degrees and central principal point.
    """
    f = 0.5 * W / np.tan(0.5 * fov * np.pi / 180.0)
    cx: float = 0.5 * W
    cy: float = 0.5 * H
    K_33: Float32[ndarray, "3 3"] = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float32)
    return K_33


def disparity_to_depth(
    disparity: Float32[ndarray, "h w"],
    focal_length: int,
    baseline: float = 1.0,
    gamma: float = 2.2,
) -> Float32[ndarray, "h w"]:
    range: float = float(np.minimum(disparity.max() / (disparity.min() + 1e-6), 100.0))
    disparity_max: float = float(disparity.max())
    min_disparity_range: float = disparity_max / range

    depth: Float32[ndarray, "h w"] = (focal_length * baseline) / np.maximum(disparity, min_disparity_range)
    # gamma correction for better visualizationg
    depth: Float32[ndarray, "h w"] = np.power(depth, 1.0 / gamma)
    return depth


def depth_to_disparity(
    depth: Float32[ndarray, "h w"], focal_length: int, baseline: float = 1.0
) -> Float32[ndarray, "h w"]:
    disparity = (focal_length * baseline) / (depth + 0.01)
    return disparity


def depth_edges_mask(depth: Float32[ndarray, "h w"], threshold: float = 0.1) -> Bool[ndarray, "h w"]:
    """Returns a mask of edges in the depth map.
    Args:
    depth: 2D numpy array of shape (H, W) with dtype float32.
    Returns:
    mask: 2D numpy array of shape (H, W) with dtype bool.
    """
    # Compute the x and y gradients of the depth map.
    depth_dx, depth_dy = np.gradient(depth)
    # Compute the gradient magnitude.
    depth_grad = np.sqrt(depth_dx**2 + depth_dy**2)
    # Compute the edge mask.
    mask: Bool[ndarray, "h w"] = depth_grad > threshold
    return mask


def depth_to_points(
    depth_1hw: Float32[ndarray, "1 h w"],
    K_33: Float32[ndarray, "3 3"],
    world_R_cam: Float32[ndarray, "3 3"] | None = None,
    world_t_cam: Float32[ndarray, "3"] | None = None,
) -> Float32[ndarray, "h w 3"]:
    K_33_inv: Float32[ndarray, "3 3"] = np.linalg.inv(K_33)
    if world_R_cam is None:
        world_R_cam: Float32[ndarray, "3 3"] = np.eye(3, dtype=np.float32)
    if world_t_cam is None:
        world_t_cam: Float32[ndarray, "3"] = np.zeros(3, dtype=np.float32)

    _, height, width = depth_1hw.shape

    # create 3d points grid
    x: Int64[ndarray, "w"] = np.arange(width)
    y: Int64[ndarray, "h"] = np.arange(height)
    coord: Int64[ndarray, "h w 2"] = np.stack(np.meshgrid(x, y), -1)
    z: Int64[ndarray, "h w 1"] = np.ones_like(coord)[:, :, [0]]
    coord = np.concatenate((coord, z), -1).astype(np.float32)  # z=1
    coord: Float32[ndarray, "1 h w 3"] = rearrange(coord, "h w c -> 1 h w c")

    # from depth to 3D points
    depth_1hw11: Float32[ndarray, "1 h w 1 1"] = rearrange(depth_1hw, "1 h w -> 1 h w 1 1")

    # back project points from pixels to camera coordinate system
    pts3D_1 = depth_1hw11 * rearrange(K_33_inv, "h w -> 1 1 1 h w") @ rearrange(coord, "1 h w c -> 1 h w c 1")

    # transform from camera to world coordinate system
    pts3D_2: Float32[ndarray, "1 h w 3 1"] = rearrange(world_R_cam, "h w -> 1 1 1 h w") @ pts3D_1 + rearrange(
        world_t_cam, "s -> 1 1 1 s 1"
    )

    # rearrange to 3D points
    pointcloud: Float32[ndarray, "h w 3"] = rearrange(pts3D_2, "1 h w c 1 -> h w c")
    return pointcloud


def multidepth_to_points(
    depth_maps: Float32[ndarray, "b h w 1"],
    world_T_cam_batch: Float32[ndarray, "b 4 4"],
    K_b33: Float32[ndarray, "b 3 3"],
) -> Float32[ndarray, "b h w 3"]:
    world_points_list: list[ndarray] = []
    depth_map: Float32[ndarray, "h w 1"]
    for idx, depth_map in enumerate(depth_maps):
        world_T_cam: Float32[ndarray, "4 4"] = world_T_cam_batch[idx]
        partial_points: Float32[ndarray, "h w 3"] = depth_to_points(
            depth_1hw=rearrange(depth_map, "h w 1 -> 1 h w"),
            K_33=K_b33[idx],
            world_R_cam=world_T_cam[:3, :3],
            world_t_cam=world_T_cam[:3, 3],
        )
        world_points_list.append(partial_points)

    world_points: Float32[ndarray, "b h w 3"] = np.stack(world_points_list, axis=0)
    return world_points


def clip_disparity(disparity):
    # Get the 95th percentile of the disparity values
    p95 = np.percentile(disparity, 99)
    # Clip the disparity values at 105% of the 95th percentile
    clipped_disparity = np.clip(disparity, a_min=None, a_max=1.01 * p95)
    # normalize the disparity to 0-1
    clipped_disparity = (clipped_disparity - clipped_disparity.min()) / (
        clipped_disparity.max() - clipped_disparity.min()
    )
    clipped_disparity = (clipped_disparity * 255).astype(np.uint8)
    return clipped_disparity
