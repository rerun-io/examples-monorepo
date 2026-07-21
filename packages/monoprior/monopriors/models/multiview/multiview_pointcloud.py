"""Point-cloud materialization for backend-neutral multi-view predictions."""

import numpy as np
from einops import rearrange
from jaxtyping import Bool, Float, Float32, Int, UInt8
from numpy import ndarray

from monopriors.depth_utils import multidepth_to_points
from monopriors.models.multiview.multiview_model import MultiviewPred


def mv_pred_to_pointcloud(
    mv_pred_list: list[MultiviewPred],
    depth_list: list[Float32[ndarray, "H W"]] | None = None,
) -> Float32[ndarray, "num_points 3"]:
    """Unproject every predicted pixel into world coordinates."""
    depths: list[Float32[ndarray, "H W"]] = (
        depth_list if depth_list is not None else [prediction.depth_map for prediction in mv_pred_list]
    )
    if len(depths) != len(mv_pred_list):
        raise ValueError("Predictions and depth maps must have the same length.")

    pointclouds: list[Float32[ndarray, "points 3"]] = []
    for prediction, depth in zip(mv_pred_list, depths, strict=True):
        K_33: Float[ndarray, "3 3"] | None = prediction.pinhole_param.intrinsics.k_matrix
        if K_33 is None:
            raise ValueError("Multi-view prediction must include camera intrinsics.")
        depth_1hw1: Float32[ndarray, "1 H W 1"] = rearrange(depth, "h w -> 1 h w 1").astype(np.float32)
        world_T_cam_144: Float32[ndarray, "1 4 4"] = rearrange(
            prediction.pinhole_param.extrinsics.world_T_cam.astype(np.float32), "h w -> 1 h w"
        )
        K_133: Float32[ndarray, "1 3 3"] = rearrange(K_33.astype(np.float32), "h w -> 1 h w")
        points: Float32[ndarray, "points 3"] = multidepth_to_points(
            depth_maps=depth_1hw1,
            world_T_cam_batch=world_T_cam_144,
            K_b33=K_133,
        ).reshape(-1, 3)
        pointclouds.append(points)

    if not pointclouds:
        return np.empty((0, 3), dtype=np.float32)
    return np.concatenate(pointclouds)


def unproject_selected_pixels(
    depth_map: Float[ndarray, "H W"],
    K_33: Float[ndarray, "3 3"],
    world_T_cam: Float[ndarray, "4 4"],
    pixel_y: Int[ndarray, "points"],
    pixel_x: Int[ndarray, "points"],
) -> Float[ndarray, "points 3"]:
    """Unproject selected depth pixels through a camera-to-world transform."""
    if pixel_y.shape != pixel_x.shape:
        raise ValueError("Pixel x/y coordinates must have the same shape.")
    pixel_coordinates: Float[ndarray, "points 3"] = np.stack(
        [pixel_x, pixel_y, np.ones_like(pixel_x)], axis=1
    ).astype(K_33.dtype)
    camera_rays: Float[ndarray, "points 3"] = pixel_coordinates @ np.linalg.inv(K_33).T
    camera_points: Float[ndarray, "points 3"] = camera_rays * depth_map[pixel_y, pixel_x, None]
    return camera_points @ world_T_cam[:3, :3].T + world_T_cam[:3, 3]


def mv_pred_to_filtered_pointcloud(
    mv_pred_list: list[MultiviewPred],
    confidence_masks: list[UInt8[ndarray, "H W"]],
    *,
    depth_list: list[Float32[ndarray, "H W"]] | None = None,
    target_points: int = 150_000,
) -> tuple[Float32[ndarray, "sampled_points 3"], UInt8[ndarray, "sampled_points 3"]]:
    """Unproject a spatially uniform, exact subset of confident source pixels."""
    if target_points <= 0:
        raise ValueError("target_points must be positive.")
    if len(mv_pred_list) != len(confidence_masks):
        raise ValueError("Predictions and confidence masks must have the same length.")
    depths: list[Float32[ndarray, "H W"]] = (
        depth_list if depth_list is not None else [prediction.depth_map for prediction in mv_pred_list]
    )
    if len(mv_pred_list) != len(depths):
        raise ValueError("Predictions and depth maps must have the same length.")

    valid_pixel_counts: list[int] = [int(np.count_nonzero(mask)) for mask in confidence_masks]
    valid_pixel_count: int = sum(valid_pixel_counts)
    selected_ranks: Int[ndarray, "sampled_points"] = np.linspace(
        0,
        max(valid_pixel_count - 1, 0),
        min(target_points, valid_pixel_count),
        dtype=np.int64,
    )
    sampled_points: list[Float32[ndarray, "points 3"]] = []
    sampled_colors: list[UInt8[ndarray, "points 3"]] = []
    rank_offset: int = 0

    for prediction, depth_map, confidence_mask, camera_valid_count in zip(
        mv_pred_list,
        depths,
        confidence_masks,
        valid_pixel_counts,
        strict=True,
    ):
        if confidence_mask.shape != depth_map.shape:
            raise ValueError("Each confidence mask must match its prediction's depth shape.")

        camera_ranks: Int[ndarray, "camera_points"] = selected_ranks[
            (selected_ranks >= rank_offset) & (selected_ranks < rank_offset + camera_valid_count)
        ] - rank_offset
        rank_offset += camera_valid_count
        if len(camera_ranks) == 0:
            continue
        sampled_flat_indices: Int[ndarray, "camera_points"] = np.flatnonzero(confidence_mask)[camera_ranks]
        sampled_y, sampled_x = np.divmod(sampled_flat_indices, depth_map.shape[1])

        K_33_raw: Float[ndarray, "3 3"] | None = prediction.pinhole_param.intrinsics.k_matrix
        if K_33_raw is None:
            raise ValueError("Multi-view prediction must include camera intrinsics.")
        world_T_cam: Float32[ndarray, "4 4"] = prediction.pinhole_param.extrinsics.world_T_cam.astype(np.float32)
        world_points: Float32[ndarray, "camera_points 3"] = unproject_selected_pixels(
            depth_map,
            K_33_raw.astype(np.float32),
            world_T_cam,
            sampled_y,
            sampled_x,
        )
        finite: Bool[ndarray, "camera_points"] = np.isfinite(world_points).all(axis=1)
        sampled_points.append(world_points[finite])
        sampled_colors.append(prediction.rgb_image[sampled_y, sampled_x][finite])

    if not sampled_points:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

    points: Float32[ndarray, "sampled_points 3"] = np.concatenate(sampled_points)
    colors: UInt8[ndarray, "sampled_points 3"] = np.concatenate(sampled_colors)
    return points, colors
