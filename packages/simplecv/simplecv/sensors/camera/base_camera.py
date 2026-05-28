"""Shared camera-space projection utilities."""

import numpy as np
from einops import rearrange
from jaxtyping import Float
from numpy import ndarray


def world_to_cam_batched(
    xyz_world: Float[ndarray, "n_frames n_points 3"],
    cam_T_world: Float[ndarray, "n_views 4 4"],
) -> Float[ndarray, "n_frames n_views n_points 3"]:
    """Transform world-frame points into each camera's coordinate system.

    Args:
        xyz_world: World-frame coordinates ``[n_frames, n_points, 3]``.
        cam_T_world: Camera-from-world transforms for each view ``[n_views, 4, 4]``.

    Returns:
        Camera-frame coordinates ``[n_frames, n_views, n_points, 3]`` with per-view poses applied.
    """

    xyz_world_hom: Float[ndarray, "n_frames n_points 4"] = np.concatenate(
        [xyz_world, np.ones((xyz_world.shape[0], xyz_world.shape[1], 1), dtype=xyz_world.dtype)],
        axis=-1,
    )

    # Expose n_views dimension for batched matmul and transpose n_points and homogeneous axis.
    xyz_world_hom: Float[ndarray, "n_frames 1 4 n_points"] = rearrange(
        xyz_world_hom, "n_frames n_points xyz_hom -> n_frames 1 xyz_hom n_points"
    )
    cam_T_world_batched: Float[ndarray, "1 n_views 4 4"] = rearrange(cam_T_world, "n_views m n -> 1 n_views m n")
    # [n_frames, 1, 4, 4] @ [1, n_views, 4, n_points] -> [n_frames, n_views, 4, n_points]
    xyz_cam_hom_unrearranged: Float[ndarray, "n_frames n_views 4 n_points"] = cam_T_world_batched @ xyz_world_hom
    xyz_cam_hom: Float[ndarray, "n_frames n_views n_points 4"] = rearrange(
        xyz_cam_hom_unrearranged,
        "n_frames n_views xyz_hom n_points -> n_frames n_views n_points xyz_hom",
    )
    xyz_cam: Float[ndarray, "n_frames n_views n_points 3"] = xyz_cam_hom[..., :3] / xyz_cam_hom[..., 3:]
    return xyz_cam


def cam_to_world_batched(
    xyz_cam: Float[ndarray, "n_frames n_views n_points 3"],
) -> Float[ndarray, "n_frames n_views n_points 3"]:
    raise NotImplementedError("cam_to_world_batched is not implemented yet.")


def filter_out_of_bounds(
    uv_batch: Float[ndarray, "n_frames n_views n_points 2"],
    xyz_cam_batch: Float[ndarray, "n_frames n_views n_points 3"],
    h: int,
    w: int,
):
    """Mask pixels projected outside the image plane or behind the camera."""

    # make sure points are within image bounds
    out_of_bounds = np.logical_or(uv_batch[..., 0] >= w, uv_batch[..., 1] >= h)
    out_of_bounds = np.logical_or(out_of_bounds, uv_batch[..., 0] < 0)
    out_of_bounds = np.logical_or(out_of_bounds, uv_batch[..., 1] < 0)
    # make sure points are in front of camera
    out_of_bounds = np.logical_or(out_of_bounds, xyz_cam_batch[..., 2] < 0)

    # if out of bounds, set to nan
    uv_batch[out_of_bounds, :] = np.nan
    return uv_batch
