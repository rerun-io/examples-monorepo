"""Batched helpers for projecting points through Kannala–Brandt fisheye models."""

from collections.abc import Sequence

import numpy as np
from einops import rearrange
from jaxtyping import Float
from numpy import ndarray

from simplecv.camera_parameters import Fisheye62Parameters, KannalaBrandtDistortion, apply_radial_tangential_distortion
from simplecv.sensors.camera.base_camera import filter_out_of_bounds, world_to_cam_batched


def arctan_cam_to_image_batched(
    xyz_cam: Float[ndarray, "n_frames n_views n_points 3"],
    K: Float[ndarray, "n_views 3 3"],
) -> Float[ndarray, "n_frames n_views n_points 2"]:
    """Map camera-frame points onto the image plane using the Kannala–Brandt arctan model.

    Args:
        xyz_cam: Camera-space coordinates ``[n_frames, n_views, n_points, 3]``.
        K: Intrinsic matrices per view ``[n_views, 3, 3]``.

    Returns:
        Undistorted pixel coordinates ``[n_frames, n_views, n_points, 2]`` produced by the arctan
        parameterisation (prior to applying the polynomial distortion terms).
    """

    x_cam: Float[ndarray, "n_frames n_views n_points"] = xyz_cam[..., 0]
    y_cam: Float[ndarray, "n_frames n_views n_points"] = xyz_cam[..., 1]
    z_cam: Float[ndarray, "n_frames n_views n_points"] = xyz_cam[..., 2]

    r_xy: Float[ndarray, "n_frames n_views n_points"] = np.sqrt(x_cam * x_cam + y_cam * y_cam)
    eps: float = float(2.0**-128)
    denom: Float[ndarray, "n_frames n_views n_points"] = np.maximum(r_xy, eps)
    theta: Float[ndarray, "n_frames n_views n_points"] = np.arctan2(r_xy, z_cam)
    scale: Float[ndarray, "n_frames n_views n_points"] = theta / denom

    xy_cam: Float[ndarray, "n_frames n_views n_points 2"] = np.zeros_like(xyz_cam[..., :2])
    xy_cam[..., 0] = x_cam * scale
    xy_cam[..., 1] = y_cam * scale

    ones: Float[ndarray, "n_frames n_views n_points 1"] = np.ones_like(scale)[..., None]
    xy_cam_hom: Float[ndarray, "n_frames n_views n_points 3"] = np.concatenate([xy_cam, ones], axis=-1)

    xy_cam_hom_batched: Float[ndarray, "n_frames n_views 3 n_points"] = rearrange(
        xy_cam_hom, "n_frames n_views n_points xyz -> n_frames n_views xyz n_points"
    )
    K_batched: Float[ndarray, "1 n_views 3 3"] = rearrange(K, "n_views n m -> 1 n_views n m")
    uv_hom: Float[ndarray, "n_frames n_views 3 n_points"] = K_batched @ xy_cam_hom_batched
    uv_hom = rearrange(uv_hom, "n_frames n_views xyz n_points -> n_frames n_views n_points xyz")

    denom_uv: Float[ndarray, "n_frames n_views n_points 1"] = uv_hom[..., 2:3]
    denom_safe: Float[ndarray, "n_frames n_views n_points 1"] = np.where(
        np.abs(denom_uv) < eps, np.sign(denom_uv) * eps, denom_uv
    )
    uv: Float[ndarray, "n_frames n_views n_points 2"] = uv_hom[..., :2] / denom_safe

    return uv


def apply_kannala_brandt_distortion_batch(
    uv_stack: Float[ndarray, "n_frames n_views n_kpts 2"],
    intrinsics_stack: Float[ndarray, "n_views 3 3"],
    distortions: Sequence[KannalaBrandtDistortion | None],
) -> Float[ndarray, "n_frames n_views n_kpts 2"]:
    """Apply per-view Kannala–Brandt distortion polynomials to image coordinates."""

    if all(distortion is None for distortion in distortions):
        return uv_stack

    uv_distorted: Float[ndarray, "n_frames n_views n_kpts 2"] = uv_stack.copy()
    K_views: Float[ndarray, "n_views 3 3"] = np.asarray(intrinsics_stack)

    fx: Float[ndarray, "n_views"] = K_views[:, 0, 0]
    fy: Float[ndarray, "n_views"] = K_views[:, 1, 1]
    cx: Float[ndarray, "n_views"] = K_views[:, 0, 2]
    cy: Float[ndarray, "n_views"] = K_views[:, 1, 2]

    uv_normalized: Float[ndarray, "n_frames n_views n_kpts 2"] = uv_distorted.copy()
    uv_normalized[..., 0] = (uv_normalized[..., 0] - cx[None, :, None]) / fx[None, :, None]
    uv_normalized[..., 1] = (uv_normalized[..., 1] - cy[None, :, None]) / fy[None, :, None]

    n_frames: int = uv_stack.shape[0]
    n_kpts: int = uv_stack.shape[2]

    for view_idx, distortion in enumerate(distortions):
        if distortion is None:
            continue
        view_norm: Float[ndarray, "n_frames n_kpts 2"] = uv_normalized[:, view_idx, :, :]
        view_norm_flat: Float[ndarray, "_ 2"] = view_norm.reshape(n_frames * n_kpts, 2)
        distorted_flat: Float[ndarray, "_ 2"] = apply_radial_tangential_distortion(distortion, view_norm_flat)
        uv_normalized[:, view_idx, :, :] = distorted_flat.reshape(n_frames, n_kpts, 2)

    uv_distorted[..., 0] = uv_normalized[..., 0] * fx[None, :, None] + cx[None, :, None]
    uv_distorted[..., 1] = uv_normalized[..., 1] * fy[None, :, None] + cy[None, :, None]

    return uv_distorted


def project_kannala_brandt_grid(
    xyz_stack_world: Float[ndarray, "n_frames n_points 3"],
    pinholes_per_view: list[Fisheye62Parameters],
    filter_invalid: bool = True,
) -> Float[ndarray, "n_frames n_views n_points 2"]:
    """
    Project world-frame keypoints through a small set of fixed fisheye pinholes (frames × views grid).

    Args:
        xyz_stack_world: World-frame coordinates ``[n_frames, n_points, 3]`` to reproject.
        pinholes_per_view: Ordered camera models defining extrinsics/intrinsics (one per view, ``len = n_views``).
        filter_invalid: When ``True`` (default) mask pixels that fall outside the image bounds or
            behind the camera. Disable to obtain the raw projection for debugging/comparison.

    Returns:
        Distorted pixel coordinates ``[n_frames, n_views, n_points, 2]``. When ``filter_invalid`` is
        ``True`` (default) points falling outside the image bounds or behind the camera are filtered.

    Notes:
        * Assumes all cameras share identical image dimensions.
        * Uses ``filter_out_of_bounds`` to drop invalid pixels when ``filter_invalid`` is enabled.
        * Intended for the multi-view case where ``n_views`` is small and fixed while ``n_frames`` is large.
          Do **not** pass one pinhole per frame here; that creates an ``n_frames × n_frames`` outer product.
          If you have a per-frame pose list, use ``project_kannala_brandt_diagonal`` instead.
    """
    # 0. Prepare intrinsics and extrinsics stacks
    cam_T_world: Float[ndarray, "n_views 4 4"] = np.stack(
        [pinhole.extrinsics.cam_T_world for pinhole in pinholes_per_view]
    )
    K_stack: Float[ndarray, "n_views 3 3"] = np.stack([pinhole.intrinsics.k_matrix for pinhole in pinholes_per_view])
    # TODO currently assumes same distortion coeffs for all cameras, should be extended to support per-camera coeffs
    # 1. Transform world coordinates to camera coordinates
    xyz_stack_cam: Float[ndarray, "n_frames n_views n_points 3"] = world_to_cam_batched(xyz_stack_world, cam_T_world)
    # 2. Project camera coordinates to image coordinates
    uv_stack: Float[ndarray, "n_frames n_views n_points 2"] = arctan_cam_to_image_batched(
        xyz_cam=xyz_stack_cam, K=K_stack
    )
    # TODO currently assumes same distortion coeffs for all cameras, should be extended to support per-camera coeffs
    # 3. Apply Kannala–Brandt distortion if coefficients are provided
    if pinholes_per_view[0].distortion is not None:
        distortions: list[KannalaBrandtDistortion | None] = [pinhole.distortion for pinhole in pinholes_per_view]
        uv_stack = apply_kannala_brandt_distortion_batch(
            uv_stack=uv_stack, intrinsics_stack=K_stack, distortions=distortions
        )
    if not filter_invalid:
        return uv_stack
    # 4. Filter out-of-bounds points (if needed)
    # check that all cameras have same image size for now, could be extended later
    h: int = pinholes_per_view[0].intrinsics.height
    w: int = pinholes_per_view[0].intrinsics.width
    assert all((pinhole.intrinsics.height == h and pinhole.intrinsics.width == w) for pinhole in pinholes_per_view), (
        "All pinhole cameras must have the same image size for batched Brown–Conrady projection."
    )
    uv_filtered: Float[ndarray, "n_frames n_views n_points 2"] = filter_out_of_bounds(
        uv_batch=uv_stack, xyz_cam_batch=xyz_stack_cam, h=h, w=w
    )
    return uv_filtered


def project_kannala_brandt_diagonal(
    xyz_stack_world: Float[ndarray, "n_frames n_points 3"],
    pinholes_per_frame: list[Fisheye62Parameters],
    filter_invalid: bool = True,
) -> Float[ndarray, "n_frames n_points 2"]:
    """
    Project per-frame points through matching per-frame fisheye pinholes (frame-aligned diagonal, O(F)).

    Use this when you have one fisheye pose per frame (``len(pinholes_per_frame) == n_frames``), e.g.,
    ego cameras over time. Avoids the frames×views outer product in ``project_kannala_brandt_grid``.
    """

    n_frames: int = min(len(pinholes_per_frame), xyz_stack_world.shape[0])
    if n_frames == 0:
        return np.zeros((0, xyz_stack_world.shape[1], 2), dtype=xyz_stack_world.dtype)

    xyz_stack_world = xyz_stack_world[:n_frames]
    cam_T_world: Float[ndarray, "n_frames 4 4"] = np.stack(
        [pinholes_per_frame[idx].extrinsics.cam_T_world for idx in range(n_frames)]
    )
    K_stack: Float[ndarray, "n_frames 3 3"] = np.stack(
        [pinholes_per_frame[idx].intrinsics.k_matrix for idx in range(n_frames)], dtype=float
    )

    # World → cam (per frame)
    xyz_world_h: Float[ndarray, "n_frames n_points 4"] = np.concatenate(
        [xyz_stack_world, np.ones((*xyz_stack_world.shape[:2], 1), dtype=xyz_stack_world.dtype)], axis=-1
    )
    xyz_cam_h: Float[ndarray, "n_frames n_points 4"] = np.einsum("fij,fkj->fki", cam_T_world, xyz_world_h)
    xyz_cam: Float[ndarray, "n_frames n_points 3"] = xyz_cam_h[..., :3] / xyz_cam_h[..., 3:]

    # Cam → image using Kannala–Brandt arctan model (per frame)
    x_cam = xyz_cam[..., 0]
    y_cam = xyz_cam[..., 1]
    z_cam = xyz_cam[..., 2]

    r_xy: Float[ndarray, "n_frames n_points"] = np.sqrt(x_cam * x_cam + y_cam * y_cam)
    eps: float = float(2.0**-128)
    denom: Float[ndarray, "n_frames n_points"] = np.maximum(r_xy, eps)
    theta: Float[ndarray, "n_frames n_points"] = np.arctan2(r_xy, z_cam)
    scale: Float[ndarray, "n_frames n_points"] = theta / denom

    xy_cam: Float[ndarray, "n_frames n_points 2"] = np.zeros_like(xyz_cam[..., :2])
    xy_cam[..., 0] = x_cam * scale
    xy_cam[..., 1] = y_cam * scale

    ones: Float[ndarray, "n_frames n_points 1"] = np.ones_like(scale)[..., None]
    xy_cam_hom: Float[ndarray, "n_frames n_points 3"] = np.concatenate([xy_cam, ones], axis=-1)
    uv_h: Float[ndarray, "n_frames n_points 3"] = np.einsum("fij,fkj->fki", K_stack, xy_cam_hom)

    denom_uv: Float[ndarray, "n_frames n_points 1"] = uv_h[..., 2:3]
    denom_safe: Float[ndarray, "n_frames n_points 1"] = np.where(
        np.abs(denom_uv) < eps, np.sign(denom_uv) * eps, denom_uv
    )
    uv: Float[ndarray, "n_frames n_points 2"] = uv_h[..., :2] / denom_safe

    distortions: list[KannalaBrandtDistortion | None] = [pinholes_per_frame[idx].distortion for idx in range(n_frames)]
    if any(distortion is not None for distortion in distortions):
        fx: Float[ndarray, "n_frames"] = K_stack[:, 0, 0]
        fy: Float[ndarray, "n_frames"] = K_stack[:, 1, 1]
        cx: Float[ndarray, "n_frames"] = K_stack[:, 0, 2]
        cy: Float[ndarray, "n_frames"] = K_stack[:, 1, 2]

        uv_norm: Float[ndarray, "n_frames n_points 2"] = uv.copy()
        uv_norm[..., 0] = (uv_norm[..., 0] - cx[:, None]) / fx[:, None]
        uv_norm[..., 1] = (uv_norm[..., 1] - cy[:, None]) / fy[:, None]

        for frame_idx, distortion in enumerate(distortions):
            if distortion is None:
                continue
            frame_flat: Float[ndarray, "_ 2"] = uv_norm[frame_idx].reshape(-1, 2)
            distorted_flat = apply_radial_tangential_distortion(distortion, frame_flat)
            uv_norm[frame_idx] = distorted_flat.reshape(-1, 2)

        uv[..., 0] = uv_norm[..., 0] * fx[:, None] + cx[:, None]
        uv[..., 1] = uv_norm[..., 1] * fy[:, None] + cy[:, None]

    if not filter_invalid:
        return uv

    h: int = pinholes_per_frame[0].intrinsics.height
    w: int = pinholes_per_frame[0].intrinsics.width
    uv_filtered: Float[ndarray, "n_frames n_points 2"] = filter_out_of_bounds(
        uv_batch=uv[:, None, :, :], xyz_cam_batch=xyz_cam[:, None, :, :], h=h, w=w
    )[:, 0, :, :]
    return uv_filtered
