from collections.abc import Sequence

import numpy as np
from einops import rearrange
from jaxtyping import Float, Int
from numpy import ndarray

from simplecv.camera_parameters import KannalaBrandtDistortion, apply_radial_tangential_distortion


def projectN3(
    kpts3d: Float[ndarray, "n_kpts 4"],
    Pall: Float[ndarray, "n_views 3 4"],
) -> Float[ndarray, "n_views nJoints 3"]:
    """Project homogeneous 3D joints through per-view pinhole matrices.

    Args:
        kpts3d: Homogeneous 3D joints ``[n_kpts, 4]`` with confidence in the last entry.
        Pall: Projection matrices ``[n_views, 3, 4]`` per camera.

    Returns:
        Homogeneous image coordinates ``[n_views, n_kpts, 3]``.
    """
    nViews: int = len(Pall)
    # Augment euclidean XYZ with a homogeneous component to match the projection matrices.
    kp3d = np.hstack((kpts3d[:, :3], np.ones((kpts3d.shape[0], 1))))
    kp2ds = []
    for nv in range(nViews):
        kp2d = Pall[nv] @ kp3d.T
        kp2d[:2, :] /= kp2d[2:, :]
        kp2ds.append(kp2d.T[None, :, :])
    kp2ds = np.vstack(kp2ds)
    # Preserve original confidences by zeroing projections of invisible joints.
    kp2ds[..., -1] = kp2ds[..., -1] * (kpts3d[None, :, -1] > 0.0)
    return kp2ds


def proj_3d_vectorized(
    xyz_hom: Float[ndarray, "n_frames n_joints 4"], P: Float[ndarray, "n_views 3 4"]
) -> Float[ndarray, "n_frames n_views n_joints 2"]:
    """
    Project homogeneous joints using a stack of pinhole projection matrices.

    Args:
        xyz_hom: Homogeneous 3D joints ``[n_frames, n_joints, 4]``.
        P: Projection matrices ``[n_views, 3, 4]`` containing ``K [R | t]``.

    Returns:
        Pixel coordinates ``[n_frames, n_views, n_joints, 2]``.
    """
    # Reshape to expose the homogeneous axis for batched matmul.
    xyz_hom: Float[ndarray, "n_frames 1 4 n_joints"] = rearrange(
        xyz_hom, "n_frames n_joints xyz_hom -> n_frames 1 xyz_hom n_joints"
    )
    P: Float[ndarray, "1 n_views 3 4"] = rearrange(P, "n_views n m -> 1 n_views n m")

    # [1, n_views, 3, 4] @ [n_frames, 1, 4, n_joints] -> [n_frames, n_views, 3, n_joints]
    uv_hom: Float[ndarray, "n_frames n_views 3 n_joints"] = P @ xyz_hom
    uv_hom = rearrange(uv_hom, "n_frames n_views xyz_hom n_joints -> n_frames n_views n_joints xyz_hom")
    # Convert back from homogeneous coordinates (divide by w).
    uv: Float[ndarray, "n_frames n_views n_joints 2"] = uv_hom[..., :2] / uv_hom[..., 2:]

    return uv


def arctan_proj_3d_vectorized(
    xyz_hom: Float[ndarray, "n_frames n_joints 4"],
    *,
    cam_T_world: Float[ndarray, "n_views 4 4"],
    K: Float[ndarray, "n_views 3 3"],
) -> Float[ndarray, "n_frames n_views n_joints 2"]:
    """Project joints with a Kannala–Brandt fisheye model.

    Args:
        xyz_hom: Homogeneous 3D joints ``[n_frames, n_joints, 4]``.
        cam_T_world: Camera-from-world transforms ``[n_views, 4, 4]``.
        K: Intrinsic matrices ``[n_views, 3, 3]``.

    Returns:
        Pixel coordinates ``[n_frames, n_views, n_joints, 2]``.
    """
    K_views: Float[ndarray, "n_views 3 3"] = np.asarray(K)
    cam_T_world_views: Float[ndarray, "n_views 4 4"] = np.asarray(cam_T_world)

    # Broadcast over frames/views to obtain camera-frame homogeneous coordinates.
    xyz_hom_batched: Float[ndarray, "n_frames 1 4 n_joints"] = rearrange(
        xyz_hom, "n_frames n_joints xyz -> n_frames 1 xyz n_joints"
    )
    cam_T_world_batched: Float[ndarray, "1 n_views 4 4"] = rearrange(cam_T_world_views, "n_views n m -> 1 n_views n m")
    # [1, n_views, 4, 4] @ [n_frames, 1, 4, n_joints] -> [n_frames, n_views, 4, n_joints]
    xyz_cam_hom: Float[ndarray, "n_frames n_views 4 n_joints"] = cam_T_world_batched @ xyz_hom_batched
    xyz_cam: Float[ndarray, "n_frames n_views n_joints 3"] = rearrange(
        xyz_cam_hom[:, :, :3, :],
        "n_frames n_views xyz n_joints -> n_frames n_views n_joints xyz",
    )

    # Decompose into camera axes for the arctan mapping.
    x_cam: Float[ndarray, "n_frames n_views n_joints"] = xyz_cam[..., 0]
    y_cam: Float[ndarray, "n_frames n_views n_joints"] = xyz_cam[..., 1]
    z_cam: Float[ndarray, "n_frames n_views n_joints"] = xyz_cam[..., 2]

    r_xy: Float[ndarray, "n_frames n_views n_joints"] = np.sqrt(x_cam * x_cam + y_cam * y_cam)
    eps: float = float(2.0**-128)
    denom: Float[ndarray, "n_frames n_views n_joints"] = np.maximum(r_xy, eps)
    theta: Float[ndarray, "n_frames n_views n_joints"] = np.arctan2(r_xy, z_cam)
    scale: Float[ndarray, "n_frames n_views n_joints"] = theta / denom

    xy_cam: Float[ndarray, "n_frames n_views n_joints 2"] = np.zeros_like(xyz_cam[..., :2])
    xy_cam[..., 0] = x_cam * scale
    xy_cam[..., 1] = y_cam * scale

    ones: Float[ndarray, "n_frames n_views n_joints 1"] = np.ones_like(scale)[..., None]
    xy_cam_hom: Float[ndarray, "n_frames n_views n_joints 3"] = np.concatenate([xy_cam, ones], axis=-1)

    xy_cam_hom_batched: Float[ndarray, "n_frames n_views 3 n_joints"] = rearrange(
        xy_cam_hom, "n_frames n_views n_joints xyz -> n_frames n_views xyz n_joints"
    )
    K_batched: Float[ndarray, "1 n_views 3 3"] = rearrange(K_views, "n_views n m -> 1 n_views n m")
    # [1, n_views, 3, 3] @ [n_frames, n_views, 3, n_joints] -> [n_frames, n_views, 3, n_joints]
    uv_hom: Float[ndarray, "n_frames n_views 3 n_joints"] = K_batched @ xy_cam_hom_batched
    uv_hom: Float[ndarray, "n_frames n_views n_joints 3"] = rearrange(
        uv_hom, "n_frames n_views xyz n_joints -> n_frames n_views n_joints xyz"
    )

    denom_uv: Float[ndarray, "n_frames n_views n_joints 1"] = uv_hom[..., 2:3]
    # Avoid infinities around the optical axis where w ≈ 0.
    denom_safe: Float[ndarray, "n_frames n_views n_joints 1"] = np.where(
        np.abs(denom_uv) < eps, np.sign(denom_uv) * eps, denom_uv
    )
    uv: Float[ndarray, "n_frames n_views n_joints 2"] = uv_hom[..., :2] / denom_safe

    return uv


def apply_kannala_brandt_distortion_batch(
    uv_stack: Float[ndarray, "n_frames n_views n_kpts 2"],
    intrinsics_stack: Float[ndarray, "n_views 3 3"],
    distortions: Sequence[KannalaBrandtDistortion | None],
) -> Float[ndarray, "n_frames n_views n_kpts 2"]:
    """Apply per-view Kannala–Brandt distortion to a stack of UV coordinates."""

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


def batch_triangulate(
    keypoints_2d: Float[ndarray, "nViews nJoints 3"],
    projection_matrices: Float[ndarray, "nViews 3 4"],
    min_views: int = 2,
) -> Float[ndarray, "nJoints 4"]:
    """Triangulate joints from multi-view correspondences with linear least squares.

    Args:
        keypoints_2d: Homogeneous 2D joints ``[n_views, n_joints, 3]``.
        projection_matrices: Camera projection matrices ``[n_views, 3, 4]``.
        min_views: Minimum number of valid views required to triangulate a joint.

    Returns:
        Homogeneous 3D joints ``[n_joints, 4]`` with aggregated confidence.
    """
    num_joints: int = keypoints_2d.shape[1]

    # Count views where each joint is visible
    visibility_count: Int[ndarray, "nJoints"] = (keypoints_2d[:, :, -1] > 0).sum(axis=0)
    valid_joints = np.where(visibility_count >= min_views)[0]

    # Filter keypoints by valid joints
    filtered_keypoints: Float[ndarray, "nViews nJoints 3"] = keypoints_2d[:, valid_joints]
    conf3d = filtered_keypoints[:, :, -1].sum(axis=0) / visibility_count[valid_joints]

    P0: Float[ndarray, "1 nViews 4"] = projection_matrices[None, :, 0, :]
    P1: Float[ndarray, "1 nViews 4"] = projection_matrices[None, :, 1, :]
    P2: Float[ndarray, "1 nViews 4"] = projection_matrices[None, :, 2, :]

    # x-coords homogenous
    u: Float[ndarray, "nJoints nViews 1"] = rearrange(filtered_keypoints[..., 0], "c j -> j c 1")
    uP2: Float[ndarray, "nJoints nViews 4"] = u * P2

    # y-coords homogenous
    v: Float[ndarray, "nJoints nViews 1"] = rearrange(filtered_keypoints[..., 1], "c j -> j c 1")
    vP2: Float[ndarray, "nJoints nViews 4"] = v * P2

    confidences: Float[ndarray, "nJoints nViews 1"] = rearrange(filtered_keypoints[..., 2], "c j -> j c 1")

    Au: Float[ndarray, "nJoints nViews 4"] = confidences * (uP2 - P0)
    Av: Float[ndarray, "nJoints nViews 4"] = confidences * (vP2 - P1)
    A: Float[ndarray, "nJoints _ 4"] = np.hstack([Au, Av])

    # Solve using SVD
    _, _, Vh = np.linalg.svd(A)
    triangulated_points = Vh[:, -1, :]
    triangulated_points /= triangulated_points[:, 3, None]

    # Construct result
    result: Float[ndarray, "nJoints 4"] = np.zeros((num_joints, 4))
    # convert from homogenous to euclidean and add confidence
    result[valid_joints, :3] = triangulated_points[:, :3]
    result[valid_joints, 3] = conf3d

    return result
