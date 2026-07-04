"""Camera projection helpers for the Brown–Conrady distortion model."""

import os
import warnings
from collections.abc import Sequence

import cv2
import numpy as np
from einops import rearrange
from jaxtyping import Float
from numpy import ndarray

from simplecv.camera_parameters import BrownConradyDistortion, PinholeParameters
from simplecv.sensors.camera.base_camera import filter_out_of_bounds, world_to_cam_batched


def cam_to_image_batched(
    xyz_cam: Float[ndarray, "n_frames n_views n_points 3"],
    K: Float[ndarray, "n_views 3 3"],
) -> Float[ndarray, "n_frames n_views n_points 2"]:
    """Project batched camera-frame points into pixel space using intrinsics.

    Args:
        xyz_cam: Camera-frame coordinates ``[n_frames, n_views, n_points, 3]``.
        K: Intrinsic matrices per view ``[n_views, 3, 3]``.

    Returns:
        Pixel coordinates ``[n_frames, n_views, n_points, 2]``.
    """
    xyz_cam: Float[ndarray, "n_frames n_views 3 n_points"] = rearrange(
        xyz_cam,
        "n_frames n_views n_points dim -> n_frames n_views dim n_points",
        dim=3,
    )
    # [1, n_views, 3, 3] @ [1, n_views, 3, n_points] -> [n_frames, n_views, 3, n_points]
    uv_hom: Float[ndarray, "n_frames n_views 3 n_points"] = K @ xyz_cam
    uv_hom: Float[ndarray, "n_frames n_views n_points 3"] = rearrange(
        uv_hom, "n_frames n_views dim n_points -> n_frames n_views n_points dim", dim=3
    )
    uv: Float[ndarray, "n_frames n_views n_points 2"] = uv_hom[..., :2] / uv_hom[..., 2:]
    return uv


def _distort_normalized_points(
    points_xy: Float[ndarray, "n 2"],
    distortion: BrownConradyDistortion,
) -> Float[ndarray, "n 2"]:
    """Apply Brown–Conrady distortion in normalized coordinates.

    The implementation mirrors OpenCV's rational model (k1–k6, p1–p2) with optional
    thin-prism (s1–s4) and tilt (tau_x, tau_y) terms. Points are assumed to already be
    normalized by the focal lengths.
    """

    x: Float[ndarray, "n"] = points_xy[:, 0]
    y: Float[ndarray, "n"] = points_xy[:, 1]

    r2: Float[ndarray, "n"] = x * x + y * y
    r4: Float[ndarray, "n"] = r2 * r2
    r6: Float[ndarray, "n"] = r4 * r2

    radial_num: Float[ndarray, "n"] = (
        1.0
        + distortion.k1 * r2
        + distortion.k2 * r4
        + distortion.k3 * r6
    )
    radial_den: Float[ndarray, "n"] = (
        1.0
        + distortion.k4 * r2
        + distortion.k5 * r4
        + distortion.k6 * r6
    )
    eps: float = float(2.0**-52)
    radial_safe: Float[ndarray, "n"] = radial_num / np.where(
        np.abs(radial_den) < eps, np.sign(radial_den) * eps, radial_den
    )

    x_radial: Float[ndarray, "n"] = x * radial_safe
    y_radial: Float[ndarray, "n"] = y * radial_safe

    x_tangential: Float[ndarray, "n"] = 2.0 * distortion.p1 * x * y + distortion.p2 * (r2 + 2.0 * x * x)
    y_tangential: Float[ndarray, "n"] = distortion.p1 * (r2 + 2.0 * y * y) + 2.0 * distortion.p2 * x * y

    x_prism: Float[ndarray, "n"] = distortion.s1 * r2 + distortion.s2 * r4
    y_prism: Float[ndarray, "n"] = distortion.s3 * r2 + distortion.s4 * r4

    x_distorted: Float[ndarray, "n"] = x_radial + x_tangential + x_prism
    y_distorted: Float[ndarray, "n"] = y_radial + y_tangential + y_prism

    if distortion.tau_x != 0.0 or distortion.tau_y != 0.0:
        # Tilt per OpenCV docs:
        # s [x''' y''' 1]^T = [[R33, 0, -R13], [0, R33, -R23], [0, 0, 1]] * R(tau) * [x'' y'' 1]^T
        c_tx: float = float(np.cos(distortion.tau_x))
        s_tx: float = float(np.sin(distortion.tau_x))
        c_ty: float = float(np.cos(distortion.tau_y))
        s_ty: float = float(np.sin(distortion.tau_y))

        # Rotation R(tau_x, tau_y)
        R00: float = c_ty
        R01: float = s_ty * s_tx
        R02: float = -s_ty * c_tx
        R10: float = 0.0
        R11: float = c_tx
        R12: float = s_tx
        R20: float = s_ty
        R21: float = -c_ty * s_tx
        R22: float = c_ty * c_tx

        x1: Float[ndarray, "n"] = R00 * x_distorted + R01 * y_distorted + R02
        y1: Float[ndarray, "n"] = R10 * x_distorted + R11 * y_distorted + R12
        z1: Float[ndarray, "n"] = R20 * x_distorted + R21 * y_distorted + R22

        R13: float = R02
        R23: float = R12
        R33: float = R22

        denom_safe: Float[ndarray, "n"] = np.where(np.abs(z1) < eps, np.sign(z1) * eps, z1)
        x_distorted = R33 * (x1 / denom_safe) - R13
        y_distorted = R33 * (y1 / denom_safe) - R23

    return np.stack((x_distorted, y_distorted), axis=-1)


def _opencv_distort_view(
    uv_view: Float[ndarray, "n_frames n_points 2"],
    K_view: Float[ndarray, "3 3"],
    distortion: BrownConradyDistortion,
) -> Float[ndarray, "n_frames n_points 2"]:
    """Reference Brown–Conrady distortion using OpenCV's ``projectPoints``.

    This is used only for optional validation to ensure the NumPy implementation
    matches OpenCV's model (including tilt). It keeps the API local to avoid a hard
    dependency on OpenCV for callers that may not have it installed.
    """

    try:
        import cv2  # Local import to avoid import-time dependency when unused
    except Exception as exc:  # pragma: no cover - defensive fallback
        warnings.warn(f"OpenCV unavailable for distortion validation: {exc}", stacklevel=2)
        return uv_view

    uv_norm: Float[ndarray, "n_frames n_points 2"] = uv_view.copy()
    fx: float = float(K_view[0, 0])
    fy: float = float(K_view[1, 1])
    cx: float = float(K_view[0, 2])
    cy: float = float(K_view[1, 2])
    uv_norm[..., 0] = (uv_norm[..., 0] - cx) / fx
    uv_norm[..., 1] = (uv_norm[..., 1] - cy) / fy

    ones: Float[ndarray, "n_frames n_points 1"] = np.ones((*uv_norm.shape[:2], 1), dtype=np.float64)
    pts_cam: Float[ndarray, "n_frames n_points 3"] = np.concatenate([uv_norm.astype(np.float64), ones], axis=-1)
    pts_flat: Float[ndarray, "_ 3"] = pts_cam.reshape(-1, 3)

    dist_vec: Float[ndarray, "14"] = np.array(
        [
            distortion.k1,
            distortion.k2,
            distortion.p1,
            distortion.p2,
            distortion.k3,
            distortion.k4,
            distortion.k5,
            distortion.k6,
            distortion.s1,
            distortion.s2,
            distortion.s3,
            distortion.s4,
            distortion.tau_x,
            distortion.tau_y,
        ],
        dtype=np.float64,
    )

    rvec = np.zeros((3, 1), dtype=np.float64)
    tvec = np.zeros((3, 1), dtype=np.float64)
    proj_flat, _ = cv2.projectPoints(pts_flat, rvec, tvec, K_view.astype(np.float64), dist_vec)
    proj_flat = proj_flat.reshape(uv_view.shape[0], uv_view.shape[1], 2)
    return proj_flat.astype(uv_view.dtype, copy=False)


def apply_brown_conrady_distortion_batch(
    uv_stack: Float[ndarray, "n_frames n_views n_kpts 2"],
    intrinsics_stack: Float[ndarray, "n_views 3 3"],
    distortions: Sequence[BrownConradyDistortion | None],
) -> Float[ndarray, "n_frames n_views n_kpts 2"]:
    """Apply per-view Brown–Conrady distortion polynomials to image coordinates."""

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

    validate: bool = bool(os.environ.get("SIMPLECV_VALIDATE_BC_DISTORT", ""))
    for view_idx, distortion in enumerate(distortions):
        if distortion is None:
            continue

        view_norm: Float[ndarray, "n_frames n_kpts 2"] = uv_normalized[:, view_idx, :, :]
        view_norm_flat: Float[ndarray, "_ 2"] = view_norm.reshape(n_frames * n_kpts, 2)

        distorted_flat: Float[ndarray, "_ 2"] = _distort_normalized_points(
            points_xy=view_norm_flat, distortion=distortion
        )
        uv_normalized[:, view_idx, :, :] = distorted_flat.reshape(n_frames, n_kpts, 2)

        if validate:
            uv_cv: Float[ndarray, "n_frames n_kpts 2"] = _opencv_distort_view(
                uv_view=uv_stack[:, view_idx, :, :],
                K_view=K_views[view_idx],
                distortion=distortion,
            )
            uv_manual_view: Float[ndarray, "n_frames n_kpts 2"] = np.empty_like(uv_cv)
            uv_manual_view[..., 0] = view_norm[..., 0] * fx[None, view_idx, None] + cx[None, view_idx, None]
            uv_manual_view[..., 1] = view_norm[..., 1] * fy[None, view_idx, None] + cy[None, view_idx, None]

            max_err: float = float(np.max(np.abs(uv_cv - uv_manual_view)))
            if max_err > 1e-3:
                warnings.warn(
                    "OpenCV/Brown–Conrady distortion mismatch (max abs err > 1e-3 px). "
                    f"view={view_idx}, max_err={max_err:.4f}",
                    stacklevel=2,
                )

    uv_distorted[..., 0] = uv_normalized[..., 0] * fx[None, :, None] + cx[None, :, None]
    uv_distorted[..., 1] = uv_normalized[..., 1] * fy[None, :, None] + cy[None, :, None]

    return uv_distorted


def undistort_brown_conrady_batch(
    uv_distorted: Float[ndarray, "n_frames n_views n_kpts 2"],
    intrinsics_stack: Float[ndarray, "n_views 3 3"],
    distortions: Sequence[BrownConradyDistortion | None],
) -> Float[ndarray, "n_frames n_views n_kpts 2"]:
    """Remove Brown–Conrady distortion in pixel space for a batched set of views using OpenCV.

    Note:
        This mirrors OpenCV's ``undistortPoints`` with ``P=K`` so outputs stay in pixel space,
        matching the projection matrices used by downstream triangulation.
    """

    if all(distortion is None for distortion in distortions):
        return uv_distorted

    uv_out: Float[ndarray, "n_frames n_views n_kpts 2"] = uv_distorted.copy().astype(float)
    n_frames, n_views, n_kpts, _ = uv_out.shape
    K_views: Float[ndarray, "n_views 3 3"] = np.asarray(intrinsics_stack, dtype=float)

    for view_idx, distortion in enumerate(distortions):
        if distortion is None:
            continue

        dvec: Float[ndarray, "14"] = np.array(
            [
                distortion.k1,
                distortion.k2,
                distortion.p1,
                distortion.p2,
                distortion.k3,
                distortion.k4,
                distortion.k5,
                distortion.k6,
                distortion.s1,
                distortion.s2,
                distortion.s3,
                distortion.s4,
                distortion.tau_x,
                distortion.tau_y,
            ],
            dtype=np.float64,
        )
        flat: Float[ndarray, "_ 2"] = uv_out[:, view_idx, :, :].reshape(-1, 2).astype(np.float64)
        undist_flat = cv2.undistortPoints(
            flat[:, None, :],
            cameraMatrix=K_views[view_idx],
            distCoeffs=dvec,
            P=K_views[view_idx],
        )
        uv_out[:, view_idx, :, :] = undist_flat.reshape(n_frames, n_kpts, 2)

    return uv_out


def project_brown_conrady_grid(
    xyz_stack_world: Float[ndarray, "n_frames n_points 3"],
    pinholes_per_view: list[PinholeParameters],
    *,
    filter_invalid: bool = True,
) -> Float[ndarray, "n_frames n_views n_points 2"]:
    """
    Project world-frame keypoints through a small set of fixed pinhole cameras (frames × views grid).

    Args:
        xyz_stack_world: World-frame coordinates ``[n_frames, n_points, 3]`` to reproject.
        pinholes_per_view: Ordered camera models defining extrinsics/intrinsics (one per view, ``len = n_views``).
        filter_invalid: When ``True`` (default) mask pixels that fall outside the image bounds or
            behind the camera. Disable to obtain the raw projection for debugging/comparison.

    Returns:
        Distorted pixel coordinates ``[n_frames, n_views, n_points, 2]``. When ``filter_invalid`` is
        ``True`` (default) points falling behind the camera or outside the image bounds are masked via
        ``filter_out_of_bounds``.

    Notes:
        * Assumes all cameras share identical image dimensions and supports per-view Brown–Conrady
          coefficients (radial, tangential, thin-prism, tilt).
        * Uses ``filter_out_of_bounds`` to drop pixels when ``filter_invalid`` is enabled.
        * Intended for the multi-view case where ``n_views`` is small and fixed while ``n_frames`` is large.
          Do **not** pass one pinhole per frame here; that creates an ``n_frames × n_frames`` outer product.
          If you have a per-frame pose list, use ``project_brown_conrady_diagonal`` instead.
    """
    # 0. Prepare intrinsics and extrinsics stacks
    cam_T_world: Float[ndarray, "n_views 4 4"] = np.stack(
        [pinhole.extrinsics.cam_T_world for pinhole in pinholes_per_view]
    )

    n_views: int = len(pinholes_per_view)
    K_stack: Float[ndarray, "n_views 3 3"] = np.empty((n_views, 3, 3), dtype=float)
    for view_idx, pinhole in enumerate(pinholes_per_view):
        k_matrix: Float[ndarray, "3 3"] | None = pinhole.intrinsics.k_matrix
        assert k_matrix is not None, "All pinholes must carry intrinsics.k_matrix"
        K_stack[view_idx] = k_matrix
    # 1. Transform world coordinates to camera coordinates
    xyz_stack_cam: Float[ndarray, "n_frames n_views n_points 3"] = world_to_cam_batched(xyz_stack_world, cam_T_world)
    # 2. Project camera coordinates to image coordinates
    uv_stack: Float[ndarray, "n_frames n_views n_points 2"] = cam_to_image_batched(xyz_cam=xyz_stack_cam, K=K_stack)
    # 3. Apply Brown–Conrady distortion if coefficients are provided
    if pinholes_per_view[0].distortion is not None:
        distortions: list[BrownConradyDistortion | None] = [pinhole.distortion for pinhole in pinholes_per_view]
        uv_stack = apply_brown_conrady_distortion_batch(
            uv_stack=uv_stack, intrinsics_stack=K_stack, distortions=distortions
        )

    if not filter_invalid:
        return uv_stack

    # 4. Filter out-of-bounds points per view — image sizes may differ across a
    # rig (e.g. MAMMA eval mixes landscape and portrait-mounted cameras). The
    # filter NaN-masks in place, and basic view slices write through to uv_stack.
    for view_idx, pinhole in enumerate(pinholes_per_view):
        filter_out_of_bounds(
            uv_batch=uv_stack[:, view_idx : view_idx + 1],
            xyz_cam_batch=xyz_stack_cam[:, view_idx : view_idx + 1],
            h=pinhole.intrinsics.height,
            w=pinhole.intrinsics.width,
        )
    return uv_stack


def project_brown_conrady_diagonal(
    xyz_stack_world: Float[ndarray, "n_frames n_points 3"],
    pinholes_per_frame: list[PinholeParameters],
    *,
    filter_invalid: bool = True,
) -> Float[ndarray, "n_frames n_points 2"]:
    """
    Project per-frame points through matching per-frame pinhole parameters (frame-aligned diagonal, O(F)).

    Use this when each frame has its own camera pose (``len(pinholes_per_frame) == n_frames``), e.g.,
    ego cameras over time. It walks the frames×frames diagonal directly—no outer-product expansion.
    If you instead have a small, fixed set of views, call ``project_brown_conrady_grid``.
    """

    n_frames: int = min(len(pinholes_per_frame), xyz_stack_world.shape[0])
    xyz_stack_world = xyz_stack_world[:n_frames]

    cam_T_world: Float[ndarray, "n_frames 4 4"] = np.stack(
        [pinholes_per_frame[idx].extrinsics.cam_T_world for idx in range(n_frames)]
    )
    k_matrices: list[Float[ndarray, "3 3"]] = []
    for idx in range(n_frames):
        k_matrix: Float[ndarray, "3 3"] | None = pinholes_per_frame[idx].intrinsics.k_matrix
        assert k_matrix is not None, "Brown-Conrady projection requires a 3x3 intrinsic matrix."
        k_matrices.append(k_matrix)
    K_stack: Float[ndarray, "n_frames 3 3"] = np.stack(k_matrices, dtype=float)

    # World → cam (per frame)
    xyz_world_h: Float[ndarray, "n_frames n_points 4"] = np.concatenate(
        [xyz_stack_world, np.ones((*xyz_stack_world.shape[:2], 1), dtype=xyz_stack_world.dtype)],
        axis=-1,
    )
    xyz_cam_h: Float[ndarray, "n_frames n_points 4"] = np.einsum("fij,fkj->fki", cam_T_world, xyz_world_h)
    xyz_cam: Float[ndarray, "n_frames n_points 3"] = xyz_cam_h[..., :3] / xyz_cam_h[..., 3:]

    # Cam → image
    uv_h: Float[ndarray, "n_frames n_points 3"] = np.einsum("fij,fkj->fki", K_stack, xyz_cam)
    uv: Float[ndarray, "n_frames n_points 2"] = uv_h[..., :2] / uv_h[..., 2:]

    # Distortion (reuse batch helper with n_views == n_frames)
    distortions: list[BrownConradyDistortion | None] = [pinholes_per_frame[idx].distortion for idx in range(n_frames)]
    if any(distortion is not None for distortion in distortions):
        fx: Float[ndarray, "n_frames"] = K_stack[:, 0, 0]
        fy: Float[ndarray, "n_frames"] = K_stack[:, 1, 1]
        cx: Float[ndarray, "n_frames"] = K_stack[:, 0, 2]
        cy: Float[ndarray, "n_frames"] = K_stack[:, 1, 2]

        uv_norm: Float[ndarray, "n_frames n_points 2"] = uv.copy()
        uv_norm[..., 0] = (uv_norm[..., 0] - cx[:, None]) / fx[:, None]
        uv_norm[..., 1] = (uv_norm[..., 1] - cy[:, None]) / fy[:, None]

        shared_distortion: BrownConradyDistortion | None = distortions[0]
        if shared_distortion is not None and all(distortion == shared_distortion for distortion in distortions):
            uv_norm_flat: Float[ndarray, "_ 2"] = uv_norm.reshape(n_frames * xyz_stack_world.shape[1], 2)
            distorted_flat: Float[ndarray, "_ 2"] = _distort_normalized_points(
                points_xy=uv_norm_flat,
                distortion=shared_distortion,
            )
            uv_norm = distorted_flat.reshape(n_frames, xyz_stack_world.shape[1], 2)
        else:
            for frame_idx, distortion in enumerate(distortions):
                if distortion is None:
                    continue
                frame_flat: Float[ndarray, "_ 2"] = uv_norm[frame_idx].reshape(-1, 2)
                distorted_flat = _distort_normalized_points(points_xy=frame_flat, distortion=distortion)
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
