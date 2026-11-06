"""The fused body of one multiview frame and its reprojection checks.

The multiview optimizer (``posekit.ops.multiview_body``) returns a global rotation, body pose and translation in the
first camera's frame. Here MHR forward kinematics turns them, with the first view's shape, scale, hands and
expression, into a world-space mesh and 70 keypoints, and three reprojection checks compare them with the detected
2D keypoints: the triangulated keypoints, each view's own MHR prediction, and the fused MHR keypoints.
"""

from dataclasses import dataclass

import numpy as np
import torch
from jaxtyping import Float32
from numpy import ndarray
from torch import Tensor

from posekit.models.sam3d_body import MhrBodyModel
from posekit.models.sam3d_body_estimator import FinalPosePrediction
from posekit.ops.multiview_body import OptimizationResult, ValidationResult, validate_reprojection
from posekit.ops.multiview_body.mhr_output import expect_mhr_tensor_tuple, expect_mhr_vertices


@dataclass(frozen=True, slots=True)
class FusedFrame:
    """World-space fused body of one frame, its projections into every view and the reprojection checks."""

    verts_world: Float32[ndarray, "n_verts 3"]
    """Fused MHR mesh in world coordinates."""
    triangulated_kpts: Float32[ndarray, "n_kpts 3"]
    """Triangulated 3D keypoints."""
    mhr_world: Float32[ndarray, "70 3"]
    """Fused MHR keypoints in world coordinates."""
    target_2d: Float32[ndarray, "n_views n_kpts 2"]
    """Detected 2D keypoints of each view."""
    projected_triangulated: Float32[ndarray, "n_views n_kpts 2"]
    """Triangulated keypoints projected into each view."""
    projected_mhr: Float32[ndarray, "n_views n_kpts 2"]
    """Each view's own MHR keypoints projected with its intrinsics."""
    projected_mhr_world: Float32[ndarray, "n_views n_kpts 2"]
    """Fused MHR keypoints projected into each view."""
    val_triangulated: ValidationResult
    """Reprojection error of the triangulated keypoints."""
    val_mhr: ValidationResult
    """Reprojection error of each view's own MHR keypoints."""
    val_mhr_world: ValidationResult
    """Reprojection error of the fused MHR keypoints."""
    mean_3d_dist: float
    """Mean 3D distance between the fused MHR and the triangulated keypoints, metres."""


def to_world(points_cam: Float32[ndarray, "n 3"], world_T_cam: Float32[ndarray, "4 4"]) -> Float32[ndarray, "n 3"]:
    """Camera-space points in world coordinates."""
    points_hom: Float32[ndarray, "n 4"] = np.concatenate([points_cam, np.ones((points_cam.shape[0], 1), dtype=np.float32)], axis=1)
    return (points_hom @ world_T_cam.T)[:, :3]


def project_to_views(points_world: Float32[ndarray, "n_kpts 3"], projection_matrices: Float32[ndarray, "n_views 3 4"]) -> Float32[ndarray, "n_views n_kpts 2"]:
    """World points projected into every view with its ``P``."""
    n_views: int = int(projection_matrices.shape[0])
    points_hom: Float32[ndarray, "n_kpts 4"] = np.concatenate(
        [points_world, np.ones((points_world.shape[0], 1), dtype=np.float32)], axis=-1
    )
    projected: Float32[ndarray, "n_views n_kpts 2"] = np.zeros((n_views, points_world.shape[0], 2), dtype=np.float32)
    for view_idx in range(n_views):
        P_view: Float32[ndarray, "3 4"] = projection_matrices[view_idx]
        uv_hom: Float32[ndarray, "n_kpts 3"] = (P_view @ points_hom.T).T
        projected[view_idx] = uv_hom[:, :2] / (uv_hom[:, 2:3] + 1e-8)
    return projected


def project_with_intrinsics(K: Float32[ndarray, "3 3"], points_cam: Float32[ndarray, "n_kpts 3"]) -> Float32[ndarray, "n_kpts 2"]:
    """Camera-space points projected with ``K``; NaN for points on the camera plane."""
    projected: Float32[ndarray, "n_kpts 2"] = np.zeros((points_cam.shape[0], 2), dtype=np.float32)
    for k in range(points_cam.shape[0]):
        xyz: Float32[ndarray, "3"] = points_cam[k]
        uv_hom: Float32[ndarray, "3"] = K @ xyz
        if abs(uv_hom[2]) > 1e-6:
            projected[k] = uv_hom[:2] / uv_hom[2]
        else:
            projected[k] = np.array([np.nan, np.nan], dtype=np.float32)
    return projected


def _fixed_params(base_pred: FinalPosePrediction, device: torch.device) -> dict[str, Tensor]:
    """The first view's shape, scale, hand and expression parameters, batched for ``mhr_forward``."""
    return {
        "hand_pose_params": torch.from_numpy(base_pred.hand_pose_params[np.newaxis, :].astype(np.float32)).to(device),
        "scale_params": torch.from_numpy(base_pred.scale_params[np.newaxis, :].astype(np.float32)).to(device),
        "shape_params": torch.from_numpy(base_pred.shape_params[np.newaxis, :].astype(np.float32)).to(device),
        "expr_params": torch.from_numpy(base_pred.expr_params[np.newaxis, :].astype(np.float32)).to(device),
    }


def fused_mesh_world(
    mhr_head: MhrBodyModel, result: OptimizationResult, base_pred: FinalPosePrediction, first_world_T_cam: Float32[ndarray, "4 4"], device: torch.device
) -> Float32[ndarray, "n_verts 3"]:
    """Fused mesh from ``mhr_forward`` (same FK as the validation keypoints), in world coordinates."""
    with torch.no_grad():
        fixed: dict[str, Tensor] = _fixed_params(base_pred, device)
        opt_global_rot_mesh: Float32[Tensor, "1 3"] = torch.from_numpy(result.global_rot[np.newaxis, :].astype(np.float32)).to(device)
        opt_body_pose_mesh: Float32[Tensor, "1 133"] = torch.from_numpy(result.body_pose[np.newaxis, :].astype(np.float32)).to(device)
        zero_trans_mesh: Float32[Tensor, "1 3"] = torch.zeros(1, 3, device=device, dtype=torch.float32)
        verts_fk: Float32[Tensor, "1 n_verts 3"] = expect_mhr_vertices(
            mhr_head.mhr_forward(
                global_trans=zero_trans_mesh,
                global_rot=opt_global_rot_mesh,
                body_pose_params=opt_body_pose_mesh,
                hand_pose_params=fixed["hand_pose_params"], scale_params=fixed["scale_params"],
                shape_params=fixed["shape_params"], expr_params=fixed["expr_params"],
                return_keypoints=False,  # Just vertices
            ),
        )
        # Apply Y/Z flip (same as keypoints)
        verts_fk = verts_fk.clone()
        verts_fk[..., 1] *= -1
        verts_fk[..., 2] *= -1
        # Add translation after FK
        opt_trans_mesh: Float32[Tensor, "3"] = torch.from_numpy(result.translation.astype(np.float32)).to(device)
        verts_cam_t: Float32[Tensor, "n_verts 3"] = verts_fk.squeeze(0) + opt_trans_mesh
    return to_world(verts_cam_t.cpu().numpy(), first_world_T_cam)


def fused_keypoints_world(
    mhr_head: MhrBodyModel, result: OptimizationResult, base_pred: FinalPosePrediction, first_world_T_cam: Float32[ndarray, "4 4"], device: torch.device
) -> Float32[ndarray, "70 3"]:
    """The 70 fused MHR keypoints from ``mhr_forward`` with the optimized parameters, in world coordinates."""
    with torch.no_grad():
        fixed: dict[str, Tensor] = _fixed_params(base_pred, device)
        opt_global_rot: Float32[Tensor, "1 3"] = torch.from_numpy(result.global_rot[np.newaxis, :].astype(np.float32)).to(device)
        opt_translation: Float32[Tensor, "1 3"] = torch.from_numpy(result.translation[np.newaxis, :].astype(np.float32)).to(device)
        opt_body_pose: Float32[Tensor, "1 133"] = torch.from_numpy(result.body_pose[np.newaxis, :].astype(np.float32)).to(device)
        # NOTE: mhr_forward uses global_trans=0 internally, pred_cam_t applied post-FK
        zero_trans_v: Float32[Tensor, "1 3"] = torch.zeros(1, 3, device=device, dtype=torch.float32)
        mhr_output_v: tuple[Tensor, ...] = expect_mhr_tensor_tuple(
            mhr_head.mhr_forward(
                global_trans=zero_trans_v,  # Always zero
                global_rot=opt_global_rot,
                body_pose_params=opt_body_pose,
                hand_pose_params=fixed["hand_pose_params"], scale_params=fixed["scale_params"],
                shape_params=fixed["shape_params"], expr_params=fixed["expr_params"],
                return_keypoints=True,
            ),
            min_len=2,
        )
        # mhr_output_v = (vertices, keypoints_308)
        keypoints_70: Float32[Tensor, "1 70 3"] = mhr_output_v[1][:, :70, :]
        # Apply camera flip
        keypoints_cam: Float32[Tensor, "70 3"] = keypoints_70.squeeze(0).clone()
        keypoints_cam[..., 1] *= -1
        keypoints_cam[..., 2] *= -1
        # Apply camera-space translation AFTER FK (same as network forward)
        keypoints_cam = keypoints_cam + opt_translation.squeeze(0)
    return to_world(keypoints_cam.cpu().numpy(), first_world_T_cam)


def fuse_frame(
    *,
    mhr_head: MhrBodyModel,
    result: OptimizationResult,
    per_view_predictions: list[FinalPosePrediction],
    keypoints_2d_stack: Float32[ndarray, "n_views n_kpts 3"],
    view_intrinsics: list[Float32[ndarray, "3 3"] | None],
    projection_matrices: Float32[ndarray, "n_views 3 4"],
    first_world_T_cam: Float32[ndarray, "4 4"],
    device: torch.device,
) -> FusedFrame:
    """The fused body of one frame (first view's prediction as base, optimized pose) and its reprojection checks.

    Args:
        view_intrinsics: ``K`` of the rig camera at each prediction's position; None when the sample lacks it.
    """
    n_views: int = int(projection_matrices.shape[0])
    base_pred: FinalPosePrediction = per_view_predictions[0]
    verts_world: Float32[ndarray, "n_verts 3"] = fused_mesh_world(mhr_head, result, base_pred, first_world_T_cam, device)

    # Project triangulated keypoints back to 2D and compare against detected 2D keypoints
    triangulated_kpts: Float32[ndarray, "n_kpts 3"] = result.triangulated_kpts_3d[:, :3]
    triangulated_conf: Float32[ndarray, "n_kpts"] = result.triangulated_kpts_3d[:, 3]
    n_kpts: int = triangulated_kpts.shape[0]
    projected_triangulated: Float32[ndarray, "n_views n_kpts 2"] = project_to_views(triangulated_kpts, projection_matrices)
    target_2d: Float32[ndarray, "n_views n_kpts 2"] = keypoints_2d_stack[:, :, :2]
    confidence_2d: Float32[ndarray, "n_views n_kpts"] = keypoints_2d_stack[:, :, 2]
    # Expand triangulated confidence to all views
    conf_expanded: Float32[ndarray, "n_views n_kpts"] = np.broadcast_to(triangulated_conf[np.newaxis, :], (n_views, n_kpts)).astype(np.float32)
    combined_conf: Float32[ndarray, "n_views n_kpts"] = conf_expanded * confidence_2d
    val_triangulated: ValidationResult = validate_reprojection(projected_2d=projected_triangulated, target_2d=target_2d, confidence=combined_conf)

    # Per-view self-projection: each view's pred_keypoints_3d + pred_cam_t projected with its K,
    # compared to that view's pred_keypoints_2d.
    projected_mhr: Float32[ndarray, "n_views n_kpts 2"] = np.zeros((n_views, base_pred.pred_keypoints_3d.shape[0], 2), dtype=np.float32)
    for view_idx, pred in enumerate(per_view_predictions):
        K_view: Float32[ndarray, "3 3"] | None = view_intrinsics[view_idx]
        if K_view is not None:
            kpts_cam: Float32[ndarray, "n_kpts 3"] = pred.pred_keypoints_3d.astype(np.float32) + pred.pred_cam_t.astype(np.float32)
            projected_mhr[view_idx] = project_with_intrinsics(K_view, kpts_cam)
    val_mhr: ValidationResult = validate_reprojection(projected_2d=projected_mhr, target_2d=target_2d, confidence=confidence_2d)

    # Multiview consistency: fused MHR keypoints (the optimizer's FK) projected into every view
    mhr_world: Float32[ndarray, "70 3"] = fused_keypoints_world(mhr_head, result, base_pred, first_world_T_cam, device)
    projected_mhr_world: Float32[ndarray, "n_views n_kpts 2"] = project_to_views(mhr_world, projection_matrices)
    val_mhr_world: ValidationResult = validate_reprojection(projected_2d=projected_mhr_world, target_2d=target_2d, confidence=confidence_2d)

    # How far the FK output is from the triangulated keypoints
    dist_3d: Float32[ndarray, "n_kpts"] = np.linalg.norm(mhr_world - triangulated_kpts, axis=1)
    return FusedFrame(
        verts_world=verts_world,
        triangulated_kpts=triangulated_kpts,
        mhr_world=mhr_world,
        target_2d=target_2d,
        projected_triangulated=projected_triangulated,
        projected_mhr=projected_mhr,
        projected_mhr_world=projected_mhr_world,
        val_triangulated=val_triangulated,
        val_mhr=val_mhr,
        val_mhr_world=val_mhr_world,
        mean_3d_dist=float(np.mean(dist_3d)),
    )


def print_frame_summary(frame_idx: int, fused: FusedFrame, n_views: int, first_pred: FinalPosePrediction, first_K: Float32[ndarray, "3 3"] | None) -> None:
    """Print the frame's reprojection checks, per-view breakdowns and the first view's MHR self-projection error."""
    val_triangulated: ValidationResult = fused.val_triangulated
    val_mhr_world: ValidationResult = fused.val_mhr_world
    print(f"  Frame {frame_idx}: triangulated   {val_triangulated}")
    print(f"  Frame {frame_idx}: MHR per-view   {fused.val_mhr}")
    print(f"  Frame {frame_idx}: MHR world      {val_mhr_world}")
    print(f"  Frame {frame_idx}: 3D dist (MHR vs triang) = {fused.mean_3d_dist:.3f}m")

    # Per-view triangulated error breakdown
    view_errors: list[str] = []
    for v_idx in range(n_views):
        view_err: float = val_triangulated.per_view_mean_errors[v_idx] if v_idx < len(val_triangulated.per_view_mean_errors) else float("nan")
        view_errors.append(f"v{v_idx}={view_err:.1f}")
    print(f"  [DEBUG] triangulated per-view: {', '.join(view_errors)}")

    # MHR world per-view error breakdown
    mhr_world_errors: list[str] = []
    for v_idx in range(n_views):
        mhr_err: float = val_mhr_world.per_view_mean_errors[v_idx] if v_idx < len(val_mhr_world.per_view_mean_errors) else float("nan")
        mhr_world_errors.append(f"v{v_idx}={mhr_err:.1f}")
    print(f"  [DEBUG] MHR world per-view: {', '.join(mhr_world_errors)}")

    # Compute filtered metrics (exclude views with error > 100px)
    OUTLIER_THRESHOLD: float = 100.0
    good_view_indices: list[int] = [v_idx for v_idx in range(n_views) if val_mhr_world.per_view_mean_errors[v_idx] < OUTLIER_THRESHOLD]
    if good_view_indices:
        good_errors: list[float] = [val_mhr_world.per_view_mean_errors[v_idx] for v_idx in good_view_indices]
        filtered_mean: float = float(np.mean(good_errors))
        print(f"  Frame {frame_idx}: MHR world (filtered, {len(good_view_indices)} views) mean={filtered_mean:.1f}px")

    # Diagnostic: the first view's pred_keypoints_3d + pred_cam_t projected with K should match its pred_keypoints_2d
    if first_K is not None:
        test_kpts_cam: Float32[ndarray, "n_kpts 3"] = first_pred.pred_keypoints_3d.astype(np.float32) + first_pred.pred_cam_t.astype(np.float32)
        projected_test: Float32[ndarray, "n_kpts 2"] = project_with_intrinsics(first_K, test_kpts_cam)
        mhr_2d: Float32[ndarray, "n_kpts 2"] = first_pred.pred_keypoints_2d.astype(np.float32)
        self_error: Float32[ndarray, "n_kpts"] = np.linalg.norm(projected_test - mhr_2d, axis=-1).astype(np.float32)
        mean_self_err: float = float(np.nanmean(self_error))
        print(f"  [DEBUG] MHR self-proj error (view 0): mean={mean_self_err:.1f}px")
