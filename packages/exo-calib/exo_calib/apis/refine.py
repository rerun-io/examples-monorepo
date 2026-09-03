"""Stage C+D: Kineo-style correspondences, robust triangulation, and staged BA.

Consumes Stage A (``exocalib_init`` layer) and Stage B (``exocalib_kp2d``
layer) straight from the catalog: applies the AssemblyHands-X confidence
post-processing (temporal median filter, crop-margin downweight,
threshold-rescale), builds spatio-temporal correspondences with Kineo's pair
rule, triangulates robustly against the initial cameras, and refines the
extrinsics with kornia-rs bundle adjustment (re-triangulating between rounds),
then one log-focal scale per camera, a rotation-update guard, and a MoGe-2
metric rescale. Logs init/refined 3D keypoint tracks with Kineo's pairwise
reprojection consensus confidence, plus the refined frusta.
"""

from typing import Literal

import numpy as np
from jaxtyping import Bool, Float64
from numpy import ndarray

from exo_calib.ba import BaResult, PosePrior, mean_reprojection_error_px, refine_extrinsics, revert_large_pose_updates
from exo_calib.cameras import RigCameras, camera_centers
from exo_calib.catalog_io import StageConfig, StageContext, stage_context
from exo_calib.confidence import drop_face_confidences, kineo_point_confidence_for_obs
from exo_calib.correspondences import ObservationSet, build_observations, under_supported_view_indices
from exo_calib.eval import rig_geometry
from exo_calib.focal import FocalRefinementResult, refine_focals
from exo_calib.keypoints import AHX_CONF_TAU, AHX_MARGIN_PX, AHX_MEDIAN_WINDOW, CameraKeypoints, GatedKeypoints, stack_postprocessed
from exo_calib.kp2d_layer import load_stage_b
from exo_calib.layer_io import (
    CALIBRATION_VARIANTS,
    EXOCALIB_ROOT,
    REFINED_LAYER,
    VARIANT_COLOR,
    CalibrationVariant,
    ClassSpec,
    RefinementDiagnostics,
    exocalib_entity,
    kp3d_entity,
    log_cameras_layer,
    log_coco133_class_context,
    log_point_tracks,
    log_refinement_diagnostics,
    new_layer_recording,
    read_rig_cameras,
    register_layer,
    write_base_calibration,
)
from exo_calib.metric_scale import estimate_metric_rescale
from exo_calib.triangulation import RobustTriangulationResult, triangulate_points, triangulate_robust

MIN_PAIR_CONFIDENCE: float = 0.90
"""Kineo pair rule: geometric-mean confidence a view pair must exceed (Kineo's default is 0.75)."""
MIN_VIEWS: int = 2
"""Minimum observing cameras per correspondence point."""
REPROJECTION_THRESHOLD_PX: float = 100.0
"""Per-observation gate for robust triangulation; wide, so Cauchy BA sees every useful observation."""
BA_ROUNDS: int = 2
"""Bundle-adjust / re-triangulate rounds."""
BA_MAX_ITERATIONS: int = 50
"""LM iterations per bundle-adjust call."""
ROBUST_KERNEL: Literal["cauchy"] = "cauchy"
"""Robust kernel for the BA residuals."""
ROBUST_SCALE_PX: float = 1.0
"""Robust-kernel scale in pixels."""
POSE_PRIOR_SIGMA_RATIO: float = 0.06
"""Soft prior anchoring camera centers to the Stage A estimate, as a fraction of the rig's median
pairwise camera distance; pins the similarity gauge without freezing rigs of any size."""
MINIMUM_VIEW_SUPPORT_RATIO: float = 0.10
"""Fix views whose robust-observation count is below this fraction of the median count."""
MAXIMUM_ROTATION_UPDATE_DEG: float = 10.0
"""Revert a camera pose when BA rotates it farther from Stage A than this data-independent safety bound."""
FOCAL_ITERATIONS: int = 300
"""Adam iterations for focal-only refinement."""
FOCAL_LEARNING_RATE: float = 0.03
"""Adam learning rate for per-camera log-focal scales."""
FOCAL_ROBUST_SCALE_PX: float = 1.0
"""Cauchy pixel scale for focal-only reprojection residuals."""
LAMBDA_DECAY: float = 1.0
"""Kineo exponential decay for the logged 3D point confidence."""


def main(config: StageConfig) -> None:
    """Run correspondence building, robust triangulation, staged BA, and layer logging."""
    context: StageContext = stage_context(config)
    dataset, segment_id = context.dataset, context.segment_id
    names: tuple[str, ...] = context.layout.exo_camera_names
    if len(names) < 2:
        raise ValueError(f"extrinsic refinement needs at least two exo cameras; found {list(names)}")
    init: RigCameras = read_rig_cameras(dataset, segment_id, names, source="init")
    per_camera: dict[str, CameraKeypoints] = load_stage_b(dataset, segment_id, names)

    gated: GatedKeypoints = stack_postprocessed(per_camera, names, AHX_MEDIAN_WINDOW, AHX_MARGIN_PX, AHX_CONF_TAU)
    kp_xy: Float64[ndarray, "v t 133 2"] = gated.xy
    conf: Float64[ndarray, "v t 133"] = drop_face_confidences(gated.conf)
    print(f"{kp_xy.shape[1]} frames x {len(names)} views; mean post conf {conf[conf > 0].mean():.3f}")

    candidates: ObservationSet = build_observations(kp_xy, conf, min_pair_conf=MIN_PAIR_CONFIDENCE, min_views=MIN_VIEWS)
    print(f"observations: {len(candidates.obs_view_idx)} over {len(candidates.point_frame_idx)} points")
    robust: RobustTriangulationResult = triangulate_robust(candidates, init.intrinsics, init.cam_T_world, reproj_threshold_px=REPROJECTION_THRESHOLD_PX)
    points_init: Float64[ndarray, "n 3"] = robust.points_xyz
    obs: ObservationSet = robust.obs
    print(f"robust triangulation kept {obs.obs_point_idx.size}/{candidates.obs_point_idx.size} observations")
    init_reprojection_px: float = mean_reprojection_error_px(obs, points_init, init.intrinsics, init.cam_T_world)
    print(f"init mean reprojection error: {init_reprojection_px:.2f} px")

    fixed_pose_indices: tuple[int, ...] = under_supported_view_indices(obs.obs_view_idx, len(names), MINIMUM_VIEW_SUPPORT_RATIO)
    if fixed_pose_indices:
        print(f"fixed under-supported camera poses: {list(fixed_pose_indices)}")

    # Soft anchors on the Stage A centres, shared by every BA call; sigma follows the rig's size.
    pairwise_distance: Float64[ndarray, "v v"] = rig_geometry(init.cam_T_world).pairwise_distance_m
    sigma_m: float = POSE_PRIOR_SIGMA_RATIO * float(np.median(pairwise_distance[np.triu_indices(len(names), k=1)]))
    prior: PosePrior = PosePrior(centers=camera_centers(init.cam_T_world), sigma_m=sigma_m)
    print(f"centre prior sigma {sigma_m:.3f} m")
    ba: BaResult = refine_extrinsics(
        init.cam_T_world,
        init.intrinsics,
        obs,
        points_init,
        rounds=BA_ROUNDS,
        robust=ROBUST_KERNEL,
        robust_scale_px=ROBUST_SCALE_PX,
        max_iterations=BA_MAX_ITERATIONS,
        pose_prior=prior,
        fixed_pose_indices=fixed_pose_indices,
    )
    reprojection_px_per_round: list[float] = list(ba.mean_reproj_px_per_round)
    print(f"BA reprojection error per round (px): {[round(e, 3) for e in reprojection_px_per_round]}, converged={ba.converged}")

    # One log-focal scale per camera with poses and points fixed, then one more BA round.
    focal: FocalRefinementResult = refine_focals(
        init.intrinsics,
        ba.cam_T_world,
        ba.points_xyz,
        obs,
        iterations=FOCAL_ITERATIONS,
        learning_rate=FOCAL_LEARNING_RATE,
        robust_scale_px=FOCAL_ROBUST_SCALE_PX,
        device="cuda",
    )
    refined_intrinsics: Float64[ndarray, "v 3 3"] = focal.intrinsics
    print(f"focal-only Cauchy loss {focal.initial_loss:.4f}->{focal.final_loss:.4f}; scales {np.round(focal.focal_scale, 5).tolist()}")
    focal_ba: BaResult = refine_extrinsics(
        ba.cam_T_world,
        refined_intrinsics,
        obs,
        ba.points_xyz,
        rounds=1,
        robust=ROBUST_KERNEL,
        robust_scale_px=ROBUST_SCALE_PX,
        max_iterations=BA_MAX_ITERATIONS,
        pose_prior=prior,
        fixed_pose_indices=fixed_pose_indices,
    )
    cam_T_world: Float64[ndarray, "v 4 4"] = focal_ba.cam_T_world.copy()
    points_xyz: Float64[ndarray, "n 3"] = focal_ba.points_xyz.copy()
    reprojection_px_per_round.extend(focal_ba.mean_reproj_px_per_round)
    print(f"post-focal BA reprojection error (px): {reprojection_px_per_round[-1]:.3f}, converged={focal_ba.converged}")

    guarded_cam_T_world, reverted = revert_large_pose_updates(cam_T_world, init.cam_T_world, maximum_rotation_deg=MAXIMUM_ROTATION_UPDATE_DEG)
    if reverted:
        cam_T_world = guarded_cam_T_world
        points_xyz = triangulate_points(obs, refined_intrinsics, cam_T_world)
        reprojection_px_per_round.append(mean_reprojection_error_px(obs, points_xyz, refined_intrinsics, cam_T_world))
        print(f"rotation-update guard reverted camera poses {list(reverted)}; reprojection {reprojection_px_per_round[-1]:.3f} px")

    first_cam: CameraKeypoints = per_camera[names[0]]
    estimated_scale: float | None = estimate_metric_rescale(dataset, segment_id, cam_T_world, refined_intrinsics, obs, points_xyz, first_cam.sample_indices, names)
    scale_fix: float = 1.0 if estimated_scale is None else estimated_scale
    cam_T_world[:, :3, 3] *= scale_fix
    points_xyz *= scale_fix

    recording, rrd_path = new_layer_recording(segment_id, context.segment_dir / f"{REFINED_LAYER}.rrd")
    log_refinement_diagnostics(
        recording,
        RefinementDiagnostics(
            init_reprojection_px=init_reprojection_px,
            ba_reprojection_px=reprojection_px_per_round,
            observation_count_per_view=np.bincount(obs.obs_view_idx, minlength=len(names)).tolist(),
            metric_rescale_fix=scale_fix,
            focal_scale=focal.focal_scale.tolist(),
        ),
    )
    # No skeleton edges: a frame's 3D track holds only the joints that triangulated, so most
    # COCO-133 links would dangle and the viewer warns about every unresolved one.
    log_coco133_class_context(
        recording,
        EXOCALIB_ROOT,
        tuple(ClassSpec(class_id, f"kp3d {variant}", VARIANT_COLOR[variant]) for class_id, variant in enumerate(CALIBRATION_VARIANTS, start=1)),
        connections=False,
    )
    video_wh: tuple[int, int] = (int(first_cam.video_wh[0]), int(first_cam.video_wh[1]))
    refined: RigCameras = RigCameras(names=init.names, intrinsics=refined_intrinsics, cam_T_world=cam_T_world)
    log_cameras_layer(refined, video_wh, exocalib_entity("refined"), recording, color=VARIANT_COLOR["refined"])
    uncalibrated_idx: list[int] = [i for i, name in enumerate(init.names) if name not in context.layout.calibrated_camera_names]
    if uncalibrated_idx:
        write_base_calibration(
            recording,
            RigCameras(
                names=tuple(init.names[i] for i in uncalibrated_idx),
                intrinsics=refined_intrinsics[uncalibrated_idx],
                cam_T_world=cam_T_world[uncalibrated_idx],
            ),
            video_wh,
        )
        print(f"base calibration written for {len(uncalibrated_idx)} camera(s) without ground truth")
    # Kineo pairwise reprojection consensus confidences for both camera sets.
    variant_rigs: dict[CalibrationVariant, tuple[RigCameras, Float64[ndarray, "n 3"]]] = {"init": (init, points_init), "refined": (refined, points_xyz)}
    for class_id, variant in enumerate(CALIBRATION_VARIANTS, start=1):
        rig, variant_points = variant_rigs[variant]
        conf_points: Float64[ndarray, " n"] = kineo_point_confidence_for_obs(variant_points, obs, rig.intrinsics, rig.cam_T_world, kp_xy, conf, LAMBDA_DECAY)
        valid: Bool[ndarray, " n"] = np.isfinite(variant_points).all(axis=1)
        print(f"{variant}: {int(valid.sum())} points, mean 3D confidence {conf_points[valid].mean():.3f}")
        log_point_tracks(recording, kp3d_entity(variant), obs, variant_points, conf_points, first_cam.times_ns, class_id)
    recording.flush(timeout_sec=30.0)
    print(f"wrote {rrd_path}")
    if config.register:
        register_layer(dataset, rrd_path, REFINED_LAYER)
        print(f"registered layer {REFINED_LAYER}")
