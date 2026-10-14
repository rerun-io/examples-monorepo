"""Stage C+D (whole segment): Kineo-style correspondences, robust triangulation, staged BA, focal, metric rescale.

Applies the AssemblyHands-X post-processing to the Stage B records, builds spatio-temporal correspondences with
Kineo's pair rule, triangulates robustly against the ORIGINAL Stage A rig, refines the extrinsics with kornia-rs bundle
adjustment (re-triangulating between rounds), then one log-focal scale per camera, a rotation-update guard against
Stage A, and a MoGe-2 metric rescale (resources ``frames``: ``SegmentFrames``, ``moge``: ``MoGeV2MetricPredictor``).
"""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
from dagkit.node import Emit, Event, PortType, RunContext
from dagkit.time import STATIC
from jaxtyping import Float32, Float64
from numpy import ndarray
from simplecv.types.rig import RigCameras

from exo_calib.ops.bundle_adjust import BaResult, PosePrior, mean_reprojection_error_px, refine_extrinsics, revert_large_pose_updates
from exo_calib.ops.confidence import drop_face_confidences, kineo_point_confidence
from exo_calib.ops.correspondences import ObservationSet, build_observations, under_supported_view_indices
from exo_calib.ops.eval import camera_centers, rig_geometry
from exo_calib.ops.focal import FocalRefinementResult, refine_focals
from exo_calib.ops.keypoints import GatedKeypoints, stack_postprocessed
from exo_calib.ops.metric_scale import estimate_metric_rescale
from exo_calib.ops.triangulation import RobustTriangulationResult, triangulate_points, triangulate_robust
from exo_calib.types import VARIANTS, BaseCalibration, Keypoints3dTracks, Kp2dOverlay, MultiViewKeypoints, RefinementDiagnostics, RigLayout

if TYPE_CHECKING:
    from monopriors.models.metric_depth.moge_v2 import MoGeV2MetricPredictor
    from simplecv.sources.catalog_video import SegmentFrames

MIN_PAIR_CONFIDENCE: float = 0.90
"""Kineo pair rule: geometric-mean confidence a view pair must exceed (Kineo's default is 0.75)."""
MIN_VIEWS: int = 2
"""Minimum observing cameras per correspondence point."""
REPROJECTION_THRESHOLD_PX: float = 100.0
"""Per-observation gate for robust triangulation; wide, so Cauchy BA sees every useful observation."""
BA_ROUNDS: int = 2
BA_MAX_ITERATIONS: int = 50
ROBUST_KERNEL: Literal["cauchy"] = "cauchy"
ROBUST_SCALE_PX: float = 1.0
POSE_PRIOR_SIGMA_RATIO: float = 0.06
"""Soft prior anchoring camera centers to Stage A, as a fraction of the rig's median pairwise camera distance;
pins the similarity gauge without freezing rigs of any size."""
MINIMUM_VIEW_SUPPORT_RATIO: float = 0.10
"""Fix views whose robust-observation count is below this fraction of the median count."""
MAXIMUM_ROTATION_UPDATE_DEG: float = 10.0
"""Revert a camera pose when BA rotates it farther from Stage A than this data-independent safety bound, also after
removing the rotation the whole rig shares (BA's softly pinned gauge)."""
FOCAL_ITERATIONS: int = 300
FOCAL_LEARNING_RATE: float = 0.03
FOCAL_ROBUST_SCALE_PX: float = 1.0
LAMBDA_DECAY: float = 1.0
"""Kineo exponential decay for the logged 3D point confidence."""
OVERLAY_LOG_STRIDE: int = 5
"""Every N-th frame gets the gated / rejected keypoint skeleton (every frame gets its boxes)."""


class RefineNode:
    """Refine the rig by bundle adjustment."""

    inputs: Mapping[str, PortType] = {"kp2d": MultiViewKeypoints, "init": RigCameras, "rig_layout": RigLayout}
    outputs: Mapping[str, PortType] = {
        "refined": RigCameras, "base_rig": BaseCalibration, "diag": RefinementDiagnostics, "kp3d": Keypoints3dTracks, "kp2d_overlay": Kp2dOverlay,
    }  # fmt: skip

    def __init__(self, device: Literal["cuda", "cpu"] = "cuda") -> None:
        self.device: Literal["cuda", "cpu"] = device
        """Where the focal stage runs."""

    def reset(self, ctx: RunContext) -> None:
        self.frames, self.moge = cast("SegmentFrames", ctx.resources["frames"]), cast("MoGeV2MetricPredictor", ctx.resources["moge"])

    def run(self, inputs: Mapping[str, Sequence[Event]], out: Emit) -> None:
        records: list[MultiViewKeypoints] = [e.value for e in inputs["kp2d"]]  # type: ignore[misc]
        times: list[int] = [e.t_ns for e in inputs["kp2d"]]
        layout, init = inputs["rig_layout"][-1].value, inputs["init"][-1].value
        assert isinstance(layout, RigLayout) and isinstance(init, RigCameras) and records, "refine needs Stage B records, the Stage A rig and the layout"
        names: tuple[str, ...] = records[0].views
        if len(names) < 2:
            raise ValueError(f"extrinsic refinement needs at least two exo cameras; found {list(names)}")
        init = init.select(names)
        gated: GatedKeypoints = stack_postprocessed(records)
        kp_xy: Float64[ndarray, "v t 133 2"] = gated.xy
        conf: Float64[ndarray, "v t 133"] = drop_face_confidences(gated.conf)
        print(f"{kp_xy.shape[1]} frames x {len(names)} views; mean post conf {conf[conf > 0].mean():.3f}")

        candidates: ObservationSet = build_observations(kp_xy, conf, min_pair_conf=MIN_PAIR_CONFIDENCE, min_views=MIN_VIEWS)
        robust: RobustTriangulationResult = triangulate_robust(candidates, init.intrinsics, init.cam_T_world, reproj_threshold_px=REPROJECTION_THRESHOLD_PX)
        points_init, obs = robust.points_xyz, robust.obs
        init_reprojection_px: float = mean_reprojection_error_px(obs, points_init, init.intrinsics, init.cam_T_world)
        print(f"robust triangulation kept {obs.obs_point_idx.size}/{candidates.obs_point_idx.size} observations; init reprojection {init_reprojection_px:.2f} px")
        fixed: tuple[int, ...] = under_supported_view_indices(obs.obs_view_idx, len(names), MINIMUM_VIEW_SUPPORT_RATIO)
        pairwise: Float64[ndarray, "v v"] = rig_geometry(init.cam_T_world).pairwise_distance_m
        prior = PosePrior(centers=camera_centers(init.cam_T_world), sigma_m=POSE_PRIOR_SIGMA_RATIO * float(np.median(pairwise[np.triu_indices(len(names), k=1)])))
        settings = dict(robust=ROBUST_KERNEL, robust_scale_px=ROBUST_SCALE_PX, max_iterations=BA_MAX_ITERATIONS, pose_prior=prior, fixed_pose_indices=fixed)
        ba: BaResult = refine_extrinsics(init.cam_T_world, init.intrinsics, obs, points_init, rounds=BA_ROUNDS, **settings)  # type: ignore[arg-type]
        # One log-focal scale per camera with poses and points fixed, then one more BA round.
        focal: FocalRefinementResult = refine_focals(
            init.intrinsics, ba.cam_T_world, ba.points_xyz, obs, iterations=FOCAL_ITERATIONS, learning_rate=FOCAL_LEARNING_RATE, robust_scale_px=FOCAL_ROBUST_SCALE_PX, device=self.device
        )
        focal_ba: BaResult = refine_extrinsics(ba.cam_T_world, focal.intrinsics, obs, ba.points_xyz, rounds=1, **settings)  # type: ignore[arg-type]
        reprojection_px: list[float] = [*ba.mean_reproj_px_per_round, *focal_ba.mean_reproj_px_per_round]
        cam_T_world, reverted = revert_large_pose_updates(focal_ba.cam_T_world, init.cam_T_world, maximum_rotation_deg=MAXIMUM_ROTATION_UPDATE_DEG)
        points_xyz: Float64[ndarray, "n 3"] = focal_ba.points_xyz.copy()
        if reverted:
            points_xyz = triangulate_points(obs, focal.intrinsics, cam_T_world)
            reprojection_px.append(mean_reprojection_error_px(obs, points_xyz, focal.intrinsics, cam_T_world))
        print(f"BA reprojection per round {np.round(reprojection_px, 3).tolist()} px; focal scales {np.round(focal.focal_scale, 5).tolist()}; reverted {list(reverted)}")

        def depth_at(view: int, indices: list[int]) -> list[Float64[ndarray, "h w"]]:
            K: ndarray = focal.intrinsics[view].astype(np.float32)
            return [self.moge(f.permute(1, 2, 0).contiguous().cpu().numpy(), K_33=K).depth_meters.astype(np.float64) for f in self.frames.frames_at(view, indices)]

        scale: float | None = estimate_metric_rescale(cam_T_world, obs, points_xyz, np.asarray([r.frame_index for r in records]), depth_at)
        cam_T_world[:, :3, 3] *= 1.0 if scale is None else scale
        points_xyz *= 1.0 if scale is None else scale

        refined = RigCameras(names, focal.intrinsics, cam_T_world, init.resolution_wh)
        out.put("refined", STATIC, refined)
        if uncalibrated := [name for name in names if name not in layout.calibrated_camera_names]:
            out.put("base_rig", STATIC, BaseCalibration(refined.select(uncalibrated)))
        counts: list[int] = np.bincount(obs.obs_view_idx, minlength=len(names)).tolist()
        out.put("diag", STATIC, RefinementDiagnostics(init_reprojection_px, reprojection_px, counts, 1.0 if scale is None else scale, focal.focal_scale.tolist()))
        rigs: list[tuple[RigCameras, Float64[ndarray, "n 3"]]] = [(init, points_init), (refined, points_xyz)]
        kp_per_view: Float64[ndarray, "v n 2"] = np.stack([kp_xy[v, obs.point_frame_idx, obs.point_joint_idx] for v in range(len(names))])
        conf_per_view: Float64[ndarray, "v n"] = np.stack([conf[v, obs.point_frame_idx, obs.point_joint_idx] for v in range(len(names))])
        point_conf: Float64[ndarray, "2 n"] = np.stack([
            kineo_point_confidence(p, kp_per_view, conf_per_view, rig.intrinsics, rig.cam_T_world, lambda_decay=LAMBDA_DECAY) for rig, p in rigs
        ])
        for variant, p, c in zip(VARIANTS, (points_init, points_xyz), point_conf, strict=True):
            print(f"{variant}: {int(np.isfinite(p).all(axis=1).sum())} points, mean 3D confidence {c[np.isfinite(p).all(axis=1)].mean():.3f}")
        for frame in np.unique(obs.point_frame_idx).tolist():
            rows = np.flatnonzero(obs.point_frame_idx == frame)
            out.put("kp3d", times[frame], Keypoints3dTracks(obs.point_joint_idx[rows], np.stack((points_init[rows], points_xyz[rows])), point_conf[:, rows]))
        # Each camera's drawing at its own sample time, in time order over the whole segment (a camera may sample first).
        draws: list[tuple[int, int, Kp2dOverlay]] = []
        for t, r in enumerate(records):
            for v, view in enumerate(names):
                skeleton: bool = t % OVERLAY_LOG_STRIDE == 0
                crop: Float32[ndarray, "4"] = np.concatenate((r.crop_origin_xy[v], r.crop_size_wh[v]))
                overlay = Kp2dOverlay(
                    view, r.bbox_xyxy[v], r.box_source[v], crop, r.kp_xy[v], r.conf[v], gated.xy[v, t] if skeleton else None, gated.conf[v, t] if skeleton else None
                )
                draws.append((int(r.cam_t_ns[v]), v, overlay))
        for t_ns, _, overlay in sorted(draws, key=lambda draw: draw[:2]):
            out.put("kp2d_overlay", t_ns, overlay)
