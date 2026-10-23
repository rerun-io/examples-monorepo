"""Stage C+D (whole segment): the rig refined through its lenses (``refine_through_lenses``), then made metric.

Applies the AssemblyHands-X post-processing to the Stage B records and refines every pose and one focal scale of the Stage A
rig from them (Kineo-style correspondences, robust triangulation, ``refine_rig``). Then a metric rescale that fuses MoGe-2
depth (resources ``frames``: ``SegmentFrames``, ``moge``: ``MoGeV2MetricPredictor``) with the person's arm bones
(``metric_rescale``). The node draws the 3D joints and every camera's Stage B overlay, each cleared at the next frame that
does not draw it again.
"""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
from dagkit.node import Emit, Event, PortType, RunContext
from dagkit.time import STATIC
from jaxtyping import Float32, Float64, Int64, UInt8
from numpy import ndarray
from simplecv.ops.framesets import clear_times
from simplecv.types.rig import RigCameras

from exo_calib.ops.bundle_adjust import LensRefinement, refine_through_lenses
from exo_calib.ops.confidence import drop_face_confidences, kineo_point_confidence
from exo_calib.ops.correspondences import observations_at
from exo_calib.ops.keypoints import GatedKeypoints, stack_postprocessed, undistort_views
from exo_calib.ops.lens import rectify
from exo_calib.ops.metric_scale import MetricRescale, metric_rescale
from exo_calib.ops.triangulation import triangulate_points
from exo_calib.types import (
    VARIANTS,
    BaseCalibration,
    Keypoints3dTracks,
    Kp2dOverlay,
    Lens,
    MultiViewKeypoints,
    RefinementDiagnostics,
    RigLayout,
)

if TYPE_CHECKING:
    from monopriors.models.metric_depth.moge_v2 import MoGeV2MetricPredictor
    from simplecv.sources.catalog_video import SegmentFrames

LAMBDA_DECAY: float = 1.0
"""Kineo exponential decay for the logged 3D point confidence."""
OVERLAY_LOG_STRIDE: int = 5
"""Every N-th frame gets the gated / rejected keypoint skeleton, the others clear it (every frame gets its boxes and raw joints)."""


class RefineNode:
    """Refine the poses and a shared focal."""

    inputs: Mapping[str, PortType] = {"kp2d": MultiViewKeypoints, "init": RigCameras}
    outputs: Mapping[str, PortType] = {
        "refined": RigCameras, "base_rig": BaseCalibration, "diag": RefinementDiagnostics, "kp3d": Keypoints3dTracks, "kp2d_overlay": Kp2dOverlay,
    }  # fmt: skip

    def __init__(self, layout: RigLayout, device: Literal["cuda", "cpu"] = "cuda") -> None:
        self.layout: RigLayout = layout
        """The segment's cameras: a refined camera without GT calibration also goes out as ``base_rig``."""
        self.device: Literal["cuda", "cpu"] = device
        """Where the refinement runs."""

    def reset(self, ctx: RunContext) -> None:
        self.frames, self.moge = cast("SegmentFrames", ctx.resources["frames"]), cast("MoGeV2MetricPredictor", ctx.resources["moge"])

    def run(self, inputs: Mapping[str, Sequence[Event]], out: Emit) -> None:
        records: list[MultiViewKeypoints] = [e.value for e in inputs["kp2d"]]  # type: ignore[misc]
        times: list[int] = [e.t_ns for e in inputs["kp2d"]]
        init = inputs["init"][-1].value
        assert isinstance(init, RigCameras) and records, "refine needs Stage B records and the Stage A rig"
        names: tuple[str, ...] = records[0].views
        if len(names) < 2:
            raise ValueError(f"extrinsic refinement needs at least two exo cameras; found {list(names)}")
        init = init.select(names)
        # Each camera's own video frame of every record: cameras on one clock that start apart (SelfCap's phones) pair by
        # time, so a record's frame_index is the first camera's only.
        video_frames: Int64[ndarray, "v t"] = np.stack(
            [np.searchsorted(self.frames.times_ns(v), [int(r.cam_t_ns[v]) for r in records], side="right") - 1 for v in range(len(names))]
        )
        gated: GatedKeypoints = stack_postprocessed(records)
        conf: Float64[ndarray, "v t 133"] = drop_face_confidences(gated.conf)
        lenses: list[Lens] = [self.layout.lens(name) for name in names]
        print(f"{gated.xy.shape[1]} frames x {len(names)} views; mean post conf {conf[conf > 0].mean():.3f}; lenses {[lens.model for lens in lenses]}")
        fit: LensRefinement = refine_through_lenses(lenses, gated.xy, conf, init, device=self.device)
        K, obs, xy = fit.intrinsics, fit.obs, fit.xy
        init_xy: Float64[ndarray, "v t 133 2"] = undistort_views(lenses, gated.xy, init.intrinsics)
        points_init: Float64[ndarray, "n 3"] = triangulate_points(observations_at(init_xy, obs), init.intrinsics, init.cam_T_world)

        def depth_at(view: int, indices: list[int]) -> list[Float64[ndarray, "h w"]]:
            """MoGe-2 metric depth of the frames rectified through the camera's lens (MoGe is a pinhole model: on raw GoPro
            frames it read 16-30 % far), so it lines up with the undistorted pixels of ``obs``."""
            frames: UInt8[ndarray, "k h w 3"] = rectify(lenses[view], self.frames.frames_at(view, indices).permute(0, 2, 3, 1).contiguous().cpu().numpy(), K[view])
            return [self.moge(frame, K_33=K[view].astype(np.float32)).depth_meters.astype(np.float64) for frame in frames]

        rescale: MetricRescale = metric_rescale(xy, conf, K, fit.cam_T_world, obs, fit.points_xyz, video_frames, depth_at)
        cam_T_world: Float64[ndarray, "v 4 4"] = fit.cam_T_world.copy()
        cam_T_world[:, :3, 3] *= rescale.factor
        points_xyz: Float64[ndarray, "n 3"] = fit.points_xyz * rescale.factor

        refined = RigCameras(names, K, cam_T_world, init.resolution_wh)
        out.put("refined", STATIC, refined)
        if uncalibrated := [name for name in names if name not in self.layout.calibrated_camera_names]:
            out.put("base_rig", STATIC, BaseCalibration(refined.select(uncalibrated)))
        counts: list[int] = np.bincount(obs.obs_view_idx, minlength=len(names)).tolist()
        focal_scale: list[float] = (K[:, 0, 0] / init.intrinsics[:, 0, 0]).tolist()
        out.put("diag", STATIC, RefinementDiagnostics(fit.init_reprojection_px, fit.ba_reprojection_px, counts, rescale.factor, focal_scale, rescale.depth, rescale.body))
        conf_per_view: Float64[ndarray, "v n"] = np.stack([conf[v, obs.point_frame_idx, obs.point_joint_idx] for v in range(len(names))])
        point_conf: Float64[ndarray, "2 n"] = np.stack([
            kineo_point_confidence(p, grid[:, obs.point_frame_idx, obs.point_joint_idx], conf_per_view, rig.intrinsics, rig.cam_T_world, lambda_decay=LAMBDA_DECAY)
            for rig, p, grid in ((init, points_init, init_xy), (refined, points_xyz, xy))
        ])
        for variant, p, c in zip(VARIANTS, (points_init, points_xyz), point_conf, strict=True):
            print(f"{variant}: {int(np.isfinite(p).all(axis=1).sum())} points, mean 3D confidence {c[np.isfinite(p).all(axis=1)].mean():.3f}")
        triangulated: Int64[ndarray, " f"] = np.unique(obs.point_frame_idx)  # only a frameset with 3D joints replaces them
        kp3d_clears: list[int | None] = clear_times(self.frames.times_ns(0), video_frames[0, triangulated])  # on the first camera's clock
        for frame, clear_ns in zip(triangulated.tolist(), kp3d_clears, strict=True):
            rows = np.flatnonzero(obs.point_frame_idx == frame)
            out.put("kp3d", times[frame], Keypoints3dTracks(obs.point_joint_idx[rows], np.stack((points_init[rows], points_xyz[rows])), point_conf[:, rows]))
            if clear_ns is not None:
                out.put("kp3d", clear_ns, Keypoints3dTracks.empty())
        # Each camera's drawing at its own sample time and its clear at the next video frame, in time order over the whole
        # segment (a camera may sample first).
        overlay_clears: list[list[int | None]] = [clear_times(self.frames.times_ns(v), video_frames[v]) for v in range(len(names))]  # every record draws in every view
        draws: list[tuple[int, int, Kp2dOverlay]] = []
        for t, r in enumerate(records):
            for v, view in enumerate(names):
                skeleton: bool = t % OVERLAY_LOG_STRIDE == 0
                crop: Float32[ndarray, "4"] = np.concatenate((r.crop_origin_xy[v], r.crop_size_wh[v]))
                overlay = Kp2dOverlay(
                    view, r.bbox_xyxy[v], r.box_source[v], crop, r.kp_xy[v], r.conf[v], gated.xy[v, t] if skeleton else None, gated.conf[v, t] if skeleton else None
                )
                draws.append((int(r.cam_t_ns[v]), v, overlay))
                if (clear_ns := overlay_clears[v][t]) is not None:
                    draws.append((clear_ns, v, Kp2dOverlay.nothing(view)))
        for t_ns, _, overlay in sorted(draws, key=lambda draw: draw[:2]):
            out.put("kp2d_overlay", t_ns, overlay)
