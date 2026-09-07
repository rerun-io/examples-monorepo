"""Rerun logging for the estimator: three trajectories, the keyframe window, the landmarks and the counters.

All Rerun logging is Python (D03), so the core returns arrays and this module
decides what they look like. The estimate is logged under ``/world/runs/slam_rs``,
beside the dataset's own ``/world/runs/gt``, and the basalt C++ reference under
``/world/runs/basalt_cpp``: the three trajectories are then one 3D view with
three colours, and the viewer's own entity tree says which is which.

The comparison is the point of the rung. Ground truth and the C++ trajectory are
both known before the replay starts, so they are drawn **up to the cursor** just
as the estimate is: at any time in the timeline the three lines have seen exactly
the same interval, which is what makes a divergence readable rather than a matter
of where the eye starts. That costs one re-logged strip per frameset, quadratic
in the frameset count, so each of the three is drawn at the frameset cadence
(:func:`at_frameset_cadence`): the ground truth runs at 917 Hz against 54 Hz of
framesets, and re-logging it whole cost 17.20 MB of the smoke recording's 54.13
MB of rows — 17x the estimate's own strip — to draw a line no viewer can
resolve. Thinned, the three strips are 1.04 MB each over the 412-frameset smoke
segment and about 100 MB each over a 4,000-frameset one, so a long segment still
wants ``--max-framesets``.

The three do not start in one frame. basalt initialises its world at the identity
with gravity along z, while the ground truth is in the capture rig's own frame,
so an unaligned overlay puts the estimate metres away from the truth it is being
compared with. The run and the C++ reference therefore carry a
:class:`rerun.Transform3D` — the rigid alignment onto the ground truth, the same
one the ATE reports — and everything under them (the trajectory, the rig, the
window and the landmarks) is logged in the estimator's own frame and drawn in the
dataset's. The alignment is refreshed every :data:`ATE_EVERY` framesets and is
the identity until enough poses have been associated, so the run visibly settles
into place over the first second.

The keyframe window is drawn as frustum wireframes rather than
:class:`rerun.Pinhole` frusta: a ``Pinhole`` carries no colour, and colour is how
a keyframe, a demoted pose block and the frames the last marginalization removed
are told apart. The estimated rig's own cameras are real ``Pinhole`` frusta,
because there the camera is what is being drawn.
"""

from dataclasses import dataclass, field

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import Bool, Float64, Int64, UInt8
from numpy import ndarray
from scipy.spatial.transform import Rotation
from simplecv.ops.umeyama import SimilarityTransform

from slam_rs import _core
from slam_rs.catalog_feed import CameraCalib
from slam_rs.frontend_log import KEYPOINT_RADIUS_PX, camera_entity, track_colors
from slam_rs.trajectory import MIN_ASSOCIATED_POSES, Association, AteResult, Trajectory, associate, ate, rigid_alignment

RUN_ENTITY: str = "/world/runs/slam_rs"
"""Where this run's estimate goes, beside the dataset's own ``/world/runs/gt``."""
GT_ENTITY: str = "/world/runs/gt"
"""The dataset's own ground-truth run, which the base recording already names."""
CPP_ENTITY: str = "/world/runs/basalt_cpp"
"""The basalt C++ reference trajectory for the same segment."""
STATS_ENTITY: str = "/stats/vio"
"""Where the per-frame counters go, off the dataset's own tree and beside the frontend's."""

ESTIMATE_COLOR: tuple[int, int, int] = (70, 220, 130)
"""The port's own trajectory: green."""
GT_COLOR: tuple[int, int, int] = (235, 235, 235)
"""Ground truth: near-white, the reference every error is measured against."""
CPP_COLOR: tuple[int, int, int] = (255, 150, 40)
"""The basalt C++ trajectory: orange."""
KEYFRAME_COLOR: tuple[int, int, int, int] = (90, 200, 255, 255)
"""A keyframe still inside the window."""
LTKF_COLOR: tuple[int, int, int, int] = (255, 235, 90, 255)
"""A long-term keyframe, which the pose budget never evicts."""
POSE_COLOR: tuple[int, int, int, int] = (150, 150, 165, 200)
"""A window frame that is not a keyframe: a state awaiting its vote, or a demoted pose block."""
MARGINALIZED_COLOR: tuple[int, int, int, int] = (255, 90, 90, 70)
"""A frame the last marginalization removed: the same wireframe, faded out."""

FRUSTUM_DEPTH_M: float = 0.08
"""How far a window frame's wireframe extends, metres. An orientation marker, not a claim about range."""
ATE_EVERY: int = 30
"""Framesets between two ATE-so-far points: about one a second, and each costs a rigid alignment."""

IDENTITY: SimilarityTransform = SimilarityTransform(dst_R_src=np.eye(3), dst_t_src=np.zeros(3), scale=1.0)
"""The alignment a run carries before enough of it has been associated with the ground truth."""


def alignment_onto(source: Trajectory, target: Trajectory) -> SimilarityTransform:
    """The rigid transform taking one trajectory into another's frame.

    The association is driven by ``source`` — each of its poses takes the nearest
    ``target`` pose within the tolerance — which is the convention the reference
    manifest's own numbers were produced with.

    Args:
        source: Trajectory to move, e.g. the estimate.
        target: Trajectory whose frame to move it into, e.g. the ground truth.

    Returns:
        The alignment, or the identity when too few poses associate for one to
        mean anything.
    """
    if len(source) == 0 or len(target) == 0:
        return IDENTITY
    association: Association = associate(source, target)
    if association.count < MIN_ASSOCIATED_POSES:
        return IDENTITY
    return rigid_alignment(
        source.position_m[association.matched],
        target.position_m[association.candidate_index[association.matched]],
    )


def at_frameset_cadence(trajectory: Trajectory, frame_t_ns: Int64[ndarray, " n_frames"]) -> Trajectory:
    """Thin a reference to one pose per frameset: the last one at or before each frame time.

    Only what is **drawn** is thinned. The ATE and the alignment are computed
    against the whole reference, because there the truth's own density is the
    thing being measured against; the strip is re-logged once per frameset, so
    every pose in it is paid for a second time on every later frameset.

    Args:
        trajectory: The reference to thin; an empty one comes back unchanged.
        frame_t_ns: The segment's frameset times, ascending.

    Returns:
        The reference's poses at the frameset cadence, in time order.
    """
    if len(trajectory) == 0:
        return trajectory
    latest: Int64[ndarray, " n_frames"] = np.searchsorted(trajectory.t_ns, frame_t_ns, side="right") - 1
    keep: Int64[ndarray, " n_kept"] = np.unique(latest[latest >= 0])
    return Trajectory(
        t_ns=trajectory.t_ns[keep],
        position_m=trajectory.position_m[keep],
        quaternion_wxyz=trajectory.quaternion_wxyz[keep],
    )


def log_alignment(entity: str, alignment: SimilarityTransform) -> None:
    """Place one run's whole subtree in the frame its alignment maps into."""
    rr.log(entity, rr.Transform3D(translation=alignment.dst_t_src, mat3x3=alignment.dst_R_src))


def frustum_strip(camera: CameraCalib, depth_m: float = FRUSTUM_DEPTH_M) -> Float64[ndarray, "10 3"]:
    """One camera's frustum wireframe in rig coordinates, as a single line strip.

    Ten points: the four image-plane corners closed into a rectangle, then the
    four rays back to the apex. Tracing it as one strip repeats the corner-to-corner
    edge once, which is invisible and cheaper than eight separate segments.

    The corners are the pinhole rays through the calibrated intrinsics. On a
    fisheye that under-states the real field of view — the wireframe says where
    the camera is and where it looks, not what it can see.

    Args:
        camera: The camera to draw, with its ``imu_T_cam``.
        depth_m: How far the wireframe extends along the optical axis.

    Returns:
        Ten points in the rig (IMU) frame, in strip order.
    """
    corners_px: Float64[ndarray, "4 2"] = np.array(
        [[0.0, 0.0], [camera.width, 0.0], [camera.width, camera.height], [0.0, camera.height]], dtype=np.float64
    )
    corners_cam: Float64[ndarray, "4 3"] = depth_m * np.column_stack(
        [(corners_px[:, 0] - camera.cx) / camera.fx, (corners_px[:, 1] - camera.cy) / camera.fy, np.ones(4)]
    )
    apex: Float64[ndarray, " 3"] = np.zeros(3)
    strip_cam: Float64[ndarray, "10 3"] = np.array(
        [corners_cam[0], corners_cam[1], corners_cam[2], corners_cam[3], corners_cam[0], apex, corners_cam[1], corners_cam[2], apex, corners_cam[3]]
    )
    return strip_cam @ camera.imu_T_cam[:3, :3].T + camera.imu_T_cam[:3, 3]


def log_run_rig(cameras: tuple[CameraCalib, ...], entity: str = RUN_ENTITY) -> None:
    """Log the static camera geometry of an estimated rig, so its pose draws as frusta.

    The dataset's own rig already carries this under ``/world/rig_00``; an
    estimated run needs its own copy under its own path, or the estimate and the
    ground truth would move the same frusta.

    Args:
        cameras: The rig's cameras, in rig order.
        entity: Run entity the rig hangs under.
    """
    for camera in cameras:
        rr.log(
            f"{entity}/rig/cam_{camera.index:02d}",
            rr.Transform3D(translation=camera.imu_T_cam[:3, 3], mat3x3=camera.imu_T_cam[:3, :3]),
            static=True,
        )
        image_from_camera: Float64[ndarray, "3 3"] = np.array(
            [[camera.fx, 0.0, camera.cx], [0.0, camera.fy, camera.cy], [0.0, 0.0, 1.0]], dtype=np.float64
        )
        rr.log(
            f"{entity}/rig/cam_{camera.index:02d}",
            rr.Pinhole(image_from_camera=image_from_camera, resolution=[camera.width, camera.height], camera_xyz=rr.ViewCoordinates.RDF),
            static=True,
        )


@dataclass(slots=True)
class VioLogger:
    """Logs one frameset's estimator output, and carries the paths between them.

    The two reference trajectories are handed over whole, on the same
    ``video_time`` clock the replay logs on; the estimate accumulates as it is
    produced. Every ``log`` call writes at the caller's time cursor and nowhere
    else.
    """

    cameras: tuple[CameraCalib, ...]
    """The rig's cameras, in rig order."""
    ground_truth: Trajectory
    """Ground truth for the whole segment, on ``video_time``; may be empty."""
    cpp: Trajectory
    """The basalt C++ trajectory for the whole segment, on ``video_time``; may be empty."""
    frame_t_ns: Int64[ndarray, " n_frames"]
    """The segment's frameset times: the cadence the two references are drawn at."""
    estimate_t_ns: list[int] = field(default_factory=list)
    """Timestamps of the poses reported so far, in replay order."""
    estimate_position_m: list[Float64[ndarray, " 3"]] = field(default_factory=list)
    """Positions of the poses reported so far."""
    estimate_quaternion_wxyz: list[Float64[ndarray, " 4"]] = field(default_factory=list)
    """Rotations of the poses reported so far, w-first as :mod:`slam_rs.trajectory` stores them."""
    window_strip: Float64[ndarray, "10 3"] = field(init=False)
    """Camera 0's frustum wireframe in rig coordinates, drawn at every window pose."""
    ground_truth_strip: Trajectory = field(init=False)
    """The ground truth thinned to the frameset cadence: what the drawn strip is taken from."""
    cpp_strip: Trajectory = field(init=False)
    """The C++ trajectory thinned the same way."""
    previous_strips: dict[int, Float64[ndarray, " 10 3"]] = field(default_factory=dict)
    """The last frameset's window wireframes by timestamp: where a marginalized frame is drawn from."""
    framesets: int = 0
    """Framesets logged, which paces the ATE-so-far."""

    def __post_init__(self) -> None:
        """Log the estimated rig's static geometry, precompute the window wireframe and thin the references."""
        log_run_rig(self.cameras)
        self.window_strip = frustum_strip(self.cameras[0])
        self.ground_truth_strip = at_frameset_cadence(self.ground_truth, self.frame_t_ns)
        self.cpp_strip = at_frameset_cadence(self.cpp, self.frame_t_ns)

    def log(self, result: _core.VioResult, snapshot: _core.VioSnapshot, frame: _core.FlowFrame, elapsed_ms: float) -> None:
        """Log one tracked frameset: the keypoints, the three paths, the rig, the window, the landmarks and the counters.

        Args:
            result: What ``track`` returned; only called where it tracked.
            snapshot: The window and the frame's statistics behind that result.
            frame: The keypoints the estimator's own frontend tracked.
            elapsed_ms: Wall time the ``track`` call took.
        """
        self.framesets += 1
        if self.framesets == 1:
            # Both the C++ trajectory and the ground truth are known before the
            # replay starts, so this alignment is a constant of the run; it is
            # written here rather than at construction because a row needs the
            # caller's time cursor. The run's own alignment has no row until the
            # first ATE, and a missing transform is the identity.
            log_alignment(CPP_ENTITY, alignment_onto(self.cpp, self.ground_truth))
        pose: Float64[ndarray, " 7"] = result.world_from_rig
        self.estimate_t_ns.append(result.t_ns)
        self.estimate_position_m.append(pose[0:3].copy())
        self.estimate_quaternion_wxyz.append(np.roll(pose[3:7], 1).copy())

        estimated: Trajectory = self.estimated()
        rr.log(f"{RUN_ENTITY}/rig", rr.Transform3D(translation=pose[0:3], quaternion=rr.Quaternion(xyzw=pose[3:7])))
        self._log_keypoints(frame)
        self._log_paths(estimated)
        self._log_window(snapshot)
        self._log_landmarks(snapshot)
        self._log_scalars(result, snapshot, elapsed_ms)
        if self.framesets % ATE_EVERY == 0:
            self._log_ate(estimated)
            log_alignment(RUN_ENTITY, alignment_onto(estimated, self.ground_truth))

    def estimated(self) -> Trajectory:
        """Everything reported so far, on the replay's ``video_time`` clock."""
        return Trajectory(
            t_ns=np.array(self.estimate_t_ns, dtype=np.int64),
            position_m=np.array(self.estimate_position_m, dtype=np.float64).reshape(-1, 3),
            quaternion_wxyz=np.array(self.estimate_quaternion_wxyz, dtype=np.float64).reshape(-1, 4),
        )

    def _log_keypoints(self, frame: _core.FlowFrame) -> None:
        """Draw the estimator's own tracked keypoints on the camera images.

        The palette and the entity paths are the frontend rung's
        (:mod:`slam_rs.frontend_log`), so a keypoint has the same colour in both
        recordings and the two views can be read side by side. The trails, the
        occupancy grid and the C++ overlay stay there: they are the frontend's
        evidence, and this rung's is the trajectory.
        """
        for index in range(frame.camera_count):
            ids: Int64[ndarray, " n_tracks"] = frame.ids(index)
            rr.log(
                f"{camera_entity(index)}/keypoints",
                rr.Points2D(frame.positions(index), colors=track_colors(ids), radii=KEYPOINT_RADIUS_PX),
            )

    def _log_paths(self, estimated: Trajectory) -> None:
        """Draw the three trajectories, each up to the current cursor.

        Each is logged in its own frame; the run entities' alignment transforms
        are what bring the three together in the dataset's world.

        Args:
            estimated: Everything reported so far, the newest pose last.
        """
        rr.log(f"{RUN_ENTITY}/trajectory", rr.LineStrips3D([estimated.position_m], colors=ESTIMATE_COLOR, radii=0.004))
        t_ns: int = int(estimated.t_ns[-1])
        for entity, trajectory, color in ((GT_ENTITY, self.ground_truth_strip, GT_COLOR), (CPP_ENTITY, self.cpp_strip, CPP_COLOR)):
            if len(trajectory) == 0:
                continue
            drawn: int = int(np.searchsorted(trajectory.t_ns, t_ns, side="right"))
            rr.log(f"{entity}/trajectory", rr.LineStrips3D([trajectory.position_m[:drawn]], colors=color, radii=0.004))

    def _log_window(self, snapshot: _core.VioSnapshot) -> None:
        """Draw a frustum wireframe at every window pose, coloured by what the frame is."""
        poses: Float64[ndarray, "n_frames 7"] = snapshot.window_poses
        # A measured frameset always leaves at least its own state in the window,
        # so this is never empty; the reshape is for the one-frame case, where
        # scipy drops the batch axis.
        rotations: Float64[ndarray, "n_frames 3 3"] = Rotation.from_quat(poses[:, 3:7]).as_matrix().reshape(-1, 3, 3)
        strips: list[Float64[ndarray, "10 3"]] = [
            self.window_strip @ rotation.T + translation for rotation, translation in zip(rotations, poses[:, 0:3], strict=True)
        ]
        t_ns: Int64[ndarray, " n_frames"] = snapshot.window_t_ns
        keyframe: Bool[ndarray, " n_frames"] = np.isin(t_ns, snapshot.kf_ids)
        long_term: Bool[ndarray, " n_frames"] = np.isin(t_ns, snapshot.ltkfs)
        colors: UInt8[ndarray, "n_frames 4"] = np.where(keyframe[:, None], KEYFRAME_COLOR, POSE_COLOR).astype(np.uint8)
        colors[long_term] = LTKF_COLOR
        rr.log(f"{RUN_ENTITY}/window", rr.LineStrips3D(strips, colors=colors, radii=0.001))

        # The frames that left this step are gone from the window above, so they
        # are drawn from the poses they held when it was taken: the previous
        # frameset's window, the last snapshot that still had them.
        leaving: set[int] = set(snapshot.marginalized.tolist())
        rr.log(
            f"{RUN_ENTITY}/marginalized",
            rr.LineStrips3D(
                [strip for held_t_ns, strip in self.previous_strips.items() if held_t_ns in leaving],
                colors=MARGINALIZED_COLOR,
                radii=0.001,
            ),
        )
        self.previous_strips = dict(zip(t_ns.tolist(), strips, strict=True))

    def _log_landmarks(self, snapshot: _core.VioSnapshot) -> None:
        """Draw the window's landmarks, coloured by the keyframe that hosts them."""
        rr.log(
            f"{RUN_ENTITY}/landmarks",
            rr.Points3D(snapshot.landmark_positions, colors=track_colors(snapshot.landmark_hosts), radii=0.008),
        )

    def _log_scalars(self, result: _core.VioResult, snapshot: _core.VioSnapshot, elapsed_ms: float) -> None:
        """Log the counters the time-series view plots."""
        rr.log(f"{RUN_ENTITY}/velocity", rr.Scalars(result.velocity))
        rr.log(f"{RUN_ENTITY}/gyro_bias", rr.Scalars(result.gyro_bias))
        rr.log(f"{RUN_ENTITY}/accel_bias", rr.Scalars(result.accel_bias))
        rr.log(f"{STATS_ENTITY}/num_landmarks", rr.Scalars(float(len(snapshot.landmark_ids))))
        rr.log(f"{STATS_ENTITY}/num_observations", rr.Scalars(float(snapshot.num_observations)))
        rr.log(f"{STATS_ENTITY}/num_keyframes", rr.Scalars(float(len(snapshot.kf_ids))))
        rr.log(f"{STATS_ENTITY}/lm_iterations", rr.Scalars(float(snapshot.lm_iterations)))
        rr.log(f"{STATS_ENTITY}/lm_lambda", rr.Scalars(snapshot.lm_lambda))
        rr.log(f"{STATS_ENTITY}/track_ms", rr.Scalars(elapsed_ms))
        for stage, milliseconds in snapshot.timings_ms.items():
            rr.log(f"{STATS_ENTITY}/stage_ms/{stage}", rr.Scalars(milliseconds))

    def _log_ate(self, estimated: Trajectory) -> None:
        """Log the rigid-aligned error of everything reported so far, against both references.

        Args:
            estimated: Everything reported so far.
        """
        for name, reference in (("gt", self.ground_truth), ("cpp", self.cpp)):
            if len(reference) == 0 or len(estimated) < MIN_ASSOCIATED_POSES:
                continue
            result: AteResult = ate(estimated, reference)
            if result.n_associated >= MIN_ASSOCIATED_POSES:
                rr.log(f"{STATS_ENTITY}/ate_cm/{name}", rr.Scalars(100.0 * result.rmse_m))


def vio_blueprint(cameras: tuple[CameraCalib, ...]) -> rrb.Blueprint:
    """One 3D view of the world, the camera images beside it, and the counters below.

    Args:
        cameras: The rig's cameras, in rig order.

    Returns:
        A blueprint with the panels collapsed, so the frame is all content.
    """
    views: list[rrb.View] = [rrb.Spatial2DView(origin=camera_entity(camera.index), name=f"cam {camera.index:02d}") for camera in cameras]
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial3DView(origin="/world", name="world"),
                rrb.Vertical(*views),
                column_shares=[2, 1],
            ),
            rrb.TimeSeriesView(origin=STATS_ENTITY, name="estimator"),
            row_shares=[3, 1],
        ),
        collapse_panels=True,
    )
