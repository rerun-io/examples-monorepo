"""Rerun logging for the estimator: the input rung, three trajectories, the keyframe window, the landmarks and the counters.

All Rerun logging is Python (D03), so the core returns arrays and this module
decides what they look like. It holds the whole drawn account of a replay — the
rig's static geometry, the frames and the inertial samples the estimator is fed
(:func:`log_frameset_inputs`), the drive that tracks them (:class:`VioStage`)
and what the estimator decided — so the two tools that replay a segment draw one
rung rather than two copies of one, and ``apis/`` is CLIs over it. The estimate is logged under ``/world/runs/slam_rs``,
beside the dataset's own ``/world/runs/gt``, and the basalt C++ reference under
``/world/runs/basalt_cpp``: the three trajectories are then one 3D view with
three colours, and the viewer's own entity tree says which is which.

The comparison is the point of the rung. Ground truth and the C++ trajectory are
both known before the replay starts, so they are drawn **up to the cursor** just
as the estimate is: at any time in the timeline the three lines have seen exactly
the same interval, which is what makes a divergence readable rather than a matter
of where the eye starts. Each frameset logs the one segment its line gained, and
:func:`vio_blueprint` gives the three ``trajectory`` entities a visible time
range running from the start of the recording to the cursor, which is what turns
those segments back into the path so far — Rerun's default for a view that is
not a time series is latest-at, under which a two-point strip renders alone. The
rung is then linear in the frameset count, where re-logging each whole strip was
quadratic: the ground truth alone cost 17.20 MB of the smoke recording's 54.13 MB
of rows, and a 4,000-frameset segment paid about 100 MB a line.

The two references are still thinned to the frameset cadence
(:func:`at_frameset_cadence`), which is also what makes a segment a segment: the
ground truth runs at 917 Hz against 54 Hz of framesets, so an unthinned line
costs 17x the estimate's rows to draw what no viewer can resolve.
:meth:`VioLogger.log_complete_paths` puts each whole path in once, static, at the
end of a replay, so a viewer that opens the file anywhere still sees where each
run went.

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
from typing import Literal, TypeAlias

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import Float64, Int64, UInt8
from numpy import ndarray
from scipy.spatial.transform import Rotation
from simplecv.ops.umeyama import SimilarityTransform

from slam_rs import _core
from slam_rs.catalog_feed import IMU_ENTITY, RIG_ENTITY, TIMELINE, CameraCalib, Frameset, ImuStream, SegmentFeed
from slam_rs.frontend_log import camera_entity, camera_views, log_keypoints, track_colors
from slam_rs.tracking import Lockstep
from slam_rs.trajectory import MIN_ASSOCIATED_POSES, Association, AteResult, Trajectory, associate, ate, rigid_alignment

FrameMode: TypeAlias = Literal["downscaled", "jpeg", "off"]
"""How a replay's input rung draws the frames it was fed.

``downscaled`` is half resolution and uncompressed, which is enough to see what
a segment is; ``jpeg`` is full resolution, which is what a stage that tracks
needs, because its keypoints are in the pixels of the frame it tracked and not
of a downscaled copy; ``off`` logs no image at all, which is how a tool measures
the estimator without paying for the encode.
"""
IMAGE_DOWNSCALE: int = 2
"""Divisor of the ``downscaled`` mode: the viewer does not need full-resolution pixels to show what was fed."""
JPEG_QUALITY: int = 85
"""Quality of the full-resolution frames the tracking stages log; 960x960 grayscale lands around 38 kB."""

RUN_ENTITY: str = "/world/runs/slam_rs"
"""Where this run's estimate goes, beside the dataset's own ``/world/runs/gt``."""
GT_ENTITY: str = "/world/runs/gt"
"""The dataset's own ground-truth run, which the base recording already names."""
CPP_ENTITY: str = "/world/runs/basalt_cpp"
"""The basalt C++ reference trajectory for the same segment."""
VIO_STATS_ENTITY: str = "/stats/vio"
"""Where the per-frame counters go, off the dataset's own tree and beside the frontend's.

Named for the rung it belongs to, not for what it is: this module already imports
three names from :mod:`slam_rs.frontend_log`, which has its own ``STATS_ENTITY``
and its own ``CPP_COLOR`` with different values, and one unqualified import of
either would have been silently wrong.
"""

ESTIMATE_COLOR: tuple[int, int, int] = (70, 220, 130)
"""The port's own trajectory: green."""
GT_COLOR: tuple[int, int, int] = (235, 235, 235)
"""Ground truth: near-white, the reference every error is measured against."""
CPP_TRAJECTORY_COLOR: tuple[int, int, int] = (255, 150, 40)
"""The basalt C++ trajectory: orange, and not the frontend rung's magenta ``CPP_COLOR``."""
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
IMAGE_PLANE_M: float = 0.1
"""How far a rig camera's ``Pinhole`` frustum extends, metres: Rerun's default grows with the scene, so they changed size as landmarks came in."""
ATE_EVERY: int = 30
"""Framesets between two ATE-so-far points: about one a second, and each costs a rigid alignment."""
ROUTE_ALPHA: int = 40
"""How opaque a whole path is (:meth:`VioLogger.log_complete_paths`) beside the trail drawn up to the cursor."""
ROUTE_RADIUS_M: float = 0.0015
"""How thick a whole path is, metres: under half the trail's, so the two read as one line and its ghost."""

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
    thing being measured against; a line drawn one segment a frameset can only
    move by one pose a frameset, so the poses between them would never be drawn
    at all.

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


def inverted(alignment: SimilarityTransform) -> SimilarityTransform:
    """The rigid alignment the other way round.

    :func:`slam_rs.trajectory.ate` solves the reference onto the estimate, which
    is the direction its residuals are measured in; a run's subtree is drawn in
    the reference's frame, which is this direction. Rigid only — the scale is
    fixed at one, so the inverse is the transposed rotation.

    Args:
        alignment: A rigid alignment, ``scale == 1``.

    Returns:
        The alignment mapping the destination frame back into the source's.
    """
    return SimilarityTransform(
        dst_R_src=alignment.dst_R_src.T,
        dst_t_src=-alignment.dst_R_src.T @ alignment.dst_t_src,
        scale=1.0,
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


def log_rig(cameras: tuple[CameraCalib, ...], entity_prefix: str, pinhole_child: str = "") -> None:
    """Log a rig's static camera geometry, so whatever moves it draws as frusta.

    Two rigs are drawn from the same calibration and this is the one place that
    spells it: the dataset's own under ``/world/rig_00``, whose ``Pinhole`` sits
    on the ``pinhole`` child the images and keypoints hang under, and the
    estimated run's own copy under its run entity — without which the estimate
    and the ground truth would move the same frusta.

    Args:
        cameras: The rig's cameras, in rig order.
        entity_prefix: What ``cam_MM`` hangs under.
        pinhole_child: Child entity the ``Pinhole`` goes on; empty puts it on the
            camera itself, which is what a rig with no images wants.
    """
    for camera in cameras:
        node: str = f"{entity_prefix}/cam_{camera.index:02d}"
        rr.log(
            node,
            rr.Transform3D(translation=camera.imu_T_cam[:3, 3], mat3x3=camera.imu_T_cam[:3, :3]),
            static=True,
        )
        image_from_camera: Float64[ndarray, "3 3"] = np.array(
            [[camera.fx, 0.0, camera.cx], [0.0, camera.fy, camera.cy], [0.0, 0.0, 1.0]], dtype=np.float64
        )
        rr.log(
            f"{node}{pinhole_child}",
            rr.Pinhole(
                image_from_camera=image_from_camera,
                resolution=[camera.width, camera.height],
                camera_xyz=rr.ViewCoordinates.RDF,
                image_plane_distance=IMAGE_PLANE_M,
            ),
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
        log_rig(self.cameras, f"{RUN_ENTITY}/rig")
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

        rr.log(f"{RUN_ENTITY}/rig", rr.Transform3D(translation=pose[0:3], quaternion=rr.Quaternion(xyzw=pose[3:7])))
        # The estimator's own frontend output, on the frontend rung's paths and
        # in its palette (:func:`slam_rs.frontend_log.log_keypoints`).
        log_keypoints(frame)
        self._log_paths()
        self._log_window(snapshot)
        self._log_landmarks(snapshot)
        self._log_scalars(snapshot, elapsed_ms)
        if self.framesets % ATE_EVERY == 0:
            # The only two readers of the whole estimate, and the reason it is
            # not built every frameset: three arrays over lists that grow with
            # the run cost 1.64 s over a 4,648-frameset clip to draw a segment
            # from the last two poses.
            estimated: Trajectory = self.estimated()
            # The ATE against the ground truth has already solved this alignment,
            # in the direction its own residuals are measured in; drawing it is
            # that solution inverted, not a second association and a second SVD.
            against_truth: AteResult | None = self._log_ate(estimated)
            log_alignment(RUN_ENTITY, IDENTITY if against_truth is None else inverted(against_truth.alignment))

    def estimated(self) -> Trajectory:
        """Everything reported so far, on the replay's ``video_time`` clock."""
        return Trajectory(
            t_ns=np.array(self.estimate_t_ns, dtype=np.int64),
            position_m=np.array(self.estimate_position_m, dtype=np.float64).reshape(-1, 3),
            quaternion_wxyz=np.array(self.estimate_quaternion_wxyz, dtype=np.float64).reshape(-1, 4),
        )

    def log_complete_paths(self) -> None:
        """Log each whole path once, static, so all three are visible at every cursor.

        The per-frameset segments show where each run had got to; these show
        where it went. Static rows have no timestamp, so they cost one copy of
        each path however long the clip is.

        They are drawn faded and thin (:data:`ROUTE_ALPHA`,
        :data:`ROUTE_RADIUS_M`) for the same reason the frames the last
        marginalization removed are: two things in one place have to be told
        apart. At full colour and radius the route covers the trail exactly, and
        a cursor halfway through a clip looks like a cursor at the end of it.
        """
        for entity, trajectory, color in (
            (f"{RUN_ENTITY}/path", self.estimated(), ESTIMATE_COLOR),
            (f"{GT_ENTITY}/path", self.ground_truth_strip, GT_COLOR),
            (f"{CPP_ENTITY}/path", self.cpp_strip, CPP_TRAJECTORY_COLOR),
        ):
            if len(trajectory) < 2:
                continue
            rr.log(entity, rr.LineStrips3D([trajectory.position_m], colors=(*color, ROUTE_ALPHA), radii=ROUTE_RADIUS_M), static=True)

    def _log_paths(self) -> None:
        """Draw the segment each of the three trajectories gained at this cursor.

        One two-point strip a line a frameset, ending on the newest pose at or
        before the cursor: the view's visible time range (:func:`vio_blueprint`)
        is what accumulates them into the path so far, and a reference that has
        no second pose yet draws nothing. Each is logged in its own frame; the
        run entities' alignment transforms are what bring the three together in
        the dataset's world.

        The estimate's segment is taken from the two poses this run last
        appended, not from :meth:`estimated`: a segment needs two poses, not the
        whole trajectory rebuilt behind them.
        """
        t_ns: int = self.estimate_t_ns[-1]
        if len(self.estimate_position_m) >= 2:
            segment: Float64[ndarray, "2 3"] = np.array(self.estimate_position_m[-2:], dtype=np.float64)
            rr.log(f"{RUN_ENTITY}/trajectory", rr.LineStrips3D([segment], colors=ESTIMATE_COLOR, radii=0.004))
        for entity, trajectory, color in ((GT_ENTITY, self.ground_truth_strip, GT_COLOR), (CPP_ENTITY, self.cpp_strip, CPP_TRAJECTORY_COLOR)):
            drawn: int = int(np.searchsorted(trajectory.t_ns, t_ns, side="right"))
            if drawn >= 2:
                rr.log(f"{entity}/trajectory", rr.LineStrips3D([trajectory.position_m[drawn - 2 : drawn]], colors=color, radii=0.004))

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
        colors: UInt8[ndarray, "n_frames 4"] = np.where(snapshot.window_keyframe[:, None], KEYFRAME_COLOR, POSE_COLOR).astype(np.uint8)
        colors[snapshot.window_long_term] = LTKF_COLOR
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

    def _log_scalars(self, snapshot: _core.VioSnapshot, elapsed_ms: float) -> None:
        """Log the counters the time-series views plot.

        The estimator's velocity and its two biases used to be logged here, under
        the 3D subtree, where no view could reach them. Plotting them would take
        three more views and not one — metres a second, radians a second and
        metres a second squared share no axis, which is the mixing C73's split
        into seven views undid — so they are dropped instead; the pose they
        belong to is on the trajectory either way.
        """
        rr.log(f"{VIO_STATS_ENTITY}/num_landmarks", rr.Scalars(float(len(snapshot.landmark_ids))))
        rr.log(f"{VIO_STATS_ENTITY}/num_observations", rr.Scalars(float(snapshot.num_observations)))
        rr.log(f"{VIO_STATS_ENTITY}/num_keyframes", rr.Scalars(float(len(snapshot.kf_ids))))
        rr.log(f"{VIO_STATS_ENTITY}/lm_iterations", rr.Scalars(float(snapshot.lm_iterations)))
        rr.log(f"{VIO_STATS_ENTITY}/lm_lambda", rr.Scalars(snapshot.lm_lambda))
        # The cost the frame started and ended the LM loop at: the pair is the
        # convergence trace, and :func:`vio_blueprint` gives it an axis of its
        # own. Both are negative on most frames, which is basalt's own convention
        # and not a sign error: the marginalization prior term deliberately drops
        # the 1/2 r^T r (``crates/slam-rs/src/estimator/optimize.rs:71``, D20).
        rr.log(f"{VIO_STATS_ENTITY}/lm_error_before", rr.Scalars(snapshot.lm_error_before))
        rr.log(f"{VIO_STATS_ENTITY}/lm_error_after", rr.Scalars(snapshot.lm_error_after))
        rr.log(f"{VIO_STATS_ENTITY}/track_ms", rr.Scalars(elapsed_ms))
        for stage, milliseconds in snapshot.timings_ms.items():
            rr.log(f"{VIO_STATS_ENTITY}/stage_ms/{stage}", rr.Scalars(milliseconds))

    def _log_ate(self, estimated: Trajectory) -> AteResult | None:
        """Log the rigid-aligned error of everything reported so far, against both references.

        Args:
            estimated: Everything reported so far.

        Returns:
            The error against the ground truth, whose alignment is also what
            places the run's subtree, or None where too few poses associated for
            either number to mean anything.
        """
        scored: AteResult | None = None
        for name, reference in (("gt", self.ground_truth), ("cpp", self.cpp)):
            if len(reference) == 0 or len(estimated) < MIN_ASSOCIATED_POSES:
                continue
            result: AteResult = ate(estimated, reference)
            if result.n_associated >= MIN_ASSOCIATED_POSES:
                rr.log(f"{VIO_STATS_ENTITY}/ate_cm/{name}", rr.Scalars(100.0 * result.rmse_m))
                if name == "gt":
                    scored = result
        return scored


def log_calibration(cameras: tuple[CameraCalib, ...]) -> None:
    """Log the rig's static geometry so the images sit in the right place in 3D.

    The ``Pinhole`` goes on the ``pinhole`` child, which is the dataset's own
    layout: the images and the keypoints hang under it, so they are the pixels of
    the camera that projects them.
    """
    rr.log("/", rr.ViewCoordinates.RUB, static=True)
    log_rig(cameras, RIG_ENTITY, pinhole_child="/pinhole")


@dataclass(slots=True)
class VioStage:
    """The whole pipeline over one segment, and the Rerun layer it draws.

    The estimator, its logger and its timings only ever exist together, on
    ``--stage vio``.
    """

    lockstep: Lockstep
    """The estimator and the D17 hold, which the V2 gate drives the same way."""
    logger: VioLogger
    """Where the trajectories, the window, the landmarks and the counters go."""

    @property
    def elapsed_ms(self) -> list[float]:
        """Wall time each ``track`` call that tracked took, in frameset order."""
        return self.lockstep.elapsed_ms

    @property
    def pending(self) -> list[Frameset]:
        """Framesets still held for want of the inertial samples that cover them."""
        return self.lockstep.pending

    def run(self, frameset: Frameset) -> None:
        """Track the frameset and everything its samples now cover, and log each one.

        Args:
            frameset: The frameset to track, with the samples since the previous one.
        """
        for held, result in self.lockstep.push(frameset):
            # The rows belong at the frameset's own time, which is the caller's
            # cursor for all but a retried one.
            rr.set_time(TIMELINE, duration=np.timedelta64(held.t_ns, "ns"))
            # Both are present on a frameset that tracked — the snapshot because it
            # measured, the keypoints because the frontend accepted it — so a
            # missing one is a broken invariant, not a rung to skip (D32).
            snapshot: _core.VioSnapshot | None = self.lockstep.vio.snapshot()
            frame: _core.FlowFrame | None = self.lockstep.vio.flow_frame()
            assert snapshot is not None, f"frameset {held.t_ns} tracked without a window snapshot"
            assert frame is not None, f"frameset {held.t_ns} tracked without the keypoints it tracked on"
            self.logger.log(result, snapshot, frame, self.lockstep.elapsed_ms[-1])

    def refuse_lost_framesets(self) -> None:
        """Stop the run when a frameset never got the inertial samples that cover it.

        Every frameset either produced a pose or is still held (D17); one still
        held at the end of a segment is a lost frameset, not a count to print,
        and both tools that drive this stage end on the same rule.

        Raises:
            SystemExit: If any frameset is still held.
        """
        if self.pending:
            raise SystemExit(f"{len(self.pending)} framesets never got the inertial samples that cover them")

    def summary(self) -> str:
        """One line on what the stage did, for the end of a replay.

        The empty case is this stage's own to report: the run that tracked
        nothing is the one whose held framesets most need naming, and a caller
        that guarded the call on ``elapsed_ms`` suppressed exactly that line.
        """
        unresolved: str = ""
        if self.pending:
            unresolved = f", {len(self.pending)} FRAMESETS NEVER COVERED BY THE IMU at {[held.t_ns for held in self.pending]}"
        if not self.elapsed_ms:
            return f"vio: {self.lockstep.imu_samples} IMU samples pushed, nothing tracked{unresolved}"
        return (
            f"vio: {self.lockstep.imu_samples} IMU samples pushed, {len(self.elapsed_ms)} tracked, "
            f"{self.lockstep.retries} retries, {np.mean(self.elapsed_ms):.1f} ms per frameset "
            f"(median {np.median(self.elapsed_ms):.1f}, max {np.max(self.elapsed_ms):.1f})"
            f"{unresolved}"
        )


def log_imu(imu: ImuStream) -> None:
    """Log one frameset's inertial samples, one column per channel.

    Two ``send_columns`` calls instead of a ``set_time`` and two ``log`` calls per
    sample — about 57 samples a frameset at 1 kHz, so 171 calls become 2, and the
    frameset's inertial rung goes from 0.446 ms to 0.037 ms. The rows are the
    same rows: each timestamp still carries its three components, which is what
    the partition says.

    Args:
        imu: The samples since the previous frameset, on the ``video_time`` clock.
    """
    if not len(imu):
        return
    times: rr.TimeColumn = rr.TimeColumn(TIMELINE, duration=imu.t_ns.astype("timedelta64[ns]"))
    components: int = imu.gyro_rad_s.shape[1]
    for entity, channel in ((f"{IMU_ENTITY}/gyro", imu.gyro_rad_s), (f"{IMU_ENTITY}/accel", imu.accel_m_s2)):
        rr.send_columns(entity, indexes=[times], columns=rr.Scalars.columns(scalars=channel.reshape(-1)).partition([components] * len(imu)))


def log_frameset_inputs(feed: SegmentFeed, frameset: Frameset, mode: FrameMode) -> None:
    """Log what the estimator is fed for one frameset, at the frameset's own time.

    Every replay draws this rung whether or not a stage consumes it, and the two
    tools that drive a stage draw the same one: a channel added here reaches both
    rather than one of them.

    Args:
        feed: The open feed, for the cameras the images belong to.
        frameset: The frameset, with the inertial samples since the previous one.
        mode: How to draw the frames; see :data:`FrameMode`.
    """
    # The samples are what the estimator is fed, so they are logged in every
    # stage whether or not one consumes them.
    log_imu(frameset.imu)
    rr.set_time(TIMELINE, duration=np.timedelta64(frameset.t_ns, "ns"))
    for camera, image in zip(feed.cameras, frameset.images, strict=True):
        entity: str = f"{camera_entity(camera.index)}/image"
        if mode == "downscaled":
            small: UInt8[ndarray, "h w"] = np.ascontiguousarray(image[::IMAGE_DOWNSCALE, ::IMAGE_DOWNSCALE])
            rr.log(entity, rr.Image(small, color_model="L"))
        elif mode == "jpeg":
            # Full resolution, or the keypoints would sit two pixels off the
            # corner they were computed on; JPEG keeps a whole segment small.
            # The encode costs 3.9 ms a frameset on the replay thread and is
            # deliberate: no lane that reports a wall time comes through here,
            # because the V2 gate drives the feed and `Vio` itself with
            # nothing logged.
            rr.log(entity, rr.Image(image, color_model="L").compress(jpeg_quality=JPEG_QUALITY))
    if frameset.ground_truth is not None:
        pose_wxyz: Float64[ndarray, " 7"] = frameset.ground_truth
        rr.log(
            RIG_ENTITY,
            rr.Transform3D(translation=pose_wxyz[0:3], quaternion=rr.Quaternion(xyzw=np.roll(pose_wxyz[3:7], -1))),
        )


def vio_blueprint(cameras: tuple[CameraCalib, ...]) -> rrb.Blueprint:
    """One 3D view of the world, the camera images beside it, and the counters below.

    The counters are one time-series view per magnitude rather than one view for
    all of them. Rerun gives a view a single y-axis, and the widest series in it
    sets the scale for every other: with all fifteen in one view the LM cost's
    -68,000 left the landmark counts, the millisecond stage times and the
    centimetre ATE as one flat line on the zero, and the legend named seven of
    the fifteen. One axis per magnitude is the only lever there is, so the split
    follows what the smoke segment actually measures — counts in the hundreds,
    keyframes and LM steps in single digits, the frame and its two dominant
    stages in tens of milliseconds, the four remaining stages in fractions of
    one, the cost in tens of thousands, the damping over seven decades, and the
    ATE in centimetres.

    The seven sit in a row across the band the single view had, so the world view
    and the two camera views keep exactly the space they had. At 1920x1080 that
    is 274x250 each, where a two-row grid in the same band would be 135 px tall:
    a plot spends its first ~60 vertical pixels on the time axis and its labels,
    so height is what a trace can least spare. Narrowing the views also shrinks
    each legend, because a view of one magnitude holds few series.

    The three trajectories are logged one segment a frameset, so each of them
    carries a visible time range reaching from the start of the recording to the
    cursor: without it Rerun's default for a 3D view is latest-at, under which
    each segment renders alone. The range goes on the three entities rather than
    on the view, because everything else the view holds — the window frusta, the
    strips of the frames the last marginalization removed, the landmarks — is a
    whole state re-logged every frameset, and a window reaching back to the start
    would draw every copy of it at once.

    Args:
        cameras: The rig's cameras, in rig order.

    Returns:
        A blueprint with the panels collapsed, so the frame is all content.
    """
    views: list[rrb.View] = camera_views(cameras)
    trail: rrb.VisibleTimeRanges = rrb.VisibleTimeRanges(
        rrb.VisibleTimeRange(TIMELINE, start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.cursor_relative())
    )
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial3DView(
                    origin="/world",
                    name="world",
                    overrides={f"{run}/trajectory": trail for run in (RUN_ENTITY, GT_ENTITY, CPP_ENTITY)},
                ),
                rrb.Vertical(*views),
                column_shares=[2, 1],
            ),
            rrb.Horizontal(
                rrb.TimeSeriesView(
                    origin=VIO_STATS_ENTITY,
                    contents=[f"{VIO_STATS_ENTITY}/num_landmarks", f"{VIO_STATS_ENTITY}/num_observations"],
                    name="counts",
                ),
                rrb.TimeSeriesView(
                    origin=VIO_STATS_ENTITY,
                    contents=[f"{VIO_STATS_ENTITY}/num_keyframes", f"{VIO_STATS_ENTITY}/lm_iterations"],
                    name="keyframes & LM steps",
                ),
                rrb.TimeSeriesView(
                    origin=VIO_STATS_ENTITY,
                    contents=[f"{VIO_STATS_ENTITY}/track_ms", f"{VIO_STATS_ENTITY}/stage_ms/measure", f"{VIO_STATS_ENTITY}/stage_ms/solver"],
                    name="timing (ms)",
                ),
                # Naming the stages rather than globbing ``stage_ms/**`` is what
                # makes a stage nobody gave a view a failing test rather than a
                # flat line: ``test_vio_log`` reads this partition off the views.
                rrb.TimeSeriesView(
                    origin=VIO_STATS_ENTITY,
                    contents=[f"{VIO_STATS_ENTITY}/stage_ms/{stage}" for stage in ("back_substitution", "error", "linearize", "marginalize")],
                    name="solve stages (ms)",
                ),
                # The frontend's four in their own view for the same reason the
                # solver's four have one: on a 640x360 fisheye rig the KLT and
                # the detector are tens of milliseconds where the preintegration
                # is a fiftieth of one, and one axis for both hides the smaller.
                rrb.TimeSeriesView(
                    origin=VIO_STATS_ENTITY,
                    contents=[f"{VIO_STATS_ENTITY}/stage_ms/frontend_{stage}" for stage in ("pyramid", "detect", "track", "imu")],
                    name="frontend stages (ms)",
                ),
                rrb.TimeSeriesView(
                    origin=VIO_STATS_ENTITY,
                    contents=[f"{VIO_STATS_ENTITY}/lm_error_before", f"{VIO_STATS_ENTITY}/lm_error_after"],
                    name="LM cost",
                ),
                # The damping is its own view because it is not a cost: it runs
                # from 1e-5 to 74 on this segment, which the cost's own axis
                # would put on the zero line just as flatly.
                rrb.TimeSeriesView(origin=VIO_STATS_ENTITY, contents=[f"{VIO_STATS_ENTITY}/lm_lambda"], name="LM damping"),
                rrb.TimeSeriesView(
                    origin=VIO_STATS_ENTITY,
                    contents=[f"{VIO_STATS_ENTITY}/ate_cm/gt", f"{VIO_STATS_ENTITY}/ate_cm/cpp"],
                    name="ATE (cm)",
                ),
            ),
            row_shares=[3, 1],
        ),
        collapse_panels=True,
    )
