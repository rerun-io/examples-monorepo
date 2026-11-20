"""Log VIO estimates, ground truth, rig and window Pinhole cameras, and tracker statistics.

The run names follow the catalog layer (f0, :mod:`slam_rs.catalog_layer`): the raw causal VIO, drawn live, is
``/world/runs/slam_rs_vio``; the final run (the raw VIO re-anchored on the back end's map after the flush) is
``/world/runs/slam_rs``, where the map, the loops and the pose graph are drawn too. Each run carries its own
alignment to the ground truth. Until the final run exists, ``slam_rs`` repeats the raw run's alignment, so the map
sits on the raw run while the replay plays.
"""

import time
from dataclasses import dataclass, field
from typing import Literal, TypeAlias

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import Float64, Int64, UInt8
from numpy import ndarray
from simplecv.ops.umeyama import SimilarityTransform
from simplecv.stage_times import StageTimes

from slam_rs import _core
from slam_rs.catalog_feed import IMU_ENTITY, RIG_ENTITY, TIMELINE, CameraCalib, Frameset, ImuStream
from slam_rs.frontend_log import camera_entity, camera_views, log_keypoints
from slam_rs.landmark_colors import LandmarkColors
from slam_rs.map_loops import RUN_ENTITY, log_loops, log_pose_graph
from slam_rs.tracking import DropReason, Lockstep
from slam_rs.trajectory import MIN_ASSOCIATED_POSES, AteResult, Trajectory, ate

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

VIO_RUN_ENTITY: str = "/world/runs/slam_rs_vio"
"""The raw causal VIO, drawn live: rig, trajectory, window, landmarks (the catalog layer's ``slam_rs_vio``)."""
GT_ENTITY: str = "/world/runs/gt"
"""The dataset's own ground-truth run, which the base recording already names."""
MAP_KEYFRAMES_ENTITY: str = f"{RUN_ENTITY}/keyframes"
"""The back end's map keyframes (DESIGN §8), one point per keyframe at its raw pose."""
MAP_POINTS_ENTITY: str = f"{RUN_ENTITY}/map/points"
"""The back end's map points in the world frame (DESIGN §8)."""
MAP_KEYFRAMES_EVERY_NS: int = 1_000_000_000
"""Least video time between two draws of the map keyframes (at most ``map_max_keyframes`` points, ~24 kB a draw)."""
MAP_POINTS_EVERY_NS: int = 5_000_000_000
"""Least video time between two draws of the map points (tens of thousands of points, ~0.5 MB a draw on MIPB08).

Each draw is the whole map, so drawing at every new keyframe would grow the recording with keyframes x map size;
at a fixed video-time cadence it grows linearly with the clip. The final map is drawn once more after the flush.
"""
VIO_STATS_ENTITY: str = "/stats/vio"
"""Entity path for per-frame VIO counters, separate from dataset and frontend stats.
"""

FINAL_COLOR: tuple[int, int, int] = (255, 120, 40)
"""The final run: orange, over the raw run's green."""
ESTIMATE_COLOR: tuple[int, int, int] = (70, 220, 130)
"""The raw VIO run: green."""
GT_COLOR: tuple[int, int, int] = (235, 235, 235)
"""Ground truth: near-white, the reference every error is measured against."""
KEYFRAME_COLOR: tuple[int, int, int, int] = (90, 200, 255, 255)
"""A keyframe still inside the window."""
LTKF_COLOR: tuple[int, int, int, int] = (255, 235, 90, 255)
"""A long-term keyframe, which the pose budget never evicts."""
POSE_COLOR: tuple[int, int, int, int] = (150, 150, 165, 200)
"""A window frame that is not a keyframe: a state awaiting its vote, or a demoted pose block."""
MARGINALIZED_COLOR: tuple[int, int, int, int] = (255, 90, 90, 70)
"""A frame the last marginalization removed: the same Pinhole frustum, faded out."""
MAP_KEYFRAME_COLOR: tuple[int, int, int, int] = (255, 160, 60, 255)
"""A map keyframe that has left the window; window keyframes keep :data:`KEYFRAME_COLOR`."""
MAP_POINT_COLOR: tuple[int, int, int, int] = (200, 200, 210, 120)
"""A map point whose image colour is unknown: faint, so the window's landmarks stay readable on top."""
LANDMARK_COLOR: tuple[int, int, int, int] = (200, 200, 210, 255)
"""A window landmark whose image colour is unknown (the frames came without :attr:`Frameset.colors`)."""
COLOR_DOWNSCALE: int = 4
"""How much smaller the RGB copy a replay samples landmark colours from is than the tracked frame."""

FRUSTUM_DEPTH_M: float = 0.08
"""Window frusta's ``image_plane_distance``, metres: an orientation marker, not a claim about range.

Pinhole rays understate a fisheye's real field of view: the frustum shows where
its camera is and where it looks, not what it can see.
"""
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


def image_from_camera(camera: CameraCalib) -> Float64[ndarray, "3 3"]:
    """Build the camera's pinhole intrinsics matrix."""
    return np.array([[camera.fx, 0.0, camera.cx], [0.0, camera.fy, camera.cy], [0.0, 0.0, 1.0]], dtype=np.float64)


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
        rr.log(
            f"{node}{pinhole_child}",
            rr.Pinhole(
                image_from_camera=image_from_camera(camera),
                resolution=[camera.width, camera.height],
                camera_xyz=rr.ViewCoordinates.RDF,
                image_plane_distance=IMAGE_PLANE_M,
            ),
            static=True,
        )


@dataclass(slots=True)
class VioLogger:
    """Logs one frameset's estimator output, and carries the paths between them.

    The two reference trajectories are handed over whole, on the estimator's
    clock the results come on (:attr:`SegmentFeed.frame_t_ns`); the estimate
    accumulates as it is produced. Every ``log`` call writes at the caller's time
    cursor and nowhere else; :func:`set_video_time` puts that cursor on the base
    video's ``video_time``.
    """

    cameras: tuple[CameraCalib, ...]
    """The rig's cameras, in rig order."""
    ground_truth: Trajectory
    """Ground truth for the whole segment, on the estimator's clock; may be empty."""
    frame_t_ns: Int64[ndarray, " n_frames"]
    """The segment's frameset times: the cadence used to draw ground truth."""
    tracked_resolution: tuple[tuple[int, int], ...]
    """``(width, height)`` of the frames each camera is tracked on (``Vio.tracked_resolution``), which the keypoints are in."""
    cam_time_offset_ns: int
    """The estimator's clock minus the base video's ``video_time`` (:attr:`ImuCalib.cam_time_offset_ns`): what
    :func:`set_video_time` removes, so every row lands on the base frame it belongs to."""
    animate_base_rig: bool = False
    """Also move the dataset's rig (``/world/rig_00``) with the raw VIO pose: only for a replay viewed on its own, on a
    segment without ground truth. Off, the rig belongs to the layers that own it: ``gt``, or the catalog ``slam_rs``
    layer, which writes the final poses there (:func:`slam_rs.catalog_layer.write_layer`)."""
    estimate_t_ns: list[int] = field(default_factory=list)
    """Timestamps of the poses reported so far, in replay order."""
    estimate_position_m: list[Float64[ndarray, " 3"]] = field(default_factory=list)
    """Positions of the poses reported so far."""
    estimate_quaternion_wxyz: list[Float64[ndarray, " 4"]] = field(default_factory=list)
    """Rotations of the poses reported so far, w-first as :mod:`slam_rs.trajectory` stores them."""
    ground_truth_strip: Trajectory = field(init=False)
    """The ground truth thinned to the frameset cadence: what the drawn strip is taken from."""
    previous_window: dict[int, Float64[ndarray, " 7"]] = field(default_factory=dict)
    """Previous window poses by timestamp, used to draw marginalized frames."""
    pinholes: dict[tuple[int, int, int, int], rr.Pinhole] = field(init=False)
    """Camera 0 frusta by role colour, reused for every slot and frameset."""
    highest_slots: dict[Literal["window", "marginalized"], int] = field(default_factory=dict)
    """Highest initialized slot index in each family; static rig offsets are logged once."""
    previous_counts: dict[Literal["window", "marginalized"], int] = field(default_factory=dict)
    """Number of slots drawn in each family on the previous frameset."""
    framesets: int = 0
    """Framesets logged, which paces the ATE-so-far."""
    map_keyframe: int | None = None
    """Newest window keyframe when the map was last read; None before the first."""
    map_keyframes_t_ns: int | None = None
    """Video time of the last map-keyframes draw; None before the first."""
    map_points_t_ns: int | None = None
    """Video time of the last map-points draw; None before the first."""
    map_keyframes_every_ns: int = MAP_KEYFRAMES_EVERY_NS
    """Least video time between two map-keyframes draws (:data:`MAP_KEYFRAMES_EVERY_NS`)."""
    map_points_every_ns: int = MAP_POINTS_EVERY_NS
    """Least video time between two map-points draws (:data:`MAP_POINTS_EVERY_NS`)."""
    landmark_colors: LandmarkColors = field(default_factory=LandmarkColors)
    """The image's colour under each landmark's keypoint where it was first tracked."""

    def __post_init__(self) -> None:
        """Log the estimated rig's static geometry and thin the references."""
        log_rig(self.cameras, f"{VIO_RUN_ENTITY}/rig")
        self.ground_truth_strip = at_frameset_cadence(self.ground_truth, self.frame_t_ns)
        camera: CameraCalib = self.cameras[0]
        camera_intrinsics: Float64[ndarray, "3 3"] = image_from_camera(camera)
        self.pinholes = {
            color: rr.Pinhole(
                image_from_camera=camera_intrinsics,
                resolution=[camera.width, camera.height],
                camera_xyz=rr.ViewCoordinates.RDF,
                image_plane_distance=FRUSTUM_DEPTH_M,
                color=color,
            )
            for color in (KEYFRAME_COLOR, LTKF_COLOR, POSE_COLOR, MARGINALIZED_COLOR)
        }

    def log(
        self,
        result: _core.VioResult,
        snapshot: _core.VioSnapshot,
        frame: _core.FlowFrame,
        elapsed_ms: float,
        map_snapshot: _core.MapSnapshot | None = None,
        colors: list[UInt8[ndarray, "ch cw 3"]] | None = None,
    ) -> None:
        """Log one tracked frameset: the keypoints, the estimated and ground-truth paths, the rig, the window, the landmarks and the counters.

        Args:
            result: What ``track`` returned; only called where it tracked.
            snapshot: The window and the frame's statistics behind that result.
            frame: The keypoints the estimator's own frontend tracked.
            elapsed_ms: Wall time the ``track`` call took.
            map_snapshot: The back end's map, when the caller read it (see :meth:`map_is_due`); drawn under the run.
            colors: The frameset's frames in RGB (:attr:`Frameset.colors`), which colour the landmarks; None draws them grey.
        """
        pose: Float64[ndarray, " 7"] | None = result.world_from_rig
        assert pose is not None, f"Tracking result at {result.t_ns} has no pose"
        self.framesets += 1
        self.estimate_t_ns.append(result.t_ns)
        self.estimate_position_m.append(pose[0:3].copy())
        self.estimate_quaternion_wxyz.append(np.roll(pose[3:7], 1).copy())

        rig_pose: rr.Transform3D = rr.Transform3D(translation=pose[0:3], quaternion=rr.Quaternion(xyzw=pose[3:7]))
        rr.log(f"{VIO_RUN_ENTITY}/rig", rig_pose)
        if self.animate_base_rig:
            rr.log(RIG_ENTITY, rig_pose)
        if colors is not None:
            for camera, color in enumerate(colors):
                self.landmark_colors.remember(frame.ids(camera), frame.positions(camera), color, self.tracked_resolution[camera])
        # The estimator's own frontend output, on the frontend rung's paths and
        # in its palette (:func:`slam_rs.frontend_log.log_keypoints`).
        log_keypoints(frame)
        self._log_paths()
        self._log_window(snapshot)
        self._log_landmarks(snapshot)
        if map_snapshot is not None:
            self._log_map(map_snapshot, snapshot.t_ns, final=False)
            self.map_keyframe = int(snapshot.kf_ids[-1]) if len(snapshot.kf_ids) else None
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
            alignment: SimilarityTransform = IDENTITY if against_truth is None else inverted(against_truth.alignment)
            # The map under the final run's entity sits on the raw run until :meth:`log_final_run` places it.
            for run in (VIO_RUN_ENTITY, RUN_ENTITY):
                log_alignment(run, alignment)

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
            (f"{VIO_RUN_ENTITY}/path", self.estimated(), ESTIMATE_COLOR),
            (f"{GT_ENTITY}/path", self.ground_truth_strip, GT_COLOR),
        ):
            if len(trajectory) < 2:
                continue
            rr.log(entity, rr.LineStrips3D([trajectory.position_m], colors=(*color, ROUTE_ALPHA), radii=ROUTE_RADIUS_M), static=True)

    def _log_paths(self) -> None:
        """Draw the segment each trajectory gained at this cursor.

        One two-point strip a line a frameset, ending on the newest pose at or
        before the cursor: the view's visible time range (:func:`vio_blueprint`)
        is what accumulates them into the path so far, and a reference that has
        no second pose yet draws nothing. Each is logged in its own frame; the
        run entities' alignment transforms are what bring them together in
        the dataset's world.

        The estimate's segment is taken from the two poses this run last
        appended, not from :meth:`estimated`: a segment needs two poses, not the
        whole trajectory rebuilt behind them.
        """
        t_ns: int = self.estimate_t_ns[-1]
        if len(self.estimate_position_m) >= 2:
            segment: Float64[ndarray, "2 3"] = np.array(self.estimate_position_m[-2:], dtype=np.float64)
            rr.log(f"{VIO_RUN_ENTITY}/trajectory", rr.LineStrips3D([segment], colors=ESTIMATE_COLOR, radii=0.004))
        drawn: int = int(np.searchsorted(self.ground_truth_strip.t_ns, t_ns, side="right"))
        if drawn < 2:
            return
        rr.log(f"{GT_ENTITY}/trajectory", rr.LineStrips3D([self.ground_truth_strip.position_m[drawn - 2 : drawn]], colors=GT_COLOR, radii=0.004))

    def _log_camera_slots(
        self, family: Literal["window", "marginalized"], poses: list[Float64[ndarray, " 7"]], colors: list[tuple[int, int, int, int]]
    ) -> None:
        """Draw poses as camera 0 frusta and clear slots that just became empty."""
        camera: CameraCalib = self.cameras[0]
        highest: int = self.highest_slots.get(family, -1)
        for k, (pose, color) in enumerate(zip(poses, colors, strict=True)):
            slot: str = f"{VIO_RUN_ENTITY}/{family}/{k:02d}"
            if k > highest:
                rr.log(
                    f"{slot}/cam_00",
                    rr.Transform3D(translation=camera.imu_T_cam[:3, 3], mat3x3=camera.imu_T_cam[:3, :3]),
                    static=True,
                )
            rr.log(slot, rr.Transform3D(translation=pose[0:3], quaternion=rr.Quaternion(xyzw=pose[3:7])))
            rr.log(f"{slot}/cam_00", self.pinholes[color])
        for k in range(len(poses), self.previous_counts.get(family, 0)):
            rr.log(f"{VIO_RUN_ENTITY}/{family}/{k:02d}", rr.Clear(recursive=True))
        self.previous_counts[family] = len(poses)
        self.highest_slots[family] = max(highest, len(poses) - 1)

    def _log_window(self, snapshot: _core.VioSnapshot) -> None:
        """Draw window and marginalized poses as Pinhole cameras, coloured by role."""
        poses: Float64[ndarray, "n_frames 7"] = snapshot.window_poses
        colors: list[tuple[int, int, int, int]] = [
            LTKF_COLOR if long_term else KEYFRAME_COLOR if keyframe else POSE_COLOR
            for keyframe, long_term in zip(snapshot.window_keyframe, snapshot.window_long_term, strict=True)
        ]
        self._log_camera_slots("window", list(poses), colors)

        # Removed frames retain the poses from the last snapshot that held them.
        leaving: set[int] = set(snapshot.marginalized.tolist())
        marginalized: list[Float64[ndarray, " 7"]] = [pose for t_ns, pose in self.previous_window.items() if t_ns in leaving]
        self._log_camera_slots("marginalized", marginalized, [MARGINALIZED_COLOR] * len(marginalized))
        self.previous_window = {t_ns: pose for t_ns, pose in zip(snapshot.window_t_ns.tolist(), poses, strict=True)}

    def _log_landmarks(self, snapshot: _core.VioSnapshot) -> None:
        """Draw the window's landmarks in the colour of the image where each was first tracked."""
        rr.log(
            f"{VIO_RUN_ENTITY}/landmarks",
            rr.Points3D(snapshot.landmark_positions, colors=self.landmark_colors.lookup(snapshot.landmark_ids, LANDMARK_COLOR), radii=0.008),
        )

    def map_is_due(self, snapshot: _core.VioSnapshot) -> bool:
        """Whether to read the map now: a keyframe joined the window since the last read, and the keyframes are due.

        The map changes only when a keyframe's stamp reaches the back end, and a long run's map holds tens of
        thousands of points, so it is read (and drawn) at most every :attr:`map_keyframes_every_ns` of video time.
        """
        if len(snapshot.kf_ids) == 0 or int(snapshot.kf_ids[-1]) == self.map_keyframe:
            return False
        return self.map_keyframes_t_ns is None or snapshot.t_ns - self.map_keyframes_t_ns >= self.map_keyframes_every_ns

    def log_final_map(self, map_snapshot: _core.MapSnapshot, t_ns: int) -> None:
        """Draw the map once more after the run's flush, keyframes and points, so the recording ends on the final map.

        Args:
            map_snapshot: The map after :meth:`slam_rs._core.Vio.flush` (current in both modes).
            t_ns: Video time of the cursor the caller set, usually the last frameset's.
        """
        self._log_map(map_snapshot, t_ns, final=True)
        log_pose_graph(map_snapshot)

    def log_final_run(self, final: Trajectory) -> None:
        """Draw the final run's whole path at the cursor the caller set (the last frameset's), under its own alignment.

        The alignment is fitted to the final run, not to the raw one, so a loop the back end closed lands on the ground
        truth. It replaces the raw run's alignment on :data:`RUN_ENTITY` from this cursor on, so the final map drawn
        there moves with it.

        Args:
            final: The final trajectory (:func:`slam_rs.tracking.final_trajectory`: GBA, then re-anchoring), on ``video_time``.
        """
        if len(final) < 2:
            return
        rr.log(f"{RUN_ENTITY}/trajectory", rr.LineStrips3D([final.position_m], colors=FINAL_COLOR, radii=0.004))
        alignment: SimilarityTransform = IDENTITY
        if len(self.ground_truth) and len(final) >= MIN_ASSOCIATED_POSES:
            against_truth: AteResult = ate(final, self.ground_truth)
            if against_truth.n_associated >= MIN_ASSOCIATED_POSES:
                alignment = inverted(against_truth.alignment)
        log_alignment(RUN_ENTITY, alignment)

    def _log_map(self, map_snapshot: _core.MapSnapshot, t_ns: int, *, final: bool) -> None:
        """Draw the back end's keyframes (window ones in the window colour) at the cursor, and its points when due or final."""
        colors: UInt8[ndarray, "k 4"] = np.where(
            map_snapshot.keyframe_active[:, None], np.array(KEYFRAME_COLOR, dtype=np.uint8), np.array(MAP_KEYFRAME_COLOR, dtype=np.uint8)
        )
        rr.log(MAP_KEYFRAMES_ENTITY, rr.Points3D(map_snapshot.keyframe_poses[:, 0:3], colors=colors, radii=0.012))
        log_loops(map_snapshot)
        self.map_keyframes_t_ns = t_ns
        if final or self.map_points_t_ns is None or t_ns - self.map_points_t_ns >= self.map_points_every_ns:
            rr.log(MAP_POINTS_ENTITY, rr.Points3D(map_snapshot.points_world, colors=self.landmark_colors.lookup(map_snapshot.point_ids, MAP_POINT_COLOR), radii=0.004))
            self.map_points_t_ns = t_ns

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
        # own. Both are negative on most frames, following the prior cost convention
        # and not a sign error: the marginalization prior term deliberately drops
        # the 1/2 r^T r (``crates/slam-rs/src/estimator/optimize.rs:71``, D20).
        rr.log(f"{VIO_STATS_ENTITY}/lm_error_before", rr.Scalars(snapshot.lm_error_before))
        rr.log(f"{VIO_STATS_ENTITY}/lm_error_after", rr.Scalars(snapshot.lm_error_after))
        rr.log(f"{VIO_STATS_ENTITY}/track_ms", rr.Scalars(elapsed_ms))
        for stage, milliseconds in snapshot.timings_ms.items():
            rr.log(f"{VIO_STATS_ENTITY}/stage_ms/{stage}", rr.Scalars(milliseconds))

    def _log_ate(self, estimated: Trajectory) -> AteResult | None:
        """Log the rigid-aligned error of everything reported so far, against ground truth.

        Args:
            estimated: Everything reported so far.

        Returns:
            The error against the ground truth, whose alignment is also what
            places the run's subtree, or None where too few poses associated for
            the number to mean anything.
        """
        if len(self.ground_truth) == 0 or len(estimated) < MIN_ASSOCIATED_POSES:
            return None
        result: AteResult = ate(estimated, self.ground_truth)
        if result.n_associated < MIN_ASSOCIATED_POSES:
            return None
        rr.log(f"{VIO_STATS_ENTITY}/ate_cm/gt", rr.Scalars(100.0 * result.rmse_m))
        return result


def log_calibration(cameras: tuple[CameraCalib, ...], *, gravity_world: bool = False) -> None:
    """Log the rig's static geometry so the images sit in the right place in 3D.

    The ``Pinhole`` goes on the ``pinhole`` child, which is the dataset's own
    layout: the images and the keypoints hang under it, so they are the pixels of
    the camera that projects them.

    Args:
        cameras: The rig's cameras.
        gravity_world: The world is the estimator's own (no ground truth to align to): gravity-aligned, z up, as the
            catalog layer states it. Otherwise the root keeps ``RUB``.
    """
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP if gravity_world else rr.ViewCoordinates.RUB, static=True)
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
    times: StageTimes = field(default_factory=StageTimes)
    """Per-frameset wall time of each stage: ``track`` (the whole call), the ``frontend`` (pyramid, detection, tracking, stereo) and its ``imu`` prediction, the ``estimator`` (the
    rest of the call: bootstrap, estimator and glue) and the ``log`` work after it."""
    keypoints: list[int] = field(default_factory=list)
    """Keypoints the frontend holds after each frameset, summed over cameras."""

    def __post_init__(self) -> None:
        """Draw the keypoints in each camera's own pixels: a rig wider than 2048 px is tracked on frames reduced by two."""
        for camera, (width, _) in zip(self.logger.cameras, self.logger.tracked_resolution, strict=True):
            if width != camera.width:
                rr.log(f"{camera_entity(camera.index)}/keypoints", rr.Transform3D(scale=2.0), static=True)

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
        started: float = time.monotonic()
        consumed: int = len(self.lockstep.consumed_ms)
        for held, result in self.lockstep.push(frameset):
            self._log_result(held, result)
        vio: _core.Vio = self.lockstep.vio
        calls: list[float] = self.lockstep.consumed_ms[consumed:]
        if calls:
            # The breakdown belongs to the last call; a frameset released from the IMU hold adds a second one.
            frontend: dict[str, float] = vio.frontend_timings_ms
            flow: float = frontend["pyramid"] + frontend["detect"] + frontend["track"] + frontend["stereo"]
            self.times.add("track", sum(calls))
            self.times.add("frontend", flow)
            self.times.add("imu", frontend["imu"])
            self.times.add("estimator", calls[-1] - flow - frontend["imu"])
        # One copy of the frontend's keypoints per frameset, for the count.
        frame: _core.FlowFrame | None = vio.flow_frame() if calls else None
        if calls and frame is not None:
            self.keypoints.append(sum(frame.num_tracks(camera) for camera in range(vio.camera_count)))
        self.times.add("log", 1e3 * (time.monotonic() - started) - sum(calls))

    def flush(self) -> None:
        """Log the last lagged frame, then the final map (the flush made it current), before completing paths or exporting poses."""
        last_t_ns: int | None = None
        for held, result in self.lockstep.flush():
            self._log_result(held, result)
            last_t_ns = held.t_ns
        final: _core.MapSnapshot | None = self.lockstep.vio.map_snapshot()
        if last_t_ns is None and self.logger.estimate_t_ns:
            last_t_ns = self.logger.estimate_t_ns[-1]
        if final is not None and last_t_ns is not None:
            set_video_time(last_t_ns, self.logger.cam_time_offset_ns)
            self.logger.log_final_map(final, last_t_ns)

    def _log_result(self, held: Frameset, result: _core.VioResult) -> None:
        """Log a result with the keypoints and timestamp of its own input frame."""
        set_video_time(held.t_ns, self.logger.cam_time_offset_ns)
        snapshot: _core.VioSnapshot | None = self.lockstep.vio.snapshot()
        frame: _core.FlowFrame = self.lockstep.flow_frame()
        assert snapshot is not None, f"frameset {held.t_ns} tracked without a window snapshot"
        assert frame.t_ns == result.t_ns
        # None with the back end off; read only when a keyframe joined and the map is due (never waits for the back end).
        map_snapshot: _core.MapSnapshot | None = self.lockstep.vio.map_snapshot() if self.logger.map_is_due(snapshot) else None
        self.logger.log(result, snapshot, frame, self.lockstep.elapsed_ms[-1], map_snapshot, held.colors)

    def refuse_lost_framesets(self) -> None:
        """Stop the run when a frameset never got the inertial samples that cover it.

        Frames waiting for stereo are consumed without a pose. An input still
        held for IMU at the end is an error; both tools use the same rule.

        Raises:
            SystemExit: If any frameset is still held.
        """
        uncovered: int = len(self.pending) + sum(drop.reason == DropReason.IMU_ENDED for drop in self.lockstep.dropped)
        if uncovered:
            raise SystemExit(f"{uncovered} framesets never got the inertial samples that cover them")

    def summary(self) -> str:
        """One line on what the stage did, for the end of a replay.

        The empty case is this stage's own to report: the run that tracked
        nothing is the one whose held framesets most need naming, and a caller
        that guarded the call on ``elapsed_ms`` suppressed exactly that line.
        """
        unresolved: str = ""
        uncovered: list[int] = [held.t_ns for held in self.pending] + [
            drop.t_ns for drop in self.lockstep.dropped if drop.reason == DropReason.IMU_ENDED
        ]
        if uncovered:
            unresolved = f", {len(uncovered)} FRAMESETS NEVER COVERED BY THE IMU at {uncovered}"
        if not self.elapsed_ms:
            return f"vio: {self.lockstep.imu_samples} IMU samples pushed, nothing tracked{unresolved}"
        return (
            f"vio: {self.lockstep.imu_samples} IMU samples pushed, {len(self.elapsed_ms)} tracked, "
            f"{self.lockstep.retries} retries, {np.mean(self.elapsed_ms):.1f} ms per frameset "
            f"(median {np.median(self.elapsed_ms):.1f}, max {np.max(self.elapsed_ms):.1f})"
            f"{unresolved}"
        )


def set_video_time(t_ns: int, cam_time_offset_ns: int) -> None:
    """Put the time cursor on the base video's frame for an estimator-clock stamp.

    The feed hands the estimator ``video_time`` plus the camera-to-IMU offset (:attr:`SegmentFeed.frame_t_ns`); a replay
    row written at that stamp would sit beside the base frame, not on it, and a latest-at query at the frame would read
    the previous row. Every replay row goes through here, so the offset is removed once, at the logging boundary.

    Args:
        t_ns: A stamp on the estimator's clock.
        cam_time_offset_ns: :attr:`ImuCalib.cam_time_offset_ns` of the feed the stamp came from.
    """
    rr.set_time(TIMELINE, duration=np.timedelta64(t_ns - cam_time_offset_ns, "ns"))


def log_imu(imu: ImuStream, cam_time_offset_ns: int) -> None:
    """Log one frameset's inertial samples, one column per channel.

    Two ``send_columns`` calls instead of a ``set_time`` and two ``log`` calls per
    sample — about 57 samples a frameset at 1 kHz, so 171 calls become 2, and the
    frameset's inertial rung goes from 0.446 ms to 0.037 ms. The rows are the
    same rows: each timestamp still carries its three components, which is what
    the partition says.

    Args:
        imu: The samples since the previous frameset, on the estimator's clock.
        cam_time_offset_ns: What :func:`set_video_time` removes, so the samples sit on the base recording's own rows.
    """
    if not len(imu):
        return
    times: rr.TimeColumn = rr.TimeColumn(TIMELINE, duration=(imu.t_ns - cam_time_offset_ns).astype("timedelta64[ns]"))
    components: int = imu.gyro_rad_s.shape[1]
    for entity, channel in ((f"{IMU_ENTITY}/gyro", imu.gyro_rad_s), (f"{IMU_ENTITY}/accel", imu.accel_m_s2)):
        rr.send_columns(entity, indexes=[times], columns=rr.Scalars.columns(scalars=channel.reshape(-1)).partition([components] * len(imu)))


MASK_CLASSES: rr.AnnotationContext = rr.AnnotationContext(
    [rr.AnnotationInfo(id=0, label="kept", color=(255, 255, 255, 0))]
    + [rr.AnnotationInfo(id=label, label="masked", color=(40, 110, 255, 140)) for label in range(1, 256)]
)
"""Kept pixels draw nothing, every nonzero label a translucent blue over the frame. Class 0 needs an explicit colour with
alpha 0 and nonzero RGB: an unlisted class 0 gets an automatic solid colour, and (0, 0, 0, 0) is stored as 0, which the
viewer also reads as "no colour"."""


class FrameMaskLog:
    """Logs the masks a replay tracked with, in the mask layer's form, under the layer's own entity name.

    Each camera's mask goes to ``<camera>/<name>`` at its own resolution, with a static ``Transform3D`` that scales it onto
    the camera's image, so the viewer draws it over the frame and the keypoints. The statics are logged once per camera
    and size, and a mask only when it differs from the camera's last one: a layer sparser than the video holds one mask
    over many frames.
    """

    def __init__(self, name: str) -> None:
        """Log under ``<camera>/<name>``.

        Args:
            name: The mask layer's entity name under each camera's ``pinhole`` (``--mask-entity``).
        """
        self.name: str = name
        self.scaled: dict[int, tuple[int, int, int, int]] = {}
        """Per camera index, the ``(image w, image h, mask w, mask h)`` its static scale was logged for."""
        self.last: dict[int, UInt8[ndarray, "mh mw"]] = {}

    def log(self, cameras: tuple[int, ...], image_sizes: list[tuple[int, int]], masks: list[UInt8[ndarray, "mh mw"] | None]) -> None:
        """Log one frameset's masks at the time already set. A camera with no mask logs nothing.

        Args:
            cameras: Rig index of each camera, in frameset order.
            image_sizes: ``(width, height)`` of each camera's image.
            masks: Each camera's mask, or None.
        """
        for index, (width, height), mask in zip(cameras, image_sizes, masks, strict=True):
            if mask is None:
                continue
            entity: str = f"{camera_entity(index)}/{self.name}"
            size: tuple[int, int, int, int] = (width, height, mask.shape[1], mask.shape[0])
            if self.scaled.get(index) != size:
                rr.log(entity, rr.Transform3D(scale=[width / mask.shape[1], height / mask.shape[0], 1.0]), MASK_CLASSES, static=True)
                self.scaled[index] = size
            last: UInt8[ndarray, "mh mw"] | None = self.last.get(index)
            if last is not None and last.shape == mask.shape and np.array_equal(last, mask):
                continue
            rr.log(entity, rr.SegmentationImage(mask))
            self.last[index] = mask


def log_frameset_inputs(
    cameras: tuple[CameraCalib, ...], cam_time_offset_ns: int, frameset: Frameset, mode: FrameMode, mask_log: FrameMaskLog | None = None
) -> None:
    """Log what the estimator is fed for one frameset, on the base video's frame of it (:func:`set_video_time`).

    Every replay draws this rung whether or not a stage consumes it, and the two
    tools that drive a stage draw the same one: a channel added here reaches both
    rather than one of them.

    Args:
        cameras: The feed's cameras (:attr:`SegmentFeed.cameras`), which the images belong to.
        cam_time_offset_ns: The feed's :attr:`ImuCalib.cam_time_offset_ns`, which :func:`set_video_time` removes.
        frameset: The frameset, with the inertial samples since the previous one.
        mode: How to draw the frames; see :data:`FrameMode`.
        mask_log: Where the frameset's masks go, drawn with the frames; None logs no mask.
    """
    # The samples are what the estimator is fed, so they are logged in every
    # stage whether or not one consumes them.
    log_imu(frameset.imu, cam_time_offset_ns)
    set_video_time(frameset.t_ns, cam_time_offset_ns)
    for camera, image in zip(cameras, frameset.images, strict=True):
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
    if mode != "off" and frameset.masks is not None and mask_log is not None:
        mask_log.log(tuple(camera.index for camera in cameras), [(image.shape[1], image.shape[0]) for image in frameset.images], frameset.masks)
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
    frusta of the frames the last marginalization removed, the landmarks — is a
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
                    contents=["/world/**"],
                    overrides={f"{run}/trajectory": trail for run in (RUN_ENTITY, VIO_RUN_ENTITY, GT_ENTITY)},
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
                    contents=[
                        f"{VIO_STATS_ENTITY}/stage_ms/{stage}"
                        for stage in ("back_substitution", "error", "linearize", "predict", "keyframe", "optimize", "marginalize")
                    ],
                    name="solve stages (ms)",
                ),
                # The frontend's four in their own view for the same reason the
                # solver's four have one: on a 640x360 fisheye rig the KLT and
                # the detector are tens of milliseconds where the preintegration
                # is a fiftieth of one, and one axis for both hides the smaller.
                rrb.TimeSeriesView(
                    origin=VIO_STATS_ENTITY,
                    contents=[f"{VIO_STATS_ENTITY}/stage_ms/frontend_{stage}" for stage in ("pyramid", "detect", "track", "stereo", "imu")],
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
                    contents=[f"{VIO_STATS_ENTITY}/ate_cm/gt"],
                    name="ATE (cm)",
                ),
            ),
            row_shares=[3, 1],
        ),
        collapse_panels=True,
        auto_views=False,
    )
