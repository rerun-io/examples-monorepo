"""Log VIO estimates, ground truth, rig and window Pinhole cameras, and tracker statistics.

The run names follow the catalog layer (f0, :mod:`slam_rs.catalog_layer`): the raw causal VIO, drawn live, is
``/world/runs/slam_rs_vio``; the final run (the raw VIO re-anchored on the back end's map after the flush) is
``/world/runs/slam_rs``, where the map, the loops and the pose graph are drawn too. Each run carries its own
alignment to the ground truth. Until the final run exists, ``slam_rs`` repeats the raw run's alignment, so the map
sits on the raw run while the replay plays.
"""

import itertools
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, TypeAlias

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import Float32, Float64, Int64, UInt8
from numpy import ndarray
from simplecv.ops.umeyama import SimilarityTransform
from simplecv.rerun_walk_path import PATH_RADIUS_UI, TRAIL, TRAJECTORY, send_trajectory
from simplecv.stage_times import StageTimes

from slam_rs import _core
from slam_rs.catalog_feed import RIG_ENTITY, TIMELINE, CameraCalib, Frameset
from slam_rs.columns import ColumnLog
from slam_rs.frontend_log import KEYPOINT_RADIUS_PX, camera_entity, camera_views, track_colors
from slam_rs.landmark_colors import LandmarkColors
from slam_rs.map_loops import PGO_CORRECTION_COLOR, PGO_CORRECTIONS_ENTITY, RUN_ENTITY, correction_segments, log_loops, log_pose_graph
from slam_rs.map_point_blocks import MapPointBlocks
from slam_rs.tracking import DropReason, Lockstep
from slam_rs.trajectory import MIN_ASSOCIATED_POSES, AteResult, PoseLog, Trajectory, ate

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
"""Least video time between two draws of the map points' changed blocks (:class:`MapPointBlocks`). The final map is
drawn whole once more after the flush."""
MAP_POINT_BLOCKS_ENTITY: str = f"{MAP_POINTS_ENTITY}/blocks"
"""Parent of the map-point blocks (:class:`slam_rs.map_point_blocks.MapPointBlocks`) the run draws while it goes on,
``blocks/0000`` on, each with its own colours and :data:`MAP_POINT_BLOCK_RADIUS_UI`. Cleared at the final draw, which
puts the whole map on :data:`MAP_POINTS_ENTITY` itself, where a layout's overrides find it."""
MAP_POINT_RADIUS_M: float = 0.004
"""Radius of the final map's points on :data:`MAP_POINTS_ENTITY`, metres (a layout may override it there)."""
MAP_POINT_BLOCK_RADIUS_UI: float = 1.5
"""Radius of the blocks' points, ui points, logged with the data: a layout's overrides cannot reach the block entities
(no path patterns), and a view default would size every radius-less cloud in the view, not only these."""
VIO_STATS_ENTITY: str = "/stats/vio"
"""Entity path for per-frame VIO counters, separate from dataset and frontend stats.
"""

FINAL_COLOR: tuple[int, int, int] = (255, 120, 40)
"""The final run: orange, over the raw run's green."""
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
BLOCK_FRAMESETS: int = 300
"""Framesets between two sends of the held rows (:class:`slam_rs.columns.ColumnLog`): ten seconds of a 30 Hz rig, so a
chunk holds hundreds of rows while a four-hour run holds at most one block in memory."""
LANDMARK_RADIUS_M: float = 0.008
"""Radius of the window's landmarks, metres."""
MAP_KEYFRAME_RADIUS_M: float = 0.012
"""Radius of the map's keyframe points, metres."""
PGO_CORRECTION_RADIUS_M: float = 0.004
"""Stroke of a pose-graph correction, metres."""
SlotFamily: TypeAlias = Literal["window", "marginalized"]
"""The two families of numbered frame slots under the run: the estimator's window, and what its last marginalization removed."""
UNIFIED_SPHERE_MODELS: tuple[str, ...] = ("mei_v3", "mei_v6", "ds")
"""Camera models whose focal is the unified sphere's ``gamma`` and whose first coefficient is ``xi`` (:func:`display_intrinsics`)."""
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


def image_from_camera(camera: CameraCalib) -> Float64[ndarray, "3 3"]:
    """Build the camera's pinhole intrinsics matrix."""
    return np.array([[camera.fx, 0.0, camera.cx], [0.0, camera.fy, camera.cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def display_intrinsics(camera: CameraCalib) -> Float64[ndarray, "3 3"]:
    """The pinhole a viewer should draw this camera's frustum (and the picture in it) with: its paraxial focal.

    A viewer's ``Pinhole`` is a plain pinhole. A unified-sphere model's focal (mei's and the double sphere's ``gamma``)
    is not what the lens projects at its axis: there a ray at angle ``t`` lands ``gamma / (1 + xi) * t`` pixels out.
    Drawn with ``gamma`` itself, the X4 Air's 200 degree lens (xi 2.31) is a 55 degree frustum; with the paraxial
    focal it is 120 degrees, which is what the middle of the picture covers. The other models' focal is paraxial
    already. For display only: a calibration reader wants :func:`image_from_camera`.
    """
    intrinsics: Float64[ndarray, "3 3"] = image_from_camera(camera)
    if camera.model in UNIFIED_SPHERE_MODELS:
        intrinsics[0, 0] /= 1.0 + camera.distortion[0]
        intrinsics[1, 1] /= 1.0 + camera.distortion[0]
    return intrinsics


def dataset_videos(camera_positions: Sequence[int]) -> tuple[str, ...]:
    """The dataset's own ``VideoStream`` of each fed camera, by its position in the rig: what :attr:`VioLogger.video_entities` wants."""
    return tuple(f"{RIG_ENTITY}/cam_{position:02d}/pinhole/video" for position in camera_positions)


def log_rig(cameras: tuple[CameraCalib, ...], entity_prefix: str, *, display: bool = False, recording: rr.RecordingStream | None = None) -> None:
    """Log a rig's static camera geometry, so whatever moves it draws as frusta.

    Two rigs are drawn from the same calibration and this is the one place that
    spells it: the dataset's own under ``/world/rig_00`` and the estimated run's
    own copy under its run entity — without which the estimate and the ground
    truth would move the same frusta. Both have the dataset's layout: the
    camera node carries the pose in the rig, its ``pinhole`` child the
    ``Pinhole``, and the pictures and keypoints hang under that.

    Args:
        cameras: The rig's cameras, in rig order.
        entity_prefix: What ``cam_MM`` hangs under.
        display: The rig is a drawn copy, not the dataset's calibration: its frusta take
            :func:`display_intrinsics`, so a wide lens's picture has the right size.
        recording: Where to log; None is the global recording.
    """
    for camera in cameras:
        node: str = f"{entity_prefix}/cam_{camera.index:02d}"
        rr.log(
            node,
            rr.Transform3D(translation=camera.imu_T_cam[:3, 3], mat3x3=camera.imu_T_cam[:3, :3]),
            static=True,
            recording=recording,
        )
        rr.log(
            f"{node}/pinhole",
            rr.Pinhole(
                image_from_camera=display_intrinsics(camera) if display else image_from_camera(camera),
                resolution=[camera.width, camera.height],
                camera_xyz=rr.ViewCoordinates.RDF,
                image_plane_distance=IMAGE_PLANE_M,
            ),
            static=True,
            recording=recording,
        )


@dataclass(slots=True)
class VioLogger:
    """Logs one frameset's estimator output, and carries the paths between them.

    The two reference trajectories are handed over whole, on the estimator's
    clock the results come on (:attr:`SegmentFeed.frame_t_ns`); the estimate
    accumulates as it is produced. Each row is stamped with its own frameset's
    base ``video_time`` (the estimator's stamp minus :attr:`cam_time_offset_ns`)
    and held in :attr:`columns`, which goes out every :attr:`block_framesets`
    framesets and at every whole-run draw; :meth:`flush` sends the rest.
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
    """The estimator's clock minus the base video's ``video_time`` (:attr:`ImuCalib.cam_time_offset_ns`): what every row's
    stamp has removed, so it lands on the base frame it belongs to: a row at the estimator's stamp would sit beside the
    base frame, and a latest-at query at the frame would read the previous row."""
    video_entities: tuple[str, ...] = ()
    """The dataset's ``VideoStream`` of each camera, in :attr:`cameras` order. With them a window keyframe's frusta
    reference the frames of its own time (``VideoFrameReference``: nothing is encoded again); empty, they stay empty."""
    camera_positions: tuple[int, ...] = ()
    """The rig position of each fed camera, in :attr:`cameras` order (:attr:`SegmentFeed.camera_positions`): the keypoints
    go on that base camera. Empty: each camera's own index, which is its position on a rig fed whole."""
    base_downscale: int = 1
    """How much smaller the decoded frames are than the base's (:attr:`RigProfile.downscale`): the keypoints are drawn in
    the base camera's own pixels."""
    loops_as_found: bool = True
    """Draw the map's loops at every map read, as the back end finds them (a live run). Off (a replay), the loops are
    drawn once, the final set at the end (:meth:`log_final_map`): a replay's loops would otherwise stand from their
    detection on, each drawn again at every read."""
    columns: ColumnLog = field(default_factory=ColumnLog)
    """The rows not sent yet, and the recording they go to."""
    block_framesets: int = BLOCK_FRAMESETS
    """Framesets between two sends of :attr:`columns` (:data:`BLOCK_FRAMESETS`)."""
    corrections_shown: bool = False
    """Whether pose-graph corrections are drawn from the last map read on (:meth:`_log_corrections`)."""
    poses: PoseLog = field(default_factory=PoseLog)
    """Every pose the run reported, for the CSV export, the ATE and the per-frameset path segment."""
    ground_truth_strip: Trajectory = field(init=False)
    """The ground truth thinned to the frameset cadence: what the drawn strip is taken from."""
    previous_window: dict[int, Float64[ndarray, " 7"]] = field(default_factory=dict)
    """Previous window poses by timestamp, used to draw marginalized frames."""
    display_intrinsics: tuple[Float64[ndarray, "3 3"], ...] = field(init=False)
    """Each camera's frustum intrinsics (:func:`display_intrinsics`), shared by every slot."""
    keypoint_scale: tuple[float, ...] = field(init=False)
    """Base pixels per tracked pixel, per camera."""
    highest_slots: dict[SlotFamily, int] = field(default_factory=dict)
    """Highest initialized slot index in each family; static rig offsets are logged once."""
    window_slots: dict[int, int] = field(default_factory=dict)
    """The slot each window frame holds, by frame timestamp. A frame keeps its slot while it is in the window, so a
    keyframe's pictures are referenced (and decoded by the viewer) once, not again whenever an older frame leaves."""
    marginalized_count: int = 0
    """Marginalized slots drawn on the previous frameset."""
    slot_frusta: dict[tuple[SlotFamily, int], tuple[int, int, int, int]] = field(default_factory=dict)
    """The role colour each slot's frusta were last drawn in; they are logged again only when it changes or the slot was cleared."""
    slot_pictures: dict[tuple[SlotFamily, int], int] = field(default_factory=dict)
    """The frame each slot's pictures reference, for the slots that show any."""
    framesets: int = 0
    """Framesets logged, which paces the ATE-so-far and the sends."""
    map_keyframe: int | None = None
    """Newest keyframe of the map as last read; None before the first. Until the map holds the window's newest
    keyframe (Realtime: the back-end thread has yet to step its stamp), the map is read again at the cadence."""
    map_keyframes_t_ns: int | None = None
    """Estimator time of the last map-keyframes draw; None before the first."""
    map_points_t_ns: int | None = None
    """Estimator time of the last map-points draw; None before the first."""
    map_keyframes_every_ns: int = MAP_KEYFRAMES_EVERY_NS
    """Least video time between two map-keyframes draws (:data:`MAP_KEYFRAMES_EVERY_NS`)."""
    map_points_every_ns: int = MAP_POINTS_EVERY_NS
    """Least video time between two map-points draws (:data:`MAP_POINTS_EVERY_NS`)."""
    landmark_colors: LandmarkColors = field(default_factory=LandmarkColors)
    """The image's colour under each landmark's keypoint where it was first tracked."""
    map_point_blocks: MapPointBlocks = field(default_factory=MapPointBlocks)
    """The map's points by host as read, and the blocks to draw again; its revision is the next read's ``since``."""

    def __post_init__(self) -> None:
        """Log the estimated rig's static geometry and thin the references."""
        recording: rr.RecordingStream | None = self.columns.recording
        log_rig(self.cameras, f"{VIO_RUN_ENTITY}/rig", display=True, recording=recording)
        self.ground_truth_strip = at_frameset_cadence(self.ground_truth, self.frame_t_ns)
        self.display_intrinsics = tuple(display_intrinsics(camera) for camera in self.cameras)
        self.keypoint_scale = tuple(self.base_downscale * camera.width / width for camera, (width, _) in zip(self.cameras, self.tracked_resolution, strict=True))
        positions: tuple[int, ...] = self.camera_positions or tuple(camera.index for camera in self.cameras)
        if len(positions) != len(self.cameras):
            raise ValueError(f"{len(positions)} camera positions for {len(self.cameras)} cameras")
        self.camera_positions = positions

    def video_t_ns(self, t_ns: int) -> int:
        """The base ``video_time`` of an estimator stamp (:attr:`cam_time_offset_ns`)."""
        return t_ns - self.cam_time_offset_ns

    def log(
        self,
        result: _core.VioResult,
        snapshot: _core.VioSnapshot,
        frame: _core.FlowFrame,
        elapsed_ms: float,
        map_delta: _core.MapDelta | None = None,
        colors: list[UInt8[ndarray, "ch cw 3"]] | None = None,
    ) -> None:
        """Log one tracked frameset: the keypoints, the ground truth's path, the window, the landmarks, the map and the counters.

        The run's own poses (rig, follow frame, trail, lens frames) are drawn from the whole trajectory at the end, by
        :func:`slam_rs.catalog_layer.log_layer`, the one writer of both runs' paths.

        Args:
            result: What ``track`` returned; only called where it tracked.
            snapshot: The window and the frame's statistics behind that result.
            frame: The keypoints the estimator's own frontend tracked.
            elapsed_ms: Wall time the ``track`` call took.
            map_delta: The back end's map changes, when the caller read them (:meth:`read_map`); drawn under the run.
            colors: The frameset's frames in RGB (:attr:`Frameset.colors`), which colour the landmarks; None draws them grey.
        """
        pose: Float64[ndarray, " 7"] | None = result.world_from_rig
        assert pose is not None, f"Tracking result at {result.t_ns} has no pose"
        self.framesets += 1
        self.poses.append(result.t_ns, pose[0:3], np.roll(pose[3:7], 1))
        t: int = self.video_t_ns(result.t_ns)
        if colors is not None:
            for camera, color in enumerate(colors):
                self.landmark_colors.remember(frame.ids(camera), frame.positions(camera), color, self.tracked_resolution[camera])
        self._log_keypoints(frame, t)
        self._log_ground_truth_trail(result.t_ns, t)
        self._log_window(snapshot, t)
        self.columns.static_once(f"{VIO_RUN_ENTITY}/landmarks", rr.Points3D.from_fields(radii=LANDMARK_RADIUS_M))
        self.columns.add_batch(f"{VIO_RUN_ENTITY}/landmarks", rr.Points3D.columns, t, positions=snapshot.landmark_positions,
                               colors=self.landmark_colors.lookup(snapshot.landmark_ids, LANDMARK_COLOR))
        if map_delta is not None:
            self._log_map_delta(map_delta, snapshot.t_ns)
            map_keyframes: Int64[ndarray, " k"] = map_delta.map_keyframes.t_ns
            self.map_keyframe = int(map_keyframes[-1]) if len(map_keyframes) else None
        self._log_scalars(snapshot, elapsed_ms, t)
        if self.framesets % ATE_EVERY == 0 and len(self.ground_truth):
            # The only two readers of the whole estimate, and the reason it is
            # not built every frameset: three arrays over lists that grow with
            # the run cost 1.64 s over a 4,648-frameset clip to draw a segment
            # from the last two poses.
            estimated: Trajectory = self.estimated()
            # The ATE against the ground truth has already solved this alignment,
            # in the direction its own residuals are measured in; drawing it is
            # that solution inverted, not a second association and a second SVD.
            against_truth: AteResult | None = self._log_ate(estimated, t)
            alignment: SimilarityTransform = IDENTITY if against_truth is None else inverted(against_truth.alignment)
            # The map under the final run's entity sits on the raw run until :meth:`log_final_run` places it.
            for run in (VIO_RUN_ENTITY, RUN_ENTITY):
                self._log_alignment(run, alignment, t)
        if self.framesets % self.block_framesets == 0:
            self.columns.flush()

    def flush(self) -> None:
        """Send every row held so far."""
        self.columns.flush()

    def estimated(self) -> Trajectory:
        """Everything reported so far, on the replay's ``video_time`` clock."""
        return self.poses.trajectory()

    def log_ground_truth_path(self) -> None:
        """Log the ground truth's whole path once, at the last pose (:mod:`simplecv.rerun_walk_path`): the trail shows where
        it had got to, this where it went, from the end of the walk on."""
        estimated: Trajectory = self.estimated()
        if len(self.ground_truth_strip) >= 2 and len(estimated):
            send_trajectory(f"{GT_ENTITY}/{TRAJECTORY}", self.video_t_ns(int(estimated.t_ns[-1])), self.ground_truth_strip.position_m, GT_COLOR,
                            recording=self.columns.recording)
        self.flush()

    def _log_keypoints(self, frame: _core.FlowFrame, t: int) -> None:
        """Each camera's tracked keypoints on its base camera, in that camera's pixels (:func:`slam_rs.frontend_log.log_keypoints`)."""
        for index, (position, scale) in enumerate(zip(self.camera_positions, self.keypoint_scale, strict=True)):
            ids: Int64[ndarray, " n_tracks"] = frame.ids(index)
            tracked: Float32[ndarray, "n_tracks 2"] = frame.positions(index)
            self.columns.static_once(f"{camera_entity(position)}/keypoints", rr.Points2D.from_fields(radii=KEYPOINT_RADIUS_PX * scale))
            self.columns.add_batch(f"{camera_entity(position)}/keypoints", rr.Points2D.columns, t,
                                   positions=tracked if scale == 1.0 else tracked * np.float32(scale), colors=track_colors(ids))

    def _log_ground_truth_trail(self, t_ns: int, t: int) -> None:
        """Draw the segment the ground truth gained by this frameset: from its newest-but-one to its newest pose at or
        before it, on the frameset's row (the view's visible time range accumulates them into the path so far). None
        before its first pose; the alignment transforms bring it and the runs together in the dataset's world.

        Args:
            t_ns: The frameset's estimator stamp.
            t: Its ``video_time``.
        """
        drawn: int = int(np.searchsorted(self.ground_truth_strip.t_ns, t_ns, side="right"))
        if drawn < 1:
            return
        reference: Float64[ndarray, "2 3"] = self.ground_truth_strip.position_m[[max(drawn - 2, 0), drawn - 1]]
        self.columns.static_once(f"{GT_ENTITY}/{TRAIL}", rr.LineStrips3D.from_fields(colors=GT_COLOR, radii=rr.Radius.ui_points(PATH_RADIUS_UI)))
        self.columns.add_batch(f"{GT_ENTITY}/{TRAIL}", rr.LineStrips3D.columns, t, strips=[reference])

    def _draw_slot(
        self, family: SlotFamily, k: int, t_ns: int, pose: Float64[ndarray, " 7"], color: tuple[int, int, int, int], t: int, *, pictures: bool
    ) -> None:
        """Draw one window or marginalized frame on its slot: the rig's pose, and a frustum a camera in the role's colour.

        The pose goes out every frameset (the window refines it). The frusta go out when the slot's role colour
        changed. With ``pictures`` and :attr:`video_entities`, each frustum references its lens's frame of ``t_ns``,
        once for as long as the frame holds the slot.

        Args:
            family: ``window`` or ``marginalized``.
            k: Slot index.
            t_ns: The frame's stamp on the estimator's clock.
            pose: ``world_from_rig`` as ``[tx, ty, tz, qx, qy, qz, qw]``.
            color: The frame's role colour.
            t: The row's ``video_time``: the frameset being logged.
            pictures: Whether the frusta show the frame's pictures.
        """
        slot: str = f"{VIO_RUN_ENTITY}/{family}/{k:02d}"
        recording: rr.RecordingStream | None = self.columns.recording
        if k > self.highest_slots.get(family, -1):
            for camera in self.cameras:
                rr.log(f"{slot}/cam_{camera.index:02d}", rr.Transform3D(translation=camera.imu_T_cam[:3, 3], mat3x3=camera.imu_T_cam[:3, :3]),
                       static=True, recording=recording)
            self.highest_slots[family] = k
        self.columns.add(slot, rr.Transform3D.columns, t, translation=pose[0:3].copy(), quaternion=pose[3:7].copy())
        if self.slot_frusta.get((family, k)) != color:
            self.slot_frusta[family, k] = color
            for position, camera in enumerate(self.cameras):
                self.columns.add(f"{slot}/cam_{camera.index:02d}/pinhole", rr.Pinhole.columns, t, image_from_camera=self.display_intrinsics[position],
                                 resolution=np.array([camera.width, camera.height], dtype=np.float32), camera_xyz=rr.ViewCoordinates.RDF,
                                 image_plane_distance=FRUSTUM_DEPTH_M, color=color)
        shown: int | None = self.slot_pictures.get((family, k))
        wanted: int | None = t_ns if pictures and self.video_entities else None
        if wanted == shown:
            return
        for camera, video in zip(self.cameras, self.video_entities, strict=False):
            entity: str = f"{slot}/cam_{camera.index:02d}/pinhole/video"
            if wanted is None:
                self.columns.add(entity, rr.Clear.columns, t, is_recursive=False)
            else:
                self.columns.add(entity, rr.VideoFrameReference.columns, t, timestamp=self.video_t_ns(wanted), video_reference=video)
        if wanted is None:
            del self.slot_pictures[family, k]
        else:
            self.slot_pictures[family, k] = wanted

    def _clear_slot(self, family: SlotFamily, k: int, t: int) -> None:
        """Clear a slot that just emptied, once: Rerun keeps the latest state, so it stays empty until it is drawn again."""
        self.columns.add(f"{VIO_RUN_ENTITY}/{family}/{k:02d}", rr.Clear.columns, t, is_recursive=True)
        self.slot_frusta.pop((family, k), None)
        self.slot_pictures.pop((family, k), None)

    def _log_window(self, snapshot: _core.VioSnapshot, t: int) -> None:
        """Draw window and marginalized frames as the rig's cameras, coloured by role.

        A window frame takes the lowest free slot when it enters and keeps it until it leaves; a keyframe's frusta
        show its pictures. The frames the last marginalization removed are drawn once more, faded, where the previous
        window had them, on slots of their own in order: they last one frameset, so they show no picture.
        """
        poses: Float64[ndarray, "n_frames 7"] = snapshot.window_poses
        stamps: list[int] = snapshot.window_t_ns.tolist()
        present: set[int] = set(stamps)
        freed: set[int] = {self.window_slots.pop(t_ns) for t_ns in [t_ns for t_ns in self.window_slots if t_ns not in present]}
        taken: set[int] = set(self.window_slots.values())
        for t_ns, pose, keyframe, long_term in zip(stamps, poses, snapshot.window_keyframe, snapshot.window_long_term, strict=True):
            if t_ns not in self.window_slots:
                self.window_slots[t_ns] = next(k for k in itertools.count() if k not in taken)
                taken.add(self.window_slots[t_ns])
            color: tuple[int, int, int, int] = LTKF_COLOR if long_term else KEYFRAME_COLOR if keyframe else POSE_COLOR
            self._draw_slot("window", self.window_slots[t_ns], t_ns, pose, color, t, pictures=bool(keyframe))
        for k in freed - taken:
            self._clear_slot("window", k, t)

        # Removed frames retain the poses from the last snapshot that held them.
        leaving: set[int] = set(snapshot.marginalized.tolist())
        marginalized: list[tuple[int, Float64[ndarray, " 7"]]] = [(t_ns, pose) for t_ns, pose in self.previous_window.items() if t_ns in leaving]
        for k, (t_ns, pose) in enumerate(marginalized):
            self._draw_slot("marginalized", k, t_ns, pose, MARGINALIZED_COLOR, t, pictures=False)
        for k in range(len(marginalized), self.marginalized_count):
            self._clear_slot("marginalized", k, t)
        self.marginalized_count = len(marginalized)
        self.previous_window = {t_ns: pose for t_ns, pose in zip(stamps, poses, strict=True)}

    def map_is_due(self, snapshot: _core.VioSnapshot) -> bool:
        """Whether to read the map now: the map does not hold the window's newest keyframe yet as last read, and the
        keyframes are due (at most every :attr:`map_keyframes_every_ns` of video time).

        The map changes when a keyframe's stamp reaches the back end. Offline that is inside ``track``, so the map is
        read once per keyframe join at the cadence. Realtime the back-end thread may not have stepped it yet: the read
        returns the newest posted delta, and the map is read again at the cadence until it holds that keyframe.
        """
        if len(snapshot.kf_ids) == 0 or int(snapshot.kf_ids[-1]) == self.map_keyframe:
            return False
        return self.map_keyframes_t_ns is None or snapshot.t_ns - self.map_keyframes_t_ns >= self.map_keyframes_every_ns

    def read_map(self, vio: _core.Vio, snapshot: _core.VioSnapshot) -> _core.MapDelta | None:
        """The map's changes since what :attr:`map_point_blocks` holds, when they are due (:meth:`map_is_due`); None
        otherwise or with the back end off. Never waits for the back end."""
        return vio.map_delta(self.map_point_blocks.revision) if self.map_is_due(snapshot) else None

    def log_final_map(self, map_snapshot: _core.MapSnapshot, t_ns: int) -> None:
        """Draw the map once more after the run's flush, keyframes and points, so the recording ends on the final map.

        Args:
            map_snapshot: The map after :meth:`slam_rs._core.Vio.flush` (current in both modes).
            t_ns: Estimator time of the frameset it is drawn at, usually the last one's.
        """
        t: int = self.video_t_ns(t_ns)
        self._log_map_keyframes(map_snapshot.map_keyframes, t_ns, loops=True)
        # The end of the run: one entity holds the whole map from here on, where a layout's overrides find it.
        self.columns.add(MAP_POINT_BLOCKS_ENTITY, rr.Clear.columns, t, is_recursive=True)
        self.columns.static_once(MAP_POINTS_ENTITY, rr.Points3D.from_fields(radii=MAP_POINT_RADIUS_M))
        self.columns.add_batch(MAP_POINTS_ENTITY, rr.Points3D.columns, t, positions=map_snapshot.points_world,
                               colors=self.landmark_colors.lookup(map_snapshot.point_ids, MAP_POINT_COLOR))
        self.map_points_t_ns = t_ns
        log_pose_graph(map_snapshot, self.columns, t)
        self.flush()

    def log_final_alignment(self, final: Trajectory) -> None:
        """Place the final run under its own alignment to the ground truth, from the last pose on.

        The alignment is fitted to the final run, not to the raw one, so a loop the back end closed lands on the ground
        truth. It replaces the raw run's alignment on :data:`RUN_ENTITY`, so the final map drawn there moves with it.
        Without ground truth nothing is drawn: the run is in the estimator's world.

        Args:
            final: The final trajectory (:func:`slam_rs.tracking.final_trajectory`: GBA, then re-anchoring), on the estimator's clock.
        """
        if len(final) < 2 or len(self.ground_truth) == 0:
            return
        alignment: SimilarityTransform = IDENTITY
        if len(final) >= MIN_ASSOCIATED_POSES:
            against_truth: AteResult = ate(final, self.ground_truth)
            if against_truth.n_associated >= MIN_ASSOCIATED_POSES:
                alignment = inverted(against_truth.alignment)
        self._log_alignment(RUN_ENTITY, alignment, self.video_t_ns(int(final.t_ns[-1])))
        self.flush()

    def _log_alignment(self, entity: str, alignment: SimilarityTransform, t: int) -> None:
        """Place one run's whole subtree in the frame its alignment maps into."""
        self.columns.add(entity, rr.Transform3D.columns, t, translation=np.asarray(alignment.dst_t_src, dtype=np.float64),
                         mat3x3=np.asarray(alignment.dst_R_src, dtype=np.float64))

    def _log_map_keyframes(self, keyframes: _core.MapKeyframes, t_ns: int, *, loops: bool) -> None:
        """Draw the back end's keyframes (window ones in the window colour), with ``loops`` its loops, and its pose-graph
        corrections at the frameset of estimator time ``t_ns``."""
        t: int = self.video_t_ns(t_ns)
        colors: UInt8[ndarray, "k 4"] = np.where(
            keyframes.active[:, None], np.array(KEYFRAME_COLOR, dtype=np.uint8), np.array(MAP_KEYFRAME_COLOR, dtype=np.uint8)
        )
        self.columns.static_once(MAP_KEYFRAMES_ENTITY, rr.Points3D.from_fields(radii=MAP_KEYFRAME_RADIUS_M))
        self.columns.add_batch(MAP_KEYFRAMES_ENTITY, rr.Points3D.columns, t, positions=keyframes.poses[:, 0:3], colors=colors)
        if loops:
            log_loops(keyframes, self.columns, t)
        self._log_corrections(keyframes, t)
        self.map_keyframes_t_ns = t_ns

    def _log_corrections(self, keyframes: _core.MapKeyframes, t: int) -> None:
        """Draw the pose-graph corrections at the frameset when there are any; clear them once when they go."""
        corrections: Float64[ndarray, "m 2 3"] = correction_segments(keyframes.raw_poses, keyframes.poses)
        if len(corrections):
            self.columns.static_once(PGO_CORRECTIONS_ENTITY, rr.LineStrips3D.from_fields(colors=PGO_CORRECTION_COLOR, radii=PGO_CORRECTION_RADIUS_M))
            self.columns.add_batch(PGO_CORRECTIONS_ENTITY, rr.LineStrips3D.columns, t, strips=list(corrections))
        elif self.corrections_shown:
            self.columns.add(PGO_CORRECTIONS_ENTITY, rr.Clear.columns, t, is_recursive=False)
        self.corrections_shown = bool(len(corrections))

    def _log_map_delta(self, map_delta: _core.MapDelta, t_ns: int) -> None:
        """While the run goes on: the keyframes and loops, the delta into :attr:`map_point_blocks`, and when the points
        are due the blocks it marked."""
        self._log_map_keyframes(map_delta.map_keyframes, t_ns, loops=self.loops_as_found)
        self.map_point_blocks.apply(
            map_delta.revision, map_delta.map_keyframes.t_ns, map_delta.changed_hosts, map_delta.point_counts, map_delta.point_ids, map_delta.points_world
        )
        if self.map_points_t_ns is not None and t_ns - self.map_points_t_ns < self.map_points_every_ns:
            return
        t: int = self.video_t_ns(t_ns)
        for block, ids, positions in self.map_point_blocks.redraw():
            path: str = f"{MAP_POINT_BLOCKS_ENTITY}/{block:04d}"
            if ids.shape[0]:
                self.columns.static_once(path, rr.Points3D.from_fields(radii=rr.Radius.ui_points(MAP_POINT_BLOCK_RADIUS_UI)))
                self.columns.add_batch(path, rr.Points3D.columns, t, positions=positions, colors=self.landmark_colors.lookup(ids, MAP_POINT_COLOR))
            else:
                self.columns.add(path, rr.Clear.columns, t, is_recursive=False)
        self.map_points_t_ns = t_ns

    def _log_scalars(self, snapshot: _core.VioSnapshot, elapsed_ms: float, t: int) -> None:
        """Log the counters the time-series views plot.

        The estimator's velocity and its two biases used to be logged here, under
        the 3D subtree, where no view could reach them. Plotting them would take
        three more views and not one — metres a second, radians a second and
        metres a second squared share no axis, which is the mixing C73's split
        into seven views undid — so they are dropped instead; the pose they
        belong to is on the trajectory either way.
        """
        counters: dict[str, float] = {
            "num_landmarks": float(len(snapshot.landmark_ids)),
            "num_observations": float(snapshot.num_observations),
            "num_keyframes": float(len(snapshot.kf_ids)),
            "lm_iterations": float(snapshot.lm_iterations),
            "lm_lambda": snapshot.lm_lambda,
            # The cost the frame started and ended the LM loop at: the pair is the
            # convergence trace, and :func:`vio_blueprint` gives it an axis of its
            # own. Both are negative on most frames, following the prior cost convention
            # and not a sign error: the marginalization prior term deliberately drops
            # the 1/2 r^T r (``crates/slam-rs/src/estimator/optimize.rs:71``, D20).
            "lm_error_before": snapshot.lm_error_before,
            "lm_error_after": snapshot.lm_error_after,
            "track_ms": elapsed_ms,
        }
        counters.update({f"stage_ms/{stage}": milliseconds for stage, milliseconds in snapshot.timings_ms.items()})
        for name, value in counters.items():
            self.columns.add(f"{VIO_STATS_ENTITY}/{name}", rr.Scalars.columns, t, scalars=value)

    def _log_ate(self, estimated: Trajectory, t: int) -> AteResult | None:
        """Log the rigid-aligned error of everything reported so far, against ground truth.

        Args:
            estimated: Everything reported so far.
            t: The row's ``video_time``.

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
        self.columns.add(f"{VIO_STATS_ENTITY}/ate_cm/gt", rr.Scalars.columns, t, scalars=100.0 * result.rmse_m)
        return result


BOOTSTRAP_ENTITY: str = f"{RUN_ENTITY}/bootstrap"
"""Where the inertial bootstrap's attempts go, one row per attempt at the frameset that ran it."""


@dataclass(slots=True)
class BootstrapLog:
    """The inertial bootstrap in the result recording: each attempt when it happens, the status as a recording property.

    The attempts are logged as the run makes them, so a bootstrap that never accepts and stops the run is still in
    the recording. Nothing is logged on the stereo start.
    """

    cam_time_offset_ns: int
    """The feed's :attr:`ImuCalib.cam_time_offset_ns`: the attempts' stamps are on the estimator's clock."""
    columns: ColumnLog
    """Where the attempts' rows are held (the run's :attr:`VioLogger.columns`), and the recording the status goes to."""
    logged: int = 0
    """Attempts already in the recording."""
    accepted: bool = False
    """Whether the recording already holds the accepted status."""

    def log_new(self, vio: _core.Vio) -> None:
        """Log the attempts made since the last call, at their own frameset time, and the status when it changed.

        Args:
            vio: The pipeline that runs the bootstrap.
        """
        report: _core.BootstrapReport | None = vio.bootstrap_report()
        if report is None:
            return
        attempts: list[_core.BootstrapAttempt] = report.attempts
        for attempt in attempts[self.logged :]:
            t: int = attempt.last_t_ns - self.cam_time_offset_ns
            self.columns.add(BOOTSTRAP_ENTITY, rr.AnyValues.columns, t, verdict=attempt.verdict, accepted=attempt.accepted, first_t_ns=attempt.first_t_ns,
                             frames=attempt.frames, tracks=attempt.tracks, tracks_used=attempt.tracks_used)
            quality: dict[str, float | None] = {
                "window_s": (attempt.last_t_ns - attempt.first_t_ns) * 1e-9,
                "excitation_mps2": attempt.excitation_mps2,
                "residual_rms_rad": attempt.residual_rms_rad,
                "condition": attempt.condition,
                "gravity_norm_unconstrained": attempt.gravity_norm_unconstrained,
                "scale_sigma": attempt.scale_sigma,
            }
            for name, value in quality.items():
                if value is not None:
                    self.columns.add(f"{BOOTSTRAP_ENTITY}/{name}", rr.Scalars.columns, t, scalars=value)
        accepted: bool = report.start_t_ns is not None
        if len(attempts) == self.logged and accepted == self.accepted:
            return
        self.logged = len(attempts)
        self.accepted = accepted
        # A recording property holds the latest value: on a run that stops this is the status where it stopped.
        rr.send_property(
            "bootstrap",
            rr.AnyValues(
                status="accepted" if accepted else "bootstrapping",
                attempts=len(attempts),
                start_t_ns=report.start_t_ns,
                accepted_at_t_ns=report.accepted_at_t_ns,
                replayed=report.replayed,
            ),
            recording=self.columns.recording,
        )

    @staticmethod
    def summary(vio: _core.Vio, first_t_ns: int) -> str | None:
        """One line on the bootstrap for the console, or None on the stereo start.

        Args:
            vio: The pipeline that ran.
            first_t_ns: The segment's first frameset time, the origin of the printed seconds.
        """
        report: _core.BootstrapReport | None = vio.bootstrap_report()
        if report is None:
            return None
        verdicts: list[str] = [attempt.verdict for attempt in report.attempts]
        if report.start_t_ns is None or report.accepted_at_t_ns is None:
            return f"inertial bootstrap: {len(verdicts)} attempts {verdicts}; no window accepted"
        return (
            f"inertial bootstrap: {len(verdicts)} attempts {verdicts}; started at {(report.start_t_ns - first_t_ns) * 1e-9:.2f} s, "
            f"first pose at {(report.accepted_at_t_ns - first_t_ns) * 1e-9:.2f} s, {report.replayed} framesets replayed"
        )


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
    calls_seen: int = 0
    """``track`` calls already in :attr:`times`."""
    bootstrap: "BootstrapLog" = field(init=False)
    """The inertial bootstrap's attempts, logged as they happen."""

    @property
    def elapsed_ms(self) -> list[float]:
        """Wall time each ``track`` call that tracked took, in frameset order."""
        return self.lockstep.elapsed_ms

    @property
    def pending(self) -> list[Frameset]:
        """Framesets still held for want of the inertial samples that cover them."""
        return self.lockstep.pending

    def __post_init__(self) -> None:
        """Start the bootstrap's log on the logger's recording."""
        self.bootstrap = BootstrapLog(self.logger.cam_time_offset_ns, self.logger.columns)

    def run(self, frameset: Frameset) -> None:
        """Track the frameset and everything its samples now cover, and log each one (a caller that drives the lockstep
        itself; :func:`slam_rs.tracking.drive` calls :meth:`tracked` and :meth:`consumed` instead).

        Args:
            frameset: The frameset to track, with the samples since the previous one.
        """
        for held, result in self.lockstep.push(frameset):
            self.tracked(held, result)
        self.consumed(frameset)

    def flush(self) -> None:
        """Log the last lagged frame, then the final map (the flush made it current), before completing paths or exporting poses."""
        for held, result in self.lockstep.flush():
            self.tracked(held, result)
        self.flushed()

    def consumed(self, frameset: Frameset) -> None:
        """After a frameset's push: the new ``track`` calls' stage times, the keypoints held and the bootstrap's attempts.

        Args:
            frameset: The frameset just pushed.
        """
        vio: _core.Vio = self.lockstep.vio
        calls: list[float] = self.lockstep.consumed_ms[self.calls_seen :]
        self.calls_seen = len(self.lockstep.consumed_ms)
        if calls:
            # The breakdown belongs to the last call; a frameset released from the IMU hold adds a second one.
            frontend: dict[str, float] = vio.frontend_timings_ms
            flow: float = frontend["pyramid"] + frontend["detect"] + frontend["track"] + frontend["stereo"]
            self.times.add("track", sum(calls))
            self.times.add("frontend", flow)
            self.times.add("imu", frontend["imu"])
            self.times.add("estimator", calls[-1] - flow - frontend["imu"])
            # One copy of the frontend's keypoints per frameset, for the count.
            frame: _core.FlowFrame | None = vio.flow_frame()
            if frame is not None:
                self.keypoints.append(sum(frame.num_tracks(camera) for camera in range(vio.camera_count)))
        self.bootstrap.log_new(vio)

    def tracked(self, held: Frameset, result: _core.VioResult) -> None:
        """Log a result with the keypoints and timestamp of its own input frame."""
        started: float = time.monotonic()
        snapshot: _core.VioSnapshot | None = self.lockstep.vio.snapshot()
        frame: _core.FlowFrame = self.lockstep.flow_frame()
        assert snapshot is not None, f"frameset {held.t_ns} tracked without a window snapshot"
        assert frame.t_ns == result.t_ns
        map_delta: _core.MapDelta | None = self.logger.read_map(self.lockstep.vio, snapshot)
        self.logger.log(result, snapshot, frame, self.lockstep.elapsed_ms[-1], map_delta, held.colors)
        self.times.add("log", 1e3 * (time.monotonic() - started))

    def flushed(self) -> None:
        """After the estimator's flush: the final map (the flush made it current) at the last pose, and every row held."""
        final: _core.MapSnapshot | None = self.lockstep.vio.map_snapshot()
        last_t_ns: int | None = self.logger.poses.last_t_ns()
        if final is not None and last_t_ns is not None:
            self.logger.log_final_map(final, last_t_ns)
        self.bootstrap.log_new(self.lockstep.vio)
        self.logger.flush()

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
                    overrides={f"{run}/{TRAIL}": trail for run in (RUN_ENTITY, VIO_RUN_ENTITY, GT_ENTITY)},
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
