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
from simplecv.stage_times import StageTimes

from slam_rs import _core
from slam_rs.catalog_feed import IMU_ENTITY, RIG_ENTITY, TIMELINE, CameraCalib, Frameset, ImuStream
from slam_rs.follow_frame import FloorGrid, FloorHeight, HeadingFilter, floor_entity, floor_grid, follow_entity, follow_yaw, yaw_quaternion_xyzw
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
"""Least video time between two draws of the map points' changed blocks (:class:`MapPointBlocks`). The final map is
drawn whole once more after the flush."""
MAP_POINT_BLOCKS_ENTITY: str = f"{MAP_POINTS_ENTITY}/blocks"
"""Parent of the map-point blocks the run draws while it goes on; cleared at the final draw, which puts the whole map
on :data:`MAP_POINTS_ENTITY` itself (where a layout's overrides find it)."""
KEYFRAMES_PER_BLOCK: int = 32
"""Host keyframes per map-point block: one entity per block keeps the viewer's per-entity cost low (~47 entities on a
24-minute walk) while a local BA or a loop closure redraws only the blocks it moved."""
MAP_POINT_TOLERANCE_M: float = 0.001
"""A block is drawn again when one of its points moved more than this since its last draw, or its point set changed."""
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
SlotFamily: TypeAlias = Literal["window", "marginalized"]
"""The two families of numbered frame slots under the run: the estimator's window, and what its last marginalization removed."""
UNIFIED_SPHERE_MODELS: tuple[str, ...] = ("mei_v3", "mei_v6", "ds")
"""Camera models whose focal is the unified sphere's ``gamma`` and whose first coefficient is ``xi`` (:func:`display_intrinsics`)."""
FLOOR_GRID_RADIUS_UI: float = 0.5
"""Stroke of a floor-grid edge, ui points: hairlines at every zoom."""
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


def log_rig(cameras: tuple[CameraCalib, ...], entity_prefix: str, *, display: bool = False) -> None:
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
    """
    for camera in cameras:
        node: str = f"{entity_prefix}/cam_{camera.index:02d}"
        rr.log(
            node,
            rr.Transform3D(translation=camera.imu_T_cam[:3, 3], mat3x3=camera.imu_T_cam[:3, :3]),
            static=True,
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
        )


@dataclass(slots=True)
class PoseLog:
    """Every pose the run reported, in arrays that double when full: the whole estimate is a slice of them.

    Lists of small arrays rebuilt into one array at every read grew with the run (every 30 framesets for the ATE and
    the alignment: ~0.85 ms a frameset by the end of a 24-minute walk). A slice copies nothing, and later appends only
    write past it, so an estimate taken earlier stays as it was.
    """

    capacity: int = 1024
    """Rows the arrays start with."""
    count: int = 0
    """Poses appended."""
    t_ns: Int64[ndarray, " c"] = field(init=False)
    """Timestamps, the first ``count`` rows used."""
    position_m: Float64[ndarray, "c 3"] = field(init=False)
    """Positions, metres."""
    quaternion_wxyz: Float64[ndarray, "c 4"] = field(init=False)
    """Orientations, scalar first."""

    def __post_init__(self) -> None:
        self.t_ns = np.zeros(self.capacity, dtype=np.int64)
        self.position_m = np.zeros((self.capacity, 3), dtype=np.float64)
        self.quaternion_wxyz = np.zeros((self.capacity, 4), dtype=np.float64)

    def __len__(self) -> int:
        return self.count

    def append(self, t_ns: int, position_m: Float64[ndarray, " 3"], quaternion_wxyz: Float64[ndarray, " 4"]) -> None:
        """Add one pose; the arrays double (a copy) when full."""
        if self.count == self.t_ns.shape[0]:
            size: int = 2 * max(self.count, 1)
            self.t_ns = np.concatenate([self.t_ns, np.zeros(size - self.count, dtype=np.int64)])
            self.position_m = np.concatenate([self.position_m, np.zeros((size - self.count, 3), dtype=np.float64)])
            self.quaternion_wxyz = np.concatenate([self.quaternion_wxyz, np.zeros((size - self.count, 4), dtype=np.float64)])
        self.t_ns[self.count] = t_ns
        self.position_m[self.count] = position_m
        self.quaternion_wxyz[self.count] = quaternion_wxyz
        self.count += 1

    def trajectory(self) -> Trajectory:
        """Every pose so far, as slices of the arrays (no copy)."""
        return Trajectory(t_ns=self.t_ns[: self.count], position_m=self.position_m[: self.count], quaternion_wxyz=self.quaternion_wxyz[: self.count])

    def last_t_ns(self) -> int | None:
        """The newest pose's timestamp; None before the first."""
        return int(self.t_ns[self.count - 1]) if self.count else None

    def last_positions(self, n: int) -> Float64[ndarray, "m 3"]:
        """The newest ``n`` positions (fewer at the start of a run)."""
        return self.position_m[max(0, self.count - n) : self.count]


@dataclass(slots=True)
class MapPointBlocks:
    """The map's points as last read, by host keyframe, and which blocks of ``keyframes_per_block`` keyframes (in the
    order they joined the map) to draw again.

    Drawing the whole map at every cadence grew the recording with clip length x map size (1.0 GB of map points on a
    24-minute walk), and reading it whole each second cost O(map) (``Vio.map_snapshot``). :meth:`apply` takes each
    ``Vio.map_delta`` (the hosts a write touched and their points) and marks their blocks; :meth:`redraw` returns the
    marked blocks whose point set changed or whose points moved past ``tolerance_m`` since their last draw.
    """

    keyframes_per_block: int = KEYFRAMES_PER_BLOCK
    """Host keyframes a block holds (:data:`KEYFRAMES_PER_BLOCK`)."""
    tolerance_m: float = MAP_POINT_TOLERANCE_M
    """Largest point move, metres, that does not redraw its block (:data:`MAP_POINT_TOLERANCE_M`)."""
    block_of_host: dict[int, int] = field(default_factory=dict)
    """Block of every host keyframe seen so far, by keyframe timestamp: a keyframe keeps its block for the whole run."""
    host_points: dict[int, tuple[Int64[ndarray, " p"], Float32[ndarray, "p 3"]]] = field(default_factory=dict)
    """Each host's point ids (ascending) and world positions as the last delta that touched it gave them."""
    marked: set[int] = field(default_factory=set)
    """Blocks a delta touched since the last :meth:`redraw`."""
    drawn: dict[int, tuple[Int64[ndarray, " p"], Float32[ndarray, "p 3"]]] = field(default_factory=dict)
    """Each block's point ids (ascending) and positions as last drawn."""

    def apply(
        self,
        keyframe_t_ns: Int64[ndarray, " k"],
        changed_hosts: Int64[ndarray, " c"],
        point_ids: Int64[ndarray, " p"],
        point_hosts: Int64[ndarray, " p"],
        points_world: Float32[ndarray, "p 3"],
    ) -> None:
        """Take one map delta: every changed host's points are replaced by the delta's (none: the host lost them all,
        or left the map), and its block is marked.

        Args:
            keyframe_t_ns: :attr:`slam_rs._core.MapDelta.keyframe_t_ns`, the map's keyframes now.
            changed_hosts: :attr:`slam_rs._core.MapDelta.changed_hosts`.
            point_ids: :attr:`slam_rs._core.MapDelta.point_ids`, the changed hosts' points.
            point_hosts: :attr:`slam_rs._core.MapDelta.point_hosts`.
            points_world: :attr:`slam_rs._core.MapDelta.points_world`.
        """
        order: Int64[ndarray, " p"] = np.lexsort((point_ids, point_hosts))
        point_ids, point_hosts, points_world = point_ids[order], point_hosts[order], points_world[order]
        hosts, starts = np.unique(point_hosts, return_index=True)
        ends: list[int] = [*starts[1:].tolist(), point_hosts.shape[0]] if starts.shape[0] else []
        delivered: dict[int, tuple[int, int]] = dict(zip(hosts.tolist(), zip(starts.tolist(), ends, strict=True), strict=True))
        present: set[int] = set(keyframe_t_ns.tolist())
        for host in changed_hosts.tolist():
            span = delivered.get(host)
            if host in present and span is not None:
                if host not in self.block_of_host:
                    self.block_of_host[host] = len(self.block_of_host) // self.keyframes_per_block
                self.host_points[host] = (point_ids[span[0] : span[1]], points_world[span[0] : span[1]])
            elif self.host_points.pop(host, None) is None:
                continue
            self.marked.add(self.block_of_host[host])

    def redraw(self) -> list[tuple[int, Int64[ndarray, " q"], Float32[ndarray, "q 3"]]]:
        """The marked blocks to draw again, ascending, with their points (ids ascending); a block left with no points
        comes back empty. Clears the marks."""
        hosts_of: dict[int, list[int]] = {}
        for host, block in self.block_of_host.items():
            if block in self.marked and host in self.host_points:
                hosts_of.setdefault(block, []).append(host)
        redraw: list[tuple[int, Int64[ndarray, " q"], Float32[ndarray, "q 3"]]] = []
        for block in sorted(self.marked):
            parts = [self.host_points[host] for host in hosts_of.get(block, [])]
            if not parts:
                if self.drawn.pop(block, None) is not None:
                    redraw.append((block, np.zeros(0, dtype=np.int64), np.zeros((0, 3), dtype=np.float32)))
                continue
            ids: Int64[ndarray, " q"] = np.concatenate([part[0] for part in parts])
            order: Int64[ndarray, " q"] = np.argsort(ids, kind="stable")
            ids, positions = ids[order], np.concatenate([part[1] for part in parts])[order]
            last = self.drawn.get(block)
            if last is None or not np.array_equal(last[0], ids) or float(np.abs(last[1] - positions).max()) > self.tolerance_m:
                self.drawn[block] = (ids, positions)
                redraw.append((block, ids, positions))
        self.marked.clear()
        return redraw


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
    layer, which writes the final poses there (:func:`slam_rs.catalog_layer.write_layer`). Each run's level follow
    frame and floor are written either way (:mod:`slam_rs.follow_frame`): the raw run's as it goes, the final run's
    with it (:meth:`log_final_run`)."""
    video_entities: tuple[str, ...] = ()
    """The dataset's ``VideoStream`` of each camera, in :attr:`cameras` order. With them the run's rig references the
    current frame of each lens under its ``pinhole`` and a window keyframe the frames of its own time
    (``VideoFrameReference``: nothing is encoded again); empty, the frusta stay empty."""
    heading: HeadingFilter = field(default_factory=HeadingFilter)
    """The follow frame's heading so far: the direction of travel, low-passed."""
    floor_height: FloorHeight = field(default_factory=FloorHeight)
    """The rig's height over the floor so far, from the landmarks each window held around it."""
    poses: PoseLog = field(default_factory=PoseLog)
    """Every pose the run reported, for the CSV export, the ATE and the per-frameset path segment."""
    ground_truth_strip: Trajectory = field(init=False)
    """The ground truth thinned to the frameset cadence: what the drawn strip is taken from."""
    previous_window: dict[int, Float64[ndarray, " 7"]] = field(default_factory=dict)
    """Previous window poses by timestamp, used to draw marginalized frames."""
    pinholes: dict[tuple[int, tuple[int, int, int, int]], rr.Pinhole] = field(init=False)
    """Each camera's frustum by position in :attr:`cameras` and role colour, reused for every slot and frameset."""
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
    map_point_blocks: MapPointBlocks = field(default_factory=MapPointBlocks)
    """The map's points by host and the blocks as last drawn, so a draw logs only what changed."""
    map_revision: int = 0
    """The map revision of the last delta read: the next read asks only for what changed after it."""

    def __post_init__(self) -> None:
        """Log the estimated rig's static geometry and thin the references."""
        log_rig(self.cameras, f"{VIO_RUN_ENTITY}/rig", display=True)
        for camera, video in zip(self.cameras, self.video_entities, strict=False):
            rr.log(f"{VIO_RUN_ENTITY}/rig/cam_{camera.index:02d}/pinhole/video", rr.VideoFrameReference.from_fields(video_reference=video), static=True)
        self.ground_truth_strip = at_frameset_cadence(self.ground_truth, self.frame_t_ns)
        self.pinholes = {
            (position, color): rr.Pinhole(
                image_from_camera=display_intrinsics(camera),
                resolution=[camera.width, camera.height],
                camera_xyz=rr.ViewCoordinates.RDF,
                image_plane_distance=FRUSTUM_DEPTH_M,
                color=color,
            )
            for position, camera in enumerate(self.cameras)
            for color in (KEYFRAME_COLOR, LTKF_COLOR, POSE_COLOR, MARGINALIZED_COLOR)
        }

    def log(
        self,
        result: _core.VioResult,
        snapshot: _core.VioSnapshot,
        frame: _core.FlowFrame,
        elapsed_ms: float,
        map_delta: _core.MapDelta | None = None,
        colors: list[UInt8[ndarray, "ch cw 3"]] | None = None,
    ) -> None:
        """Log one tracked frameset: the keypoints, the estimated and ground-truth paths, the rig, the window, the landmarks and the counters.

        Args:
            result: What ``track`` returned; only called where it tracked.
            snapshot: The window and the frame's statistics behind that result.
            frame: The keypoints the estimator's own frontend tracked.
            elapsed_ms: Wall time the ``track`` call took.
            map_delta: The back end's map changes since :attr:`map_revision`, when the caller read them (see
                :meth:`map_is_due`); drawn under the run.
            colors: The frameset's frames in RGB (:attr:`Frameset.colors`), which colour the landmarks; None draws them grey.
        """
        pose: Float64[ndarray, " 7"] | None = result.world_from_rig
        assert pose is not None, f"Tracking result at {result.t_ns} has no pose"
        self.framesets += 1
        self.poses.append(result.t_ns, pose[0:3], np.roll(pose[3:7], 1))

        rig_pose: rr.Transform3D = rr.Transform3D(translation=pose[0:3], quaternion=rr.Quaternion(xyzw=pose[3:7]))
        rr.log(f"{VIO_RUN_ENTITY}/rig", rig_pose)
        for camera, _ in zip(self.cameras, self.video_entities, strict=False):
            rr.log(f"{VIO_RUN_ENTITY}/rig/cam_{camera.index:02d}/pinhole/video", rr.VideoFrameReference(nanoseconds=result.t_ns - self.cam_time_offset_ns))
        if self.animate_base_rig:
            rr.log(RIG_ENTITY, rig_pose)
        half_yaw: float = 0.5 * self.heading.update(result.t_ns, pose[0:3])
        rr.log(follow_entity(VIO_RUN_ENTITY), rr.Transform3D(translation=pose[0:3], quaternion=rr.Quaternion(xyzw=[0.0, 0.0, np.sin(half_yaw), np.cos(half_yaw)])))
        self.floor_height.add(pose[0:3], snapshot.landmark_positions.astype(np.float64))
        if colors is not None:
            for camera, color in enumerate(colors):
                self.landmark_colors.remember(frame.ids(camera), frame.positions(camera), color, self.tracked_resolution[camera])
        # The estimator's own frontend output, on the frontend rung's paths and
        # in its palette (:func:`slam_rs.frontend_log.log_keypoints`).
        log_keypoints(frame)
        self._log_paths()
        self._log_window(snapshot)
        self._log_landmarks(snapshot)
        if map_delta is not None:
            self._log_map_delta(map_delta, snapshot.t_ns)
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
        return self.poses.trajectory()

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
        self.log_floor_grid(VIO_RUN_ENTITY, self.estimated())

    def log_floor_grid(self, run_entity: str, trajectory: Trajectory) -> None:
        """Lay the floor grid under one run's walk, when the landmarks around the rig gave a floor.

        The grid lies the rig's carrying height (:attr:`floor_height`) under ``trajectory``. Its edges go on
        ``video_time`` at the time the walk passed them, a row per stamp, so a view chooses how much of the floor it
        shows around the cursor.

        Args:
            run_entity: The run the floor belongs to (:func:`slam_rs.follow_frame.floor_entity`).
            trajectory: That run's poses, on the estimator's clock.
        """
        height: float | None = self.floor_height.height_m()
        if height is None or len(trajectory) < 2:
            return
        grid: FloorGrid = floor_grid(trajectory.t_ns, trajectory.position_m, height)
        stamps, counts = np.unique(grid.t_ns, return_counts=True)
        entity: str = floor_entity(run_entity)
        rr.send_columns(
            entity,
            indexes=[rr.TimeColumn(TIMELINE, duration=(stamps - self.cam_time_offset_ns).astype("timedelta64[ns]"))],
            columns=rr.LineStrips3D.columns(strips=grid.strips, colors=grid.colors).partition(counts.tolist()),
        )
        rr.log(entity, rr.LineStrips3D.from_fields(radii=rr.Radius.ui_points(FLOOR_GRID_RADIUS_UI)), static=True)

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
        t_ns: int = self.poses.last_t_ns() or 0
        if len(self.poses) >= 2:
            segment: Float64[ndarray, "2 3"] = self.poses.last_positions(2).copy()
            rr.log(f"{VIO_RUN_ENTITY}/trajectory", rr.LineStrips3D([segment], colors=ESTIMATE_COLOR, radii=0.004))
        drawn: int = int(np.searchsorted(self.ground_truth_strip.t_ns, t_ns, side="right"))
        if drawn < 2:
            return
        rr.log(f"{GT_ENTITY}/trajectory", rr.LineStrips3D([self.ground_truth_strip.position_m[drawn - 2 : drawn]], colors=GT_COLOR, radii=0.004))

    def _draw_slot(self, family: SlotFamily, k: int, t_ns: int, pose: Float64[ndarray, " 7"], color: tuple[int, int, int, int], *, pictures: bool) -> None:
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
            pictures: Whether the frusta show the frame's pictures.
        """
        slot: str = f"{VIO_RUN_ENTITY}/{family}/{k:02d}"
        if k > self.highest_slots.get(family, -1):
            for camera in self.cameras:
                rr.log(f"{slot}/cam_{camera.index:02d}", rr.Transform3D(translation=camera.imu_T_cam[:3, 3], mat3x3=camera.imu_T_cam[:3, :3]), static=True)
            self.highest_slots[family] = k
        rr.log(slot, rr.Transform3D(translation=pose[0:3], quaternion=rr.Quaternion(xyzw=pose[3:7])))
        if self.slot_frusta.get((family, k)) != color:
            self.slot_frusta[family, k] = color
            for position, camera in enumerate(self.cameras):
                rr.log(f"{slot}/cam_{camera.index:02d}/pinhole", self.pinholes[position, color])
        shown: int | None = self.slot_pictures.get((family, k))
        wanted: int | None = t_ns if pictures and self.video_entities else None
        if wanted == shown:
            return
        for camera, video in zip(self.cameras, self.video_entities, strict=False):
            entity: str = f"{slot}/cam_{camera.index:02d}/pinhole/video"
            if wanted is None:
                rr.log(entity, rr.Clear(recursive=False))
            else:
                rr.log(entity, rr.VideoFrameReference(nanoseconds=wanted - self.cam_time_offset_ns, video_reference=video))
        if wanted is None:
            del self.slot_pictures[family, k]
        else:
            self.slot_pictures[family, k] = wanted

    def _clear_slot(self, family: SlotFamily, k: int) -> None:
        """Clear a slot that just emptied, once: Rerun keeps the latest state, so it stays empty until it is drawn again."""
        rr.log(f"{VIO_RUN_ENTITY}/{family}/{k:02d}", rr.Clear(recursive=True))
        self.slot_frusta.pop((family, k), None)
        self.slot_pictures.pop((family, k), None)

    def _log_window(self, snapshot: _core.VioSnapshot) -> None:
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
            self._draw_slot("window", self.window_slots[t_ns], t_ns, pose, color, pictures=bool(keyframe))
        for k in freed - taken:
            self._clear_slot("window", k)

        # Removed frames retain the poses from the last snapshot that held them.
        leaving: set[int] = set(snapshot.marginalized.tolist())
        marginalized: list[tuple[int, Float64[ndarray, " 7"]]] = [(t_ns, pose) for t_ns, pose in self.previous_window.items() if t_ns in leaving]
        for k, (t_ns, pose) in enumerate(marginalized):
            self._draw_slot("marginalized", k, t_ns, pose, MARGINALIZED_COLOR, pictures=False)
        for k in range(len(marginalized), self.marginalized_count):
            self._clear_slot("marginalized", k)
        self.marginalized_count = len(marginalized)
        self.previous_window = {t_ns: pose for t_ns, pose in zip(stamps, poses, strict=True)}

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
        self._log_map_keyframes(map_snapshot, t_ns)
        # The end of the run: one entity holds the whole map from here on, where a layout's overrides find it.
        rr.log(MAP_POINT_BLOCKS_ENTITY, rr.Clear(recursive=True))
        rr.log(MAP_POINTS_ENTITY, rr.Points3D(map_snapshot.points_world, colors=self.landmark_colors.lookup(map_snapshot.point_ids, MAP_POINT_COLOR), radii=0.004))
        self.map_points_t_ns = t_ns
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
        # What a view of the final run needs at every cursor, not only from the last frameset on: the whole path
        # (static, as the raw run's), its level follow frame and its floor, on the walk's own times.
        rr.log(f"{RUN_ENTITY}/path", rr.LineStrips3D([final.position_m], colors=(*FINAL_COLOR, ROUTE_ALPHA), radii=ROUTE_RADIUS_M), static=True)
        rr.send_columns(
            follow_entity(RUN_ENTITY),
            indexes=[rr.TimeColumn(TIMELINE, duration=(final.t_ns - self.cam_time_offset_ns).astype("timedelta64[ns]"))],
            columns=rr.Transform3D.columns(translation=final.position_m, quaternion=yaw_quaternion_xyzw(follow_yaw(final.t_ns, final.position_m))),
        )
        self.log_floor_grid(RUN_ENTITY, final)
        alignment: SimilarityTransform = IDENTITY
        if len(self.ground_truth) and len(final) >= MIN_ASSOCIATED_POSES:
            against_truth: AteResult = ate(final, self.ground_truth)
            if against_truth.n_associated >= MIN_ASSOCIATED_POSES:
                alignment = inverted(against_truth.alignment)
        log_alignment(RUN_ENTITY, alignment)

    def _log_map_keyframes(self, map_view: _core.MapSnapshot | _core.MapDelta, t_ns: int) -> None:
        """Draw the back end's keyframes (window ones in the window colour) and its loops at the cursor."""
        colors: UInt8[ndarray, "k 4"] = np.where(
            map_view.keyframe_active[:, None], np.array(KEYFRAME_COLOR, dtype=np.uint8), np.array(MAP_KEYFRAME_COLOR, dtype=np.uint8)
        )
        rr.log(MAP_KEYFRAMES_ENTITY, rr.Points3D(map_view.keyframe_poses[:, 0:3], colors=colors, radii=0.012))
        log_loops(map_view)
        self.map_keyframes_t_ns = t_ns

    def _log_map_delta(self, map_delta: _core.MapDelta, t_ns: int) -> None:
        """While the run goes on: the keyframes and loops, the delta into :attr:`map_point_blocks`, and when the points
        are due only the blocks whose points changed (their size comes from the view: no radii logged)."""
        self._log_map_keyframes(map_delta, t_ns)
        self.map_point_blocks.apply(map_delta.keyframe_t_ns, map_delta.changed_hosts, map_delta.point_ids, map_delta.point_hosts, map_delta.points_world)
        self.map_revision = map_delta.revision
        if self.map_points_t_ns is not None and t_ns - self.map_points_t_ns < self.map_points_every_ns:
            return
        for block, ids, positions in self.map_point_blocks.redraw():
            path: str = f"{MAP_POINT_BLOCKS_ENTITY}/{block:04d}"
            if ids.shape[0]:
                rr.log(path, rr.Points3D(positions, colors=self.landmark_colors.lookup(ids, MAP_POINT_COLOR)))
            else:
                rr.log(path, rr.Clear(recursive=False))
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
    log_rig(cameras, RIG_ENTITY)


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
        if last_t_ns is None:
            last_t_ns = self.logger.poses.last_t_ns()
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
        map_delta: _core.MapDelta | None = self.lockstep.vio.map_delta(self.logger.map_revision) if self.logger.map_is_due(snapshot) else None
        self.logger.log(result, snapshot, frame, self.lockstep.elapsed_ms[-1], map_delta, held.colors)

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
                    # The map-point blocks log no radii, so each view (and a dataset's layout) sizes them.
                    defaults=[rr.Points3D.from_fields(radii=0.004)],
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
