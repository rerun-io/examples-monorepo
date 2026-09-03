"""Run LAMP on four outward Robocap cameras streamed from a Rerun catalog."""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Protocol, runtime_checkable

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
from jaxtyping import Float32, Float64, Int64, UInt8
from numpy import ndarray
from posekit.predictions import BoxDetections, Keypoints2d
from posekit.skeletons import COCO_17
from rerun.catalog import CatalogClient, DatasetEntry, DatasetView
from scipy.spatial.transform import Rotation, Slerp
from simplecv.camera_parameters import Fisheye62Parameters
from simplecv.rerun_dataloader import open_segment_decoder
from simplecv.rerun_log_utils import RerunTyroConfig
from simplecv.rerun_rig_logger import log_rig_static
from simplecv.rig import CameraSensor, Rig, RigCalibration
from simplecv.rrd_query_utils import first_valid_value

from lamptrack.cameras import RigCamera
from lamptrack.catalog_rig import RIG, TIMELINE, read_fisheye_camera, read_rig_poses
from lamptrack.models.lamp import AnnotatedLampTrackerUnion, Frameset, LampConfig, LampStep, LampTracker, PersonState
from lamptrack.third_party.lamp.core.types import SMPL_SKELETON_EDGES, color_from_id


@runtime_checkable
class VideoFrame(Protocol):
    """Decoded RGB frame returned by TorchCodec."""

    data: UInt8[torch.Tensor, "3 h w"]


@runtime_checkable
class SegmentVideoDecoder(Protocol):
    """TorchCodec method used by the catalog loop."""

    def get_frame_at(self, index: int) -> VideoFrame:
        """Decode one frame by sample index."""
        ...


@dataclass(frozen=True, slots=True)
class Config:
    """Robocap segment, sampling, tracking, and Rerun output configuration."""

    catalog_url: str = "rerun+http://127.0.0.1:51235"
    """Rerun catalog endpoint."""
    dataset: str = "robocap"
    """Catalog dataset name."""
    segment_id: str = "robocap__f408193e6447b3b0__s00000029"
    """Long street-walk segment with visible pedestrians."""
    cams: tuple[str, ...] = ("cam_00", "cam_01", "cam_04", "cam_05")
    """LAMP view order: left-front, right-front, left, right."""
    fps: float = 10.0
    """Uniform output sampling rate."""
    start_s: float = 1152.0
    """Absolute ``video_time`` second at which processing starts."""
    max_seconds: float | None = 120.0
    """Maximum processed duration; ``None`` runs to the shared video end."""
    floor_z: float | None = None
    """Optional floor height in SLAM-world metres; unknown by default."""
    log_rays: bool = False
    """Log native KB4 keypoint rays in the 3D view."""
    tracker: AnnotatedLampTrackerUnion = field(default_factory=LampConfig)
    """LAMP tracker and PoseKit model configuration."""
    rr_config: RerunTyroConfig = field(
        default_factory=lambda: RerunTyroConfig(application_id="lamp_robocap_catalog", headless=True)
    )
    """Rerun viewer, save, or connection behavior."""

    def __post_init__(self) -> None:
        """Reject values that would produce an invalid sampling loop."""
        if len(self.cams) != 4 or len(set(self.cams)) != 4:
            raise ValueError(f"cams must contain four distinct camera IDs, got {self.cams}")
        if self.fps <= 0.0:
            raise ValueError(f"fps must be positive, got {self.fps}")
        if self.max_seconds is not None and self.max_seconds <= 0.0:
            raise ValueError(f"max_seconds must be positive or None, got {self.max_seconds}")


@dataclass(frozen=True, slots=True)
class RunMetrics:
    """Measured stage timing and people counts for one catalog run."""

    frames: int
    unique_people: int
    frames_with_people: int
    detector_mean_ms: float
    pose_mean_ms: float
    tracker_mean_ms: float
    lifter_mean_ms: float
    smoothing_mean_ms: float
    total_mean_ms: float


def build_time_grid(
    video_times: list[np.ndarray],
    *,
    fps: float,
    start_s: float,
    max_seconds: float | None,
) -> Int64[ndarray, "frames"]:
    """Build a uniform nanosecond grid inside every selected video stream."""
    if not video_times or any(len(times) == 0 for times in video_times):
        raise ValueError("Every selected camera must contain at least one video sample")
    if fps <= 0.0:
        raise ValueError(f"fps must be positive, got {fps}")
    starts = [int(np.asarray(times[0]).astype("timedelta64[ns]").astype(np.int64)) for times in video_times]
    ends = [int(np.asarray(times[-1]).astype("timedelta64[ns]").astype(np.int64)) for times in video_times]
    first_ns = max(max(starts), round(start_s * 1e9))
    end_ns = min(ends)
    if max_seconds is not None:
        end_ns = min(end_ns, first_ns + round(max_seconds * 1e9))
    if first_ns >= end_ns:
        raise ValueError(f"Requested interval [{first_ns}, {end_ns}) has no shared video samples")
    step_ns = round(1e9 / fps)
    return np.arange(first_ns, end_ns, step_ns, dtype=np.int64)


def interpolate_pose(
    pose_times_ns: Int64[ndarray, "poses"],
    world_T_rig: Float64[ndarray, "poses 4 4"],
    timestamp_ns: int,
) -> Float64[ndarray, "4 4"]:
    """Interpolate translation linearly and rotation with quaternion slerp."""
    times = np.asarray(pose_times_ns, dtype=np.int64)
    poses = np.asarray(world_T_rig, dtype=np.float64)
    if len(times) == 0 or poses.shape != (len(times), 4, 4):
        raise ValueError(f"Expected matching pose arrays, got {times.shape} and {poses.shape}")
    right = int(np.searchsorted(times, timestamp_ns, side="right"))
    if right == 0:
        return poses[0].copy()
    if right == len(times):
        return poses[-1].copy()
    left = right - 1
    if times[left] == timestamp_ns:
        return poses[left].copy()
    alpha = float(timestamp_ns - times[left]) / float(times[right] - times[left])
    output = np.eye(4, dtype=np.float64)
    output[:3, 3] = (1.0 - alpha) * poses[left, :3, 3] + alpha * poses[right, :3, 3]
    slerp = Slerp(np.asarray([0.0, 1.0]), Rotation.from_matrix(poses[[left, right], :3, :3]))
    output[:3, :3] = slerp(alpha).as_matrix()
    return output


def best_detection_window(
    sample_seconds: Float64[ndarray, "samples"],
    detection_counts: Int64[ndarray, "samples"],
    *,
    window_seconds: float,
) -> tuple[float, int]:
    """Return the earliest sampled start with the largest half-open window sum."""
    times = np.asarray(sample_seconds, dtype=np.float64)
    counts = np.asarray(detection_counts, dtype=np.int64)
    if times.ndim != 1 or counts.shape != times.shape or len(times) == 0:
        raise ValueError(f"Expected non-empty aligned vectors, got {times.shape} and {counts.shape}")
    if window_seconds <= 0.0:
        raise ValueError(f"window_seconds must be positive, got {window_seconds}")
    prefix = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(counts, dtype=np.int64)])
    best_start = float(times[0])
    best_total = -1
    for left, start in enumerate(times):
        right = int(np.searchsorted(times, start + window_seconds, side="left"))
        total = int(prefix[right] - prefix[left])
        if total > best_total:
            best_start = float(start)
            best_total = total
    return best_start, best_total


def build_blueprint(cams: tuple[str, ...]) -> rrb.Blueprint:
    """Show the moving rig and people beside a two-by-two camera grid."""
    camera_views = [rrb.Spatial2DView(origin=f"{RIG}/{cam}/pinhole", name=cam) for cam in cams]
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin="world", name="rig + tracked people", contents=["$origin/**"]),
            rrb.Vertical(
                rrb.Horizontal(camera_views[0], camera_views[1]),
                rrb.Horizontal(camera_views[2], camera_views[3]),
            ),
            column_shares=[3, 2],
        ),
        rrb.TimePanel(timeline=TIMELINE),
        collapse_panels=True,
    )


def _read_video_codec(view: DatasetView, cam: str) -> rr.VideoCodec:
    """Read and type the catalog's static video codec component."""
    entity = f"{RIG}/{cam}/pinhole/video"
    table = view.filter_contents(entity).reader(index=None).select(f"/{entity}:VideoStream:codec").to_arrow_table()
    value = first_valid_value(table.column(0), component_name="VideoStream:codec")
    return rr.VideoCodec(int(np.asarray(value).ravel()[0]))


def _codec_name(codec: rr.VideoCodec) -> str:
    """Map Rerun's typed codec value to the demuxer's codec name."""
    if codec == rr.VideoCodec.H264:
        return "h264"
    if codec == rr.VideoCodec.AV1:
        return "av1"
    raise ValueError(f"Unsupported catalog video codec: {codec}")


def _frame_at(
    times: np.ndarray,
    decoder: SegmentVideoDecoder,
    timestamp_ns: int,
) -> UInt8[ndarray, "h w 3"]:
    """Decode the latest camera frame at or before one grid timestamp."""
    index = int(np.searchsorted(times, np.timedelta64(timestamp_ns, "ns"), side="right")) - 1
    if index < 0:
        raise ValueError(f"Timestamp {timestamp_ns} precedes the first video sample")
    return decoder.get_frame_at(index).data.permute(1, 2, 0).contiguous().cpu().numpy()


def _rgb_color(track_id: int) -> tuple[int, int, int]:
    """Convert LAMP's deterministic float color to Rerun bytes."""
    red, green, blue, _ = color_from_id(track_id)
    return round(red * 255.0), round(green * 255.0), round(blue * 255.0)


def _log_camera_observations(
    cam: str,
    image: UInt8[ndarray, "h w 3"],
    boxes: BoxDetections,
    keypoints: Keypoints2d,
    keypoint_conf_min: float,
) -> None:
    """Log one image with per-track COCO-17 boxes, joints, and edges."""
    root = f"{RIG}/{cam}/pinhole"
    rr.log(f"{root}/image", rr.Image(image, color_model=rr.ColorModel.RGB).compress(jpeg_quality=85))
    rr.log(f"{root}/detections", rr.Clear(recursive=True))
    boxes_xyxy = boxes.xyxy_numpy()
    points_xy = keypoints.xy_numpy()
    scores = keypoints.scores_numpy()
    track_ids = (
        boxes.track_ids.detach().cpu().numpy().astype(np.int64, copy=False)
        if boxes.track_ids is not None
        else np.full(len(boxes_xyxy), -1, dtype=np.int64)
    )
    edges = np.asarray(COCO_17.links, dtype=np.int64)
    for row, track_id_raw in enumerate(track_ids):
        track_id = int(track_id_raw)
        color = _rgb_color(track_id)
        entity = f"{root}/detections/person_{track_id}"
        rr.log(
            f"{entity}/box",
            rr.Boxes2D(
                array=boxes_xyxy[row : row + 1],
                array_format=rr.Box2DFormat.XYXY,
                labels=f"person_{track_id}",
                colors=color,
                show_labels=True,
            ),
        )
        visible = scores[row] >= keypoint_conf_min
        points = points_xy[row].copy()
        points[~visible] = np.nan
        rr.log(f"{entity}/keypoints", rr.Points2D(points, colors=color, radii=-3.0))
        valid_edges = visible[edges].all(axis=1)
        rr.log(f"{entity}/skeleton", rr.LineStrips2D(points_xy[row][edges[valid_edges]], colors=color, radii=-1.5))


def _log_person(
    tracker: LampTracker,
    state: PersonState,
    trails: dict[int, list[Float32[ndarray, "3"]]],
    faces_logged: set[int],
) -> None:
    """Log one person's latest SMPL joints, mesh, and pelvis trail."""
    track_id = state.track_id
    color = _rgb_color(track_id)
    root = f"world/people/{track_id}"
    joints = state.joints_world[-1]
    edges = np.asarray(SMPL_SKELETON_EDGES, dtype=np.int64)
    rr.log(f"{root}/skeleton", rr.Points3D(joints, colors=color, radii=0.025))
    rr.log(f"{root}/skeleton/edges", rr.LineStrips3D(joints[edges], colors=color, radii=0.012))
    trail = trails.setdefault(track_id, [])
    trail.append(joints[0].copy())
    rr.log(f"{root}/pelvis_trail", rr.LineStrips3D([np.stack(trail)], colors=color, radii=0.01))
    if track_id not in faces_logged:
        rr.log(f"{root}/mesh", rr.Mesh3D.from_fields(triangle_indices=tracker.faces), static=True)
        faces_logged.add(track_id)
    vertices = tracker.smpl_vertices(state)[-1]
    rr.log(f"{root}/mesh", rr.Mesh3D.from_fields(vertex_positions=vertices, albedo_factor=color))


def _log_rays(
    cam: str,
    camera: RigCamera,
    world_T_rig: Float64[ndarray, "4 4"],
    boxes: BoxDetections,
    keypoints: Keypoints2d,
    keypoint_conf_min: float,
) -> None:
    """Log unit world-frame rays for every visible keypoint."""
    world_T_camera = world_T_rig @ np.linalg.inv(camera.cam_T_rig)
    track_ids = boxes.track_ids.detach().cpu().numpy() if boxes.track_ids is not None else np.arange(boxes.num_detections)
    points = keypoints.xy_numpy()
    scores = keypoints.scores_numpy()
    for row, track_id_raw in enumerate(track_ids):
        visible = scores[row] >= keypoint_conf_min
        if not visible.any():
            continue
        rays_camera = camera.unproject(points[row][visible])
        directions = rays_camera @ world_T_camera[:3, :3].T
        origins = np.repeat(world_T_camera[None, :3, 3], len(directions), axis=0)
        track_id = int(track_id_raw)
        rr.log(
            f"world/rays/{cam}/person_{track_id}",
            rr.Arrows3D(origins=origins, vectors=directions, colors=_rgb_color(track_id)),
        )


def run(config: Config) -> RunMetrics:
    """Run the configured Robocap interval and return measured stage summaries."""
    dataset: DatasetEntry = CatalogClient(config.catalog_url).get_dataset(config.dataset)
    view: DatasetView = dataset.filter_segments(config.segment_id)
    camera_parameters: list[Fisheye62Parameters] = [read_fisheye_camera(view, cam) for cam in config.cams]
    cameras = [RigCamera(parameters) for parameters in camera_parameters]
    pose_times, world_T_rig_values = read_rig_poses(view)
    pose_times_ns = pose_times.astype("timedelta64[ns]").astype(np.int64)
    device = torch.device("cuda")
    decoders: dict[str, tuple[np.ndarray, list[bytes], list[bool], SegmentVideoDecoder]] = {}
    for cam in config.cams:
        codec = _read_video_codec(view, cam)
        times, samples, keyframes, decoder = open_segment_decoder(
            dataset,
            config.segment_id,
            f"{RIG}/{cam}/pinhole/video",
            TIMELINE,
            device,
            30,
            _codec_name(codec),
        )
        decoders[cam] = (times, samples, keyframes, decoder)
    grid_ns = build_time_grid(
        [entry[0] for entry in decoders.values()],
        fps=config.fps,
        start_s=config.start_s,
        max_seconds=config.max_seconds,
    )

    rr.send_blueprint(build_blueprint(config.cams))
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rig = Rig(
        index=0,
        calibration=RigCalibration(
            cameras=[
                CameraSensor(index=int(cam.split("_")[1]), name=parameters.name, kind="rgb", pinhole=parameters)
                for cam, parameters in zip(config.cams, camera_parameters, strict=True)
            ],
            reference_index=int(config.cams[0].split("_")[1]),
        ),
        image_plane_distance=0.15,
    )
    log_rig_static(rig)

    tracker_config = replace(config.tracker, floor_z=config.floor_z)
    tracker = tracker_config.setup(device=device)
    tracker.configure_cameras(cameras)
    trails: dict[int, list[Float32[ndarray, "3"]]] = {}
    faces_logged: set[int] = set()
    timing_rows: list[Float64[ndarray, "6"]] = []
    unique_people: set[int] = set()
    frames_with_people = 0

    for frame_index, timestamp_ns_raw in enumerate(grid_ns):
        timestamp_ns = int(timestamp_ns_raw)
        world_T_rig = interpolate_pose(pose_times_ns, world_T_rig_values, timestamp_ns)
        images = np.stack(
            [_frame_at(decoders[cam][0], decoders[cam][3], timestamp_ns) for cam in config.cams]
        ).astype(np.uint8, copy=False)
        step: LampStep = tracker.step(Frameset(timestamp_ns=timestamp_ns, images=images, world_T_rig=world_T_rig))
        rr.set_time(TIMELINE, duration=np.timedelta64(timestamp_ns, "ns"))
        rr.log(RIG, rr.Transform3D(mat3x3=world_T_rig[:3, :3], translation=world_T_rig[:3, 3]))
        for camera_index, (cam, camera, image) in enumerate(zip(config.cams, cameras, images, strict=True)):
            boxes = step.boxes_by_camera[camera_index]
            keypoints = step.keypoints_by_camera[camera_index]
            _log_camera_observations(cam, image, boxes, keypoints, tracker_config.keypoint_conf_min)
            if config.log_rays:
                _log_rays(cam, camera, world_T_rig, boxes, keypoints, tracker_config.keypoint_conf_min)
        current_people = [state for state in step.people.values() if int(state.timestamps_ns[-1]) == timestamp_ns]
        if current_people:
            frames_with_people += 1
        for state in current_people:
            unique_people.add(state.track_id)
            _log_person(tracker, state, trails, faces_logged)
        timing_rows.append(
            np.asarray(
                [
                    step.timings.detector_ms,
                    step.timings.pose_ms,
                    step.timings.tracker_ms,
                    step.timings.lifter_ms,
                    step.timings.smoothing_ms,
                    step.timings.total_ms,
                ],
                dtype=np.float64,
            )
        )
        if frame_index % 100 == 0:
            print(
                f"frame={frame_index}/{len(grid_ns)} video_time={timestamp_ns / 1e9:.1f}s "
                f"detections={sum(boxes.num_detections for boxes in step.boxes_by_camera.values())} "
                f"people={len(current_people)} total_ms={step.timings.total_ms:.1f}"
            )

    means = np.mean(np.stack(timing_rows), axis=0)
    return RunMetrics(
        frames=len(grid_ns),
        unique_people=len(unique_people),
        frames_with_people=frames_with_people,
        detector_mean_ms=float(means[0]),
        pose_mean_ms=float(means[1]),
        tracker_mean_ms=float(means[2]),
        lifter_mean_ms=float(means[3]),
        smoothing_mean_ms=float(means[4]),
        total_mean_ms=float(means[5]),
    )


def main(config: Config) -> None:
    """Run the catalog app and print its measured timing and people counts."""
    started = time.perf_counter()
    metrics = run(config)
    print(f"frames={metrics.frames}")
    print(f"unique_people={metrics.unique_people}")
    print(f"frames_with_people={metrics.frames_with_people}")
    print(f"detector_mean_ms={metrics.detector_mean_ms:.3f}")
    print(f"pose_mean_ms={metrics.pose_mean_ms:.3f}")
    print(f"tracker_mean_ms={metrics.tracker_mean_ms:.3f}")
    print(f"lifter_mean_ms={metrics.lifter_mean_ms:.3f}")
    print(f"smoothing_mean_ms={metrics.smoothing_mean_ms:.3f}")
    print(f"total_mean_ms={metrics.total_mean_ms:.3f}")
    print(f"wall_seconds={time.perf_counter() - started:.3f}")


__all__ = (
    "Config",
    "RunMetrics",
    "best_detection_window",
    "build_blueprint",
    "build_time_grid",
    "interpolate_pose",
    "main",
    "run",
)
