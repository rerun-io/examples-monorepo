"""Read a rig, its video and an initial trajectory from a Rerun catalog segment, into colsfm's input contract.

cuSFM-style mapping starts from a trajectory: colsfm refines it, it does not find one. A catalog segment in the
`exoego:v2` layout carries everything the pipeline needs — the cameras (`/world/rig_NN/cam_MM`, static extrinsic,
pinhole and distortion), their `VideoStream`s, and a temporal `Transform3D` that gives `world_T_rig` (the `slam_rs`
layer on RoboCap, the `gt` layer on VSLAM-LAB, or a `/world/runs/<source>/rig` prediction layer such as the KITTI
`cuvslam` one). `prepare_catalog_input` reads those through the catalog and writes metadata only: the cuSFM
`frames_meta.json` the stages read (`colsfm.frames_meta`) and a `frame_source.json` naming each keyframe's sample in its
camera's `VideoStream`. No pixel is decoded or written here; the run decodes the keyframes once on NVDEC straight from the
catalog (`colsfm.frames.VideoStreamFrames`), so the matcher sees the original video, not a re-encoded JPEG.

Every camera pose handed to the pipeline is `world_T_rig(t) @ rig_T_cam` at the camera's own frame time, the rig pose
interpolated (slerp + linear) from the trajectory stream; frames outside the trajectory's time span are dropped, not
extrapolated.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Literal, TypeAlias

import numpy as np
import pyarrow as pa
from jaxtyping import Float64, Int64
from numpy import ndarray
from rerun.catalog import CatalogClient, DatasetEntry
from scipy.spatial.transform import Rotation, Slerp
from simplecv.catalog_rig_layout import CatalogCamera, catalog_components, parse_rig_layout
from simplecv.catalog_video import CatalogVideo, read_catalog_videos
from simplecv.rrd_query_utils import first_valid_value

from colsfm import DATA_DIR
from colsfm.alignment import GROUND_TRUTH_FILE_NAME
from colsfm.export import write_tum_file
from colsfm.frame_manifest import FRAME_SOURCE_FILE, FrameSourceManifest, StreamKeyframes, write_manifest
from colsfm.frames_meta import FRAMES_META_NAME
from colsfm.geometry import TumPose, rigid3d_from_matrix

DEFAULT_CATALOG_URL: Final[str] = "rerun+http://dgx-spark:51235"
"""The shared catalog the dataforge corpora are registered on."""

DEFAULT_TIMELINE: Final[str] = "video_time"
"""The timeline exoego:v2 segments index their video and poses on."""

ProjectionModel: TypeAlias = Literal["PINHOLE", "DISTORTED_PINHOLE", "OPENCV_FISHEYE"]
"""The cuSFM camera projection models a catalog camera maps onto (COLMAP PINHOLE, FULL_OPENCV, OPENCV_FISHEYE)."""

PROJECTION_BY_DISTORTION_MODEL: Final[dict[str, tuple[ProjectionModel, int]]] = {
    "kannala_brandt": ("OPENCV_FISHEYE", 4),
    "fisheye62": ("OPENCV_FISHEYE", 4),
    "kb4": ("OPENCV_FISHEYE", 4),
    "opencv_fisheye": ("OPENCV_FISHEYE", 4),
    "brown_conrady": ("DISTORTED_PINHOLE", 8),
}
"""simplecv `DistortionModel` name -> cuSFM projection model and how many leading coefficients it takes.

Kannala-Brandt's first four slots are k1..k4. `brown_conrady` is OpenCV's coefficient vector (k1 k2 p1 p2 k3 k4 k5 k6
...), whose first eight are COLMAP FULL_OPENCV's distortion parameters in the same order. A camera without a
`DistortionModel` (or with an unknown one) is PINHOLE."""

Transform: TypeAlias = Float64[ndarray, "4 4"]
"""A homogeneous rigid transform."""


@dataclass(frozen=True)
class CatalogInputConfig:
    """Which segment, which cameras, and how densely to sample it."""

    dataset: str
    """Catalog dataset name, e.g. `robocap` or `dataforge-vslam-lab-kitti`."""
    segment: str
    """Segment id, e.g. `robocap__f408193e6447b3b0__s00000021`."""
    catalog_url: str = DEFAULT_CATALOG_URL
    """Catalog server."""
    cameras: tuple[str, ...] = ()
    """Camera names (the camera node's static `name`) fed to the pipeline; empty means every video camera."""
    stereo_pairs: tuple[tuple[str, str], ...] = ()
    """Camera-name pairs declared as `stereo_pair` (rectifiable stereo only; the matcher pairs them per rig frame)."""
    trajectory_entity: str = "/world/rig_00"
    """Entity whose temporal `Transform3D` is `world_T_rig` (the initial trajectory)."""
    timeline: str = DEFAULT_TIMELINE
    """Timeline the video and the trajectory are read on."""
    ground_truth_entity: str | None = None
    """Entity whose temporal `Transform3D` is the ground-truth `world_T_rig` (e.g. `/world/rig_00` from a `gt` layer);
    when set, it is written as `ground_truth.txt` (TUM, at each rig frame's reference timestamp) so the run is scored."""
    frame_stride: int = 1
    """Use every Nth frame of the reference camera."""
    trim_start_seconds: float = 0.0
    """Drop this much from the start of the reference camera's video."""
    trim_end_seconds: float = 0.0
    """Drop this much from the end."""
    max_samples: int | None = None
    """Cap on rig frames after striding (`None` = all)."""
    sync_tolerance_ns: int = 5_000_000
    """Largest |dt| between the reference frame and another camera's frame of the same rig frame."""
    output_root: Path = field(default_factory=lambda: DATA_DIR / "inputs")
    """Where `frames_meta.json`, `frame_source.json` and the ground truth go, one directory per segment and sampling."""

    @property
    def input_dir(self) -> Path:
        """The pipeline input directory for this segment and sampling.

        Returns:
            `<output_root>/<segment>_s<stride>[_n<max>]`.
        """
        cap: str = "" if self.max_samples is None else f"_n{self.max_samples}"
        return self.output_root / f"{self.segment}_s{self.frame_stride}{cap}"


@dataclass(frozen=True)
class CatalogCameraModel:
    """One camera as the pipeline needs it."""

    entity: str
    """Camera node entity path, e.g. `/world/rig_00/cam_02`."""
    name: str
    """The camera's static `name`, also its image folder."""
    rig_T_cam: Transform
    """Static extrinsic: camera pose in the rig frame."""
    k_matrix: Float64[ndarray, "3 3"]
    """Pinhole intrinsics."""
    width: int
    """Image width in pixels."""
    height: int
    """Image height in pixels."""
    projection: ProjectionModel
    """From the camera's `DistortionModel` (`PROJECTION_BY_DISTORTION_MODEL`); `PINHOLE` without one."""
    distortion: tuple[float, ...]
    """k1..k4 for `OPENCV_FISHEYE`, OpenCV's first eight for `DISTORTED_PINHOLE`; empty for `PINHOLE`."""


@dataclass(frozen=True)
class CatalogTrajectory:
    """The initial trajectory, `world_T_rig` per timestamp."""

    t_ns: Int64[ndarray, " n"]
    """Strictly increasing timestamps."""
    rotations: Rotation
    """`world_R_rig` per timestamp."""
    translations: Float64[ndarray, "n 3"]
    """`world_t_rig` per timestamp."""

    def at(self, query_ns: Int64[ndarray, " m"]) -> Float64[ndarray, "m 4 4"]:
        """Interpolate the rig pose at `query_ns` (inside the span; the caller drops the rest).

        Args:
            query_ns: Timestamps inside `[t_ns[0], t_ns[-1]]`.

        Returns:
            `world_T_rig` per query.
        """
        source: Float64[ndarray, " n"] = self.t_ns.astype(np.float64)
        query: Float64[ndarray, " m"] = query_ns.astype(np.float64)
        poses: Float64[ndarray, "m 4 4"] = np.tile(np.eye(4), (len(query_ns), 1, 1))
        poses[:, :3, :3] = Slerp(source, self.rotations)(query).as_matrix()
        poses[:, :3, 3] = np.stack([np.interp(query, source, self.translations[:, axis]) for axis in range(3)], axis=-1)
        return poses


@dataclass(frozen=True)
class PreparedInput:
    """What `prepare_catalog_input` wrote."""

    input_dir: Path
    """Directory holding `frames_meta.json`, `frame_source.json` and the ground truth."""
    cameras: tuple[CatalogCameraModel, ...]
    """The cameras handed to the pipeline, in entity order."""
    num_rig_frames: int
    """Rig frames written (synced samples)."""
    num_keyframes: int
    """Keyframes recorded over all cameras."""
    seconds: float
    """Wall time of the catalog reads and the metadata writes."""


def _static_values(table: pa.Table, column: str) -> list[float] | list[str] | None:
    if column not in table.column_names:
        return None
    values = table.column(column).drop_null()
    if len(values) == 0:
        return None
    return list(np.asarray(first_valid_value(values, component_name=column)).reshape(-1).tolist())


def read_catalog_cameras(dataset: DatasetEntry, segment: str, names: Sequence[str]) -> tuple[CatalogCameraModel, ...]:
    """Read every video camera's static calibration, keeping the named ones.

    Args:
        dataset: Catalog dataset.
        segment: Segment id.
        names: Camera names to keep, in any order; empty keeps all.

    Returns:
        The cameras in rig / camera index order.

    Raises:
        KeyError: When a requested name is not on the segment.
        ValueError: When a camera lacks its calibration.
    """
    layout_cameras: tuple[CatalogCamera, ...] = parse_rig_layout(catalog_components(dataset.schema())).cameras
    entities: list[str] = [camera.transform_entity for camera in layout_cameras]
    table: pa.Table = (
        dataset.filter_segments(segment).filter_contents([f"{entity}/**" for entity in entities] + entities).reader(index=None).to_arrow_table()
    )
    cameras: list[CatalogCameraModel] = []
    for camera in layout_cameras:
        entity: str = camera.transform_entity
        name_values = _static_values(table, f"{entity}:name")
        name: str = str(name_values[0]) if name_values else camera.camera
        rotation = _static_values(table, f"{entity}:Transform3D:mat3x3")
        translation = _static_values(table, f"{entity}:Transform3D:translation")
        k_values = _static_values(table, f"{camera.pinhole_entity}:Pinhole:image_from_camera")
        resolution = _static_values(table, f"{camera.pinhole_entity}:Pinhole:resolution")
        if rotation is None or translation is None or k_values is None or resolution is None:
            raise ValueError(f"{segment} {entity}: missing extrinsic or pinhole")
        # The camera node's static Transform3D is logged child-from-parent (cam_T_rig); Rerun matrices are column-major.
        cam_T_rig: Transform = np.eye(4)
        cam_T_rig[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3).T
        cam_T_rig[:3, 3] = np.asarray(translation, dtype=np.float64)
        model_values = _static_values(table, f"{camera.pinhole_entity}:simplecv.components.DistortionModel")
        coefficient_values = _static_values(table, f"{camera.pinhole_entity}:simplecv.components.DistortionCoefficients")
        model_name: str = str(model_values[0]).lower() if model_values else ""
        projection: ProjectionModel = "PINHOLE"
        distortion: tuple[float, ...] = ()
        if model_name in PROJECTION_BY_DISTORTION_MODEL and coefficient_values is not None:
            projection, count = PROJECTION_BY_DISTORTION_MODEL[model_name]
            distortion = tuple(float(value) for value in coefficient_values[:count])
        cameras.append(
            CatalogCameraModel(
                entity=entity,
                name=name,
                rig_T_cam=np.linalg.inv(cam_T_rig),
                k_matrix=np.asarray(k_values, dtype=np.float64).reshape(3, 3).T,
                width=int(float(resolution[0])),
                height=int(float(resolution[1])),
                projection=projection,
                distortion=distortion,
            )
        )
    if names:
        by_name: dict[str, CatalogCameraModel] = {camera.name: camera for camera in cameras}
        missing: list[str] = [name for name in names if name not in by_name]
        if missing:
            raise KeyError(f"{segment}: cameras {missing} not on the segment (have {sorted(by_name)})")
        cameras = [camera for camera in cameras if camera.name in set(names)]
    return tuple(cameras)


def read_catalog_trajectory(dataset: DatasetEntry, segment: str, entity: str, timeline: str) -> CatalogTrajectory:
    """Read a temporal `Transform3D` stream (quaternion or `mat3x3`, plus translation) as `world_T_<entity>`.

    Args:
        dataset: Catalog dataset.
        segment: Segment id.
        entity: Entity carrying the temporal transform.
        timeline: Index timeline.

    Returns:
        The trajectory, sorted and de-duplicated by time.

    Raises:
        ValueError: When the entity carries no temporal pose on this segment.
    """
    quaternion_column: str = f"{entity}:Transform3D:quaternion"
    matrix_column: str = f"{entity}:Transform3D:mat3x3"
    translation_column: str = f"{entity}:Transform3D:translation"
    table: pa.Table = dataset.filter_segments(segment).filter_contents([entity]).reader(index=timeline).to_arrow_table()
    rotation_column: str = quaternion_column if quaternion_column in table.column_names else matrix_column
    if rotation_column not in table.column_names or translation_column not in table.column_names:
        raise ValueError(f"{segment}: {entity} carries no temporal Transform3D on {timeline}")
    table = table.filter(table[rotation_column].is_valid())
    if table.num_rows == 0:
        raise ValueError(f"{segment}: {entity} carries no temporal Transform3D on {timeline}")
    t_ns: Int64[ndarray, " n"] = table[timeline].combine_chunks().to_numpy(zero_copy_only=False).view(np.int64)
    flat_rotation: Float64[ndarray, " k"] = np.asarray(
        table[rotation_column].combine_chunks().flatten().flatten().to_numpy(zero_copy_only=False), dtype=np.float64
    )
    rotations: Rotation = (
        Rotation.from_quat(flat_rotation.reshape(-1, 4))
        if rotation_column == quaternion_column
        else Rotation.from_matrix(flat_rotation.reshape(-1, 3, 3).transpose(0, 2, 1))  # Rerun stores Mat3x3 column-major
    )
    translations: Float64[ndarray, "n 3"] = np.asarray(
        table[translation_column].combine_chunks().flatten().flatten().to_numpy(zero_copy_only=False), dtype=np.float64
    ).reshape(-1, 3)
    order: Int64[ndarray, " n"] = np.argsort(t_ns, kind="stable")
    t_ns, rotations, translations = t_ns[order], rotations[order], translations[order]
    keep: np.ndarray = np.concatenate([[True], np.diff(t_ns) > 0])
    return CatalogTrajectory(t_ns=t_ns[keep], rotations=rotations[keep], translations=translations[keep])


def keyframe_image_name(camera_name: str, t_ns: int) -> str:
    """The name a catalog keyframe carries in `frames_meta.json`, the database and the frame-source manifest.

    A key, not a file: the pixels come from the catalog's video stream (`colsfm.frames.VideoStreamFrames`).

    Args:
        camera_name: The camera.
        t_ns: The sample's index value (`CatalogVideo.t_ns`).

    Returns:
        `<camera>/<t_ns>`.
    """
    return f"{camera_name}/{t_ns}"


def _axis_angle(transform: Transform) -> dict[str, dict[str, float]]:
    rotation_vector: Float64[ndarray, " 3"] = Rotation.from_matrix(transform[:3, :3]).as_rotvec()
    angle: float = float(np.linalg.norm(rotation_vector))
    axis: Float64[ndarray, " 3"] = rotation_vector / angle if angle > 1e-12 else np.array([0.0, 0.0, 1.0])
    return {
        "axis_angle": {"x": float(axis[0]), "y": float(axis[1]), "z": float(axis[2]), "angle_degrees": float(np.degrees(angle))},
        "translation": {"x": float(transform[0, 3]), "y": float(transform[1, 3]), "z": float(transform[2, 3])},
    }


def frames_meta_document(
    cameras: Sequence[CatalogCameraModel],
    rig_frames: Sequence[dict[str, tuple[int, Transform]]],
    stereo_pairs: Sequence[tuple[str, str]],
) -> dict[str, object]:
    """Build cuSFM's `frames_meta.json` for the sampled rig frames.

    Args:
        cameras: The cameras, in entity order; their index is the `camera_params_id`.
        rig_frames: Per rig frame, the camera name -> (frame timestamp ns, `world_T_cam`) of each camera present.
        stereo_pairs: Camera-name pairs declared as stereo.

    Returns:
        The document, ready for `json.dumps`.
    """
    camera_params: dict[str, object] = {}
    for index, camera in enumerate(cameras):
        calibration: dict[str, object] = {"image_width": camera.width, "image_height": camera.height}
        if camera.projection != "PINHOLE":
            calibration["camera_matrix"] = {"data": camera.k_matrix.reshape(-1).tolist(), "row_count": 3, "column_count": 3}
            calibration["distortion_coefficients"] = {"data": list(camera.distortion), "row_count": 1, "column_count": len(camera.distortion)}
        else:
            projection: Float64[ndarray, "3 4"] = np.hstack([camera.k_matrix, np.zeros((3, 1))])
            calibration["projection_matrix"] = {"data": projection.reshape(-1).tolist(), "row_count": 3, "column_count": 4}
        camera_params[str(index)] = {
            "sensor_meta_data": {
                "sensor_id": index,
                "sensor_type": "CAMERA",
                "sensor_name": camera.name,
                "sensor_to_vehicle_transform": _axis_angle(camera.rig_T_cam),
            },
            "calibration_parameters": calibration,
            "camera_projection_model_type": camera.projection,
        }
    params_id: dict[str, str] = {camera.name: str(index) for index, camera in enumerate(cameras)}
    keyframes: list[dict[str, object]] = []
    for sample_index, frame in enumerate(rig_frames):
        for camera in cameras:
            if camera.name not in frame:
                continue
            t_ns, world_T_cam = frame[camera.name]
            keyframes.append(
                {
                    "id": str(len(keyframes)),
                    "camera_params_id": params_id[camera.name],
                    "timestamp_microseconds": str(int(t_ns) // 1000),
                    "image_name": keyframe_image_name(camera.name, int(t_ns)),
                    "camera_to_world": _axis_angle(world_T_cam),
                    "synced_sample_id": str(sample_index),
                }
            )
    by_name: dict[str, CatalogCameraModel] = {camera.name: camera for camera in cameras}
    pairs: list[dict[str, object]] = [
        {
            "left_camera_param_id": params_id[left],
            "right_camera_param_id": params_id[right],
            "baseline_meters": float(np.linalg.norm(by_name[left].rig_T_cam[:3, 3] - by_name[right].rig_T_cam[:3, 3])),
        }
        for left, right in stereo_pairs
        if left in params_id and right in params_id
    ]
    return {
        "keyframes_metadata": keyframes,
        "initial_pose_type": "EGO_MOTION",
        "camera_params_id_to_camera_params": camera_params,
        "stereo_pair": pairs,
    }


def write_ground_truth(
    path: Path,
    ground_truth: CatalogTrajectory,
    rig_frames: Sequence[dict[str, tuple[int, Transform]]],
    cameras: Sequence[CatalogCameraModel],
) -> int:
    """Write the ground truth as TUM at each rig frame's reference timestamp, truncated to microseconds.

    The reference keyframe of a rig frame is its first camera present, as `colsfm.frames_meta` groups them, and the
    pipeline joins against ground truth within 20 us, so the file carries exactly those instants.

    Args:
        path: Destination (`ground_truth.txt`).
        ground_truth: The ground-truth trajectory.
        rig_frames: Per rig frame, the camera name -> (frame timestamp, pose) of each camera present.
        cameras: The cameras in entity order.

    Returns:
        Lines written (rig frames inside the ground truth's span).
    """
    stamps_us: list[int] = []
    for frame in rig_frames:
        present: list[CatalogCameraModel] = [camera for camera in cameras if camera.name in frame]
        if present:
            stamps_us.append(int(frame[present[0].name][0]) // 1000)
    query_ns: Int64[ndarray, " m"] = np.asarray([stamp * 1000 for stamp in stamps_us], dtype=np.int64)
    inside: np.ndarray = (query_ns >= ground_truth.t_ns[0]) & (query_ns <= ground_truth.t_ns[-1])
    poses: Float64[ndarray, "m 4 4"] = ground_truth.at(query_ns[inside])
    written: list[TumPose] = [
        TumPose(int(stamp), rigid3d_from_matrix(pose[:3, :3], pose[:3, 3])) for stamp, pose in zip(np.asarray(stamps_us)[inside], poses, strict=True)
    ]
    write_tum_file(path, written)
    return len(written)


def prepare_catalog_input(config: CatalogInputConfig) -> PreparedInput:
    """Read the segment and write the pipeline's input directory: `frames_meta.json`, `frame_source.json`, ground truth.

    Args:
        config: Segment, cameras and sampling.

    Returns:
        What was written.

    Raises:
        ValueError: When no rig frame falls inside the trajectory's time span.
    """
    started: float = time.perf_counter()
    dataset: DatasetEntry = CatalogClient(config.catalog_url).get_dataset(name=config.dataset)
    cameras: tuple[CatalogCameraModel, ...] = read_catalog_cameras(dataset, config.segment, config.cameras)
    trajectory: CatalogTrajectory = read_catalog_trajectory(dataset, config.segment, config.trajectory_entity, config.timeline)
    videos: tuple[CatalogVideo, ...] = read_catalog_videos(
        dataset, config.segment, [f"{camera.entity}/pinhole/video" for camera in cameras], config.timeline
    )
    reference: CatalogVideo = videos[0]
    times: Int64[ndarray, " n"] = reference.t_ns
    first: int = int(np.searchsorted(times, times[0] + int(config.trim_start_seconds * 1e9)))
    last: int = int(np.searchsorted(times, times[-1] - int(config.trim_end_seconds * 1e9), side="right"))
    candidates: list[int] = [
        index for index in range(first, last, config.frame_stride) if trajectory.t_ns[0] <= times[index] <= trajectory.t_ns[-1]
    ]
    if config.max_samples is not None:
        candidates = candidates[: config.max_samples]
    if not candidates:
        raise ValueError(f"{config.segment}: no reference frame inside the trajectory span")

    # Per camera, the packet nearest each reference frame, within the sync tolerance.
    selection: list[dict[int, int]] = []
    for video in videos:
        chosen: dict[int, int] = {}
        for sample, reference_index in enumerate(candidates):
            target: int = int(times[reference_index])
            nearest: int = int(np.clip(np.searchsorted(video.t_ns, target), 1, len(video.t_ns) - 1))
            nearest = nearest if abs(int(video.t_ns[nearest]) - target) < abs(int(video.t_ns[nearest - 1]) - target) else nearest - 1
            frame_t: int = int(video.t_ns[nearest])
            if abs(frame_t - target) <= config.sync_tolerance_ns and trajectory.t_ns[0] <= frame_t <= trajectory.t_ns[-1]:
                chosen[sample] = nearest
        selection.append(chosen)

    input_dir: Path = config.input_dir
    input_dir.mkdir(parents=True, exist_ok=True)
    rig_frames: list[dict[str, tuple[int, Transform]]] = [{} for _ in candidates]
    streams: list[StreamKeyframes] = []
    written: int = 0
    for camera, video, chosen in zip(cameras, videos, selection, strict=True):
        frame_times: Int64[ndarray, " m"] = np.asarray([video.t_ns[index] for index in chosen.values()], dtype=np.int64)
        world_T_rig: Float64[ndarray, "m 4 4"] = trajectory.at(frame_times) if len(frame_times) else np.zeros((0, 4, 4))
        for (sample, packet_index), pose in zip(chosen.items(), world_T_rig, strict=True):
            rig_frames[sample][camera.name] = (int(video.t_ns[packet_index]), pose @ camera.rig_T_cam)
            written += 1
        # No pixels here: the run decodes these samples straight from the catalog on NVDEC (`colsfm.frames`).
        streams.append(
            StreamKeyframes(
                entity=f"{camera.entity.lstrip('/')}/pinhole/video",
                names=[keyframe_image_name(camera.name, int(video.t_ns[index])) for index in chosen.values()],
                t_ns=[int(video.t_ns[index]) for index in chosen.values()],
            )
        )
        print(f"[colsfm.catalog] {camera.name}: {len(chosen)} keyframes")
    write_manifest(
        input_dir / FRAME_SOURCE_FILE,
        FrameSourceManifest(catalog_url=config.catalog_url, dataset=config.dataset, segment=config.segment, timeline=config.timeline, streams=streams),
    )

    if config.ground_truth_entity is not None:
        ground_truth: CatalogTrajectory = read_catalog_trajectory(dataset, config.segment, config.ground_truth_entity, config.timeline)
        write_ground_truth(input_dir / GROUND_TRUTH_FILE_NAME, ground_truth, rig_frames, cameras)
    document: dict[str, object] = frames_meta_document(cameras, rig_frames, config.stereo_pairs)
    (input_dir / FRAMES_META_NAME).write_text(json.dumps(document, indent=1))
    seconds: float = time.perf_counter() - started
    print(f"[colsfm.catalog] {config.segment}: {len(candidates)} rig frames, {written} keyframes, {len(cameras)} cameras in {seconds:.1f} s")
    return PreparedInput(input_dir=input_dir, cameras=cameras, num_rig_frames=len(candidates), num_keyframes=written, seconds=seconds)
