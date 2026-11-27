"""Calibrate an uncalibrated rig from a catalog segment: each camera's intrinsics and its pose in the rig.

`colsfm calibrate` only makes the calibration the normal path needs; it does not refine a trajectory. On a window of
rig frames sampled from the segment it runs colsfm's own features and matcher (RaCo + FlexGlue on TensorRT, decoded
from the catalog) and matches every camera against every camera of its own and the nearby rig frames. COLMAP's global
mapper then maps the images without a rig, intrinsics estimated from a seed; `apply_rig_config` derives the rig from
that model, and the global mapper maps again on the rig, with the rig extrinsics and every intrinsic free. The result
is a calibration layer (`colsfm.calibration_layer.write_calibration_layer`).

What it assumes instead of solving:

- the rig frame is the first selected camera (its `rig_T_cam` is the identity);
- the scale is the reconstruction's own, not metres: rotations and intrinsics are what the solve found, translations
  share one unknown factor. The layer says so on its root (`metric_scale=False`). A run that starts from a trajectory composes each
  camera as `world_T_rig @ rig_T_cam`, so the layer combines only with a trajectory in the same frame and scale; a metric
  trajectory in another rig frame (slam_rs: the IMU frame, metres) does not combine with it as it is, and `run catalog`
  refuses it (`colsfm.catalog.refuse_unscaled_calibration`).

Why the order: COLMAP's mappers discard a rig whose `sensor_from_rig` is unknown, so the first pass maps single images;
without the cross-camera pairs of the same and nearby rig frames it splits into one model per camera (pysfm's
measurement: 601 vs 2601 cross pairs, 0 vs 196 of 200 images registered).
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, TypeAlias

import numpy as np
import pycolmap
import rerun as rr
from jaxtyping import Float64, Int64
from numpy import ndarray
from rerun.catalog import CatalogClient, DatasetEntry
from simplecv.catalog_video_reader import StreamIndex, read_video_index

from colsfm import DATA_DIR
from colsfm.calibration_layer import write_calibration_layer
from colsfm.catalog import CatalogCameraModel, CatalogInputConfig, ProjectionModel, nearest_packets, read_camera_names
from colsfm.database import ImagePair
from colsfm.features import ExtractionReport, FeatureOptions, extract_features
from colsfm.frame_manifest import FrameSourceManifest, StreamKeyframes
from colsfm.frames import VideoStreamFrames
from colsfm.matching import MatchingOptions, MatchReport, match_pairs
from colsfm.pairs import normalise_pair
from colsfm.reconstruction import camera_sensor_id, rig_sensor_from_rig

CalibrationModel: TypeAlias = Literal["OPENCV_FISHEYE", "OPENCV"]
"""The COLMAP model every camera is estimated in: Kannala-Brandt k1..k4 (fisheye), or OpenCV k1 k2 p1 p2."""

PROJECTION_BY_MODEL: Final[dict[CalibrationModel, ProjectionModel]] = {"OPENCV_FISHEYE": "OPENCV_FISHEYE", "OPENCV": "DISTORTED_PINHOLE"}
"""How the estimated model is written: `OPENCV`'s four coefficients are the first four of `DISTORTED_PINHOLE`'s eight."""


@dataclass(frozen=True)
class CalibrateOptions:
    """How many rig frames, which model, and where the outputs go."""

    camera_model: CalibrationModel = "OPENCV_FISHEYE"
    """The model every camera is estimated in."""
    num_rig_frames: int = 300
    """Rig frames sampled evenly over the segment (after the source's trims), on the reference camera's packets."""
    overlap: int = 5
    """Each image is matched against every camera of its own rig frame and of the next `overlap` ones."""
    work_dir: Path | None = None
    """The database and the two models go here; `packages/colsfm/data/calibrate/<segment>` when None."""
    layer_path: Path | None = None
    """The calibration layer; `<work_dir>/<segment>.colsfm_calibrate.rrd` when None."""


@dataclass(frozen=True)
class CalibrationResult:
    """The calibration and how well it fits."""

    cameras: tuple[CatalogCameraModel, ...]
    """Every selected camera, in the rig frame of the first one, in the reconstruction's scale."""
    layer_path: Path
    """Where the calibration layer was written."""
    registered_images: dict[str, int]
    """Registered images per camera name."""
    mean_reprojection_error_px: float
    """Of the rig model."""
    seconds: float
    """Wall time of the whole calibration."""


def rig_window_pairs(image_ids: Sequence[Sequence[int | None]], overlap: int) -> list[ImagePair]:
    """Every image against every camera of its own rig frame and of the next `overlap` rig frames.

    Args:
        image_ids: Per rig frame, per camera, the image id (None where the camera has no image in that rig frame).
        overlap: Rig-frame lookahead.

    Returns:
        The normalised, deduplicated, sorted pairs.
    """
    pairs: set[ImagePair] = set()
    for first, row in enumerate(image_ids):
        for later in image_ids[first : first + overlap + 1]:
            pairs.update(normalise_pair(one, other) for one in row for other in later if one is not None and other is not None and one != other)
    return sorted(pairs)


def seed_camera(camera_id: int, model: CalibrationModel, width: int, height: int) -> pycolmap.Camera:
    """The camera the first mapping starts from.

    A pinhole-like lens starts from COLMAP's own default focal (1.2 x the longer side) as a guess. A fisheye starts from
    an equidistant 180-degree lens across the longer side (`r = f * theta`), as a prior: COLMAP marks a fisheye pair
    without a focal prior degenerate (a fundamental matrix cannot recover it), so verification fits an essential matrix
    on the seed's rays instead.
    """
    fisheye: bool = model == "OPENCV_FISHEYE"
    focal: float = max(width, height) / math.pi if fisheye else 1.2 * max(width, height)
    camera: pycolmap.Camera = pycolmap.Camera.create_from_model_name(camera_id, model, focal, width, height)
    camera.has_prior_focal_length = fisheye
    return camera


def _global_map(database_path: Path, work_dir: Path, name: str, *, rig: bool) -> pycolmap.Reconstruction:
    """COLMAP's global mapper on the database, the largest model kept; with `rig`, the rig extrinsics and every intrinsic free.

    Raises:
        RuntimeError: When the mapper registers nothing.
    """
    options: pycolmap.GlobalPipelineOptions = pycolmap.GlobalPipelineOptions()
    options.random_seed = 0  # the same matches give the same calibration
    options.mapper.refine_sensor_from_rig = rig
    options.mapper.bundle_adjustment.refine_sensor_from_rig = rig
    options.mapper.bundle_adjustment.refine_principal_point = rig
    (work_dir / name).mkdir(parents=True, exist_ok=True)
    models: dict[int, pycolmap.Reconstruction] = pycolmap.global_mapping(database_path, work_dir, work_dir / name, options)
    if not models:
        raise RuntimeError(f"{database_path}: the {name} mapping registered no image")
    return max(models.values(), key=lambda model: model.num_reg_images())


def solve_rig(database_path: Path, names: Sequence[str], work_dir: Path) -> pycolmap.Reconstruction:
    """Map the database without a rig, derive the rig from that model, and map again on the rig.

    Args:
        database_path: Cameras, images named `<camera>/<rig frame>`, keypoints and verified matches; no rig or frames
            yet (it receives them).
        names: Camera names, the first one the reference.
        work_dir: Where the two models are written.

    Returns:
        The rig model.
    """
    bootstrap: pycolmap.Reconstruction = _global_map(database_path, work_dir, "bootstrap", rig=False)
    # Same-named images across cameras are one rig frame: what `apply_rig_config` groups by.
    rig_config: pycolmap.RigConfig = pycolmap.RigConfig(
        cameras=[pycolmap.RigConfigCamera(image_prefix=f"{name}/", ref_sensor=index == 0) for index, name in enumerate(names)]
    )
    with pycolmap.Database.open(database_path) as database:
        pycolmap.apply_rig_config([rig_config], database, bootstrap)
    del bootstrap  # the rig is in the database now; the second mapping reads only that
    return _global_map(database_path, work_dir, "rig", rig=True)


def cameras_from_rig(
    reconstruction: pycolmap.Reconstruction, camera_ids: dict[str, int], entities: dict[str, str], model: CalibrationModel
) -> tuple[CatalogCameraModel, ...]:
    """Each camera's intrinsics and `rig_T_cam` from a reconstruction that carries one rig.

    Args:
        reconstruction: The rig model.
        camera_ids: Camera name -> COLMAP camera id (also its sensor id), the reference first.
        entities: Camera name -> the segment's camera node.
        model: The model the cameras were estimated in.

    Returns:
        The cameras, in `camera_ids` order.
    """
    rig: pycolmap.Rig = next(iter(reconstruction.rigs.values()))
    cameras: list[CatalogCameraModel] = []
    for name, camera_id in camera_ids.items():
        sensor: pycolmap.sensor_t = camera_sensor_id(camera_id)
        rig_T_cam: Float64[ndarray, "4 4"] = np.eye(4)
        if not rig.is_ref_sensor(sensor):
            rig_T_cam[:3, :] = rig_sensor_from_rig(rig, sensor).inverse().matrix()
        camera: pycolmap.Camera = reconstruction.cameras[camera_id]
        distortion: tuple[float, ...] = tuple(float(value) for value in camera.params[4:8])
        cameras.append(
            CatalogCameraModel(
                entity=entities[name],
                name=name,
                rig_T_cam=rig_T_cam,
                k_matrix=np.asarray(camera.calibration_matrix(), dtype=np.float64),
                width=int(camera.width),
                height=int(camera.height),
                projection=PROJECTION_BY_MODEL[model],
                distortion=distortion if model == "OPENCV_FISHEYE" else (*distortion, 0.0, 0.0, 0.0, 0.0),
            )
        )
    return tuple(cameras)


def window_streams(
    entities: dict[str, str], videos: Sequence[StreamIndex], num_rig_frames: int, tolerance_ns: int, trims_s: tuple[float, float]
) -> list[StreamKeyframes]:
    """The window's keyframes per camera: rig frames on the reference camera's packets, evenly over the trimmed span.

    Args:
        entities: Camera name -> camera node, the reference first.
        videos: Each camera's stream index, in `entities` order.
        num_rig_frames: How many rig frames.
        tolerance_ns: Largest |dt| between a camera's packet and its rig frame's reference time.
        trims_s: Seconds dropped from the start and the end of the reference stream.

    Returns:
        Per camera, its keyframes named `<camera>/<rig frame>`: the same name across cameras is one rig frame.

    Raises:
        ValueError: When the trimmed window is empty, a camera has no packet in it, or one packet would serve two rig frames.
    """
    times: Int64[ndarray, " n"] = videos[0].t_ns
    first_ns: int = int(times[0]) + int(trims_s[0] * 1e9)
    last_ns: int = int(times[-1]) - int(trims_s[1] * 1e9)
    first: int = int(np.searchsorted(times, first_ns))
    last: int = int(np.searchsorted(times, last_ns, side="right")) - 1
    if last < first:
        raise ValueError(f"the trims {trims_s} s leave no frame of {videos[0].entity}")
    # Unique: a window shorter than `num_rig_frames` packets must not put one packet in two rig frames.
    indices: Int64[ndarray, " k"] = np.unique(np.linspace(first, last, num_rig_frames).round().astype(np.int64))
    selection: list[dict[int, int]] = nearest_packets(videos, [int(times[index]) for index in indices], tolerance_ns, (first_ns, last_ns))
    for name, chosen in zip(entities, selection, strict=True):
        if not chosen:
            raise ValueError(f"{name} has no packet within {tolerance_ns} ns of any rig frame of the window")
        if len(set(chosen.values())) < len(chosen):
            raise ValueError(f"{name}: a packet is the nearest of two rig frames; lower the sync tolerance ({tolerance_ns} ns)")
    return [
        StreamKeyframes(
            entity=f"{entity.lstrip('/')}/pinhole/video",
            names=[f"{name}/{sample:06d}" for sample in chosen],
            t_ns=[int(video.t_ns[index]) for index in chosen.values()],
            mask_entity=None,
        )
        for (name, entity), video, chosen in zip(entities.items(), videos, selection, strict=True)
    ]


def calibrate(source: CatalogInputConfig, options: CalibrateOptions) -> CalibrationResult:
    """Estimate the rig's intrinsics and extrinsics on a window of the segment and write the calibration layer.

    Reads `dataset`, `segment`, `catalog_url`, `cameras`, `timeline`, the trims and `sync_tolerance_ns` of `source`;
    its trajectory, stride and the recorded calibration are not read.

    Args:
        source: The segment (already served, for a local input) and its cameras.
        options: Window, model and outputs.

    Returns:
        The calibration and its fit.

    Raises:
        RuntimeError: When a mapping registers nothing, or the rig model leaves a camera out.
    """
    started: float = time.perf_counter()
    dataset: DatasetEntry = CatalogClient(source.catalog_url).get_dataset(name=source.dataset)
    entities: dict[str, str] = {name: entity for entity, name in read_camera_names(dataset, source.segment, source.cameras).items()}
    videos: tuple[StreamIndex, ...] = read_video_index(
        dataset, source.segment, [f"{entity}/pinhole/video" for entity in entities.values()], source.timeline
    )
    streams: list[StreamKeyframes] = window_streams(
        entities, videos, options.num_rig_frames, source.sync_tolerance_ns, (source.trim_start_seconds, source.trim_end_seconds)
    )
    manifest: FrameSourceManifest = FrameSourceManifest(
        catalog_url=source.catalog_url, dataset=source.dataset, segment=source.segment, timeline=source.timeline, streams=streams
    )
    work_dir: Path = options.work_dir or DATA_DIR / "calibrate" / source.segment
    work_dir.mkdir(parents=True, exist_ok=True)
    database_path: Path = work_dir / "database.db"
    database_path.unlink(missing_ok=True)

    names: list[str] = [image_name for stream in streams for image_name in stream.names]
    camera_ids: dict[str, int] = {}
    image_ids: list[list[int | None]] = [[None] * len(entities) for _ in range(options.num_rig_frames)]
    frames: VideoStreamFrames = VideoStreamFrames(manifest)
    try:
        with pycolmap.Database.open(database_path) as database:
            for camera_index, (name, stream) in enumerate(zip(entities, streams, strict=True)):
                _, rgb = next(iter(frames.frames(stream.names[:1])))  # the camera's size, from its first frame
                camera_ids[name] = database.write_camera(
                    seed_camera(camera_index + 1, options.camera_model, int(rgb.shape[2]), int(rgb.shape[1])), True
                )
                for image_name in stream.names:
                    image_ids[int(image_name.split("/")[1])][camera_index] = database.write_image(
                        pycolmap.Image(name=image_name, camera_id=camera_ids[name])
                    )
        extraction: ExtractionReport = extract_features(database_path, work_dir, names, FeatureOptions(backend="raco"), frames=frames)
        pairs: list[ImagePair] = rig_window_pairs(image_ids, options.overlap)
        report: MatchReport = match_pairs(database_path, pairs, MatchingOptions(backend="flexglue", image_dir=work_dir), frames)
    finally:
        frames.close()
    print(
        f"[colsfm.calibrate] {len(names)} images, {sum(extraction.keypoint_counts.values())} keypoints, "
        f"{len(pairs)} pairs, {len(pairs) - report.empty_pairs} verified"
    )

    reconstruction: pycolmap.Reconstruction = solve_rig(database_path, list(entities), work_dir)
    registered: dict[str, int] = {name: 0 for name in entities}
    for image in reconstruction.images.values():
        if image.has_pose:
            registered[image.name.split("/")[0]] += 1
    unregistered: list[str] = [name for name, count in registered.items() if count == 0]
    if unregistered:
        raise RuntimeError(f"{source.segment}: the rig model registered no image of {unregistered}; their extrinsics are unknown")
    cameras: tuple[CatalogCameraModel, ...] = cameras_from_rig(reconstruction, camera_ids, entities, options.camera_model)
    error_px: float = float(reconstruction.compute_mean_reprojection_error())
    layer_path: Path = options.layer_path or work_dir / f"{source.segment}.colsfm_calibrate.rrd"
    write_calibration_layer(
        layer_path,
        source.segment,
        cameras,
        rr.AnyValues(
            camera_model=options.camera_model,
            reference_camera=next(iter(entities)),
            num_rig_frames=len(streams[0].names),
            registered_images=sum(registered.values()),
            mean_reprojection_error_px=error_px,
        ),
        metric_scale=False,
    )
    seconds: float = time.perf_counter() - started
    print(f"[colsfm.calibrate] {registered} registered, {error_px:.3f} px, layer {layer_path} in {seconds:.1f} s")
    return CalibrationResult(
        cameras=cameras, layer_path=layer_path, registered_images=registered, mean_reprojection_error_px=error_px, seconds=seconds
    )
