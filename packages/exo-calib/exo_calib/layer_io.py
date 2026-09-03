"""The exocalib catalog layers: names, entity paths, and each stage's record.

Every stage writes one layer onto the source segment and later stages query it
back — the catalog is the pipeline's only stage store. This module owns that
contract: which layer carries what, under which entity, with which components.
Stage B's keypoint record lives in :mod:`exo_calib.kp2d_layer`.

Base layer (written by the dataset converter; read here for the cameras):
- exo videos:   ``/world/rig_XX/cam_YY/pinhole/video`` (``VideoStream``)
- intrinsics:   ``/world/rig_XX/cam_YY/pinhole:Pinhole:image_from_camera``
- extrinsics:   ``/world/rig_XX/cam_YY:Transform3D:{mat3x3,translation}`` — logged by
  simplecv ``log_pinhole`` with ``from_parent=True``, so they ARE cam_R_world / cam_t_world.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NamedTuple, TypeAlias, get_args

import numpy as np
import pyarrow as pa
import rerun as rr
from jaxtyping import Bool, Float64, Int64
from numpy import ndarray
from rerun.catalog import DatasetEntry, OnDuplicateSegmentLayer
from simplecv.camera_parameters import Extrinsics, Intrinsics, PinholeParameters
from simplecv.catalog_calibration import read_camera_calibration
from simplecv.data.skeleton.coco_133 import COCO_133_ID2NAME, COCO_133_LINKS
from simplecv.rerun_custom_types import Points3DWithConfidence
from simplecv.rerun_log_utils import log_pinhole
from simplecv.rrd_query_utils import first_valid_value_as

from exo_calib.cameras import RigCameras
from exo_calib.correspondences import ObservationSet

TIMELINE: str = "video_time"
APPLICATION_ID: str = "exocalib"
EXOCALIB_ROOT: str = "/world/exocalib"
INIT_LAYER: str = "exocalib_init"
KP2D_LAYER: str = "exocalib_kp2d"
REFINED_LAYER: str = "exocalib_refined"
ALIGN_LAYER: str = "exocalib_align"
PIPELINE_CALIBRATION_MARKER: str = "exocalib_written"
"""Static ``AnyValues`` field the refine stage logs on a base camera node whose
``Transform3D``/``Pinhole`` it wrote itself (no dataset ground truth there).
Discovery excludes such cameras from ``calibrated_camera_names``."""
IMAGE_PLANE_DISTANCE_M: float = 0.1
"""Frustum size of the pipeline's camera layers; the base layer's GT frusta use the same, so the two compare visually."""

CalibrationVariant: TypeAlias = Literal["init", "refined"]
"""The two camera sets the pipeline writes: Stage A's estimate and the refined rig."""
CALIBRATION_VARIANTS: tuple[CalibrationVariant, ...] = get_args(CalibrationVariant)
VARIANT_COLOR: dict[CalibrationVariant, tuple[int, int, int]] = {"init": (255, 214, 0), "refined": (0, 200, 255)}
"""One colour per variant for everything the viewer draws about it: frusta, 3D keypoints, error lines."""
CalibrationSource: TypeAlias = CalibrationVariant | Literal["ground_truth"]
"""Where :func:`read_rig_cameras` reads from: a pipeline layer, or the dataset's base layer (evaluation only)."""


class ClassSpec(NamedTuple):
    """One class of a COCO-133 annotation context: id, viewer label, skeleton colour."""

    class_id: int
    label: str
    color: tuple[int, int, int]


def pinhole_entity(camera_name: str) -> str:
    """Log-side entity path of a camera's pinhole node."""
    return f"/world/{camera_name}/pinhole"


def exocalib_entity(variant: CalibrationVariant) -> str:
    """Entity path of an exocalib camera-rig variant."""
    return f"{EXOCALIB_ROOT}/{variant}"


def kp3d_entity(variant: CalibrationVariant) -> str:
    """Entity path of an exocalib triangulated-keypoint variant."""
    return f"{EXOCALIB_ROOT}/kp3d_{variant}"


def new_layer_recording(segment_id: str, rrd_path: Path) -> tuple[rr.RecordingStream, Path]:
    """Create a recording that saves near ``rrd_path`` and registers as a layer of ``segment_id``.

    The recording id must equal the source segment id so the catalog attaches the
    layer to the existing segment (mv-api ``catalog_prediction_layer`` pattern).
    If ``rrd_path`` exists, a ``-N``-suffixed sibling is written instead: the
    in-memory catalog server caches layer-file descriptors, so overwriting a
    registered rrd in place serves stale data. Returns the recording and the
    path actually written.
    """
    rrd_path.parent.mkdir(parents=True, exist_ok=True)
    actual_path: Path = rrd_path
    counter: int = 1
    while actual_path.exists():
        actual_path = rrd_path.with_name(f"{rrd_path.stem}-{counter}{rrd_path.suffix}")
        counter += 1
    recording: rr.RecordingStream = rr.RecordingStream(application_id=APPLICATION_ID, recording_id=segment_id)
    recording.save(actual_path)
    return recording, actual_path


def register_layer(dataset: DatasetEntry, rrd_path: Path, layer_name: str) -> None:
    """Register a generated RRD as a catalog layer, replacing any existing one."""
    dataset.register([rrd_path.resolve().as_uri()], layer_name=layer_name, on_duplicate=OnDuplicateSegmentLayer.REPLACE).wait()


def log_coco133_class_context(recording: rr.RecordingStream, entity: str, classes: tuple[ClassSpec, ...], connections: bool = True) -> None:
    """Log a COCO-133 annotation context the simplecv exoego way.

    One class per skeleton instance: keypoint links render in the class color,
    so each logged keypoint set (GT / init / refined / 2D detections) is
    distinguishable by the color of its skeleton. ``connections=False`` keeps
    the keypoint names (hover) but draws no skeleton edges.
    """
    recording.log(
        entity,
        rr.AnnotationContext(
            [
                rr.ClassDescription(
                    info=rr.AnnotationInfo(id=spec.class_id, label=spec.label, color=spec.color),
                    keypoint_annotations=[rr.AnnotationInfo(id=i, label=n) for i, n in COCO_133_ID2NAME.items()],
                    keypoint_connections=COCO_133_LINKS if connections else [],
                )
                for spec in classes
            ]
        ),
        static=True,
    )


def log_cameras_layer(
    cameras: RigCameras, resolution_wh: tuple[int, int], entity_prefix: str, recording: rr.RecordingStream, *, color: tuple[int, int, int]
) -> None:
    """Log a camera rig as static pinhole frusta under ``entity_prefix``, drawn in ``color`` (see :data:`VARIANT_COLOR`)."""
    for i, name in enumerate(cameras.names):
        pinhole_intrinsics: Intrinsics = Intrinsics(
            camera_conventions="RDF", k_matrix=cameras.intrinsics[i], width=resolution_wh[0], height=resolution_wh[1]
        )
        extrinsics: Extrinsics = Extrinsics(cam_R_world=cameras.cam_T_world[i][:3, :3], cam_t_world=cameras.cam_T_world[i][:3, 3])
        pinhole: PinholeParameters = PinholeParameters(name=name.replace("/", "_"), intrinsics=pinhole_intrinsics, extrinsics=extrinsics)
        log_pinhole(pinhole, Path(f"{entity_prefix}/{name}"), image_plane_distance=IMAGE_PLANE_DISTANCE_M, static=True, recording=recording)
        recording.log(f"{entity_prefix}/{name}/pinhole", rr.Pinhole.from_fields(color=color), static=True)


def read_rig_cameras(dataset: DatasetEntry, segment_id: str, names: tuple[str, ...], *, source: CalibrationSource) -> RigCameras:
    """Read a rig's intrinsics and extrinsics back from a catalog layer.

    Args:
        dataset: Catalog dataset entry.
        segment_id: Segment to read.
        names: Camera entity names, e.g. ``rig_00/cam_00``.
        source: ``"init"`` / ``"refined"`` read the pipeline's own frusta layers (the stage store for
            camera parameters); ``"ground_truth"`` reads the dataset's base layer — evaluation only,
            never inside the pipeline.

    Returns:
        The rig's camera parameters in float64.
    """
    root: str = "/world" if source == "ground_truth" else exocalib_entity(source)
    calibration: dict[str, PinholeParameters] = read_camera_calibration(dataset, segment_id, [f"{root}/{name}" for name in names])
    intrinsics: Float64[ndarray, "v 3 3"] = np.stack([np.asarray(calibration[f"{root}/{name}"].intrinsics.k_matrix, dtype=np.float64) for name in names])
    cam_T_world: Float64[ndarray, "v 4 4"] = np.stack([np.asarray(calibration[f"{root}/{name}"].extrinsics.cam_T_world, dtype=np.float64) for name in names])
    return RigCameras(names=names, intrinsics=intrinsics, cam_T_world=cam_T_world)


@dataclass(slots=True, frozen=True)
class RefinementDiagnostics:
    """What the refine stage reports about its own run; static ``AnyValues`` on the refined rig entity, copied into ``eval.json``."""

    init_reprojection_px: float
    """Mean reprojection error of the robustly triangulated points against the Stage A cameras."""
    ba_reprojection_px: list[float]
    """Mean reprojection error after each bundle-adjustment round, then after the focal stage (and the rotation guard, when it fired)."""
    observation_count_per_view: list[int]
    """Observations that survived robust triangulation, per camera."""
    metric_rescale_fix: float
    """Global scale applied after BA from MoGe-2 depth; ``1.0`` when the estimate was skipped."""
    focal_scale: list[float]
    """Per-camera focal multiplier from the focal stage."""


def log_refinement_diagnostics(recording: rr.RecordingStream, diagnostics: RefinementDiagnostics) -> None:
    """Write the refinement diagnostics next to the refined cameras."""
    recording.log(
        exocalib_entity("refined"),
        rr.AnyValues(
            init_reprojection_px=np.array([diagnostics.init_reprojection_px]),
            ba_reprojection_px=np.asarray(diagnostics.ba_reprojection_px, dtype=np.float64),
            observation_count_per_view=np.asarray(diagnostics.observation_count_per_view, dtype=np.int64),
            metric_rescale_fix=np.array([diagnostics.metric_rescale_fix]),
            focal_scale=np.asarray(diagnostics.focal_scale, dtype=np.float64),
        ),
        static=True,
    )


def read_refinement_diagnostics(dataset: DatasetEntry, segment_id: str) -> RefinementDiagnostics:
    """Read the diagnostics written by :func:`log_refinement_diagnostics`.

    Raises:
        ValueError: If the refined layer is missing a diagnostic column (rerun the refine stage).
    """
    entity: str = exocalib_entity("refined")
    table: pa.Table = dataset.filter_segments(segment_id).filter_contents([entity]).reader(index=None).to_arrow_table()

    def static_list(field_name: str) -> list[float]:
        column_name: str = f"{entity}:{field_name}"
        if column_name not in table.column_names:
            raise ValueError(f"{column_name} is missing from segment {segment_id}; rerun the refine stage")
        return np.asarray(first_valid_value_as(table.column(column_name), list, component_name=column_name), dtype=np.float64).reshape(-1).tolist()

    return RefinementDiagnostics(
        init_reprojection_px=static_list("init_reprojection_px")[0],
        ba_reprojection_px=static_list("ba_reprojection_px"),
        observation_count_per_view=[int(item) for item in static_list("observation_count_per_view")],
        metric_rescale_fix=static_list("metric_rescale_fix")[0],
        focal_scale=static_list("focal_scale"),
    )


def log_point_tracks(
    recording: rr.RecordingStream,
    entity: str,
    obs: ObservationSet,
    points_xyz: Float64[ndarray, "n 3"],
    conf: Float64[ndarray, " n"],
    times_ns: Int64[ndarray, "t"],
    class_id: int,
) -> None:
    """Log triangulated points frame-by-frame with per-point Kineo confidences."""
    for frame in np.unique(obs.point_frame_idx):
        rows: Int64[ndarray, " m"] = np.nonzero(obs.point_frame_idx == frame)[0]
        points: Float64[ndarray, "m 3"] = points_xyz[rows]
        keep: Bool[ndarray, " m"] = np.isfinite(points).all(axis=1)
        if not keep.any():
            continue
        recording.set_time(TIMELINE, duration=1e-9 * float(times_ns[int(frame)]))
        recording.log(
            entity,
            Points3DWithConfidence(
                positions=points[keep],
                confidences=conf[rows][keep],
                class_ids=class_id,
                keypoint_ids=obs.point_joint_idx[rows][keep].astype(np.int64),
                radii=0.008,
            ),
        )


def write_base_calibration(recording: rr.RecordingStream, cameras: RigCameras, resolution_wh: tuple[int, int]) -> None:
    """Write a refined calibration onto base camera nodes that carry no dataset ground truth.

    Datasets without ground truth (wildcap) leave the base camera nodes bare, so
    their video and 2D keypoints cannot be placed in 3D. The marker tells rig
    discovery that this calibration is the pipeline's own, not ground truth.
    """
    log_cameras_layer(cameras, resolution_wh, "/world", recording, color=VARIANT_COLOR["refined"])
    for name in cameras.names:
        recording.log(f"/world/{name}", rr.AnyValues(**{PIPELINE_CALIBRATION_MARKER: True}), static=True)
