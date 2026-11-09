"""Ground-truth hands of dataforge's HOT3D and UmeTrack segments, read from a Rerun catalog.

Every per-dataset read is one catalog round trip across all requested segments
(grouped locally by ``rerun_segment_id``); only the video packets are read per segment.
Ground truth is dataforge's ``hand_pose`` layer: 21 UmeTrack forward-kinematics
landmarks per hand in world metres (COCO-WholeBody hand slots, MediaPipe order), moved
into the camera with the headset pose and ``cam_from_rig``. The ``projections`` layer
supplies dataforge's full-lens 2D projection of the same joints.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from jaxtyping import Bool, Float32, Float64, Int64
from numpy import ndarray
from rerun.catalog import DatasetEntry, DatasetSelection
from scipy.spatial.transform import Rotation
from simplecv.catalog_video import CatalogVideo, read_catalog_videos

from posekit.cameras import CalibratedCamera
from posekit.predictions import HandSide

GtDataset: TypeAlias = Literal["dataforge-hot3d-quest3", "dataforge-umetrack"]
"""Catalog datasets the evaluation reads."""
RIG: str = "/world/rig_00"
GT_XYZ: str = "/world/gt/coco133_xyz"
SIDE_SLOTS: dict[HandSide, int] = {"left": 91, "right": 112}
"""First COCO-WholeBody slot of each hand; 21 slots in MediaPipe order follow."""
SIDES: tuple[HandSide, HandSide] = ("left", "right")
DERIVED_JOINTS: tuple[int, ...] = (1,)
"""dataforge derives hand joint 1 (thumb CMC) as the wrist/CMC midpoint; it is not scored."""
TIME_TOLERANCE_NS: int = 2_000_000
"""Largest gap between a video sample and its pose/label rows."""


def camera_paths(camera: str) -> tuple[str, str, str, str, str]:
    """Camera, pinhole, video, 2D projection and QA-flag entities of ``cam_MM``."""
    entity: str = f"{RIG}/{camera}"
    return entity, f"{entity}/pinhole", f"{entity}/pinhole/video", f"{entity}/pinhole/coco133_uv_projected", f"/world/gt/quality/rig_00/{camera}/qa_pass"


@dataclass(frozen=True, slots=True)
class HandGtSegment:
    """Evaluation frames of one segment: video samples, camera and both hands' ground truth."""

    segment_id: str
    """Catalog segment id."""
    camera: CalibratedCamera
    """Calibration of the evaluated camera (FISHEYE624 with dataforge's eight terms, prism zero)."""
    video: CatalogVideo
    """The camera's compressed video, all samples."""
    sample_indices: Int64[ndarray, "f"]
    """Video sample of each evaluation frame."""
    video_time_ns: Int64[ndarray, "f"]
    """``video_time`` of each evaluation frame."""
    xyz_camera: Float32[ndarray, "f 2 21 3"]
    """Ground-truth joints (left, right) in the camera frame, metres; NaN where missing."""
    uv: Float32[ndarray, "f 2 21 2"]
    """dataforge's full-lens projection of those joints, source pixels; NaN where missing."""
    valid: Bool[ndarray, "f 2 21"]
    """Scoreable joints: finite 3D and 2D, inside the image, not a derived joint."""


def _column(table: pa.Table, name: str) -> list:
    return table[name].to_pylist() if name in table.column_names else [None] * table.num_rows


def calibrated_camera(statics: pa.Table, row: int, camera: str) -> tuple[CalibratedCamera, Float64[ndarray, "4 4"]]:
    """The camera's lens and ``cam_from_rig`` from one statics row (Rerun matrices are column-major)."""
    entity, pinhole, _, _, _ = camera_paths(camera)
    image_from_camera: Float64[ndarray, "3 3"] = np.asarray(statics[f"{pinhole}:Pinhole:image_from_camera"][row].as_py()[0], np.float64).reshape(3, 3).T
    width, height = (int(v) for v in statics[f"{pinhole}:Pinhole:resolution"][row].as_py()[0])
    coefficients: Float64[ndarray, "8"] = np.asarray(statics[f"{pinhole}:simplecv.components.DistortionCoefficients"][row].as_py()[0], np.float64)
    if statics[f"{pinhole}:simplecv.components.DistortionModel"][row].as_py()[0] != "kannala_brandt" or coefficients.shape != (8,):
        raise ValueError(f"{camera}: expected dataforge's 8-term kannala_brandt (Fisheye62) lens")
    cam_from_rig: Float64[ndarray, "4 4"] = np.eye(4)
    cam_from_rig[:3, :3] = np.asarray(statics[f"{entity}:Transform3D:mat3x3"][row].as_py()[0], np.float64).reshape(3, 3).T
    cam_from_rig[:3, 3] = statics[f"{entity}:Transform3D:translation"][row].as_py()[0]
    if statics[f"{entity}:Transform3D:relation"][row].as_py()[0] != 2:
        raise ValueError(f"{camera}: expected a ChildFromParent (cam_from_rig) transform")
    camera_model = CalibratedCamera(image_from_camera, np.concatenate([coefficients, np.zeros(4)]), width, height, lens="FISHEYE624")
    return camera_model, cam_from_rig


def _nearest(times: Int64[ndarray, "n"], targets: Int64[ndarray, "m"]) -> Int64[ndarray, "m"]:
    """Index of the nearest ``times`` row per target, -1 beyond the tolerance."""
    if not len(times):
        return np.full(len(targets), -1, dtype=np.int64)
    right: Int64[ndarray, "m"] = np.clip(np.searchsorted(times, targets), 0, len(times) - 1)
    left: Int64[ndarray, "m"] = np.clip(right - 1, 0, len(times) - 1)
    best: Int64[ndarray, "m"] = np.where(np.abs(times[left] - targets) <= np.abs(times[right] - targets), left, right)
    return np.where(np.abs(times[best] - targets) <= TIME_TOLERANCE_NS, best, -1).astype(np.int64)


RotationKind: TypeAlias = Literal["quaternion", "mat3x3"]


def rig_rotation(value: Sequence[float], kind: RotationKind) -> Float64[ndarray, "3 3"]:
    """A rig's rotation from one ``Transform3D`` cell: a quaternion (x, y, z, w) or a 3x3 matrix, which Rerun stores column-major."""
    flat: Float64[ndarray, "n"] = np.asarray(value, np.float64).reshape(-1)
    return Rotation.from_quat(flat.reshape(4)).as_matrix() if kind == "quaternion" else flat.reshape(3, 3).T


def read_hand_gt(
    dataset: DatasetEntry, name: GtDataset, segment_ids: Sequence[str], camera: str, *, stride: int, max_frames: int
) -> list[HandGtSegment]:
    """Ground truth of every requested segment for one camera.

    Args:
        dataset: The catalog dataset.
        name: Its dataforge name (HOT3D adds a per-camera QA flag).
        segment_ids: Segments to read.
        camera: Camera entity name, e.g. ``cam_00``.
        stride: Keep every ``stride``-th labelled frame.
        max_frames: Keep at most this many frames per segment.

    Returns:
        One ``HandGtSegment`` per segment, in ``segment_ids`` order.
    """
    entity, pinhole, video_entity, uv_entity, qa_entity = camera_paths(camera)
    scoped: DatasetSelection = dataset.filter_segments(list(segment_ids))
    statics: pa.Table = scoped.filter_contents([entity, pinhole]).reader(index=None).to_arrow_table()
    present: set[str] = {field.name for field in dataset.arrow_schema()}
    kind: RotationKind = "quaternion" if f"{RIG}:Transform3D:quaternion" in present else "mat3x3"  # dataforge now writes mat3x3
    temporal_entities: list[str] = [RIG, GT_XYZ, uv_entity] + ([qa_entity] if name == "dataforge-hot3d-quest3" else [])
    columns: list[str] = [
        f"{RIG}:Transform3D:translation", f"{RIG}:Transform3D:{kind}", f"{GT_XYZ}:Points3D:positions",
        f"{GT_XYZ}:simplecv.KeypointConfidence3D:confidences", f"{uv_entity}:Points2D:positions",
    ] + ([f"{qa_entity}:Scalars:scalars"] if name == "dataforge-hot3d-quest3" else [])
    temporal: pa.Table = scoped.filter_contents(temporal_entities).reader(index="video_time").select("rerun_segment_id", "video_time", *columns).to_arrow_table()
    segments: list[HandGtSegment] = []
    for segment_id in segment_ids:
        static_rows: list[int] = [i for i, s in enumerate(statics["rerun_segment_id"].to_pylist()) if s == segment_id]
        static_row: int = next(i for i in static_rows if statics[f"{pinhole}:Pinhole:image_from_camera"][i].is_valid)
        camera_model, cam_from_rig = calibrated_camera(statics, static_row, camera)
        rows: pa.Table = temporal.filter(pc.field("rerun_segment_id") == segment_id)
        times: Int64[ndarray, "r"] = rows["video_time"].cast(pa.int64()).to_numpy()

        def stream(column: str, rows: pa.Table = rows, times: Int64[ndarray, "r"] = times) -> tuple[Int64[ndarray, "k"], list]:
            """One column's non-null samples in time order."""
            values: list = _column(rows, column)
            keep: list[int] = [i for i, v in enumerate(values) if v is not None]
            order: Int64[ndarray, "k"] = np.asarray(sorted(keep, key=lambda i: times[i]), dtype=np.int64)
            return times[order], [values[i] for i in order]

        rig_t, rig_translation = stream(f"{RIG}:Transform3D:translation")
        _, rig_rotations = stream(f"{RIG}:Transform3D:{kind}")
        gt_t, gt_xyz = stream(f"{GT_XYZ}:Points3D:positions")
        _, gt_conf = stream(f"{GT_XYZ}:simplecv.KeypointConfidence3D:confidences")
        uv_t, uv_rows = stream(f"{uv_entity}:Points2D:positions")
        qa_t, qa_rows = stream(f"{qa_entity}:Scalars:scalars") if name == "dataforge-hot3d-quest3" else (np.empty(0, np.int64), [])

        video: CatalogVideo = read_catalog_videos(dataset, segment_id, [video_entity], "video_time")[0]
        sample_times: Int64[ndarray, "s"] = video.t_ns.astype(np.int64)
        gt_at: Int64[ndarray, "s"] = _nearest(gt_t, sample_times)
        rig_at: Int64[ndarray, "s"] = _nearest(rig_t, sample_times)
        uv_at: Int64[ndarray, "s"] = _nearest(uv_t, sample_times)
        ok: Bool[ndarray, "s"] = (gt_at >= 0) & (rig_at >= 0) & (uv_at >= 0)
        if name == "dataforge-hot3d-quest3":
            qa_at: Int64[ndarray, "s"] = _nearest(qa_t, sample_times)
            ok &= np.asarray([index >= 0 and float(np.asarray(qa_rows[index]).reshape(-1)[0]) > 0.5 for index in qa_at])
        chosen: Int64[ndarray, "c"] = np.flatnonzero(ok)[::stride][:max_frames]

        xyz: Float32[ndarray, "f 2 21 3"] = np.full((len(chosen), 2, 21, 3), np.nan, np.float32)
        uv: Float32[ndarray, "f 2 21 2"] = np.full((len(chosen), 2, 21, 2), np.nan, np.float32)
        valid: Bool[ndarray, "f 2 21"] = np.zeros((len(chosen), 2, 21), bool)
        for out, sample in enumerate(chosen):
            world_from_rig: Float64[ndarray, "4 4"] = np.eye(4)
            world_from_rig[:3, :3] = rig_rotation(rig_rotations[rig_at[sample]], kind)
            world_from_rig[:3, 3] = np.asarray(rig_translation[rig_at[sample]], np.float64).reshape(3)
            cam_from_world: Float64[ndarray, "4 4"] = cam_from_rig @ np.linalg.inv(world_from_rig)
            points: Float64[ndarray, "133 3"] = np.asarray(gt_xyz[gt_at[sample]], np.float64).reshape(133, 3)
            confidence: Float64[ndarray, "133"] = np.asarray(gt_conf[gt_at[sample]], np.float64).reshape(133)
            pixels: Float64[ndarray, "133 2"] = np.asarray(uv_rows[uv_at[sample]], np.float64).reshape(133, 2)
            for side_index, side in enumerate(SIDES):
                slots: slice = slice(SIDE_SLOTS[side], SIDE_SLOTS[side] + 21)
                xyz[out, side_index] = points[slots] @ cam_from_world[:3, :3].T + cam_from_world[:3, 3]
                uv[out, side_index] = pixels[slots]
                inside: Bool[ndarray, "21"] = (
                    np.isfinite(uv[out, side_index]).all(1)
                    & (uv[out, side_index, :, 0] >= 0) & (uv[out, side_index, :, 0] < camera_model.width)
                    & (uv[out, side_index, :, 1] >= 0) & (uv[out, side_index, :, 1] < camera_model.height)
                )
                valid[out, side_index] = inside & np.isfinite(xyz[out, side_index]).all(1) & (confidence[slots] > 0) & (xyz[out, side_index, :, 2] > 0)
                valid[out, side_index, list(DERIVED_JOINTS)] = False
        segments.append(HandGtSegment(segment_id, camera_model, video, chosen.astype(np.int64), sample_times[chosen], xyz, uv, valid))
    return segments
