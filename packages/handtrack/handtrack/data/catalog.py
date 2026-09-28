"""The catalog side of the training data: segment listing, our splits, and one segment's rig and hand-pose timeline.

Per segment the stream runs one static query (cameras, lenses, hand profile) and one temporal query on
``video_time`` (headset pose and both hands' confidence, joint angles and wrist); the video packets come
through ``simplecv.catalog_video.read_catalog_videos``. Large Arrow columns are read through their list
offsets into numpy, never with a per-row ``to_pylist`` loop.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from typing import Literal, TypeAlias

import numpy as np
import pyarrow as pa
import torch
from jaxtyping import Bool, Float32, Int64
from numpy import ndarray
from rerun.catalog import DatasetEntry
from simplecv.umetrack_temp.generic_hand_model_torch import HandModelTorch
from torch import Tensor

from handtrack.geometry.camera import CameraRig
from handtrack.geometry.letterbox import Letterbox, letterbox_for
from handtrack.hand.pose import HandPose, Side, generic_hand_model, hand_model_from_profile
from handtrack.labels.keypoint_input import hand_scale

CATALOG_URL: str = "rerun+http://127.0.0.1:51235"
TIMELINE: str = "video_time"

DatasetName: TypeAlias = Literal["dataforge-umetrack", "dataforge-show3d", "dataforge-show3d-sample"]
Domain: TypeAlias = Literal["real", "synthetic", "show3d"]
SplitName: TypeAlias = Literal["train", "val", "test"]
"""Our splits: ``val`` is the held-out training users (UmeTrack) or subjects (SHOW3D)."""

UMETRACK: DatasetName = "dataforge-umetrack"
SHOW3D: DatasetName = "dataforge-show3d"
SHOW3D_SAMPLE: DatasetName = "dataforge-show3d-sample"

UMETRACK_VALIDATION_USERS: tuple[str, ...] = ("user_10", "user_28", "user_46")
"""Held out of UmeTrack ``training`` for validation: positions 7, 22 and 37 of the 44 sorted training users (evenly spread).

Every UmeTrack user has a real and a synthetic copy of each recording, so holding out a user holds out both domains:
148 recordings, 62,442 frames (7.1% of the 874,264 training frames; 31,311 real, 31,131 synthetic).
"""
SHOW3D_HELDOUT_SUBJECTS: tuple[str, ...] = ("HLU829", "MMO925", "XYZ109")
"""Held out of SHOW3D ``train`` for a separate SHOW3D score: 3 of the 32 subjects, spread through the sorted list, each of
middling size (19, 40 and 32 scenes; 185,294 frames, 5.3% of the labelled frames). PRI519 was skipped as the largest subject."""

SHOW3D_FPS: int = 60
"""SHOW3D headset video rate; its segment table carries no fps property."""

HAND_ROOT: str = "/world/gt/hands"
PROFILE_COLUMN: str = f"{HAND_ROOT}/profile:TextDocument:text"
SIDE_NAMES: tuple[str, str] = ("left", "right")
"""Schema side names in ``Side`` order."""


@dataclass(frozen=True, slots=True)
class DatasetLayout:
    """Where one dataset keeps its headset and how the stream samples it."""

    rig: str
    """Headset rig entity (its Transform3D is world_from_rig)."""
    cameras: tuple[str, ...]
    """Headset camera entities used for training."""
    pool_stride: int
    """Frames between kept frames of the 5 fps pool."""
    tracker_step: int
    """Frames per tracker step at 30 Hz: the gap between the poses the keypoint input extrapolates from."""
    camera_offset: int
    """First camera id of this dataset in ids unique across datasets (UmeTrack 0-3, SHOW3D 4-5)."""


UMETRACK_LAYOUT: DatasetLayout = DatasetLayout("/world/rig_00", tuple(f"/world/rig_00/cam_0{i}" for i in range(4)), pool_stride=6, tracker_step=1, camera_offset=0)
SHOW3D_LAYOUT: DatasetLayout = DatasetLayout("/world/rig_01", ("/world/rig_01/cam_00", "/world/rig_01/cam_01"), pool_stride=12, tracker_step=2, camera_offset=4)
"""SHOW3D runs at 60 Hz, so a 30 Hz tracker step is 2 frames (our choice; the tracker rate is UmeTrack's)."""


def is_show3d(dataset: str) -> bool:
    return dataset.startswith("dataforge-show3d")


def layout_for(dataset: str) -> DatasetLayout:
    return SHOW3D_LAYOUT if is_show3d(dataset) else UMETRACK_LAYOUT


@dataclass(frozen=True, slots=True)
class SegmentInfo:
    """One catalog segment, from the segment table's properties."""

    dataset: str
    segment_id: str
    domain: Domain
    interaction: str
    """UmeTrack ``hand_hand`` / ``separate_hand``; SHOW3D's episode action."""
    split: str
    """The source split: UmeTrack ``training`` / ``testing``, SHOW3D ``train`` / ``test``."""
    subject: str
    """UmeTrack user or SHOW3D subject id."""
    num_frames: int
    fps: int


def segment_infos(dataset: str, table: pa.Table) -> tuple[SegmentInfo, ...]:
    """Parse a ``segment_table()``; SHOW3D keeps only scenes with a ``hand_pose`` layer (the others have no hand labels)."""
    show3d: bool = is_show3d(dataset)
    columns: dict[str, list[list[str | int] | None]] = {name: table[name].to_pylist() for name in table.column_names if name.startswith("property:")}
    ids: list[str] = [str(segment) for segment in table["rerun_segment_id"].to_pylist()]
    layers: list[list[str] | None] = table["rerun_layer_names"].to_pylist()

    def prop(key: str, row: int, default: str | int) -> str | int:
        values: list[str | int] | None = columns.get(f"property:{key}", [None] * table.num_rows)[row]
        return values[0] if values else default

    infos: list[SegmentInfo] = []
    for row, segment in enumerate(ids):
        if show3d:
            if "hand_pose" not in (layers[row] or []):
                continue
            infos.append(
                SegmentInfo(
                    dataset=dataset,
                    segment_id=segment,
                    domain="show3d",
                    interaction=str(prop("episode:action", row, "")),
                    split=str(prop("episode:split", row, "")),
                    subject=str(prop("episode:subject_id", row, "")),
                    num_frames=int(prop("capture:num_frames", row, 0)),
                    fps=SHOW3D_FPS,
                )
            )
            continue
        domain: str = str(prop("episode:domain", row, ""))
        if domain not in ("real", "synthetic"):
            raise ValueError(f"{dataset} {segment}: unexpected domain {domain!r}")
        infos.append(
            SegmentInfo(
                dataset=dataset,
                segment_id=segment,
                domain="real" if domain == "real" else "synthetic",
                interaction=str(prop("episode:interaction", row, "")),
                split=str(prop("episode:split", row, "")),
                subject=str(prop("episode:user", row, "")),
                num_frames=int(prop("capture:num_frames", row, 0)),
                fps=int(prop("capture:fps", row, 30)),
            )
        )
    return tuple(sorted(infos, key=lambda info: info.segment_id))


def list_segments(dataset: DatasetEntry, name: str) -> tuple[SegmentInfo, ...]:
    """Every usable segment of a catalog dataset, sorted by id."""
    return segment_infos(name, dataset.segment_table().to_arrow_table())


def select_split(segments: Sequence[SegmentInfo], split: SplitName) -> tuple[SegmentInfo, ...]:
    """Our split of a dataset's segments.

    UmeTrack: train = ``training`` minus ``UMETRACK_VALIDATION_USERS``, val = those users, test = ``testing`` (real and
    synthetic; report them separately by ``domain``). SHOW3D: train = ``train`` minus ``SHOW3D_HELDOUT_SUBJECTS``,
    val = those subjects; SHOW3D ``test`` has no hand labels, so asking for it raises.
    """
    selected: list[SegmentInfo] = []
    for info in segments:
        if info.domain == "show3d":
            if split == "test":
                raise ValueError("SHOW3D's test scenes carry no hand labels; use split 'val' (the held-out subjects) for a SHOW3D score")
            held_out: bool = info.subject in SHOW3D_HELDOUT_SUBJECTS
            if info.split == "train" and held_out == (split == "val"):
                selected.append(info)
        elif split == "test":
            if info.split == "testing":
                selected.append(info)
        elif info.split == "training" and (info.subject in UMETRACK_VALIDATION_USERS) == (split == "val"):
            selected.append(info)
    return tuple(selected)


# --- Arrow columns -> numpy ------------------------------------------------------------------------------------


def list_rows(column: pa.ChunkedArray, width: int) -> Float32[ndarray, "r w"]:
    """The first value of each row of a ``list<fixed_size_list<float>[width]>`` column (``list<float>`` when width is 1).

    Rows that are null or empty come back NaN, as do null values inside a row.
    """
    lists: pa.ListArray = column.combine_chunks()
    out: Float32[ndarray, "r w"] = np.full((len(lists), width), np.nan, dtype=np.float32)
    if len(lists) == 0:
        return out
    offsets: Int64[ndarray, "r1"] = np.asarray(lists.offsets, dtype=np.int64)
    has_row: Bool[ndarray, "r"] = np.asarray(lists.is_valid(), dtype=bool) & (offsets[1:] > offsets[:-1])
    values: pa.Array = lists.values
    if pa.types.is_fixed_size_list(values.type):
        values = values.flatten()
    flat: Float32[ndarray, "v w"] = np.asarray(values.to_numpy(zero_copy_only=False), dtype=np.float32).reshape(-1, width)
    out[has_row] = flat[offsets[:-1][has_row]]
    return out


def point_rows(column: pa.ChunkedArray, count: int, width: int) -> Float32[ndarray, "r n w"]:
    """Rows of a ``list<fixed_size_list<float>[width]>`` point batch (e.g. ``Points3D:positions``) with ``count`` points each.

    Rows that are null or hold another number of points come back NaN.
    """
    lists: pa.ListArray = column.combine_chunks()
    out: Float32[ndarray, "r n w"] = np.full((len(lists), count, width), np.nan, dtype=np.float32)
    if len(lists) == 0:
        return out
    offsets: Int64[ndarray, "r1"] = np.asarray(lists.offsets, dtype=np.int64)
    full: Bool[ndarray, "r"] = np.asarray(lists.is_valid(), dtype=bool) & (offsets[1:] - offsets[:-1] == count)
    flat: Float32[ndarray, "v w"] = np.asarray(lists.values.flatten().to_numpy(zero_copy_only=False), dtype=np.float32).reshape(-1, width)
    starts: Int64[ndarray, "m"] = offsets[:-1][full]
    out[full] = flat[starts[:, None] + np.arange(count)[None, :]]
    return out


def bool_rows(column: pa.ChunkedArray) -> tuple[Bool[ndarray, "r"], Bool[ndarray, "r"]]:
    """(value, present) of a ``list<bool>`` column; absent rows read False."""
    lists: pa.ListArray = column.combine_chunks()
    offsets: Int64[ndarray, "r1"] = np.asarray(lists.offsets, dtype=np.int64)
    present: Bool[ndarray, "r"] = np.asarray(lists.is_valid(), dtype=bool) & (offsets[1:] > offsets[:-1])
    value: Bool[ndarray, "r"] = np.zeros(len(lists), dtype=bool)
    if present.any():
        flat: Bool[ndarray, "v"] = np.asarray(lists.values.to_numpy(zero_copy_only=False), dtype=bool)
        value[present] = flat[offsets[:-1][present]]
    return value, present


def rotation_from_quaternion_xyzw(quaternion: Float32[ndarray, "*b 4"]) -> Float32[ndarray, "*b 3 3"]:
    """Rotation matrices from xyzw quaternions (normalised first); NaN propagates."""
    q: Float32[ndarray, "*b 4"] = quaternion / np.linalg.norm(quaternion, axis=-1, keepdims=True)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    rows: list[Float32[ndarray, "*b 3"]] = [
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], axis=-1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], axis=-1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], axis=-1),
    ]
    return np.stack(rows, axis=-2).astype(np.float32)


def column_major_3x3(values: Float32[ndarray, "*b 9"]) -> Float32[ndarray, "*b 3 3"]:
    """Rerun stores mat3x3 (and a Pinhole's K) column-major."""
    return np.swapaxes(values.reshape(*values.shape[:-1], 3, 3), -1, -2).copy()


# --- statics: rig, letterboxes, hand model ---------------------------------------------------------------------


def static_entities(layout: DatasetLayout) -> list[str]:
    return [*layout.cameras, *(f"{camera}/pinhole" for camera in layout.cameras), f"{HAND_ROOT}/profile"]


def read_statics(dataset: DatasetEntry, info: SegmentInfo) -> pa.Table:
    """The segment's static row (cameras, lenses, hand profile): one query. Static reads need ``index=None``."""
    return dataset.filter_segments(info.segment_id).filter_contents(static_entities(layout_for(info.dataset))).reader(index=None).to_arrow_table()


def _static_value(statics: pa.Table, column: str, where: str) -> object:
    if column not in statics.column_names or statics.num_rows == 0 or not statics[column][0].is_valid:
        raise ValueError(f"{where}: static column {column} is missing")
    values: list[object] = statics[column][0].as_py()
    if not values:
        raise ValueError(f"{where}: static column {column} is empty")
    return values[0]


def static_floats(statics: pa.Table, column: str, where: str) -> list[float]:
    """A static list-valued component (a mat3x3, a translation, a resolution, distortion coefficients)."""
    value: object = _static_value(statics, column, where)
    if not isinstance(value, list) or not all(isinstance(v, int | float) for v in value):
        raise ValueError(f"{where}: static column {column} is not a list of numbers")
    return [float(v) for v in value]


def static_number(statics: pa.Table, column: str, where: str) -> float:
    value: object = _static_value(statics, column, where)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{where}: static column {column} is not a number")
    return float(value)


def static_text(statics: pa.Table, column: str, where: str) -> str:
    value: object = _static_value(statics, column, where)
    if not isinstance(value, str):
        raise ValueError(f"{where}: static column {column} is not text")
    return value


def read_rig(statics: pa.Table, info: SegmentInfo) -> tuple[CameraRig, tuple[Letterbox, ...]]:
    """The headset cameras (cam_from_rig, K, Fisheye62 or pinhole) and each camera's letterbox into the net frame."""
    layout: DatasetLayout = layout_for(info.dataset)
    where: str = f"{info.dataset} {info.segment_id}"
    sizes: list[list[float]] = []
    cam_from_rig: list[Float32[ndarray, "4 4"]] = []
    focal: list[list[float]] = []
    principal: list[list[float]] = []
    distortion: list[list[float] | None] = []
    for camera in layout.cameras:
        if static_number(statics, f"{camera}:Transform3D:relation", where) != 2:
            raise ValueError(f"{where}: {camera} extrinsic is not ChildFromParent (cam_from_rig)")
        transform: Float32[ndarray, "4 4"] = np.eye(4, dtype=np.float32)
        transform[:3, :3] = column_major_3x3(np.asarray(static_floats(statics, f"{camera}:Transform3D:mat3x3", where), dtype=np.float32))
        transform[:3, 3] = np.asarray(static_floats(statics, f"{camera}:Transform3D:translation", where), dtype=np.float32)
        cam_from_rig.append(transform)
        intrinsics: Float32[ndarray, "3 3"] = column_major_3x3(np.asarray(static_floats(statics, f"{camera}/pinhole:Pinhole:image_from_camera", where), dtype=np.float32))
        focal.append([float(intrinsics[0, 0]), float(intrinsics[1, 1])])
        principal.append([float(intrinsics[0, 2]), float(intrinsics[1, 2])])
        sizes.append(static_floats(statics, f"{camera}/pinhole:Pinhole:resolution", where))
        coefficients_column: str = f"{camera}/pinhole:simplecv.components.DistortionCoefficients"
        if coefficients_column in statics.column_names and statics[coefficients_column][0].is_valid and statics[coefficients_column][0].as_py():
            model: str = static_text(statics, f"{camera}/pinhole:simplecv.components.DistortionModel", where)
            coefficients: list[float] = static_floats(statics, coefficients_column, where)
            if model != "kannala_brandt" or len(coefficients) != 8:
                raise ValueError(f"{where}: {camera} has {model} with {len(coefficients)} coefficients; expected Fisheye62 [k1..k6, p1, p2]")
            distortion.append(coefficients)
        else:
            distortion.append(None)
    if any(d is None for d in distortion) and not all(d is None for d in distortion):
        raise ValueError(f"{where}: mixed pinhole and fisheye cameras")
    rig: CameraRig = CameraRig(
        names=layout.cameras,
        image_size=torch.tensor(sizes, dtype=torch.float32),
        cam_from_rig=torch.from_numpy(np.stack(cam_from_rig)),
        focal=torch.tensor(focal, dtype=torch.float32),
        principal=torch.tensor(principal, dtype=torch.float32),
        fisheye62=None if distortion[0] is None else torch.tensor(distortion, dtype=torch.float32),
    )
    letterboxes: tuple[Letterbox, ...] = tuple(letterbox_for(int(width), int(height)) for width, height in sizes)
    return rig, letterboxes


@cache
def _generic_model() -> HandModelTorch:
    return generic_hand_model()


# --- the hand timeline -----------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HandTimeline:
    """A segment's per-frame headset pose and ground-truth hands, on CPU, one row per ``video_time`` value."""

    video_time_ns: Int64[ndarray, "f"]
    world_from_rig: Float32[Tensor, "f 4 4"]
    """NaN where the headset pose is missing or invalid."""
    headset_valid: Bool[Tensor, "f"]
    """UmeTrack: a finite pose and ``/world/rig_00:untracked`` False; SHOW3D: a finite ``/world/rig_01`` pose."""
    poses: tuple[HandPose, HandPose]
    """Left and right hand poses [f], NaN where the hand has no pose."""
    confidence: Float32[Tensor, "f 2"]
    """The stored per-hand confidence; 0 where it is missing."""
    has_pose: Bool[Tensor, "f 2"]
    """Joint angles and wrist are both present and finite."""
    hand_model: HandModelTorch
    """The subject's hand model from ``/world/gt/hands/profile``."""
    hand_scale: float
    """ϕ: the subject's size relative to UmeTrack's generic hand (``labels.keypoint_input.hand_scale``)."""


def timeline_columns(layout: DatasetLayout, show3d: bool) -> list[str]:
    rig_columns: list[str] = (
        [f"{layout.rig}:Transform3D:mat3x3", f"{layout.rig}:Transform3D:translation"]
        if show3d
        else [f"{layout.rig}:Transform3D:quaternion", f"{layout.rig}:Transform3D:translation", f"{layout.rig}:untracked"]
    )
    hand_columns: list[str] = [
        f"{HAND_ROOT}/{side}/{component}"
        for side in SIDE_NAMES
        for component in ("confidence:Scalars:scalars", "joint_angles:joint_angles", "wrist:Transform3D:quaternion", "wrist:Transform3D:translation")
    ]
    return rig_columns + hand_columns


def read_timeline_table(dataset: DatasetEntry, info: SegmentInfo) -> pa.Table:
    """All label columns of a segment on ``video_time``: one query."""
    layout: DatasetLayout = layout_for(info.dataset)
    entities: list[str] = [layout.rig, *(f"{HAND_ROOT}/{side}/{part}" for side in SIDE_NAMES for part in ("confidence", "joint_angles", "wrist"))]
    columns: list[str] = timeline_columns(layout, is_show3d(info.dataset))
    return dataset.filter_segments(info.segment_id).filter_contents(entities).reader(index=TIMELINE).select(TIMELINE, *columns).to_arrow_table()


def hand_timeline(table: pa.Table, statics: pa.Table, info: SegmentInfo) -> HandTimeline:
    """Parse the timeline query (rows in ``video_time`` order) and the profile into a ``HandTimeline``."""
    layout: DatasetLayout = layout_for(info.dataset)
    show3d: bool = is_show3d(info.dataset)
    where: str = f"{info.dataset} {info.segment_id}"
    times: Int64[ndarray, "f"] = np.asarray(table[TIMELINE].combine_chunks().to_numpy(zero_copy_only=False)).view(np.int64)
    if len(times) == 0 or not np.all(np.diff(times) > 0):
        raise ValueError(f"{where}: expected label rows with strictly increasing {TIMELINE}")
    frames: int = len(times)
    world_from_rig: Float32[ndarray, "f 4 4"] = np.zeros((frames, 4, 4), dtype=np.float32)
    world_from_rig[:, 3, 3] = 1.0
    if show3d:
        world_from_rig[:, :3, :3] = column_major_3x3(list_rows(table[f"{layout.rig}:Transform3D:mat3x3"], 9))
    else:
        world_from_rig[:, :3, :3] = rotation_from_quaternion_xyzw(list_rows(table[f"{layout.rig}:Transform3D:quaternion"], 4))
    world_from_rig[:, :3, 3] = list_rows(table[f"{layout.rig}:Transform3D:translation"], 3)
    headset_valid: Bool[ndarray, "f"] = np.isfinite(world_from_rig).all(axis=(1, 2))
    if not show3d:
        untracked: Bool[ndarray, "f"] = bool_rows(table[f"{layout.rig}:untracked"])[0]
        headset_valid &= ~untracked
    world_from_rig[~headset_valid] = np.nan
    poses: list[HandPose] = []
    confidence: Float32[ndarray, "f 2"] = np.zeros((frames, 2), dtype=np.float32)
    has_pose: Bool[ndarray, "f 2"] = np.zeros((frames, 2), dtype=bool)
    for side in Side:
        prefix: str = f"{HAND_ROOT}/{SIDE_NAMES[side]}"
        confidence[:, side] = np.nan_to_num(list_rows(table[f"{prefix}/confidence:Scalars:scalars"], 1)[:, 0], nan=0.0)
        joint_angles: Float32[ndarray, "f 22"] = list_rows(table[f"{prefix}/joint_angles:joint_angles"], 22)
        rotation: Float32[ndarray, "f 3 3"] = rotation_from_quaternion_xyzw(list_rows(table[f"{prefix}/wrist:Transform3D:quaternion"], 4))
        translation: Float32[ndarray, "f 3"] = list_rows(table[f"{prefix}/wrist:Transform3D:translation"], 3)
        present: Bool[ndarray, "f"] = np.isfinite(joint_angles).all(axis=1) & np.isfinite(rotation).all(axis=(1, 2)) & np.isfinite(translation).all(axis=1)
        has_pose[:, side] = present
        rotation[~present] = np.nan
        translation[~present] = np.nan
        joint_angles[~present] = np.nan
        poses.append(HandPose(rotation=torch.from_numpy(rotation), translation=torch.from_numpy(translation), joint_angles=torch.from_numpy(joint_angles)))
    model: HandModelTorch = hand_model_from_profile(static_text(statics, PROFILE_COLUMN, where))
    return HandTimeline(
        video_time_ns=times,
        world_from_rig=torch.from_numpy(world_from_rig),
        headset_valid=torch.from_numpy(headset_valid),
        poses=(poses[0], poses[1]),
        confidence=torch.from_numpy(confidence),
        has_pose=torch.from_numpy(has_pose),
        hand_model=model,
        hand_scale=hand_scale(model, _generic_model()),
    )


def read_hand_timeline(dataset: DatasetEntry, info: SegmentInfo, statics: pa.Table) -> HandTimeline:
    """One temporal query plus the already-read statics (for the hand profile)."""
    return hand_timeline(read_timeline_table(dataset, info), statics, info)
