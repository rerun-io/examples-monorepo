"""Read one explicit SelfCap segment on a selected catalog clock.

Uses the same shared rig discovery as exo-calib. This module exposes no
registration, segment enumeration, or catalog mutation operation.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pyarrow as pa
from dataforge.datasets.selfcap_raw import FactoryCalibration
from dataforge.records import decode
from dataforge.schema import (
    CAPTURE_TIMELINE,
    cam_path,
    capture_property,
    coco133_uv_path,
    coco133_xyz_path,
    gyro_path,
    pinhole_path,
    rig_path,
    video_path,
)
from jaxtyping import Float64
from numpy import ndarray
from rerun.catalog import DatasetEntry
from simplecv.catalog_rig_layout import CatalogRigLayout, catalog_components, parse_rig_layout, read_rig_kinds
from simplecv.catalog_video import CatalogVideo, read_catalog_videos
from simplecv.rrd_query_utils import first_valid_value

from selfcap_curate.links import SpeedSeries, pose_speed
from selfcap_curate.records import CatalogHands, EpisodeKey, PhoneDetections, SelfcapLayout, TimeSync, read_frozen_sync

CATALOG_URL: str = "rerun+http://127.0.0.1:51235"
DATASET_NAME: str = "dataforge-selfcap"


@dataclass(frozen=True, slots=True)
class Segment:
    """An explicit catalog segment bound to its complete discovered device layout."""

    dataset: DatasetEntry
    """Read-only catalog dataset."""
    id: str
    """Explicit catalog segment id."""
    layout: SelfcapLayout
    """Device assignment discovered once for this segment."""


@dataclass(frozen=True, slots=True)
class SessionClock:
    """Ingested session identity and OAK drift origin."""

    session: str
    """Full raw-source session UUID."""
    start_s: float | None
    """First OAK left session-global frame counter divided by 24."""


def read_session_clock(segment: Segment) -> SessionClock:
    """Read ingested metadata for one episode; never consult raw CSVs."""
    dataset, segment_id, oak_rig = segment.dataset, segment.id, segment.layout.oak_rig
    camera: str = cam_path(oak_rig, 1)
    table: pa.Table = dataset.filter_segments(segment_id).filter_contents([camera, "/__properties/capture"]).reader(index=None).to_arrow_table()
    values = (
        table[f"{camera}:first_frame_idx"].drop_null().combine_chunks().flatten().to_numpy()
        if f"{camera}:first_frame_idx" in table.column_names
        else np.empty(0)
    )
    if not values.size:
        native = scalar(table, f"{camera}:num_native_frames", allow_none=True)
        if native is not None and int(native) > 0:
            raise ValueError(f"{segment_id}: missing ingested first_frame_idx; reconvert the base")
    session: str = str(scalar(table, capture_property("session")))
    return SessionClock(session, round(int(values.reshape(-1)[0]) / 24.0, 3) if values.size else None)


def select_layout(layout: CatalogRigLayout, phones: tuple[str, ...]) -> SelfcapLayout:
    """Select SelfCap devices from segment-local rig metadata."""
    exo: set[str] = {c.rig for c in layout.cameras if c.rig_kind == "exo"}
    oak: set[str] = {c.rig for c in layout.cameras if c.rig_kind == "ego"}
    quest: set[str] = {c.rig for c in layout.cameras if c.rig_kind == "quest"}
    count: int = len(exo)
    if count not in (3, 4) or oak != {f"rig_{count:02d}"} or quest != {f"rig_{count + 1:02d}"}:
        raise ValueError(f"not a SelfCap rig layout: exo={exo}, oak={oak}, quest={quest}")
    result: SelfcapLayout = SelfcapLayout(count, count + 1, phones)
    available: set[str] = {c.video_entity for c in layout.cameras}
    if not set(result.video_entities) <= available:
        raise ValueError("SelfCap segment is missing a required camera")
    return result


@dataclass(frozen=True, slots=True)
class EpisodeInputs:
    """Catalog-only inputs for CPU rotation-speed links."""

    layout: SelfcapLayout
    """Discovered device assignment."""
    gyro: SpeedSeries
    """OAK gyro magnitude on the requested catalog clock."""
    pose: SpeedSeries
    """Quest angular speed on the requested catalog clock."""
    videos: tuple[CatalogVideo, ...]
    """OAK left, RGB, Quest right encoded video samples."""
    focal_px: tuple[float, ...]
    """Original horizontal focal lengths in video order."""
    width_px: tuple[int, ...]
    """Original image widths in video order."""


def rig_layout(dataset: DatasetEntry, segment_id: str) -> CatalogRigLayout:
    """Discover camera and rig metadata within one explicitly selected segment."""
    view = dataset.filter_segments(segment_id)
    components = catalog_components(view.schema())
    codecs: list[str] = [f"{item.entity_path}:VideoStream:codec" for item in components if item.archetype == "VideoStream" and item.field == "codec"]
    table: pa.Table = view.reader(index=None).select(*codecs).to_arrow_table()
    present: set[str] = {name.removesuffix(":VideoStream:codec") for name in codecs if table[name].drop_null().length() > 0}
    components = [item for item in components if item.archetype != "VideoStream" or item.entity_path in present]
    return parse_rig_layout(components, read_rig_kinds(dataset, segment_id, parse_rig_layout(components).rigs))


def read_episode(
    segment: Segment,
    *,
    timeline: Literal["capture_time", "video_time"] = CAPTURE_TIMELINE,
) -> EpisodeInputs:
    """Read one named segment; never enumerate the shared corpus or substitute clocks."""
    identity = EpisodeKey.parse(segment.id)
    if segment.id != identity.segment:
        raise ValueError("expected an explicit selfcap__date__sid8__epNNN segment id")
    dataset, segment_id, layout = segment.dataset, segment.id, segment.layout
    view = dataset.filter_segments(segment_id)
    gyro_entity: str = gyro_path(layout.oak_rig, 0)
    pose_entity: str = rig_path(layout.quest_rig)
    table: pa.Table = view.filter_contents([gyro_entity, pose_entity]).reader(index=timeline).to_arrow_table()
    gyro_column: str = f"{gyro_entity}:Scalars:scalars"
    pose_column: str = f"{pose_entity}:Transform3D:quaternion"
    if pose_column not in table.column_names:
        raise ValueError(f"{segment_id}: missing Quest quaternion track on {timeline}")
    gyro_table: pa.Table = table.filter(table[gyro_column].is_valid())
    pose_table: pa.Table = table.filter(table[pose_column].is_valid())
    tg: Float64[ndarray, "g"] = seconds(gyro_table[timeline])
    tp: Float64[ndarray, "p"] = seconds(pose_table[timeline])
    gyro_values: Float64[ndarray, "g 3"] = floats(gyro_table[gyro_column], -1, 3)
    quaternions: Float64[ndarray, "p 4"] = floats(pose_table[pose_column], -1, 4)
    pinholes: list[str] = [pinhole_path(rig, cam) for rig, cam in ((layout.oak_rig, 1), (layout.oak_rig, 0), (layout.quest_rig, 1))]
    calibration: pa.Table = view.filter_contents(pinholes).reader(index=None).to_arrow_table()
    focals: list[float] = []
    widths: list[int] = []
    for pinhole in pinholes:
        matrix: Float64[ndarray, "3 3"] = (
            np.asarray(first_valid_value(calibration[f"{pinhole}:Pinhole:image_from_camera"]), dtype=np.float64).reshape(3, 3).T
        )
        resolution: Float64[ndarray, "2"] = np.asarray(first_valid_value(calibration[f"{pinhole}:Pinhole:resolution"]), dtype=np.float64).reshape(2)
        focals.append(float(matrix[0, 0]))
        widths.append(int(resolution[0]))
    videos: tuple[CatalogVideo, ...] = read_catalog_videos(dataset, segment_id, layout.video_entities, timeline)
    return EpisodeInputs(
        layout, SpeedSeries(tg, np.linalg.norm(gyro_values, axis=1)), pose_speed(tp, quaternions), videos, tuple(focals), tuple(widths)
    )


def read_catalog_hands(segment: Segment) -> CatalogHands:
    """Read only one base's raw hand/rig streams on capture_time, never video_time.

    Calibration layers may repeat hands on video_time. Selecting capture_time
    keeps the migrated base as the source of raw samples and avoids that copy.
    """
    dataset, segment_id, layout = segment.dataset, segment.id, segment.layout
    view = dataset.filter_segments(segment_id)
    head: str = rig_path(layout.quest_rig)
    entity: str = coco133_xyz_path()
    points_column: str = f"{entity}:Points3D:positions"
    rotation_column: str = f"{head}:Transform3D:quaternion"
    translation_column: str = f"{head}:Transform3D:translation"
    table: pa.Table = (
        view.filter_contents([entity, head])
        .reader(index=CAPTURE_TIMELINE)
        .select(CAPTURE_TIMELINE, points_column, rotation_column, translation_column)
        .to_arrow_table()
    )
    hands: pa.Table = table.filter(table[points_column].is_valid())
    poses: pa.Table = table.filter(table[rotation_column].is_valid())
    if not len(hands):
        raise ValueError(f"{segment_id}: base Quest hands absent on capture_time; migrate the base first")
    return CatalogHands(
        seconds(hands[CAPTURE_TIMELINE]),
        floats(hands[points_column], -1, 133, 3),
        seconds(poses[CAPTURE_TIMELINE]),
        floats(poses[translation_column], -1, 3),
        floats(poses[rotation_column], -1, 4),
    )


def read_phone_videos(
    segment: Segment,
    *,
    timeline: Literal["capture_time", "video_time"] = CAPTURE_TIMELINE,
) -> dict[str, CatalogVideo]:
    """Discover named iPhones in one segment on the requested sample clock."""
    dataset, segment_id, layout = segment.dataset, segment.id, segment.layout
    videos = read_catalog_videos(dataset, segment_id, [video_path(rig, 0) for rig in range(len(layout.phones))], timeline)
    return dict(zip(layout.phones, videos, strict=True))


def read_factories(segment: Segment) -> dict[str, FactoryCalibration]:
    """Parse only one segment's static raw vendor values; never read calibration files."""
    dataset, segment_id, layout = segment.dataset, segment.id, segment.layout
    view = dataset.filter_segments(segment_id)
    rigs = tuple(f"rig_{i:02d}" for i in range(layout.quest_rig + 1))
    kinds = {rig: "ego" if i == layout.oak_rig else "quest" if i == layout.quest_rig else "exo" for i, rig in enumerate(rigs)}
    paths = {rig: rig_path(int(rig.removeprefix("rig_"))) for rig in rigs}
    table = view.filter_contents(list(paths.values())).reader(index=None).to_arrow_table()
    result: dict[str, FactoryCalibration] = {}
    for rig in rigs:
        column = f"{paths[rig]}:raw_calibration_json"
        if column not in table.column_names or not len(table[column].drop_null()):
            raise ValueError(f"{segment_id}/{rig}: missing raw_calibration_json; migrate or reconvert the base")
        text = str(scalar(table, column))
        factory = decode(FactoryCalibration, text, source=f"{segment_id}/{rig}")
        if (kind := kinds[rig]) is not None and kind in ("ego", "quest"):
            result[kind] = factory
    return result


def read_phone_names(dataset: DatasetEntry, segment_id: str, layout: CatalogRigLayout) -> tuple[str, ...]:
    """Read physical phone names without fetching the video samples."""
    paths = [f"/world/{c.rig}" for c in layout.cameras if c.rig_kind == "exo"]
    metadata = dataset.filter_segments(segment_id).filter_contents(paths).reader(index=None).to_arrow_table()
    return tuple(str(scalar(metadata, f"{path}:name")) for path in paths)


class MissingSegmentError(ValueError):
    """The requested segment is not registered in the selected dataset."""


def open_segment(dataset: DatasetEntry, segment_id: str) -> Segment:
    """Read the complete segment-local physical device assignment."""
    if segment_id not in dataset.segment_ids():
        raise MissingSegmentError(f"segment {segment_id} is not registered in dataset {dataset.name}")
    layout = rig_layout(dataset, segment_id)
    phones = read_phone_names(dataset, segment_id, layout)
    # Phone-only bases remain usable for keypoint preflight; wearable slots are reserved by the ingest convention.
    if layout.cameras and all(camera.rig_kind == "exo" for camera in layout.cameras):
        return Segment(dataset, segment_id, SelfcapLayout(len(phones), len(phones) + 1, phones))
    return Segment(dataset, segment_id, select_layout(layout, phones))


def read_time_sync(dataset: DatasetEntry, segment_id: str) -> TimeSync:
    """Decode one segment's embedded alignment record."""
    table = dataset.filter_segments(segment_id).filter_contents(["/__properties/time_sync"]).reader(index=None).to_arrow_table()
    column = next(name for name in table.column_names if name.endswith(":time_sync:record_json"))
    record = read_frozen_sync(Path(segment_id), text=str(scalar(table, column)))
    if record.episode_key != EpisodeKey.parse(segment_id):
        raise ValueError(f"{segment_id}: time_sync identity belongs to {record.key}")
    return record


def read_gyro(segment: Segment) -> tuple[Float64[ndarray, "n"], Float64[ndarray, "n 3"]]:
    """Read OAK angular velocity on the raw capture clock."""
    dataset, segment_id, layout = segment.dataset, segment.id, segment.layout
    entity = gyro_path(layout.oak_rig, 0)
    column = f"{entity}:Scalars:scalars"
    table = (
        dataset.filter_segments(segment_id).filter_contents([entity]).reader(index=CAPTURE_TIMELINE).select(CAPTURE_TIMELINE, column).to_arrow_table()
    )
    table = table.filter(table[column].is_valid())
    return seconds(table[CAPTURE_TIMELINE]), floats(table[column], -1, 3)


def scalar(table: pa.Table, column: str, *, allow_none: bool = False) -> str | int | float | None:
    """Unwrap one static scalar with the existing Arrow/numpy semantics."""
    value = np.asarray(first_valid_value(table[column], allow_none=allow_none)).reshape(-1).item(0)
    return value


def read_phone_detections(segment: Segment) -> dict[str, PhoneDetections]:
    """Read this segment's Sapiens2 samples on presented capture_time.

    Missing layers return no detections; a different segment's schema must never
    make a camera or stream appear here. Null rows from other cameras are skipped.
    """
    dataset, segment_id, layout = segment.dataset, segment.id, segment.layout
    view = dataset.filter_segments(segment_id)
    available = {(c.entity_path, c.archetype, c.field) for c in catalog_components(view.schema())}
    entities = {phone: coco133_uv_path(rig, 0, pred="sapiens2") for rig, phone in enumerate(layout.phones)}
    entities = {phone: entity for phone, entity in entities.items() if (entity, "Points2D", "positions") in available}
    if not entities:
        return {}
    table = view.filter_contents(list(entities.values())).reader(index=CAPTURE_TIMELINE).to_arrow_table()
    result: dict[str, PhoneDetections] = {}
    for phone, entity in entities.items():
        rows = table.filter(table[f"{entity}:Points2D:positions"].is_valid())
        if not len(rows):
            continue
        stamps = np.asarray(rows[CAPTURE_TIMELINE].cast(pa.int64()), dtype=np.int64)
        order = np.argsort(stamps)
        xy = np.asarray(rows[f"{entity}:Points2D:positions"].to_pylist(), dtype=np.float64).reshape(-1, 133, 2)
        ids = np.asarray(rows[f"{entity}:Points2D:keypoint_ids"].to_pylist(), dtype=np.int64).reshape(-1, 133)
        scores = np.asarray(rows[f"{entity}:confidence"].to_pylist(), dtype=np.float64).reshape(-1, 133)
        if not np.all(ids == np.arange(133)):
            raise ValueError(f"{segment_id}: Sapiens2 requires all 133 ordered keypoints")
        result[phone] = PhoneDetections(stamps[order] / 1e9, xy[order], scores[order])
    return result


def seconds(column: pa.ChunkedArray) -> Float64[ndarray, "n"]:
    """Convert catalog nanoseconds to float seconds, preserving cast order."""
    return column.to_numpy().astype(np.int64).astype(np.float64) * 1e-9


def floats(column: pa.ChunkedArray, *shape: int) -> Float64[ndarray, "..."]:
    """Flatten nested Arrow lists into float64 values with the requested shape."""
    values = column.combine_chunks()
    while pa.types.is_list(values.type) or pa.types.is_fixed_size_list(values.type):
        values = values.flatten()
    return values.to_numpy(zero_copy_only=False).astype(np.float64).reshape(*shape)
