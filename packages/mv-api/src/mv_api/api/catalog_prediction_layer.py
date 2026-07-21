"""Catalog-native MVAPI prediction layer helpers."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from time import perf_counter, strftime
from typing import Any, Literal

import numpy as np
import pyarrow as pa
import torch
from einops import rearrange
from jaxtyping import Bool, Float32, Float64, Int, UInt8
from numpy import ndarray
from simplecv.camera_parameters import PinholeParameters
from simplecv.data.skeleton.coco_133 import COCO_133_IDS

CATALOG_DATASET_NAME: str = "assembly101"
CATALOG_TIMELINE: str = "video_time"
PREDICTION_2D_ENTITY_TEMPLATE: str = "/world/exo/{camera_name}/pinhole/pred/mvapi/coco133_uv"
PREDICTION_3D_ENTITY: str = "/world/pred/mvapi/coco133_xyz"
PREDICTION_VISUALIZATION_RGB: tuple[int, int, int] = (255, 0, 0)
_NATIVE_FPS_SAMPLE_LIMIT: int = 2048
"""Max ``video_time`` packets read per exo stream to detect native fps. The median inter-packet
gap is stable on any uniform prefix, so bounding the read keeps detection O(1) in clip length."""


@dataclass(frozen=True, slots=True)
class CatalogSegment:
    """One catalog segment selected for MVAPI prediction."""

    dataset: str
    """Catalog dataset name."""
    sequence_key: str
    """Slash-separated sequence key inside the dataset."""
    recording_id: str
    """Rerun catalog segment recording id."""
    recording_uri: str
    """Rerun catalog URL for the segment."""
    path: Path
    """Local RRD path for diagnostics and artifact naming."""


@dataclass(frozen=True, slots=True)
class ExoCameraStream:
    """One discovered exo camera video stream in a catalog segment."""

    name: str
    """Camera name from the exo entity path."""
    video_entity: str
    """Rerun entity path for the camera video."""
    field_path: str
    """Dataloader field path for the video sample component."""
    pinhole_entity: str
    """Rerun entity path for the camera pinhole."""
    transform_entity: str
    """Rerun entity path for the camera transform."""


@dataclass(frozen=True, slots=True)
class ViewerScreenshotTarget:
    """One required screenshot for validating an exo prediction overlay."""

    camera_name: str
    """Exo camera name to validate."""
    overlay_entity: str
    """Prediction entity that must be visible in the screenshot."""
    blueprint_path: Path
    """Per-camera Rerun blueprint path used to isolate this 2D view."""
    screenshot_path: Path
    """Filesystem path where the screenshot should be saved."""


@dataclass(frozen=True, slots=True)
class PredictionRecordingInfo:
    """Rerun recording identity for the generated prediction layer."""

    application_id: str
    """Application id for the prediction recording."""
    recording_id: str
    """Recording id copied from the source catalog segment."""


@dataclass(frozen=True, slots=True)
class CatalogPredictionLayerConfig:
    """Configuration for the catalog-native MVAPI prediction layer tool."""

    rrd_root: Path = Path("/mnt/8tb/data/exoego-forge-catalog")
    """Root directory containing the full ExoEgo Forge RRD catalog."""
    catalog_url: str = "rerun+http://127.0.0.1:9988"
    """URL of the running Rerun catalog server to connect to. Start it with the
    ``simplecv-catalog-serve`` task and register the v1 catalog with ``simplecv-catalog-register``."""
    dataset_name: str = CATALOG_DATASET_NAME
    """Catalog dataset name to process. v1 is expected to remain ``assembly101``."""
    assembly101_row_id: int = 120
    """Default Assembly101 row id after sorting rows by sequence key; row 120 has non-null AV1 VideoStream samples."""
    sequence_key: str | None = None
    """Optional exact Assembly101 sequence key override."""
    max_frames: int | None = 10
    """Maximum number of frames to process; ``None`` means full selected segment."""
    native_fps_override: float | None = None
    """Sampling rate (Hz) for the Rerun dataloader over ``video_time``. ``None`` auto-detects each
    segment's native exo frame rate from packet spacing. The dataloader always samples at (or above)
    the native rate: a sub-native grid drops reference packets before AV1 decode (RR-5087), which can
    cause deterministic ``InvalidDataError`` windows or silently wrong pixels."""
    fetch_size: int = 64
    """Number of samples fetched per Rerun catalog query."""
    video_codec: str = "av1"
    """Video codec passed to Rerun's ``VideoFrameDecoder``."""
    keyframe_interval: int = 300
    """Keyframe interval passed to Rerun's ``VideoFrameDecoder``."""
    tracker_mode: Literal["lightweight", "balanced", "performance", "wholebody"] = "wholebody"
    """MVAPI tracker model preset."""
    tracker_device: Literal["cpu", "cuda"] = "cuda"
    """Device requested by the MVAPI ONNX runtime backend."""
    tracker_backend: Literal["onnxruntime"] = "onnxruntime"
    """Inference backend requested by the MVAPI tracker."""
    keypoint_threshold: float = 0.7
    """Minimum keypoint confidence retained in logged prediction layers."""
    output_root: Path = Path("artifacts/catalog_layers")
    """Root directory for generated prediction layer RRDs."""
    layer_name: str = "mvapi_coco133_upper_body_v1"
    """Catalog layer name for generated predictions."""
    application_id: str = "assembly101_mvapi_coco133"
    """Rerun application id used by generated prediction recordings."""
    register_layer: bool = True
    """Whether to register the generated RRD back into the catalog immediately."""
    validation_root: Path = Path("artifacts/rerun-viewer-validation")
    """Root directory for Rerun Viewer screenshot validation artifacts."""
    capture_native_viewer_screenshots: bool = True
    """Capture one native Rerun Viewer screenshot per exo camera after layer registration."""
    capture_open_viewer_screenshots: bool = False
    """Capture screenshots from an already-running Rerun Viewer via ``ViewerClient``."""
    viewer_addr: str = "127.0.0.1:9876"
    """Rerun Viewer control address used when ``capture_open_viewer_screenshots`` is true."""
    viewer_window_size: str = "1920x1080"
    """Native Viewer screenshot window size documented for manual validation commands."""
    viewer_visual_bounds_width: float = 1920.0
    """Width used to bound native Viewer 2D validation blueprints."""
    viewer_visual_bounds_height: float = 1080.0
    """Height used to bound native Viewer 2D validation blueprints."""
    viewer_screenshot_timeout_seconds: float = 120.0
    """Maximum seconds to wait for one native Viewer screenshot before failing."""


@dataclass(frozen=True, slots=True)
class CatalogPredictionLayerResult:
    """Summary returned by the catalog-native prediction layer run."""

    segment: CatalogSegment
    """Selected catalog segment."""
    rrd_path: Path
    """Generated prediction RRD path."""
    layer_name: str
    """Registered catalog layer name."""
    validation_targets: list[ViewerScreenshotTarget]
    """Expected exo 2D screenshot targets for viewer validation."""
    validation_notes_path: Path
    """Path to the screenshot validation notes file."""


def select_catalog_segment(
    rows: list[CatalogSegment],
    *,
    row_id: int = 0,
    sequence_key: str | None = None,
) -> CatalogSegment:
    """Select one Assembly101 catalog segment from sorted catalog rows.

    Args:
        rows: Candidate catalog segments.
        row_id: Row index after sorting by sequence key.
        sequence_key: Optional exact sequence key override.

    Returns:
        Selected catalog segment.

    Raises:
        ValueError: If there are no rows, no matching sequence key, or row id is out of range.
    """
    if not rows:
        raise ValueError("No catalog segments are available for selection.")

    sorted_rows: list[CatalogSegment] = sorted(rows, key=lambda row: row.sequence_key)
    if sequence_key is not None:
        for row in sorted_rows:
            if row.sequence_key == sequence_key:
                return row
        raise ValueError(f"No catalog segment has sequence_key={sequence_key!r}.")

    if row_id < 0 or row_id >= len(sorted_rows):
        raise ValueError(f"Catalog row id {row_id} is outside the available range [0, {len(sorted_rows) - 1}].")
    return sorted_rows[row_id]


def _optional_segment_column_values(table: pa.Table, column_name: str) -> list[Any | None]:
    """Read optional one-row segment metadata values from a catalog segment table."""
    if column_name not in table.schema.names:
        return [None] * table.num_rows

    values: list[Any | None] = []
    for value in table.column(column_name).to_pylist():
        if isinstance(value, list):
            values.append(value[0] if value else None)
        else:
            values.append(value)
    return values


def _sequence_key_from_recording_id(dataset_name: str, recording_id: str) -> str:
    """Recover a slash-separated sequence key from a registered catalog recording id."""
    prefix: str = f"{dataset_name}__"
    if recording_id.startswith(prefix):
        return recording_id[len(prefix) :].replace("__", "/")
    return recording_id.replace("__", "/")


def catalog_segments_from_dataset(
    dataset_entry: Any,
    *,
    dataset_dir: Path,
    dataset_name: str,
) -> list[CatalogSegment]:
    """Build selectable catalog segments from a registered Rerun dataset.

    Args:
        dataset_entry: Rerun catalog dataset entry exposing ``segment_table`` and ``segment_url``.
        dataset_dir: Local dataset root for diagnostic RRD paths.
        dataset_name: Catalog dataset name.

    Returns:
        Catalog segments sorted by sequence key.

    Raises:
        FileNotFoundError: If the registered dataset has no segment rows.
    """
    segment_batches: list[pa.RecordBatch] = dataset_entry.segment_table().collect()
    if not segment_batches:
        raise FileNotFoundError(f"Registered {dataset_name} dataset has no segments.")

    segment_table: pa.Table = pa.Table.from_batches(segment_batches)
    if segment_table.num_rows == 0:
        raise FileNotFoundError(f"Registered {dataset_name} dataset has no segments.")

    recording_ids: list[str] = [str(recording_id) for recording_id in segment_table.column("rerun_segment_id").to_pylist()]
    sequence_key_values: list[Any | None] = _optional_segment_column_values(segment_table, "property:info:sequence_key")
    rows: list[CatalogSegment] = []
    for recording_id, sequence_key_value in zip(recording_ids, sequence_key_values, strict=True):
        sequence_key: str = (
            str(sequence_key_value)
            if sequence_key_value is not None
            else _sequence_key_from_recording_id(dataset_name, recording_id)
        )
        row: CatalogSegment = CatalogSegment(
            dataset=dataset_name,
            sequence_key=sequence_key,
            recording_id=recording_id,
            recording_uri=str(dataset_entry.segment_url(recording_id)),
            path=(dataset_dir / f"{sequence_key}.rrd").resolve(),
        )
        rows.append(row)
    return sorted(rows, key=lambda row: row.sequence_key)


def _descriptor_component_name(descriptor: Any) -> str | None:
    component_raw: Any = getattr(descriptor, "component", None)
    if isinstance(component_raw, str):
        return component_raw

    name_raw: Any = getattr(descriptor, "name", None)
    if isinstance(name_raw, str):
        return name_raw

    archetype_raw: Any = getattr(descriptor, "archetype", None)
    component_type_raw: Any = getattr(descriptor, "component_type", None)
    if isinstance(archetype_raw, str) and isinstance(component_type_raw, str):
        archetype_name: str = archetype_raw.rsplit(".", 1)[-1]
        component_type_name: str = component_type_raw.rsplit(".", 1)[-1]
        return f"{archetype_name}:{component_type_name}"
    if isinstance(component_type_raw, str):
        return component_type_raw
    return None


def _schema_descriptors(schema: Any) -> list[Any]:
    component_columns: Any = schema.component_columns()
    if isinstance(component_columns, dict):
        return list(component_columns.keys())
    return list(component_columns)


def discover_exo_camera_streams(schema: Any) -> list[ExoCameraStream]:
    """Discover sorted exo camera VideoStream sample fields from a catalog schema.

    Args:
        schema: Rerun catalog schema-like object exposing ``component_columns()``.

    Returns:
        Exo camera streams sorted by camera name.

    Raises:
        ValueError: If no exo ``VideoStream:sample`` fields are found.
    """
    streams_by_entity: dict[str, ExoCameraStream] = {}
    asset_video_entities: list[str] = []
    for descriptor in _schema_descriptors(schema):
        entity_path_raw: Any = getattr(descriptor, "entity_path", None)
        component_name: str | None = _descriptor_component_name(descriptor)
        if entity_path_raw is None or component_name is None:
            continue

        entity_path: str = f"/{str(entity_path_raw).lstrip('/')}"
        entity_no_slash: str = entity_path.lstrip("/")
        if not entity_no_slash.startswith("world/exo/"):
            continue
        if component_name.endswith("AssetVideo:blob"):
            asset_video_entities.append(entity_path)
            continue
        if not component_name.endswith("VideoStream:sample"):
            continue

        video_path: PurePosixPath = PurePosixPath(entity_no_slash)
        pinhole_entity: str = f"/{video_path.parent}"
        transform_entity: str = f"/{video_path.parent.parent}"
        camera_name: str = video_path.parent.parent.name
        field_path: str = f"{entity_path}:VideoStream:sample"
        streams_by_entity[entity_path] = ExoCameraStream(
            name=camera_name,
            video_entity=entity_path,
            field_path=field_path,
            pinhole_entity=pinhole_entity,
            transform_entity=transform_entity,
        )

    streams: list[ExoCameraStream] = sorted(streams_by_entity.values(), key=lambda stream: stream.name)
    if not streams:
        if asset_video_entities:
            raise ValueError(
                "Found exo AssetVideo:blob fields but v1 requires VideoStream:sample fields: "
                f"{sorted(asset_video_entities)}"
            )
        raise ValueError("No exo VideoStream:sample fields were found in the catalog schema.")
    return streams


def _has_component(descriptors: list[Any], *, entity_path: str, component_suffix: str) -> bool:
    normalized_entity: str = f"/{entity_path.lstrip('/')}"
    for descriptor in descriptors:
        entity_path_raw: Any = getattr(descriptor, "entity_path", None)
        component_name: str | None = _descriptor_component_name(descriptor)
        if entity_path_raw is None or component_name is None:
            continue

        descriptor_entity: str = f"/{str(entity_path_raw).lstrip('/')}"
        if descriptor_entity == normalized_entity and component_name.endswith(component_suffix):
            return True
    return False


def validate_exo_camera_calibration(schema: Any, streams: list[ExoCameraStream]) -> None:
    """Validate that each selected exo camera has catalog pinhole and transform data.

    Args:
        schema: Rerun catalog schema-like object exposing ``component_columns()``.
        streams: Selected exo camera streams.

    Raises:
        ValueError: If any selected camera is missing required calibration components.
    """
    descriptors: list[Any] = _schema_descriptors(schema)
    missing_fields: list[str] = []
    for stream in streams:
        required_components: tuple[tuple[str, str], ...] = (
            (stream.pinhole_entity, "Pinhole:image_from_camera"),
            (stream.pinhole_entity, "Pinhole:camera_xyz"),
            (stream.pinhole_entity, "Pinhole:resolution"),
            (stream.transform_entity, "Transform3D:translation"),
            (stream.transform_entity, "Transform3D:mat3x3"),
        )
        for entity_path, component_suffix in required_components:
            if not _has_component(descriptors, entity_path=entity_path, component_suffix=component_suffix):
                missing_fields.append(f"{stream.name}: {entity_path}:{component_suffix}")

    if missing_fields:
        raise ValueError(f"Selected exo cameras are missing calibration components: {missing_fields}")


def build_prediction_rrd_path(
    *,
    output_root: Path,
    segment: CatalogSegment,
    layer_name: str,
) -> Path:
    """Build the prediction RRD path for one catalog segment and layer.

    Args:
        output_root: Root directory for generated prediction layer artifacts.
        segment: Selected catalog segment.
        layer_name: Catalog layer name.

    Returns:
        Path for the generated prediction RRD.
    """
    sequence_path: Path = Path(*PurePosixPath(segment.sequence_key).parts)
    rrd_path: Path = output_root / segment.dataset / sequence_path / f"{layer_name}.rrd"
    return rrd_path


def build_viewer_screenshot_targets(
    *,
    run_dir: Path,
    streams: list[ExoCameraStream],
) -> list[ViewerScreenshotTarget]:
    """Build required exo 2D screenshot targets for Viewer validation.

    Args:
        run_dir: Rerun Viewer validation run directory.
        streams: Selected exo camera streams.

    Returns:
        One screenshot target per selected exo camera.
    """
    targets: list[ViewerScreenshotTarget] = []
    for stream in streams:
        overlay_entity: str = f"/world/exo/{stream.name}/pinhole/pred/mvapi/coco133_uv"
        blueprint_path: Path = run_dir / f"exo_{stream.name}.rbl"
        screenshot_path: Path = run_dir / f"exo_{stream.name}.png"
        targets.append(
            ViewerScreenshotTarget(
                camera_name=stream.name,
                overlay_entity=overlay_entity,
                blueprint_path=blueprint_path,
                screenshot_path=screenshot_path,
            )
        )
    return targets


def index_value_to_time_ns(index_value: int | np.integer[Any] | np.datetime64 | np.timedelta64) -> int:
    """Convert a Rerun dataloader sample index value to nanoseconds for ``rr.set_time``.

    Args:
        index_value: Value returned by ``SampleIndex.global_to_local``.

    Returns:
        Nanoseconds suitable for ``np.timedelta64(value, "ns")``.
    """
    if isinstance(index_value, np.datetime64):
        return int(index_value.astype("datetime64[ns]").astype(np.int64))
    if isinstance(index_value, np.timedelta64):
        return int(index_value.astype("timedelta64[ns]").astype(np.int64))
    return int(index_value)


def write_viewer_validation_notes(
    *,
    run_dir: Path,
    command: str,
    catalog_url: str,
    segment: CatalogSegment,
    rrd_path: Path,
    layer_name: str,
    targets: list[ViewerScreenshotTarget],
) -> Path:
    """Write the Rerun Viewer screenshot validation manifest.

    Args:
        run_dir: Validation run directory.
        command: Command or API flow used to launch/capture screenshots.
        catalog_url: Catalog URL used for validation.
        segment: Selected source segment.
        rrd_path: Generated prediction layer RRD path.
        layer_name: Registered prediction layer name.
        targets: Required screenshot targets.

    Returns:
        Path to the generated notes file.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    notes_path: Path = run_dir / "notes.md"
    lines: list[str] = [
        "# Rerun Viewer Validation",
        "",
        f"- command: `{command}`",
        f"- catalog_url: `{catalog_url}`",
        f"- segment_url: `{segment.recording_uri}`",
        f"- output_rrd: `{rrd_path}`",
        f"- layer_name: `{layer_name}`",
        "",
        "## Required Exo 2D Screenshots",
        "",
    ]
    for target in targets:
        lines.append(
            f"- {target.camera_name}: `{target.screenshot_path.name}` "
            f"using `{target.blueprint_path.name}` must show `{target.overlay_entity}` over the camera image. "
            "status: pending"
        )
    notes_path.write_text("\n".join(lines) + "\n")
    return notes_path


def build_prediction_recording_info(
    *,
    config: CatalogPredictionLayerConfig,
    segment: CatalogSegment,
) -> PredictionRecordingInfo:
    """Build the Rerun recording identity for the prediction layer.

    Args:
        config: Runtime configuration.
        segment: Selected source catalog segment.

    Returns:
        Recording identity that preserves the source segment id for layer registration.
    """
    return PredictionRecordingInfo(
        application_id=config.application_id,
        recording_id=segment.recording_id,
    )


def register_prediction_layer(
    dataset_entry: Any,
    *,
    rrd_path: Path,
    layer_name: str,
) -> Any:
    """Register the generated prediction RRD as a catalog layer, replacing any existing layer.

    Args:
        dataset_entry: Rerun catalog dataset entry.
        rrd_path: Generated prediction RRD path.
        layer_name: Catalog layer name.

    Returns:
        Registration handle after ``wait`` has completed.
    """
    from rerun.catalog import OnDuplicateSegmentLayer

    registration_handle: Any = dataset_entry.register(
        [rrd_path.resolve().as_uri()], layer_name=layer_name, on_duplicate=OnDuplicateSegmentLayer.REPLACE
    )
    registration_handle.wait()
    return registration_handle


def save_exo_viewer_blueprint(
    target: ViewerScreenshotTarget,
    *,
    visual_bounds_width: float = 1920.0,
    visual_bounds_height: float = 1080.0,
) -> Path:
    """Save a per-exo-camera Rerun blueprint for native screenshot validation."""
    import rerun.blueprint as rrb
    from rerun import bindings
    from rerun.recording_stream import RecordingStream

    origin: str = f"/world/exo/{target.camera_name}/pinhole"
    blueprint: rrb.Blueprint = rrb.Blueprint(
        rrb.Spatial2DView(
            origin=origin,
            name=f"exo {target.camera_name}",
            contents=f"{origin}/**",
            visual_bounds=rrb.VisualBounds2D(
                x_range=[0.0, visual_bounds_width],
                y_range=[0.0, visual_bounds_height],
            ),
        ),
        collapse_panels=True,
    )
    blueprint_stream: RecordingStream = RecordingStream._from_native(
        bindings.new_blueprint(
            application_id="mvapi-catalog-validation",
            make_default=False,
            make_thread_default=False,
            default_enabled=True,
        )
    )
    blueprint_stream.set_time("blueprint", sequence=0)
    blueprint._log_to_stream(blueprint_stream)
    target.blueprint_path.parent.mkdir(parents=True, exist_ok=True)
    target.blueprint_path.write_bytes(blueprint_stream.memory_recording().drain_as_bytes())
    return target.blueprint_path


def build_native_viewer_screenshot_command(
    *,
    target: ViewerScreenshotTarget,
    segment: CatalogSegment,
    window_size: str,
) -> list[str]:
    """Build a native Rerun screenshot command isolated from existing viewers."""
    display_wrapper: list[str] = []
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY") or os.environ.get("WAYLAND_SOCKET")) and shutil.which(
        "xvfb-run"
    ) is not None:
        display_wrapper = ["xvfb-run", "-a"]

    return [
        *display_wrapper,
        "rerun",
        "--new",
        "--window-size",
        window_size,
        "--screenshot-to",
        str(target.screenshot_path),
        segment.recording_uri,
        str(target.blueprint_path),
    ]


def capture_native_viewer_screenshots(
    *,
    targets: list[ViewerScreenshotTarget],
    segment: CatalogSegment,
    window_size: str,
    timeout_seconds: float = 120.0,
    visual_bounds_width: float = 1920.0,
    visual_bounds_height: float = 1080.0,
) -> None:
    """Capture one native Rerun Viewer screenshot for each exo camera target."""
    for target in targets:
        save_exo_viewer_blueprint(
            target,
            visual_bounds_width=visual_bounds_width,
            visual_bounds_height=visual_bounds_height,
        )
        target.screenshot_path.parent.mkdir(parents=True, exist_ok=True)
        completed_process: subprocess.CompletedProcess[bytes] = subprocess.run(
            build_native_viewer_screenshot_command(target=target, segment=segment, window_size=window_size),
            check=False,
            timeout=timeout_seconds,
        )
        if completed_process.returncode != 0 and (
            not target.screenshot_path.exists() or target.screenshot_path.stat().st_size <= 0
        ):
            completed_process.check_returncode()


def _first_valid_value(column: pa.ChunkedArray | pa.Array, *, allow_none: bool = False, component_name: str) -> Any:
    values: list[Any] = column.combine_chunks().to_pylist() if isinstance(column, pa.ChunkedArray) else column.to_pylist()
    for value in values:
        if value is None:
            continue
        while isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
            value = value[0]
        return value
    if allow_none:
        return None
    raise ValueError(f"Expected at least one non-null value in column {component_name!r}.")


def _arrow_time_column_to_ns(column: pa.ChunkedArray | pa.Array) -> Int[ndarray, "time"]:
    array: pa.Array = column.combine_chunks() if isinstance(column, pa.ChunkedArray) else column
    values: ndarray = np.asarray(array.to_numpy(zero_copy_only=False))
    if values.size == 0:
        raise ValueError("Expected at least one timestamp value.")
    if np.issubdtype(values.dtype, np.datetime64):
        return values.astype("datetime64[ns]").astype(np.int64)
    if np.issubdtype(values.dtype, np.timedelta64):
        return values.astype("timedelta64[ns]").astype(np.int64)
    return np.asarray(values, dtype=np.int64)


def _catalog_entity_packet_ns(
    *,
    segment_view: Any,
    entity_path: str,
    timeline: str,
) -> Int[ndarray, "time"]:
    """Read one entity's sorted unique video-packet timestamps (ns) on ``timeline``.

    Args:
        segment_view: Catalog view already filtered to the selected segment.
        entity_path: Rerun entity path of the video stream.
        timeline: Catalog duration timeline to read.

    Returns:
        Sorted unique packet timestamps (ns), read from up to ``_NATIVE_FPS_SAMPLE_LIMIT`` packets.

    Raises:
        ValueError: If the entity has no samples on ``timeline``.
    """
    entity_view: Any = segment_view.filter_contents(entity_path)
    table: pa.Table = entity_view.reader(index=timeline).select(timeline).limit(_NATIVE_FPS_SAMPLE_LIMIT).to_arrow_table()
    if table.num_rows == 0:
        raise ValueError(f"No {timeline!r} samples were found for {entity_path}.")
    return np.unique(_arrow_time_column_to_ns(table.column(timeline)))


def native_fps_from_packet_ns(packet_ns: Int[ndarray, "time"]) -> float:
    """Detect a video stream's native frame rate (Hz) from its packet timestamps.

    The Rerun dataloader must sample at or above the native rate: sub-native ``FixedRateSampling``
    makes multiple stored packets collide in grid slots, and ``fill_latest_at`` silently drops one
    packet from each collision (RR-5087). Decode windows crossing a dropped reference then fail
    deterministically with ``InvalidDataError`` until the next keyframe, or can return silently
    wrong pixels. To avoid silently under-sampling, the rate is derived from the median inter-packet
    gap and only trusted when the spacing is near-uniform.

    Args:
        packet_ns: Sorted unique packet timestamps in nanoseconds on a duration timeline.

    Returns:
        The native frame rate in Hz (``1e9 / median inter-packet gap``).

    Raises:
        ValueError: If fewer than two packets are given, the packets are not strictly increasing,
            or the spacing is too irregular to trust (``max_gap / median_gap >= 1.5``).
    """
    if packet_ns.shape[0] < 2:
        raise ValueError(f"Need >=2 video_time packets to detect native fps, got {packet_ns.shape[0]}.")
    gaps_ns: Int[ndarray, "gap"] = np.diff(packet_ns)
    if np.any(gaps_ns <= 0):
        raise ValueError("video_time packets must be strictly increasing to detect native fps.")
    median_gap_ns: float = float(np.median(gaps_ns))
    max_gap_ns: float = float(np.max(gaps_ns))
    if max_gap_ns / median_gap_ns >= 1.5:
        raise ValueError(
            f"video_time packet spacing is too irregular to detect a native fps "
            f"(max_gap={max_gap_ns:.0f} ns vs median={median_gap_ns:.0f} ns); "
            "pass native_fps_override to sample at a known rate."
        )
    return 1_000_000_000.0 / median_gap_ns


def detect_uniform_native_fps(
    *,
    dataset_entry: Any,
    segment: CatalogSegment,
    streams: list[ExoCameraStream],
    timeline: str = CATALOG_TIMELINE,
) -> float:
    """Detect the single native frame rate shared by every selected exo stream.

    One ``FixedRateSampling`` grid drives all camera fields at a shared timestamp, so it only keeps
    the cameras in multiview lock-step when they share a native rate. A mixed-rate rig needs explicit
    temporal alignment semantics rather than this one-rate grid, so mismatched rates are rejected
    outright.

    Args:
        dataset_entry: Rerun catalog dataset entry.
        segment: Selected catalog segment.
        streams: Selected exo camera streams.
        timeline: Catalog duration timeline used for inference.

    Returns:
        The native frame rate in Hz shared by every exo stream.

    Raises:
        ValueError: If the exo streams do not all share one native frame rate.
    """
    segment_view: Any = dataset_entry.filter_segments(segment.recording_id)
    per_stream_fps: dict[str, float] = {}
    for stream in streams:
        packet_ns: Int[ndarray, "time"] = _catalog_entity_packet_ns(
            segment_view=segment_view, entity_path=stream.video_entity, timeline=timeline
        )
        per_stream_fps[stream.name] = native_fps_from_packet_ns(packet_ns)

    reference_fps: float = next(iter(per_stream_fps.values()))
    mismatched: dict[str, float] = {name: fps for name, fps in per_stream_fps.items() if abs(fps - reference_fps) / reference_fps > 0.02}
    if mismatched:
        raise ValueError(
            f"Selected exo streams do not share one native fps (reference {reference_fps:.3f} Hz); a single "
            f"sample grid cannot align them without explicit mixed-rate semantics. Per-stream fps: {per_stream_fps}."
        )
    return reference_fps


def _read_first_catalog_component(
    *,
    segment_view: Any,
    entity_path: str,
    component: str,
    index: str | None,
    allow_none: bool = False,
) -> Any:
    selector: str = f"{entity_path}:{component}"
    projection: list[str] = [selector] if index is None else [index, selector]
    available_columns: set[str] = set(segment_view.filter_contents(entity_path).arrow_schema().names)
    if any(column not in available_columns for column in projection):
        if allow_none:
            return None
        missing_columns: list[str] = [column for column in projection if column not in available_columns]
        raise ValueError(f"Missing catalog columns for {entity_path}: {missing_columns}")

    table: pa.Table = segment_view.filter_contents(entity_path).reader(index=index).select(*projection).to_arrow_table()
    if table.num_rows == 0:
        if allow_none:
            return None
        raise ValueError(f"No rows available for catalog component {selector!r}.")
    column_index: int = 0 if index is None else 1
    return _first_valid_value(
        table.column(column_index),
        allow_none=allow_none,
        component_name=selector,
    )


def _load_catalog_intrinsics(
    *,
    segment_view: Any,
    stream: ExoCameraStream,
    timeline: str = CATALOG_TIMELINE,
) -> Any:
    from simplecv.camera_parameters import Intrinsics

    k_value: Any = _read_first_catalog_component(
        segment_view=segment_view,
        entity_path=stream.pinhole_entity,
        component="Pinhole:image_from_camera",
        index=None,
        allow_none=True,
    )
    camera_xyz_value: Any = _read_first_catalog_component(
        segment_view=segment_view,
        entity_path=stream.pinhole_entity,
        component="Pinhole:camera_xyz",
        index=None,
        allow_none=True,
    )
    resolution_value: Any = _read_first_catalog_component(
        segment_view=segment_view,
        entity_path=stream.pinhole_entity,
        component="Pinhole:resolution",
        index=None,
        allow_none=True,
    )

    if k_value is None:
        k_value = _read_first_catalog_component(
            segment_view=segment_view,
            entity_path=stream.pinhole_entity,
            component="Pinhole:image_from_camera",
            index=timeline,
        )
        camera_xyz_value = _read_first_catalog_component(
            segment_view=segment_view,
            entity_path=stream.pinhole_entity,
            component="Pinhole:camera_xyz",
            index=timeline,
            allow_none=True,
        )
        resolution_value = _read_first_catalog_component(
            segment_view=segment_view,
            entity_path=stream.pinhole_entity,
            component="Pinhole:resolution",
            index=timeline,
            allow_none=True,
        )

    k_matrix: Float32[ndarray, "3 3"] = np.asarray(k_value, dtype=np.float32).reshape(3, 3, order="F")
    camera_conventions: Literal["RDF", "RUB"] = "RDF"
    if camera_xyz_value is not None:
        camera_xyz: Int[ndarray, "3"] = np.asarray(camera_xyz_value, dtype=np.int32).reshape(-1)
        if camera_xyz.size == 3 and tuple(int(value) for value in camera_xyz) == (3, 5, 2):
            camera_conventions = "RUB"

    width: int | None = None
    height: int | None = None
    if resolution_value is not None:
        resolution: Float32[ndarray, "res"] = np.asarray(resolution_value, dtype=np.float32).reshape(-1)
        if resolution.size >= 2:
            width = int(round(float(resolution[0])))
            height = int(round(float(resolution[1])))

    if width is None:
        width = int(round(2.0 * float(k_matrix[0, 2])))
    if height is None:
        height = int(round(2.0 * float(k_matrix[1, 2])))

    return Intrinsics(
        camera_conventions=camera_conventions,
        fl_x=float(k_matrix[0, 0]),
        fl_y=float(k_matrix[1, 1]),
        cx=float(k_matrix[0, 2]),
        cy=float(k_matrix[1, 2]),
        width=width,
        height=height,
    )


def _load_catalog_extrinsics(
    *,
    segment_view: Any,
    stream: ExoCameraStream,
    timeline: str = CATALOG_TIMELINE,
) -> Any:
    from simplecv.camera_parameters import Extrinsics

    translation_value: Any = _read_first_catalog_component(
        segment_view=segment_view,
        entity_path=stream.transform_entity,
        component="Transform3D:translation",
        index=None,
        allow_none=True,
    )
    rotation_value: Any = _read_first_catalog_component(
        segment_view=segment_view,
        entity_path=stream.transform_entity,
        component="Transform3D:mat3x3",
        index=None,
        allow_none=True,
    )

    if translation_value is None:
        translation_value = _read_first_catalog_component(
            segment_view=segment_view,
            entity_path=stream.transform_entity,
            component="Transform3D:translation",
            index=timeline,
        )
    if rotation_value is None:
        rotation_value = _read_first_catalog_component(
            segment_view=segment_view,
            entity_path=stream.transform_entity,
            component="Transform3D:mat3x3",
            index=timeline,
        )

    translation: Float32[ndarray, "3"] = np.asarray(translation_value, dtype=np.float32).reshape(-1).astype(np.float32)
    rotation: Float32[ndarray, "3 3"] = np.asarray(rotation_value, dtype=np.float32).reshape(3, 3, order="F")
    return Extrinsics(cam_R_world=rotation, cam_t_world=translation)


def load_catalog_pinhole_params(
    *,
    dataset_entry: Any,
    segment: CatalogSegment,
    streams: list[ExoCameraStream],
    timeline: str = CATALOG_TIMELINE,
) -> list[PinholeParameters]:
    """Load SimpleCV pinhole parameters for discovered exo streams from the catalog."""
    from simplecv.camera_parameters import PinholeParameters

    segment_view: Any = dataset_entry.filter_segments(segment.recording_id)
    pinholes: list[PinholeParameters] = []
    for stream in streams:
        intrinsics: Any = _load_catalog_intrinsics(segment_view=segment_view, stream=stream, timeline=timeline)
        extrinsics: Any = _load_catalog_extrinsics(segment_view=segment_view, stream=stream, timeline=timeline)
        pinholes.append(PinholeParameters(name=stream.name, intrinsics=intrinsics, extrinsics=extrinsics))
    return pinholes


def build_rerun_iterable_dataset(
    *,
    dataset_entry: Any,
    segment: CatalogSegment,
    streams: list[ExoCameraStream],
    config: CatalogPredictionLayerConfig,
) -> Any:
    """Create the Rerun PyTorch iterable dataset sampled at the exo streams' native fps.

    Samples ``video_time`` at the exo cameras' shared native frame rate with the public
    ``RerunIterableDataset`` + ``FixedRateSampling``. Sampling every packet (``rate_hz`` == native
    fps) gave each packet its own grid slot on these uniformly spaced streams, so frames decode
    reliably with no exact-packet targeting, no ``Field.window``, and no private-attribute
    injection. Decimated (sub-native) sampling is deliberately avoided: grid-slot collisions drop
    reference packets before decode (RR-5087), causing deterministic ``InvalidDataError`` windows
    or silently wrong pixels.

    Args:
        dataset_entry: Rerun catalog dataset entry.
        segment: Selected catalog segment.
        streams: Discovered exo camera streams (must share one native fps).
        config: Runtime configuration.

    Returns:
        A stock ``RerunIterableDataset`` sampling ``video_time`` at the detected native fps.
    """
    from rerun.experimental.dataloader import (
        DataSource,
        Field,
        FixedRateSampling,
        RerunIterableDataset,
        VideoFrameDecoder,
    )

    native_fps: float = (
        config.native_fps_override
        if config.native_fps_override is not None
        else detect_uniform_native_fps(dataset_entry=dataset_entry, segment=segment, streams=streams)
    )
    fields: dict[str, Any] = {
        stream.name: Field(
            path=stream.field_path,
            decode=VideoFrameDecoder(
                codec=config.video_codec,
                keyframe_interval=config.keyframe_interval,
                fps_estimate=native_fps,
            ),
        )
        for stream in streams
    }
    source: Any = DataSource(dataset=dataset_entry, segments=[segment.recording_id])
    return RerunIterableDataset(
        source,
        index=CATALOG_TIMELINE,
        fields=fields,
        timeline_sampling=FixedRateSampling(rate_hz=native_fps),
        fetch_size=config.fetch_size,
        shuffle=False,
    )


def build_torch_loader(rerun_dataset: Any) -> Any:
    """Wrap the Rerun iterable dataset in a PyTorch DataLoader with the v1 contract."""
    from torch.utils.data import DataLoader

    return DataLoader(
        rerun_dataset,
        batch_size=1,
        num_workers=0,
        collate_fn=lambda batch: batch[0],
    )


def none_decoded_exo_stream_names(sample: dict[str, Any], streams: list[ExoCameraStream]) -> list[str]:
    """Return exo stream names that are present in a dataloader sample but not decoded yet."""
    none_streams: list[str] = []
    for stream in streams:
        if stream.name not in sample:
            raise ValueError(f"Rerun dataloader sample is missing required exo camera key {stream.name!r}.")
        if sample[stream.name] is None:
            none_streams.append(stream.name)
    return none_streams


def _upper_body_indices() -> Int[ndarray, "upper_body_plus"]:
    from simplecv.data.skeleton.coco_133 import FACE_IDX, LEFT_HAND_IDX, RIGHT_HAND_IDX

    upper_body_filter_idx: Int[ndarray, "upper_body"] = np.array([5, 6, 7, 8, 9, 10], dtype=np.int64)
    combined_indices: Int[ndarray, "upper_body_plus"] = np.concatenate(
        [upper_body_filter_idx, FACE_IDX, LEFT_HAND_IDX, RIGHT_HAND_IDX]
    )
    return combined_indices


def prediction_visualization_colors(num_keypoints: int) -> UInt8[ndarray, "n_kpts 3"]:
    """Build fixed red RGB colors for catalog prediction overlays."""
    if num_keypoints < 0:
        raise ValueError(f"num_keypoints must be non-negative, got {num_keypoints}.")
    red_rgb: UInt8[ndarray, "3"] = np.array(PREDICTION_VISUALIZATION_RGB, dtype=np.uint8)
    colors: UInt8[ndarray, "n_kpts 3"] = np.repeat(red_rgb[np.newaxis, :], num_keypoints, axis=0)
    return colors


def _log_prediction_frame(
    *,
    mv_state: Any,
    streams: list[ExoCameraStream],
    top_half_mask: Bool[ndarray, "n_kpts"],
    keypoint_threshold: float,
    timestamp_ns: int,
    recording: Any,
) -> None:
    import rerun as rr
    from simplecv.rerun_custom_types import Points2DWithConfidence, Points3DWithConfidence

    timestamps_seconds: Float64[ndarray, "1"] = np.array([float(timestamp_ns) * 1e-9], dtype=np.float64)

    if mv_state.xyzc_t is not None:
        xyz: Float32[ndarray, "n_kpts 3"] = mv_state.xyzc_t[:, :3].astype(np.float32, copy=True)
        scores_3d: Float32[ndarray, "n_kpts"] = mv_state.xyzc_t[:, 3].astype(np.float32, copy=True)
        invalid_3d: Bool[ndarray, "n_kpts"] = np.asarray(
            (~top_half_mask) | (scores_3d < keypoint_threshold) | ~np.isfinite(xyz).all(axis=1),
            dtype=bool,
        )
        xyz[invalid_3d, :] = np.nan
        scores_3d[invalid_3d] = np.nan
        prediction_rgb: UInt8[ndarray, "n_kpts 3"] = prediction_visualization_colors(xyz.shape[0])
        keypoint_lengths_3d: Int[ndarray, "1"] = np.array([xyz.shape[0]], dtype=np.int32)
        rr.send_columns(
            PREDICTION_3D_ENTITY,
            indexes=[rr.TimeColumn(CATALOG_TIMELINE, duration=timestamps_seconds)],
            columns=[
                *Points3DWithConfidence.columns(
                    positions=xyz,
                    confidences=scores_3d,
                    colors=prediction_rgb,
                ).partition(keypoint_lengths_3d),
            ],
            recording=recording,
        )

    if mv_state.uvc_t is not None:
        for stream, uvc_view in zip(streams, mv_state.uvc_t, strict=True):
            uv: Float32[ndarray, "n_kpts 2"] = uvc_view[:, :2].astype(np.float32, copy=True)
            scores_2d: Float32[ndarray, "n_kpts"] = uvc_view[:, 2].astype(np.float32, copy=True)
            invalid_2d: Bool[ndarray, "n_kpts"] = np.asarray(
                (~top_half_mask) | (scores_2d < keypoint_threshold) | ~np.isfinite(uv).all(axis=1),
                dtype=bool,
            )
            uv[invalid_2d, :] = np.nan
            scores_2d[invalid_2d] = np.nan
            prediction_rgb_2d: UInt8[ndarray, "n_kpts 3"] = prediction_visualization_colors(uv.shape[0])
            keypoint_lengths_2d: Int[ndarray, "1"] = np.array([uv.shape[0]], dtype=np.int32)
            rr.send_columns(
                PREDICTION_2D_ENTITY_TEMPLATE.format(camera_name=stream.name),
                indexes=[rr.TimeColumn(CATALOG_TIMELINE, duration=timestamps_seconds)],
                columns=[
                    *Points2DWithConfidence.columns(
                        positions=uv,
                        confidences=scores_2d,
                        colors=prediction_rgb_2d,
                    ).partition(keypoint_lengths_2d),
                ],
                recording=recording,
            )


def _log_prediction_static_metadata(
    *,
    streams: list[ExoCameraStream],
    recording: Any,
) -> None:
    import rerun as rr
    from simplecv.data.skeleton.coco133_layers import Coco133AnnotationLayer
    from simplecv.rerun_custom_types import Points2DWithConfidence, Points3DWithConfidence

    rr.log(
        PREDICTION_3D_ENTITY,
        Points3DWithConfidence.from_fields(
            class_ids=int(Coco133AnnotationLayer.TRIANGULATED_3D),
            keypoint_ids=COCO_133_IDS,
            show_labels=False,
        ),
        static=True,
        recording=recording,
    )
    for stream in streams:
        rr.log(
            PREDICTION_2D_ENTITY_TEMPLATE.format(camera_name=stream.name),
            Points2DWithConfidence.from_fields(
                class_ids=int(Coco133AnnotationLayer.RAW_2D),
                keypoint_ids=COCO_133_IDS,
                show_labels=False,
            ),
            static=True,
            recording=recording,
        )


def _run_mvapi_inference(
    *,
    dataset_entry: Any,
    segment: CatalogSegment,
    streams: list[ExoCameraStream],
    pinholes: list[PinholeParameters],
    config: CatalogPredictionLayerConfig,
    rrd_path: Path,
) -> None:
    import rerun as rr

    from mv_api.api.full_exoego_pipeline import set_annotation_context
    from mv_api.multiview_pose_estimator import MultiviewBodyTracker, MultiviewBodyTrackerConfig, MVHistory

    if len(streams) != len(pinholes):
        raise ValueError(f"Discovered {len(streams)} streams but loaded {len(pinholes)} pinhole calibrations.")

    rerun_dataset: Any = build_rerun_iterable_dataset(
        dataset_entry=dataset_entry,
        segment=segment,
        streams=streams,
        config=config,
    )
    data_loader: Any = build_torch_loader(rerun_dataset)
    total_samples: int = int(len(rerun_dataset))
    total_to_process: int = total_samples if config.max_frames is None else min(total_samples, int(config.max_frames))
    if total_to_process <= 0:
        raise ValueError(f"Selected segment {segment.recording_id!r} has no dataloader samples on {CATALOG_TIMELINE!r}.")

    upper_body_filter_idx: Int[ndarray, "upper_body_plus"] = _upper_body_indices()
    top_half_mask: Bool[ndarray, "n_kpts"] = np.isin(np.arange(133), upper_body_filter_idx)
    tracker_config: MultiviewBodyTrackerConfig = MultiviewBodyTrackerConfig(
        mode=config.tracker_mode,
        backend=config.tracker_backend,
        device=config.tracker_device,
        keypoint_threshold=config.keypoint_threshold,
        cams_for_detection_idx=None,
        use_wilor=False,
        perform_tracking=True,
        verbose=False,
    )
    pose_tracker: MultiviewBodyTracker = MultiviewBodyTracker(
        tracker_config,
        filter_body_idxes=upper_body_filter_idx,
    )
    if pose_tracker.num_keypoints != len(COCO_133_IDS):
        raise ValueError(
            f"The catalog prediction layer logs COCO-133 overlays; tracker mode {config.tracker_mode!r} "
            f"yields {pose_tracker.num_keypoints} keypoints. Use tracker_mode='wholebody'."
        )
    mv_state: MVHistory = MVHistory()

    rrd_path.parent.mkdir(parents=True, exist_ok=True)
    recording_info: PredictionRecordingInfo = build_prediction_recording_info(config=config, segment=segment)
    recording: rr.RecordingStream = rr.RecordingStream(
        application_id=recording_info.application_id,
        recording_id=recording_info.recording_id,
    )
    recording.save(rrd_path)
    set_annotation_context(recording=recording)
    _log_prediction_static_metadata(streams=streams, recording=recording)

    processed_frames: int = 0
    skipped_decode_frames: int = 0
    inference_start_s: float = perf_counter()
    for sample_idx, sample in enumerate(data_loader):
        if processed_frames >= total_to_process:
            break
        none_streams: list[str] = none_decoded_exo_stream_names(sample, streams)
        if none_streams:
            skipped_decode_frames += 1
            continue

        _segment_meta, index_value = rerun_dataset.sample_index.global_to_local(sample_idx)
        timestamp_ns: int = index_value_to_time_ns(index_value)
        # Dataloader samples are RGB CHW uint8 tensors already — upload once,
        # then view as NHWC for the tracker (a wrong dtype fails loudly in the
        # tracker's frame validation rather than being silently cast here).
        frames_rgb: UInt8[torch.Tensor, "n_views h w 3"] = rearrange(
            torch.stack([torch.as_tensor(sample[stream.name]) for stream in streams]).to(config.tracker_device, non_blocking=True),
            "views c h w -> views h w c",
        )

        mv_state = pose_tracker(
            frames_rgb=frames_rgb,
            pinhole_list=pinholes,
            pred_state=mv_state,
            recording=recording,
        )
        _log_prediction_frame(
            mv_state=mv_state,
            streams=streams,
            top_half_mask=top_half_mask,
            keypoint_threshold=config.keypoint_threshold,
            timestamp_ns=timestamp_ns,
            recording=recording,
        )
        if mv_state.xyzc_t is not None:
            mv_state.xyzc_t[~top_half_mask, :] = np.nan
        processed_frames += 1
        if processed_frames == 1 or processed_frames % 50 == 0 or processed_frames == total_to_process:
            elapsed_s: float = perf_counter() - inference_start_s
            fps: float = float(processed_frames) / elapsed_s if elapsed_s > 0.0 else 0.0
            print(
                f"Processed {processed_frames}/{total_to_process} catalog MVAPI frames "
                f"({fps:.3f} frames/s, skipped decode={skipped_decode_frames})",
                flush=True,
            )

    if processed_frames == 0:
        raise ValueError(
            f"Selected segment {segment.recording_id!r} produced no fully decoded exo frame sets on "
            f"{CATALOG_TIMELINE!r}; skipped {skipped_decode_frames} samples with undecoded video frames. "
            "Increase keyframe_interval or verify VideoStream keyframes are present."
        )

    flush = getattr(recording, "flush", None)
    if callable(flush):
        flush()


def capture_open_viewer_screenshots(
    *,
    targets: list[ViewerScreenshotTarget],
    viewer_addr: str,
) -> None:
    """Capture required screenshots from an already-configured native Rerun Viewer."""
    from rerun.experimental import ViewerClient

    client: Any = ViewerClient(viewer_addr)
    for target in targets:
        target.screenshot_path.parent.mkdir(parents=True, exist_ok=True)
        client.save_screenshot(str(target.screenshot_path))


def run_catalog_prediction_layer(config: CatalogPredictionLayerConfig) -> CatalogPredictionLayerResult:
    """Run the catalog-native MVAPI prediction-layer pipeline."""
    from rerun.catalog import CatalogClient

    client: Any = CatalogClient(config.catalog_url)
    dataset_entry: Any = client.get_dataset(config.dataset_name)
    rows: list[CatalogSegment] = catalog_segments_from_dataset(
        dataset_entry,
        dataset_dir=config.rrd_root.expanduser().resolve() / config.dataset_name,
        dataset_name=config.dataset_name,
    )
    segment: CatalogSegment = select_catalog_segment(
        rows,
        row_id=config.assembly101_row_id,
        sequence_key=config.sequence_key,
    )
    segment_view: Any = dataset_entry.filter_segments(segment.recording_id)
    schema: Any = segment_view.schema()
    streams: list[ExoCameraStream] = discover_exo_camera_streams(schema)
    validate_exo_camera_calibration(schema, streams)
    pinholes: list[PinholeParameters] = load_catalog_pinhole_params(
        dataset_entry=dataset_entry,
        segment=segment,
        streams=streams,
        timeline=CATALOG_TIMELINE,
    )
    rrd_path: Path = build_prediction_rrd_path(
        output_root=config.output_root,
        segment=segment,
        layer_name=config.layer_name,
    )
    _run_mvapi_inference(
        dataset_entry=dataset_entry,
        segment=segment,
        streams=streams,
        pinholes=pinholes,
        config=config,
        rrd_path=rrd_path,
    )

    if config.register_layer:
        register_prediction_layer(
            dataset_entry,
            rrd_path=rrd_path,
            layer_name=config.layer_name,
        )

    validation_run_dir: Path = config.validation_root / strftime("%Y%m%d-%H%M%S")
    targets: list[ViewerScreenshotTarget] = build_viewer_screenshot_targets(
        run_dir=validation_run_dir,
        streams=streams,
    )
    screenshot_command: str = (
        "pixi run -e mv-api-catalog --frozen xvfb-run -a rerun --new "
        f"--window-size {config.viewer_window_size} "
        "--screenshot-to <target-screenshot.png> "
        f"{segment.recording_uri} <target-blueprint.rbl>"
    )
    notes_path: Path = write_viewer_validation_notes(
        run_dir=validation_run_dir,
        command=screenshot_command,
        catalog_url=config.catalog_url,
        segment=segment,
        rrd_path=rrd_path,
        layer_name=config.layer_name,
        targets=targets,
    )

    if config.capture_open_viewer_screenshots:
        capture_open_viewer_screenshots(targets=targets, viewer_addr=config.viewer_addr)
    if config.capture_native_viewer_screenshots:
        capture_native_viewer_screenshots(
            targets=targets,
            segment=segment,
            window_size=config.viewer_window_size,
            timeout_seconds=config.viewer_screenshot_timeout_seconds,
            visual_bounds_width=config.viewer_visual_bounds_width,
            visual_bounds_height=config.viewer_visual_bounds_height,
        )

    return CatalogPredictionLayerResult(
        segment=segment,
        rrd_path=rrd_path,
        layer_name=config.layer_name,
        validation_targets=targets,
        validation_notes_path=notes_path,
    )


def main(config: CatalogPredictionLayerConfig) -> None:
    """CLI entrypoint for the catalog-native MVAPI prediction-layer tool."""
    result: CatalogPredictionLayerResult = run_catalog_prediction_layer(config)
    print(f"Generated prediction layer: {result.rrd_path}", flush=True)
    print(f"Layer name: {result.layer_name}", flush=True)
    print(f"Segment: {result.segment.recording_id}", flush=True)
    print(f"Viewer validation notes: {result.validation_notes_path}", flush=True)
    print("Required exo screenshots:", flush=True)
    for target in result.validation_targets:
        print(f"  {target.camera_name}: {target.screenshot_path}", flush=True)
