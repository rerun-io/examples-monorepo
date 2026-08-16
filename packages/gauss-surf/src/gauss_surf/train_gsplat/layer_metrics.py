"""Reference loading and metric definitions for fused layer publication."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pyarrow as pa
from arkitscenes_download.ingest.paths import (
    DEPTH_ULTRAWIDE_RECT,
    NORMALS_ULTRAWIDE_RECT,
    RGB_ULTRAWIDE_RECT,
    TIMELINE,
)
from datafusion import col
from jaxtyping import Bool, Float32, UInt8
from numpy import ndarray
from rerun.catalog import DatasetView

from gauss_surf.catalog import table_timestamps
from gauss_surf.contracts import CameraTag
from gauss_surf.render_io import blob_bytes

RGB_ERROR_MAXIMUM: float = 0.5
"""Maximum mean-absolute RGB error mapped to the top of the colormap."""
DEPTH_ERROR_MAXIMUM_M: float = 0.5
"""Maximum absolute depth error, in metres, mapped to the top of the colormap."""
NORMAL_ERROR_MAXIMUM_DEGREES: float = 45.0
"""Maximum angular normal error, in degrees, mapped to the top of the colormap."""
ULTRAWIDE_REFERENCE_COLUMN: str = f"/{DEPTH_ULTRAWIDE_RECT}:EncodedDepthImage:blob"
"""Mesh-raycast reference component used for rectified-ultrawide metrics."""
RGB_REFERENCE_COLUMN: str = f"/{RGB_ULTRAWIDE_RECT}:EncodedImage:blob"
"""Rectified-ultrawide RGB component used for triage metrics."""
DEPTH_REFERENCE_COLUMN: str = f"/{DEPTH_ULTRAWIDE_RECT}:EncodedDepthImage:blob"
"""Rectified-ultrawide mesh-depth component used for triage metrics."""
NORMAL_REFERENCE_COLUMN: str = f"/{NORMALS_ULTRAWIDE_RECT}:EncodedImage:blob"
"""Rectified-ultrawide normal component used for triage metrics."""


@dataclass(frozen=True, slots=True)
class CameraSummary:
    """Coverage and depth-error summary for one camera."""

    camera: CameraTag
    """Wide or rectified-ultrawide camera tag."""
    frame_count: int
    """Number of full-grid product frames encoded."""
    compared_frames: int
    """Number of frames with an exact provenance-depth reference."""
    coverage_min: float
    """Minimum per-frame splat valid-pixel fraction."""
    coverage_mean: float
    """Mean per-frame splat valid-pixel fraction."""
    coverage_max: float
    """Maximum per-frame splat valid-pixel fraction."""
    joint_valid_fraction: float
    """Joint-valid splat/reference pixels divided by all compared pixels."""
    joint_valid_pixels: int
    """Number of jointly valid pixels contributing to MAE."""
    total_pixels: int
    """Total splat pixels considered for joint coverage."""
    mae_m: float
    """Mean absolute z-depth error over jointly valid pixels, in metres."""
    resized_reference_frames: int
    """Reference frames resized by nearest-neighbor sampling to splat resolution."""


@dataclass(frozen=True, slots=True)
class TriageReferenceBlobs:
    """Source ultrawide signals at one exact chosen timestamp."""

    timestamp_ns: int
    """Exact duration since recording start, in nanoseconds."""
    rgb_blob: bytes
    """Stored rectified RGB image bytes."""
    depth_blob: bytes
    """Stored mesh-raycast depth PNG bytes."""
    normal_blob: bytes
    """Stored away-from-camera MoGe normal PNG bytes."""


@dataclass(frozen=True, slots=True)
class TriageErrors:
    """Three error images plus their valid-pixel means."""

    rgb_map_hw3: UInt8[ndarray, "h w 3"]
    """Viridis-like mean-absolute RGB error, scaled from 0 to 0.5."""
    depth_map_hw3: UInt8[ndarray, "h w 3"]
    """Viridis-like absolute depth error, scaled from 0 to 0.5 metres."""
    normal_map_hw3: UInt8[ndarray, "h w 3"]
    """Viridis-like angular normal error, scaled from 0 to 45 degrees."""
    rgb_mean: float
    """Mean normalized RGB error over splat-valid pixels."""
    depth_mean_m: float
    """Mean absolute depth error over jointly valid pixels, in metres."""
    normal_mean_degrees: float
    """Mean normal angle over jointly valid pixels, in degrees."""


@dataclass(frozen=True, slots=True)
class MetricSummary:
    """Distribution of per-frame means for one error signal."""

    mean: float
    """Mean of finite per-frame means."""
    minimum: float
    """Minimum finite per-frame mean."""
    maximum: float
    """Maximum finite per-frame mean."""


@dataclass(frozen=True, slots=True)
class TriageWriteStats:
    """Product and diagnostic counts plus chosen-frame error summaries."""

    product_frames: int
    """Number of full-grid splat RGB and normal frames written."""
    diagnostic_frames: int
    """Number of chosen-frame error-map rows written."""
    rgb: MetricSummary
    """Per-frame mean RGB-error distribution on chosen frames."""
    depth: MetricSummary
    """Per-frame mean depth-error distribution on chosen frames."""
    normal: MetricSummary
    """Per-frame mean normal-error distribution on chosen frames."""


@dataclass(frozen=True, slots=True)
class SplatTriageWriteStats:
    """Wide-normal and ultrawide triage counts from one recording write."""

    wide_normal_frames: int
    """Number of full-grid wide splat-normal frames written."""
    ultrawide: TriageWriteStats
    """Ultrawide product and chosen-diagnostic statistics."""


def reference_blobs_at_component_timestamps(segment_view: DatasetView, component_name: str) -> dict[int, bytes]:
    """Read encoded reference depth at every exact component timestamp."""
    available_columns: set[str] = set(segment_view.arrow_schema().names)
    if component_name not in available_columns:
        raise ValueError(f"catalog segment has no required comparison component: {component_name}")
    table: pa.Table = (
        segment_view.reader(index=TIMELINE, fill_latest_at=False)
        .filter(col(f'"{component_name}"').is_not_null())
        .select(TIMELINE, component_name)
        .sort(TIMELINE)
        .to_arrow_table()
    )
    timestamps_ns_n: ndarray = table_timestamps(table).astype(np.int64)
    component_values: list[Any] = table.column(component_name).to_pylist()
    references: dict[int, bytes] = {}
    for timestamp_ns, value in zip(timestamps_ns_n, component_values, strict=True):
        timestamp: int = int(timestamp_ns)
        if timestamp in references:
            raise ValueError(f"reference component {component_name!r} has duplicate timestamp {timestamp}")
        references[timestamp] = blob_bytes(value, component_name)
    return references


def resize_reference_nearest(
    reference_m_hw: Float32[ndarray, "source_h source_w"], target_hw: tuple[int, int]
) -> Float32[ndarray, "target_h target_w"]:
    """Resize reference depth with nearest-neighbor sampling to preserve invalid zero pixels."""
    target_height: int = target_hw[0]
    target_width: int = target_hw[1]
    if reference_m_hw.shape == target_hw:
        return reference_m_hw
    source_height: int = reference_m_hw.shape[0]
    source_width: int = reference_m_hw.shape[1]
    source_y_h: ndarray = np.minimum((np.arange(target_height) * source_height) // target_height, source_height - 1)
    source_x_w: ndarray = np.minimum((np.arange(target_width) * source_width) // target_width, source_width - 1)
    return reference_m_hw[np.ix_(source_y_h, source_x_w)].astype(np.float32, copy=False)


def viridis_like(values: Float32[ndarray, "..."]) -> UInt8[ndarray, "... 3"]:
    """Map normalized values to a small piecewise-linear viridis approximation."""
    positions_n: Float32[ndarray, "n=6"] = np.asarray([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], dtype=np.float32)
    colors_n3: Float32[ndarray, "n=6 3"] = np.asarray(
        [[68.0, 1.0, 84.0], [65.0, 68.0, 135.0], [42.0, 120.0, 142.0], [34.0, 168.0, 132.0], [122.0, 209.0, 81.0], [253.0, 231.0, 37.0]],
        dtype=np.float32,
    )
    clipped: Float32[ndarray, "..."] = np.clip(values, 0.0, 1.0).astype(np.float32, copy=False)
    rgb_float: Float32[ndarray, "... 3"] = np.empty((*clipped.shape, 3), dtype=np.float32)
    for channel in range(3):
        rgb_float[..., channel] = np.interp(clipped, positions_n, colors_n3[:, channel]).astype(np.float32)
    return np.rint(rgb_float).astype(np.uint8)


def angular_error_degrees(
    first_hw3: Float32[ndarray, "h w 3"],
    second_hw3: Float32[ndarray, "h w 3"],
) -> tuple[Float32[ndarray, "h w"], Bool[ndarray, "h w"]]:
    """Compute unsigned angles between two normal maps."""
    if first_hw3.shape != second_hw3.shape or first_hw3.ndim != 3 or first_hw3.shape[-1] != 3:
        raise ValueError(f"normal shapes must match as (H, W, 3), got {first_hw3.shape} and {second_hw3.shape}")
    first_norm_hw: Float32[ndarray, "h w"] = np.linalg.norm(first_hw3, axis=-1).astype(np.float32)
    second_norm_hw: Float32[ndarray, "h w"] = np.linalg.norm(second_hw3, axis=-1).astype(np.float32)
    valid_hw: Bool[ndarray, "h w"] = (
        np.all(np.isfinite(first_hw3), axis=-1)
        & np.all(np.isfinite(second_hw3), axis=-1)
        & (first_norm_hw > 1e-6)
        & (second_norm_hw > 1e-6)
    )
    safe_first_norm_hw: Float32[ndarray, "h w"] = np.where(valid_hw, first_norm_hw, 1.0).astype(np.float32)
    safe_second_norm_hw: Float32[ndarray, "h w"] = np.where(valid_hw, second_norm_hw, 1.0).astype(np.float32)
    cosine_hw: Float32[ndarray, "h w"] = np.sum(first_hw3 * second_hw3, axis=-1) / (safe_first_norm_hw * safe_second_norm_hw)
    angle_hw: Float32[ndarray, "h w"] = np.rad2deg(np.arccos(np.clip(cosine_hw, -1.0, 1.0))).astype(np.float32)
    return np.where(valid_hw, angle_hw, 0.0).astype(np.float32), valid_hw


def colorize_error(
    error_hw: Float32[ndarray, "h w"], valid_hw: Bool[ndarray, "h w"], *, maximum: float
) -> UInt8[ndarray, "h w 3"]:
    """Colorize an error image and paint invalid pixels black."""
    if error_hw.shape != valid_hw.shape:
        raise ValueError(f"error and validity shapes differ: {error_hw.shape} != {valid_hw.shape}")
    if maximum <= 0.0 or not np.isfinite(maximum):
        raise ValueError(f"maximum must be positive and finite, got {maximum}")
    normalized_hw: Float32[ndarray, "h w"] = (error_hw / maximum).astype(np.float32)
    color_hw3: UInt8[ndarray, "h w 3"] = viridis_like(normalized_hw)
    color_hw3[~valid_hw] = 0
    return color_hw3


def _masked_mean(values_hw: Float32[ndarray, "h w"], valid_hw: Bool[ndarray, "h w"]) -> float:
    """Return one valid-pixel mean or NaN for an empty mask."""
    return float(np.mean(values_hw[valid_hw], dtype=np.float64)) if np.any(valid_hw) else float("nan")


def compute_triage_errors(
    source_rgb_hw3: UInt8[ndarray, "h w 3"],
    splat_rgb_hw3: UInt8[ndarray, "h w 3"],
    mesh_depth_m_hw: Float32[ndarray, "h w"],
    splat_depth_m_hw: Float32[ndarray, "h w"],
    moge_normal_hw3: Float32[ndarray, "h w 3"],
    splat_normal_hw3: Float32[ndarray, "h w 3"],
) -> TriageErrors:
    """Build RGB, depth, and normal error maps for one joined frame."""
    expected_hw: tuple[int, int] = splat_depth_m_hw.shape
    expected_hw3: tuple[int, int, int] = (*expected_hw, 3)
    signal_shapes: tuple[tuple[int, ...], ...] = (
        source_rgb_hw3.shape,
        splat_rgb_hw3.shape,
        mesh_depth_m_hw.shape,
        moge_normal_hw3.shape,
        splat_normal_hw3.shape,
    )
    if signal_shapes != (expected_hw3, expected_hw3, expected_hw, expected_hw3, expected_hw3):
        raise ValueError(f"triage signal shapes do not align to splat depth {expected_hw}: {signal_shapes}")
    splat_valid_hw: Bool[ndarray, "h w"] = np.isfinite(splat_depth_m_hw) & (splat_depth_m_hw > 0.0)
    source_rgb_float_hw3: Float32[ndarray, "h w 3"] = source_rgb_hw3.astype(np.float32) / 255.0
    splat_rgb_float_hw3: Float32[ndarray, "h w 3"] = splat_rgb_hw3.astype(np.float32) / 255.0
    rgb_error_hw: Float32[ndarray, "h w"] = np.mean(np.abs(source_rgb_float_hw3 - splat_rgb_float_hw3), axis=-1).astype(np.float32)
    mesh_valid_hw: Bool[ndarray, "h w"] = np.isfinite(mesh_depth_m_hw) & (mesh_depth_m_hw > 0.0)
    depth_valid_hw: Bool[ndarray, "h w"] = splat_valid_hw & mesh_valid_hw
    depth_error_m_hw: Float32[ndarray, "h w"] = np.abs(splat_depth_m_hw - mesh_depth_m_hw).astype(np.float32)
    normal_error_degrees_hw: Float32[ndarray, "h w"]
    normal_valid_hw: Bool[ndarray, "h w"]
    normal_error_degrees_hw, normal_valid_hw = angular_error_degrees(splat_normal_hw3, moge_normal_hw3)
    normal_valid_hw &= splat_valid_hw
    return TriageErrors(
        rgb_map_hw3=colorize_error(rgb_error_hw, splat_valid_hw, maximum=RGB_ERROR_MAXIMUM),
        depth_map_hw3=colorize_error(depth_error_m_hw, depth_valid_hw, maximum=DEPTH_ERROR_MAXIMUM_M),
        normal_map_hw3=colorize_error(normal_error_degrees_hw, normal_valid_hw, maximum=NORMAL_ERROR_MAXIMUM_DEGREES),
        rgb_mean=_masked_mean(rgb_error_hw, splat_valid_hw),
        depth_mean_m=_masked_mean(depth_error_m_hw, depth_valid_hw),
        normal_mean_degrees=_masked_mean(normal_error_degrees_hw, normal_valid_hw),
    )


def load_reference_blobs(segment_view: DatasetView) -> list[TriageReferenceBlobs]:
    """Load stored source images at every chosen ultrawide timestamp."""
    required_columns: tuple[str, ...] = (RGB_REFERENCE_COLUMN, DEPTH_REFERENCE_COLUMN, NORMAL_REFERENCE_COLUMN)
    available_columns: set[str] = set(segment_view.arrow_schema().names)
    missing_columns: list[str] = [name for name in required_columns if name not in available_columns]
    if missing_columns:
        raise ValueError(f"catalog segment has no required triage components: {missing_columns}")
    table: pa.Table = (
        segment_view.reader(index=TIMELINE, fill_latest_at=False)
        .filter(col(f'"{RGB_REFERENCE_COLUMN}"').is_not_null())
        .select(TIMELINE, RGB_REFERENCE_COLUMN, DEPTH_REFERENCE_COLUMN, NORMAL_REFERENCE_COLUMN)
        .sort(TIMELINE)
        .to_arrow_table()
    )
    timestamps_ns_n: ndarray = table_timestamps(table).astype(np.int64)
    rgb_values: list[Any] = table.column(RGB_REFERENCE_COLUMN).to_pylist()
    depth_values: list[Any] = table.column(DEPTH_REFERENCE_COLUMN).to_pylist()
    normal_values: list[Any] = table.column(NORMAL_REFERENCE_COLUMN).to_pylist()
    return [
        TriageReferenceBlobs(
            timestamp_ns=int(timestamp_ns),
            rgb_blob=blob_bytes(rgb_value, RGB_REFERENCE_COLUMN),
            depth_blob=blob_bytes(depth_value, DEPTH_REFERENCE_COLUMN),
            normal_blob=blob_bytes(normal_value, NORMAL_REFERENCE_COLUMN),
        )
        for timestamp_ns, rgb_value, depth_value, normal_value in zip(
            timestamps_ns_n, rgb_values, depth_values, normal_values, strict=True
        )
    ]


def summarize_metrics(values: list[float]) -> MetricSummary:
    """Summarize finite per-frame means."""
    finite_n: Float32[ndarray, "n"] = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float32)
    if finite_n.size == 0:
        return MetricSummary(float("nan"), float("nan"), float("nan"))
    return MetricSummary(float(np.mean(finite_n)), float(np.min(finite_n)), float(np.max(finite_n)))
