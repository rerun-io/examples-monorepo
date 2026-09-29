"""Typed reference metrics and validation-only circle calibration records."""
import hashlib
from dataclasses import dataclass, field, fields
from io import BytesIO
from pathlib import Path
from typing import Literal, TypeAlias, cast
from zipfile import BadZipFile

import numpy as np
from jaxtyping import Bool, Float64, Int64
from numpy import ndarray
from serde import serde

from handtrack.eval.segment import PositionScore
from handtrack.train.checkpoint import atomic_write

Mode: TypeAlias = Literal["gt_pose", "gt_circle", "detnet", "track"]


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class Calibration:
    segments: list[str]
    """Validation-only source segments, sorted."""
    source_sha256: str
    """Digest of the upstream code used for calibration."""
    samples: int
    """Valid hand-camera observations."""
    median: float
    """Frozen circle_scale: median upstream focal / unit-circle focal."""
    q25: float
    """First quartile of focal ratios."""
    q75: float
    """Third quartile of focal ratios."""
    axis_median_deg: float
    """Median angle between crop optical axes."""
    axis_q25_deg: float
    """First quartile of optical-axis angles."""
    axis_q75_deg: float
    """Third quartile of optical-axis angles."""
    stride: int
    """Validation timeline rows between samples."""

    def __post_init__(self) -> None:
        from handtrack.data.catalog import UMETRACK_VALIDATION_USERS
        if not self.segments or any("__training__" not in segment or segment.split("__")[-2] not in UMETRACK_VALIDATION_USERS for segment in self.segments):
            raise ValueError("Circle calibration must use only held-out validation users from training")
        values: list[float] = [self.median, self.q25, self.q75, self.axis_median_deg, self.axis_q25_deg, self.axis_q75_deg]
        if self.samples < 1 or self.stride < 1 or not np.isfinite(values).all() or not 0 < self.q25 <= self.median <= self.q75:
            raise ValueError("Invalid circle calibration statistics")


def calibration_summary(segments: list[str], source: str, ratios: list[float], angles: list[float], stride: int) -> Calibration:
    if not ratios or len(ratios) != len(angles):
        raise ValueError("No paired calibration observations")
    ratio: Float64[ndarray, "3"] = np.quantile(ratios, [.25, .5, .75])
    angle: Float64[ndarray, "3"] = np.quantile(angles, [.25, .5, .75])
    return Calibration(segments, source, len(ratios), float(ratio[1]), float(ratio[0]), float(ratio[2]),
                       float(angle[1]), float(angle[0]), float(angle[2]), stride)


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class PositionStatistics:
    median_mm: float | None
    """Median finite per-hand error, mean over 21 landmarks."""
    p90_mm: float | None
    """90th percentile, NumPy linear interpolation."""
    below_20: float | None
    """GT-present hand-frames posed with error strictly below 20 mm / GT-present."""
    below_50: float | None
    """GT-present hand-frames posed with error strictly below 50 mm / GT-present."""
    wild: float | None
    """Scored errors strictly above 200 mm / all posed hand-frames, including false poses."""
    scored: int
    """Finite errors contributing to quantiles (posed and GT-valid)."""
    posed_denominator: int
    """All posed hand-frames; false poses cannot enter the wild numerator without GT."""


def position_statistics(errors: Float64[ndarray, "f 2"], posed: Bool[ndarray, "f 2"], valid: Bool[ndarray, "f 2"]) -> PositionStatistics:
    """Summarize concatenated frame samples; misses remain in success denominators."""
    values: Float64[ndarray, "n"] = errors[valid & posed & np.isfinite(errors)]
    present: int = int(valid.sum())
    count: int = int(posed.sum())
    return PositionStatistics(float(np.median(values)) if values.size else None,
        float(np.quantile(values, .9)) if values.size else None,
        int((values < 20).sum()) / present if present else None,
        int((values < 50).sum()) / present if present else None,
        int((values > 200).sum()) / count if count else None, int(values.size), count)


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class CameraStatistics:
    camera: int
    """Zero-based camera index, in the input rig order."""
    visible_pairs: int
    """Hand-camera frames with at least 19 GT landmarks in front and inside."""
    error_pairs: int
    """Visible pairs with finite predicted and GT circles; independent of presence."""
    centre_mean_px: float | None
    """Mean centre distance, native pixels."""
    centre_median_px: float | None
    """Median centre distance, native pixels."""
    centre_p90_px: float | None
    """90th percentile centre distance, native pixels; linear interpolation."""
    radius_mean_px: float | None
    """Mean absolute radius error, native pixels."""
    radius_median_px: float | None
    """Median absolute radius error, native pixels."""
    detections: int
    """Visible pairs with DetNet presence strictly above 0.5."""
    detection_rate: float | None
    """detections / visible_pairs."""
    empty_pairs: int
    """Pairs with exactly zero GT visible landmarks; excludes unavailable GT (-1)."""
    false_detections: int
    """Empty pairs with presence strictly above 0.5."""
    false_detection_rate: float | None
    """false_detections / empty_pairs."""
    selected: int
    """L2 selected hand-camera pairs before pose fitting; zero for other modes."""
    selected_visible: int
    """L2 selections with at least 19 visible GT landmarks."""
    wrong_view: int
    """L2 selections without a visible hand (selected - selected_visible)."""


def circle_statistics(circles: Float64[ndarray, "f 2 4 3"], probability: Float64[ndarray, "f 2 4"],
                      target: Float64[ndarray, "f 2 4 3"], visible: Int64[ndarray, "f 2 4"],
                      selected: Bool[ndarray, "f 2 4"]) -> list[CameraStatistics]:
    """Restrict circle errors to upstream-visible pairs, without thresholding DetNet."""
    result: list[CameraStatistics] = []
    for camera in range(4):
        relevant: Bool[ndarray, "f 2"] = visible[:, :, camera] >= 19
        empty: Bool[ndarray, "f 2"] = visible[:, :, camera] == 0
        paired: Bool[ndarray, "f 2"] = relevant & np.isfinite(circles[:, :, camera]).all(axis=-1) & np.isfinite(target[:, :, camera]).all(axis=-1)
        delta: Float64[ndarray, "n 3"] = circles[:, :, camera][paired] - target[:, :, camera][paired]
        centre: Float64[ndarray, "n"] = np.linalg.norm(delta[:, :2], axis=-1)
        radius: Float64[ndarray, "n"] = np.abs(delta[:, 2])
        count: int = int(relevant.sum())
        absent: int = int(empty.sum())
        detected: Bool[ndarray, "f 2"] = probability[:, :, camera] > .5
        hits: int = int((detected & relevant).sum())
        false: int = int((detected & empty).sum())
        chosen: int = int(selected[:, :, camera].sum())
        good: int = int((selected[:, :, camera] & relevant).sum())
        result.append(CameraStatistics(camera, count, int(paired.sum()),
            float(np.mean(centre)) if centre.size else None, float(np.median(centre)) if centre.size else None,
            float(np.quantile(centre, .9)) if centre.size else None,
            float(np.mean(radius)) if radius.size else None, float(np.median(radius)) if radius.size else None,
            hits, hits / count if count else None, absent, false, false / absent if absent else None, chosen, good, chosen - good))
    return result


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class ReferenceMetrics:
    segment: str
    """Catalog segment id; aggregate records use group names."""
    mode: Mode
    """Crop source."""
    track_end: str
    """geometry, detnet, or none for nontracking modes."""
    identity: str
    """Run identity sha256."""
    frames: int
    """Timeline rows processed."""
    position: PositionScore
    """MKPE and MKA, including their pooling denominators."""
    gt_present: int
    """GT-valid hand-frames."""
    posed: int
    """GT-valid hand-frames receiving a pose."""
    false_poses: int
    """Poses on GT-absent hand-frames."""
    coverage: float | None
    """posed / gt_present; null if no GT hands."""
    circle_samples: int
    """Finite DetNet / GT circle pairs with >=19 visible GT landmarks, irrespective of DetNet presence."""
    centre_error_px: float | None
    """Mean Euclidean circle centre error in native pixels."""
    radius_error_px: float | None
    """Mean absolute circle radius error in native pixels."""

    robust: PositionStatistics | None = None
    """Per-frame robust position statistics."""
    cameras: list[CameraStatistics] = field(default_factory=list)
    """DetNet diagnostics by camera, empty when DetNet was not run."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class Summary:
    identity: str
    """Run identity sha256."""
    selected: int
    """Selected segments."""
    metrics: list[ReferenceMetrics]
    """Pooled by mode, end policy, domain and interaction."""


@dataclass(frozen=True, slots=True)
class ReferenceFrames:
    """Frame streams; mode axis follows the segment JSON metrics order, hands left/right.

    Camera axis follows the input rig order. Circle data is shared by every mode;
    unavailable detector output is NaN, unavailable GT visibility is -1.
    """
    video_time_ns: Int64[ndarray, "f"]
    """Source timeline timestamps."""
    error_mm: Float64[ndarray, "f m 2"]
    """Mean 21-landmark position error, NaN without both a pose and valid GT."""
    posed: Bool[ndarray, "f m 2"]
    """Finite pose output, including GT-absent false poses."""
    gt_valid: Bool[ndarray, "f 2"]
    """GT-present hands, the position success-rate denominator."""
    detnet_circle: Float64[ndarray, "f 2 4 3"]
    """Native pixel (centre x, centre y, radius), regardless of presence."""
    detnet_presence: Float64[ndarray, "f 2 4"]
    """DetNet presence probability; NaN when not run."""
    gt_circle: Float64[ndarray, "f 2 4 3"]
    """GT 21-landmark enclosing circle in native pixels."""
    visible_landmarks: Int64[ndarray, "f 2 4"]
    """Count in front and inside upstream image bounds; -1 if GT unavailable."""
    selected: Bool[ndarray, "f m 2 4"]
    """Actual crop camera selections before pose fitting."""

    def __post_init__(self) -> None:
        count: int = len(self.video_time_ns)
        for item in fields(self):
            value = getattr(self, item.name)
            if not isinstance(value, cast(type, item.type)) or value.shape[0] != count:
                raise ValueError(f"Invalid {item.name}: expected {item.type} with {count} frames")
        if self.posed.shape != self.error_mm.shape or self.selected.shape[:3] != self.error_mm.shape:
            raise ValueError("Invalid mode axis: errors, posed and selected must agree")
        if np.any((self.visible_landmarks < -1) | (self.visible_landmarks > 21)):
            raise ValueError("Invalid visible landmark count")
        scored: Bool[ndarray, "f m 2"] = self.posed & self.gt_valid[:, None, :]
        if not np.array_equal(np.isfinite(self.error_mm), scored) or np.any(self.error_mm[scored] < 0) or not np.isnan(self.error_mm[~scored]).all():
            raise ValueError("Invalid errors: finite nonnegative values required exactly on posed GT-valid hands")


def save_frames(frames: ReferenceFrames, path: Path) -> str:
    """Atomically save fixed-key arrays; return the payload SHA256 for the JSON record."""
    buffer: BytesIO = BytesIO()
    np.savez(buffer, **{item.name: getattr(frames, item.name) for item in fields(frames)})
    payload: bytes = buffer.getvalue()
    atomic_write(path, payload)
    return hashlib.sha256(payload).hexdigest()


def load_frames(path: Path, sha256: str | None = None) -> ReferenceFrames:
    """Load streams without pickle or dtype coercion, validating shape and optional hash."""
    try:
        payload: bytes = path.read_bytes()
        if sha256 is not None and hashlib.sha256(payload).hexdigest() != sha256:
            raise ValueError("Frame stream digest mismatch")
        with np.load(BytesIO(payload), allow_pickle=False) as arrays:
            names: set[str] = {item.name for item in fields(ReferenceFrames)}
            if set(arrays.files) != names:
                raise ValueError("Invalid frame stream keys")
            # Check before construction so malformed input raises a source-naming ValueError,
            # even under the dev environment's dataclass type instrumentation.
            for item in fields(ReferenceFrames):
                if not isinstance(arrays[item.name], cast(type, item.type)):
                    raise ValueError(f"Invalid {item.name}: expected {item.type}")
            return ReferenceFrames(**{name: arrays[name] for name in names})
    except (ValueError, KeyError, OSError, BadZipFile, EOFError) as error:
        raise ValueError(f"Invalid reference frames {path}: {error}") from error
