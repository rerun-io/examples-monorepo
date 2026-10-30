"""Read one bounded SelfCap window without catalog writes."""

from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
import pyarrow as pa
from datafusion import col, lit
from jaxtyping import Float64
from numpy import ndarray
from rerun.catalog import CatalogClient
from scipy.spatial.transform import Rotation, Slerp
from serde import serde
from serde.json import from_json

from selfcap_soma.observations import quest_sample_times

ExoSource: TypeAlias = Literal["rtmw", "sapiens2"]


def exo_columns(columns: list[str], source: ExoSource) -> tuple[str, str]:
    """Discover a unique triangulated source carrying per-joint confidences."""
    matches = []
    for column in columns:
        if not column.endswith("/coco133_xyz:Points3D:positions"):
            continue
        entity = column.split(":", 1)[0]
        if (source == "rtmw" and "/rtmw_triangulated/" not in entity) or (source == "sapiens2" and "/sapiens2/" not in entity):
            continue
        confidence = entity + ":simplecv.KeypointConfidence3D:confidences"
        if confidence in columns:
            matches.append((column, confidence))
    if len(matches) != 1:
        raise ValueError(f"expected a unique {source} triangulation with confidence, found {matches}")
    return matches[0]


@serde
@dataclass(frozen=True, slots=True)
class SyncRecord:
    """Only the source fields used here; the third-party record can grow."""

    offsets_ms: dict[str, float]
    """Offsets from capture time to video time."""


@dataclass(frozen=True, slots=True)
class Observations:
    times: Float64[ndarray, "n"]
    """Video seconds on the exo sample grid."""
    exo: Float64[ndarray, "n 133 3"]
    """World points; missing points are NaN."""
    confidence: Float64[ndarray, "n 133"]
    """Mean inlier detector confidence, not inverse variance."""
    hands: Float64[ndarray, "n 133 3"]
    """Latency-corrected, validity-gated Quest world points."""
    head_positions: Float64[ndarray, "n 3"]
    """Tracked Quest headset positions; not anatomical eye measurements."""
    head_rotations: Float64[ndarray, "n 3 3"]
    """World rotations of the tracked Quest headset frame; anatomical origin unresolved."""
    offset_ms: float
    """Fixed catalog Quest offset."""


def seconds(values: pa.ChunkedArray) -> Float64[ndarray, "n"]:
    """Convert Arrow duration/timestamp values to seconds."""
    return values.cast(pa.int64()).to_numpy().astype(np.float64) / 1e9


def interpolate_hands(
    times: Float64[ndarray, "q"], points: Float64[ndarray, "q 133 3"], sample: Float64[ndarray, "n"]
) -> Float64[ndarray, "n 133 3"]:
    """Interpolate valid bracketing samples; reject jumps, freezes and gaps."""
    result: Float64[ndarray, "n 133 3"] = np.full((len(sample), 133, 3), np.nan)
    hi = np.clip(np.searchsorted(times, sample), 1, len(times) - 1)
    lo = hi - 1
    dt = times[hi] - times[lo]
    alpha = (sample - times[lo]) / np.maximum(dt, 1e-9)
    for base in (91, 112):
        hand = points[:, base : base + 21]
        finite = np.isfinite(hand).all((1, 2)) & (np.abs(hand).sum((1, 2)) > 0.0)
        speed = np.linalg.norm(np.diff(hand[:, 0], axis=0), axis=1) / np.maximum(np.diff(times), 1e-9)
        jump = np.r_[False, speed > 4.0]
        frozen = np.r_[False, np.abs(np.diff(hand, axis=0)).sum((1, 2)) == 0.0]
        finite &= ~(jump | np.r_[jump[1:], False] | (frozen & np.r_[False, frozen[:-1]]))
        valid = finite[lo] & finite[hi] & (dt <= 0.04) & (alpha >= 0.0) & (alpha <= 1.0)
        result[valid, base : base + 21] = hand[lo[valid]] * (1.0 - alpha[valid, None, None]) + hand[hi[valid]] * alpha[valid, None, None]
    return result


def interpolate_head(
    times: Float64[ndarray, "p"],
    positions: Float64[ndarray, "p 3"],
    quaternions: Float64[ndarray, "p 4"],
    samples: Float64[ndarray, "n"],
) -> tuple[Float64[ndarray, "n 3"], Float64[ndarray, "n 3 3"]]:
    """Interpolate head poses; missing endpoints or gaps have no position target."""
    right = np.clip(np.searchsorted(times, samples), 1, len(times) - 1)
    valid = (samples >= times[0]) & (samples <= times[-1]) & ((times[right] - times[right - 1]) <= 0.06)
    result = np.column_stack([np.interp(samples, times, positions[:, axis]) for axis in range(3)])
    result[~valid] = np.nan
    rotations = Slerp(times, Rotation.from_quat(quaternions))(np.clip(samples, times[0], times[-1])).as_matrix()
    return result, rotations


def read_window(url: str, dataset: str, segment: str, start: float, end: float, stride: int, source: ExoSource = "rtmw") -> Observations:
    """Use one materialization per entity group, with window pushdown."""
    view = CatalogClient(url).get_dataset(dataset).filter_segments(segment)
    sync_table = view.filter_contents(["/__properties/time_sync"]).reader(index=None).to_arrow_table()
    sync_column = next(name for name in sync_table.column_names if name.endswith(":time_sync:record_json"))
    sync_text = sync_table[sync_column].drop_null()[0].as_py()[0]
    sync = from_json(SyncRecord, sync_text)
    offset = sync.offsets_ms["quest_pose"]

    position, confidence = exo_columns(view.schema().column_names(), source)
    exo_path = position.split(":", 1)[0]
    table = (
        view.filter_contents([exo_path])
        .reader(index="video_time")
        .filter(col("video_time").cast(pa.int64()) >= lit(int(start * 1e9)))
        .filter(col("video_time").cast(pa.int64()) <= lit(int(end * 1e9)))
        .select("video_time", position, confidence)
        .to_arrow_table()
    )
    table = table.filter(table[position].is_valid()).take(np.arange(0, len(table), stride))
    times = seconds(table["video_time"])
    if len(times) < 3:
        raise ValueError("window needs at least three triangulated frames")
    exo = np.asarray(table[position].to_pylist(), dtype=np.float64)
    scores = np.asarray(table[confidence].to_pylist(), dtype=np.float64).reshape(-1, 133)
    head_sample, hand_sample = quest_sample_times(times, offset)

    hand_path, head_path = "/world/gt/coco133_xyz", "/world/rig_05"
    hands_column = f"{hand_path}:Points3D:positions"
    translation = f"{head_path}:Transform3D:translation"
    quaternion = f"{head_path}:Transform3D:quaternion"
    raw = (
        view.filter_contents([hand_path, head_path])
        .reader(index="capture_time")
        .filter(col("capture_time").cast(pa.int64()) >= lit(int((head_sample[0] - 0.1) * 1e9)))
        .filter(col("capture_time").cast(pa.int64()) <= lit(int((hand_sample[-1] + 0.1) * 1e9)))
        .select("capture_time", hands_column, translation, quaternion)
        .to_arrow_table()
    )
    hand_table = raw.filter(raw[hands_column].is_valid())
    head_table = raw.filter(raw[quaternion].is_valid())
    if len(hand_table) < 2 or len(head_table) < 2:
        raise ValueError(f"{dataset}/{segment}: need at least two Quest hand and head samples")
    ht = seconds(head_table["capture_time"])
    hp = np.asarray(head_table[translation].to_pylist(), dtype=np.float64).reshape(-1, 3)
    hq = np.asarray(head_table[quaternion].to_pylist(), dtype=np.float64).reshape(-1, 4)
    keep = np.r_[True, np.diff(ht) > 0.0]
    ht, hp, hq = ht[keep], hp[keep], hq[keep]
    hands = interpolate_hands(seconds(hand_table["capture_time"]), np.asarray(hand_table[hands_column].to_pylist(), dtype=np.float64), hand_sample)
    positions, rotations = interpolate_head(ht, hp, hq, head_sample)
    return Observations(times, exo, scores, hands, positions, rotations, offset)
