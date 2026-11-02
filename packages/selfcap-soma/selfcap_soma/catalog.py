"""Read one bounded SelfCap window without catalog writes."""

from dataclasses import dataclass, replace
from typing import Literal, TypeAlias

import numpy as np
import pyarrow as pa
from datafusion import col, lit
from jaxtyping import Bool, Float64, Int64
from numpy import ndarray
from rerun.catalog import CatalogClient, DatasetEntry
from scipy.spatial.transform import Rotation, Slerp
from serde import serde
from serde.json import from_json

from selfcap_soma.observations import quest_sample_times

ExoSource: TypeAlias = Literal["rtmw", "sapiens2"]


@dataclass(frozen=True, slots=True)
class RigPaths:
    exo: tuple[str, ...]
    """Calibrated exocentric rig paths."""
    oak: str
    """Egocentric OAK rig path."""
    quest: str
    """Quest tracked-head rig path."""


def rig_paths_from_kinds(kinds: dict[str, str]) -> RigPaths:
    """Resolve roles from converter metadata, never from a fixed phone count."""
    exo = tuple(sorted(path for path, kind in kinds.items() if kind == "exo"))
    oak = [path for path, kind in kinds.items() if kind == "ego"]
    quest = [path for path, kind in kinds.items() if kind == "quest"]
    if not exo or len(oak) != 1 or len(quest) != 1:
        raise ValueError(f"expected exo rigs and exactly one OAK/Quest rig, got {kinds}")
    return RigPaths(exo, oak[0], quest[0])


def read_rigs(dataset: DatasetEntry, segment: str) -> RigPaths:
    """Read static rig-kind metadata in one query."""
    view = dataset.filter_segments(segment)
    columns = [name for name in view.schema().column_names() if name.startswith("/world/rig_") and name.endswith(":kind") and name.count("/") == 2]
    table = view.filter_contents([name.split(":")[0] for name in columns]).reader(index=None).select(*columns).to_arrow_table()
    kinds = {name.split(":")[0]: str(table[name].drop_null()[0].as_py()[0]) for name in columns if len(table[name].drop_null())}
    return rig_paths_from_kinds(kinds)


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


def exo_view_track(columns: list[str], camera: str, source: ExoSource) -> str | None:
    """One camera's 2D detections for a source; None when it has none.

    Align writes Sapiens2 detections to ``pinhole/pred/sapiens2`` and the keypoints layer writes the ones it
    triangulated to ``pinhole/sapiens2``. The keypoints layer's track wins when a dataset holds both.
    """
    candidates: list[str] = [
        name.split(":", 1)[0]
        for name in columns
        if name.startswith(camera + "/pinhole/")
        and name.endswith("/coco133_uv:Points2D:positions")
        and f"/{source}/" in name
        and name.split(":", 1)[0] + ":simplecv.KeypointConfidence2D:confidences" in columns
    ]
    tracks: list[str] = [name for name in candidates if "/pred/" not in name] or candidates
    if len(tracks) > 1:
        raise ValueError(f"ambiguous {source} 2D tracks for {camera}: {tracks}")
    return tracks[0] if tracks else None


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
    """The source layer's per-joint confidence, not inverse variance: Kineo pairwise reprojection score in current keypoints layers, mean inlier detector score in older ones."""
    hands: Float64[ndarray, "n 133 3"]
    """Latency-corrected, validity-gated Quest world points."""
    head_positions: Float64[ndarray, "n 3"]
    """Tracked Quest headset positions; not anatomical eye measurements."""
    head_rotations: Float64[ndarray, "n 3 3"]
    """World rotations of the tracked Quest headset frame; anatomical origin unresolved."""
    offset_ms: float
    """Fixed catalog Quest offset."""
    support: Int64[ndarray, "n 133"] | None = None
    """Geometric inlier-view counts, when supplied by the triangulation layer."""


def subset(data: Observations, mask: Bool[ndarray, "n"]) -> Observations:
    """Keep the masked frames of every per-frame observation stream."""
    return replace(
        data,
        times=data.times[mask],
        exo=data.exo[mask],
        confidence=data.confidence[mask],
        hands=data.hands[mask],
        head_positions=data.head_positions[mask],
        head_rotations=data.head_rotations[mask],
        support=data.support[mask] if data.support is not None else None,
    )


TORSO_JOINTS: list[int] = [5, 6, 11, 12]
"""COCO shoulders and hips."""
SHOULDER_JOINTS: list[int] = [5, 6]
MIN_SHOULDER_VIEWS: int = 3
"""Fewest supporting views on each shoulder of a well-observed frame. The hips are only required to be triangulated:
they reach three views on 3-6.5 % of frames (easy26, d31), so a three-view hip rule selected 0-1 frames per segment."""


def well_observed_torso(data: Observations) -> Bool[ndarray, "n"]:
    """Frames whose four torso joints are all finite and whose shoulders each have at least MIN_SHOULDER_VIEWS views."""
    if data.support is None:
        raise ValueError("the torso gate needs support_view_count, which this triangulation layer does not hold")
    return np.isfinite(data.exo[:, TORSO_JOINTS]).all((1, 2)) & (data.support[:, SHOULDER_JOINTS].min(1) >= MIN_SHOULDER_VIEWS)


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


def rdf_head(quaternions: Float64[ndarray, "p 4"]) -> Float64[ndarray, "p 4"]:
    """Head orientations (xyzw) in RDF. Some captures (2025-12-11/12) store them with y up and z back; a worn headset is
    never upside down, so a camera "down" axis pointing up the Y-up world on median marks them. Turning by 180 degrees
    about x is q * (1, 0, 0, 0) = (w, z, -y, -x)."""
    down_y = Rotation.from_quat(quaternions).as_matrix()[:, 1, 1]
    if np.median(down_y) <= 0.0:
        return quaternions
    x, y, z, w = quaternions.T
    return np.stack([w, z, -y, -x], axis=1)


def read_window(url: str, dataset: str, segment: str, start: float, end: float, stride: int, source: ExoSource = "rtmw") -> Observations:
    """Use one materialization per entity group, with window pushdown."""
    entry = CatalogClient(url).get_dataset(dataset)
    view = entry.filter_segments(segment)
    rigs = read_rigs(entry, segment)
    sync_table = view.filter_contents(["/__properties/time_sync"]).reader(index=None).to_arrow_table()
    sync_column = next(name for name in sync_table.column_names if name.endswith(":time_sync:record_json"))
    sync_text = sync_table[sync_column].drop_null()[0].as_py()[0]
    sync = from_json(SyncRecord, sync_text)
    offset = sync.offsets_ms["quest_pose"]

    columns = view.schema().column_names()
    position, confidence = exo_columns(columns, source)
    exo_path = position.split(":", 1)[0]
    support_column = exo_path + ":support_view_count"
    selected = ["video_time", position, confidence]
    if support_column in columns:
        selected.append(support_column)
    table = (
        view.filter_contents([exo_path])
        .reader(index="video_time")
        .filter(col("video_time").cast(pa.int64()) >= lit(int(start * 1e9)))
        .filter(col("video_time").cast(pa.int64()) <= lit(int(end * 1e9)))
        .select(*selected)
        .to_arrow_table()
    )
    table = table.filter(table[position].is_valid()).take(np.arange(0, len(table), stride))
    times = seconds(table["video_time"])
    if len(times) < 3:
        raise ValueError("window needs at least three triangulated frames")
    exo = np.asarray(table[position].to_pylist(), dtype=np.float64)
    scores = np.asarray(table[confidence].to_pylist(), dtype=np.float64).reshape(-1, 133)
    head_sample, hand_sample = quest_sample_times(times, offset)

    hand_path, head_path = "/world/gt/coco133_xyz", rigs.quest
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
    ht, hp, hq = ht[keep], hp[keep], rdf_head(hq[keep])
    hands = interpolate_hands(seconds(hand_table["capture_time"]), np.asarray(hand_table[hands_column].to_pylist(), dtype=np.float64), hand_sample)
    positions, rotations = interpolate_head(ht, hp, hq, head_sample)
    support = np.asarray(table[support_column].to_pylist(), dtype=np.int64) if support_column in table.column_names else None
    return Observations(times, exo, scores, hands, positions, rotations, offset, support)
