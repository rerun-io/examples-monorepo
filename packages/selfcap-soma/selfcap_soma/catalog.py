"""Read one bounded SelfCap window without catalog writes."""

from dataclasses import dataclass, field, replace
from typing import Literal, TypeAlias

import numpy as np
import pyarrow as pa
from dataforge.interpolation import bracket, lerp, slerp
from dataforge.quest_joint_poses import QUEST_HAND_SIDES, QuestJointPoses, joint_poses_from_table, quest_hand_path
from dataforge.schema import coco133_xyz_path
from dataforge.selfcap_schema import HeadConvention, SyncRecord, rdf_head
from datafusion import col, lit
from jaxtyping import Bool, Float64, Int64
from numpy import ndarray
from rerun.catalog import DatasetEntry, DatasetSelection
from scipy.spatial.transform import Rotation
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


def read_rigs(view: DatasetSelection, columns: list[str]) -> RigPaths:
    """Read static rig-kind metadata using the segment schema already fetched."""
    columns = [name for name in columns if name.startswith("/world/rig_") and name.endswith(":kind") and name.count("/") == 2]
    table = view.filter_contents([name.split(":")[0] for name in columns]).reader(index=None).select(*columns).to_arrow_table()
    kinds = {name.split(":")[0]: str(table[name].drop_null()[0].as_py()[0]) for name in columns if len(table[name].drop_null())}
    return rig_paths_from_kinds(kinds)


@dataclass(frozen=True, slots=True)
class SegmentCatalog:
    view: DatasetSelection
    """One segment selection reused by every observation and overlay query."""
    columns: list[str]
    """Schema column names fetched once for this segment."""
    rigs: RigPaths
    """Physical roles from the segment's static rig metadata."""


def open_segment(dataset: DatasetEntry, segment: str) -> SegmentCatalog:
    """Open a segment and discover its schema and rig roles once."""
    view = dataset.filter_segments(segment)
    columns = view.schema().column_names()
    return SegmentCatalog(view, columns, read_rigs(view, columns))


def floats(column: pa.ChunkedArray, *shape: int) -> Float64[ndarray, "..."]:
    """Read nested numeric Arrow buffers, retaining slice offsets and the requested shape."""
    values = column.combine_chunks()
    while pa.types.is_list(values.type) or pa.types.is_fixed_size_list(values.type):
        values = values.flatten()
    return values.to_numpy(zero_copy_only=False).astype(np.float64).reshape(*shape)


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
    hand_poses: dict[str, QuestJointPoses] = field(default_factory=dict)
    """Raw joint poses from the same capture_time materialization as hands and head."""
    head_convention: HeadConvention = "RDF"
    """Orientation axes stored in the base, before read-side RDF normalization."""


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


def seconds(values: pa.ChunkedArray) -> Float64[ndarray, "n"]:
    """Convert Arrow duration/timestamp values to seconds."""
    return values.cast(pa.int64()).to_numpy().astype(np.float64) / 1e9


HAND_MAX_GAP_S: float = 0.04
"""Hand positions accept intervals up to and including 40 ms."""
HEAD_MAX_GAP_S: float = 0.06
"""Head positions accept intervals up to and including 60 ms."""


def interpolate_hands(
    times: Float64[ndarray, "q"], points: Float64[ndarray, "q 133 3"], sample: Float64[ndarray, "n"]
) -> Float64[ndarray, "n 133 3"]:
    """Interpolate valid bracketing samples; reject jumps, freezes and gaps."""
    result: Float64[ndarray, "n 133 3"] = np.full((len(sample), 133, 3), np.nan)
    samples = bracket(times, sample, HAND_MAX_GAP_S, denominator_floor=1e-9)
    lo, hi = samples.left, samples.right
    for base in (91, 112):
        hand = points[:, base : base + 21]
        finite = np.isfinite(hand).all((1, 2)) & (np.abs(hand).sum((1, 2)) > 0.0)
        speed = np.linalg.norm(np.diff(hand[:, 0], axis=0), axis=1) / np.maximum(np.diff(times), 1e-9)
        jump = np.r_[False, speed > 4.0]
        frozen = np.r_[False, np.abs(np.diff(hand, axis=0)).sum((1, 2)) == 0.0]
        finite &= ~(jump | np.r_[jump[1:], False] | (frozen & np.r_[False, frozen[:-1]]))
        valid = finite[lo] & finite[hi] & samples.valid
        result[valid, base : base + 21] = lerp(hand, samples)[valid]
    return result


def interpolate_head(
    times: Float64[ndarray, "p"],
    positions: Float64[ndarray, "p 3"],
    quaternions: Float64[ndarray, "p 4"],
    samples: Float64[ndarray, "n"],
) -> tuple[Float64[ndarray, "n 3"], Float64[ndarray, "n 3 3"]]:
    """Interpolate head poses; missing endpoints or gaps have no position target."""
    samples_bracket = bracket(times, samples, HEAD_MAX_GAP_S, denominator_floor=0.0)
    result = lerp(positions, samples_bracket)
    result[~samples_bracket.valid] = np.nan
    rotations = slerp(times, Rotation.from_quat(quaternions), samples)
    return result, rotations


def read_window(catalog: SegmentCatalog, start: float, end: float, stride: int, source: ExoSource = "rtmw") -> Observations:
    """Push down the triangulation window and read all Quest streams in one query."""
    view, rigs = catalog.view, catalog.rigs
    sync_table = view.filter_contents(["/__properties/time_sync"]).reader(index=None).to_arrow_table()
    sync_column = next(name for name in sync_table.column_names if name.endswith(":time_sync:record_json"))
    sync_text = sync_table[sync_column].drop_null()[0].as_py()[0]
    sync = from_json(SyncRecord, sync_text)
    offset = sync.require_offset("quest_pose")

    columns = catalog.columns
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
    exo = floats(table[position], -1, 133, 3)
    scores = floats(table[confidence], -1, 133)
    head_sample, hand_sample = quest_sample_times(times, offset)

    hand_path, head_path = coco133_xyz_path(), rigs.quest
    hands_column = f"{hand_path}:Points3D:positions"
    translation = f"{head_path}:Transform3D:translation"
    quaternion = f"{head_path}:Transform3D:quaternion"
    joint_entities = {side: quest_hand_path(side) for side in QUEST_HAND_SIDES}
    joint_columns = [name for name in columns if any(name.startswith(entity + ":InstancePoses3D:") for entity in joint_entities.values())]
    raw = (
        view.filter_contents([hand_path, head_path, *joint_entities.values()])
        .reader(index="capture_time")
        .select("capture_time", hands_column, translation, quaternion, *joint_columns)
        .to_arrow_table()
    )
    hand_poses = {
        side: joint_poses_from_table(raw, entity, "capture_time")
        for side, entity in joint_entities.items()
        if entity + ":InstancePoses3D:translations" in raw.column_names
    }
    head_rows = raw.filter(raw[quaternion].is_valid())
    axes = rdf_head(floats(head_rows[quaternion], -1, 4))
    head_stamps = head_rows["capture_time"].cast(pa.int64()).to_numpy()
    stamps = raw["capture_time"].cast(pa.int64()).to_numpy()
    raw = raw.take(np.flatnonzero((stamps >= int((head_sample[0] - 0.1) * 1e9)) & (stamps <= int((hand_sample[-1] + 0.1) * 1e9))))
    hand_table = raw.filter(raw[hands_column].is_valid())
    head_table = raw.filter(raw[quaternion].is_valid())
    if len(hand_table) < 2 or len(head_table) < 2:
        raise ValueError("need at least two Quest hand and head samples")
    ht = seconds(head_table["capture_time"])
    hp = floats(head_table[translation], -1, 3)
    hq = axes.quaternions[(head_stamps >= int((head_sample[0] - 0.1) * 1e9)) & (head_stamps <= int((hand_sample[-1] + 0.1) * 1e9))]
    keep = np.r_[True, np.diff(ht) > 0.0]
    ht, hp, hq = ht[keep], hp[keep], hq[keep]
    hands = interpolate_hands(seconds(hand_table["capture_time"]), floats(hand_table[hands_column], -1, 133, 3), hand_sample)
    positions, rotations = interpolate_head(ht, hp, hq, head_sample)
    support = floats(table[support_column], -1, 133).astype(np.int64) if support_column in table.column_names else None
    return Observations(times, exo, scores, hands, positions, rotations, offset, support, hand_poses, axes.convention)
