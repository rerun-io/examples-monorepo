"""Read-only frozen-v2 adapters used only by reference tests."""

import json
import os
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pytest
from rerun.chunk import RrdReader
from rrd_fixtures import private_server
from serde import SerdeError, from_dict, serde
from serde.json import from_json

from selfcap_curate.catalog_io import DATASET_NAME
from selfcap_curate.patch_base import patch
from selfcap_curate.records import (
    CatalogHands,
    Lag,
    Links,
    RepeatPattern,
    SelfcapLayout,
    SessionEpisode,
    TimeSync,
    finite,
    gyro_head_offsets,
    oak_quest_offset_ms,
    read_sync,
)
from selfcap_curate.session_fit import EpisodeLinks


@dataclass(frozen=True, slots=True)
class FrozenEpisodeLinks(EpisodeLinks):
    duration_s: float = 0.0
    """Frozen duration, used only to compare RGB at the midpoint."""
    search_half_width_s: dict[str, float] = field(default_factory=dict)
    """Legacy search widths, not production search configuration."""


@serde
@dataclass(frozen=True, slots=True)
class FrozenLinkFile:
    """Partial adapter for third-party v2 link JSON, including failed episodes."""

    key: str
    """Episode identity."""
    duration_s: float = 0.0
    """Source video duration."""
    window_s: tuple[float, float] = (0.0, 0.0)
    """Frozen motion window."""
    t_start_session_s: float | None = None
    """Source session start."""
    lag_left_to_pose: Lag | None = None
    """Coarse left-to-pose link."""
    lag_rgb_to_left: Lag | None = None
    """Coarse RGB-to-left seed."""
    lag_quest_video_to_pose: Lag | None = None
    """Quest video-to-pose link."""
    lag_left_to_gyro: Lag | None = None
    """Left-to-gyro link."""
    lag_gyro_to_pose: Lag | None = None
    """Gyro-to-pose link."""
    lag_left_to_quest_video: Lag | None = None
    """Optional direct left-to-Quest video link."""
    oak_left_pattern: RepeatPattern | None = None
    """Left resampling pattern."""
    oak_rgb_pattern: RepeatPattern | None = None
    """RGB resampling pattern."""
    search_half_width_s: dict[str, float] | None = None
    """Older files omit this field."""
    failed: bool = False
    """Failed link run."""


@serde
@dataclass(frozen=True, slots=True)
class FrozenFineLinks:
    """Partial adapter for third-party v2 links_rgb JSON."""

    left_to_gyro_s: float | None = None
    """Fine left lag."""
    rgb_to_gyro_s: float | None = None
    """Fine RGB lag."""
    left_peak: float | None = None
    """Fine left correlation peak."""
    rgb_peak: float | None = None
    """Fine RGB correlation peak."""
    edge: bool = False
    """Refinement hit an edge."""


def clean_lag(lag: Lag | None) -> Lag:
    """Normalize legacy NaN cells at the boundary, preserving clip evidence."""
    return (
        Lag(None, None, None, 0.0)
        if lag is None
        else replace(lag, lag_s=finite(lag.lag_s), peak=finite(lag.peak), second_peak=finite(lag.second_peak))
    )


def load_frozen_links(path: Path, fine_path: Path | None = None) -> FrozenEpisodeLinks:
    """Adapt v2 records once. stdlib JSON accepts legacy NaN; typed objects do not keep it."""
    try:
        source: FrozenLinkFile = from_dict(FrozenLinkFile, json.loads(path.read_text()))
        fine: FrozenFineLinks = (
            from_dict(FrozenFineLinks, json.loads(fine_path.read_text())) if fine_path is not None and fine_path.exists() else FrozenFineLinks()
        )
    except (SerdeError, json.JSONDecodeError) as error:
        raise ValueError(f"{path}: invalid frozen link record: {error}") from error
    links: Links = Links(
        clean_lag(source.lag_left_to_pose),
        clean_lag(source.lag_rgb_to_left),
        clean_lag(source.lag_quest_video_to_pose),
        clean_lag(source.lag_left_to_gyro),
        clean_lag(source.lag_gyro_to_pose),
        source.oak_left_pattern,
        source.oak_rgb_pattern,
        None if source.lag_left_to_quest_video is None else clean_lag(source.lag_left_to_quest_video),
        None if finite(fine.left_to_gyro_s) is None else Lag(fine.left_to_gyro_s, finite(fine.left_peak), None, 0.0, fine.edge),
        None if finite(fine.rgb_to_gyro_s) is None else Lag(fine.rgb_to_gyro_s, finite(fine.rgb_peak), None, 0.0, fine.edge),
    )
    widths: dict[str, float] = source.search_half_width_s or {
        "left_to_pose": 3.0,
        "rgb_to_left": 0.4,
        "quest_video_to_pose": 1.0,
        "left_to_quest_video": 3.0,
        "left_to_gyro": 4.0,
        "gyro_to_pose": 4.0,
    }
    return clean_observation(
        FrozenEpisodeLinks(source.key, links, source.window_s, finite(source.t_start_session_s), source.failed, source.duration_s, widths)
    )


def clean_observation(observation: FrozenEpisodeLinks) -> FrozenEpisodeLinks:
    """links_clipped_at_search_edge: exclude a peak within 5 ms of its search edge.

    A clipped peak can have high correlation while the true lag lies outside
    the search. Keeping its score would incorrectly outrank an interior link.
    """
    changes: dict[str, Lag] = {}
    for name, width in observation.search_half_width_s.items():
        lag: Lag | None = getattr(observation.links, f"lag_{name}")
        if lag is not None and lag.lag_s is not None and abs(lag.lag_s) >= width - 0.005:
            changes[f"lag_{name}"] = replace(lag, lag_s=None, peak=None, clipped=True, clipped_at_s=lag.lag_s)
    if not changes:
        return observation
    links = observation.links
    return replace(
        observation,
        links=replace(
            links,
            lag_left_to_pose=changes.get("lag_left_to_pose", links.lag_left_to_pose),
            lag_rgb_to_left=changes.get("lag_rgb_to_left", links.lag_rgb_to_left),
            lag_quest_video_to_pose=changes.get("lag_quest_video_to_pose", links.lag_quest_video_to_pose),
            lag_left_to_gyro=changes.get("lag_left_to_gyro", links.lag_left_to_gyro),
            lag_gyro_to_pose=changes.get("lag_gyro_to_pose", links.lag_gyro_to_pose),
            lag_left_to_quest_video=changes.get("lag_left_to_quest_video", links.lag_left_to_quest_video),
        ),
    )


ROOT = Path("/home/pablo/exoego-data/dataforge-selfcap-aligned")
GOLDEN_REFS = ROOT / "golden-refs"
SERVED = Path("/mnt/nas/datasets/dataforge-selfcap-rrd")
RAW = Path("/mnt/nas/datasets/exoego-self-collected/main/cut")


def fixture_sync(**changes) -> TimeSync:
    """Pilot typed record with explicit synthetic overrides."""
    return replace(read_sync(Path(__file__).parent / "fixtures/2025-11-24__027d8215__ep002.sync.json"), **changes)


@contextmanager
def private_catalog(syncs, directory: Path, *, layers: bool = True):
    """Patch explicit read-only episodes into a private temporary catalog."""
    bases, calibrations, detections = [], [], []
    for sync in syncs:
        raw = RAW / sync.date / sync.session / "episodes" / f"episode-{sync.episode[2:]}"
        base = SERVED / "base" / f"selfcap__{sync.key}.rrd"
        layer = SERVED / "calib-v2" / base.name
        for asset in (
            SERVED / "sapiens2" / sync.episode_key.rrd_name,
            base,
            raw / "ego/left.mp4",
            raw / "quest/left_hand_poses.csv",
            *((layer,) if layers else ()),
        ):
            if not os.access(asset, os.R_OK):
                pytest.skip(f"golden asset absent or unreadable: {asset}")
        bases.append(patch(base, raw, sync, output_path=directory / base.name, mode="migrate-keep-clock"))
        calibrations.append(layer)
        detections.append(SERVED / "sapiens2" / sync.episode_key.rrd_name)
    with private_server({DATASET_NAME: bases}) as server:
        assert ":51235" not in server.url()
        if layers:
            server.client().get_dataset(DATASET_NAME).register([p.resolve().as_uri() for p in calibrations], layer_name="calib-v2").wait()
        server.client().get_dataset(DATASET_NAME).register([p.resolve().as_uri() for p in detections], layer_name="sapiens2").wait()
        yield server




@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class ClockExclusionReference:
    session: str
    """The only affected session in the selected golden set."""
    source_comment: str
    """Independent pre-fix solver and trusted-subset provenance."""
    excluded_flags: dict[str, list[str]]
    """Exactly the episodes and unsynced flags allowed to differ from served."""
    oak_median_px: float
    """Trusted-only pre-fix OAK fit median, pixels."""


def oak_median_reference(syncs, served):
    """Keep served parity except for the explicitly recorded unmeasured-clock episode."""
    affected = {
        s.key: [f for f in s.flags if f in ("quest_unsynced_kept_as_cut", "quest_pose_unsynced", "oak_imu_unsynced")]
        for s in syncs
        if oak_quest_offset_ms(s) is None or gyro_head_offsets(s) is None
    }
    if not affected:
        return served
    reference = from_json(ClockExclusionReference, (Path(__file__).parent / "fixtures/measured-clock-exclusions.json").read_text())
    assert {s.key.rsplit("__", 1)[0] for s in syncs} == {reference.session}
    assert affected == reference.excluded_flags
    assert "pre-fix" in reference.source_comment
    return reference.oak_median_px








ENTITY = "/world/pred/rtmw_triangulated/coco133_xyz"


def dense(path: Path, entity: str, dimension: int):
    rows = {}
    for chunk in RrdReader(path).stream().filter(content=entity):
        batch = chunk.to_record_batch()
        if chunk.is_static or "video_time" not in batch.schema.names:
            continue
        kind = "Points3D" if dimension == 3 else "Points2D"
        for stamp, positions, ids in zip(
            batch.column("video_time").cast("int64").to_pylist(),
            batch.column(f"{kind}:positions").to_pylist(),
            batch.column(f"{kind}:keypoint_ids").to_pylist(),
            strict=True,
        ):
            array = np.full((133, dimension), np.nan)
            if ids:
                array[np.asarray(ids).reshape(-1)] = np.asarray(positions).reshape(-1, dimension)
            rows[stamp] = array
    return np.asarray(sorted(rows), dtype=np.int64), np.stack([rows[t] for t in sorted(rows)])


def chunk_bytes(path):
    return {c.to_record_batch().schema.metadata[b"rerun:id"]: c.to_record_batch().serialize().to_pybytes() for c in RrdReader(path).stream()}




def session_episode(sync, hands=None, gyro_times=None, gyro=None, *, detections=None, quest=None, layout=None, oak=None):
    """Build synthetic calibration inputs without requiring unrelated streams."""

    empty = np.empty(0)
    hands = hands if hands is not None else CatalogHands(empty, np.empty((0, 133, 3)), empty, np.empty((0, 3)), np.empty((0, 4)))
    kwargs = {} if oak is None else {"oak": oak}
    return SessionEpisode(
        sync,
        hands,
        empty if gyro_times is None else gyro_times,
        np.empty((0, 3)) if gyro is None else gyro,
        {} if quest is None else quest,
        layout or SelfcapLayout(len(sync.exo_fit), len(sync.exo_fit) + 1, tuple(sorted(sync.exo_fit))),
        {} if detections is None else detections,
        **kwargs,
    )




