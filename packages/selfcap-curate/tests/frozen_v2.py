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
from rrd_references import KeypointReference, OakReference, reference_tables
from scipy.spatial.transform import Rotation
from serde import SerdeError, from_dict, serde
from serde.json import from_json

from selfcap_curate.calib_layer import CalibSession, write_calib_layer
from selfcap_curate.catalog_io import DATASET_NAME, open_segment
from selfcap_curate.exo_calibration import Camera, EpisodeCalibration, PhoneCalibration, SessionCalibration
from selfcap_curate.factory import cy_shift_px, intrinsic_matrix, read_quest_block
from selfcap_curate.keypoints import Keypoints2D, ViewDetections
from selfcap_curate.patch_base import patch
from selfcap_curate.quest_fit import read_session_episode
from selfcap_curate.records import (
    CatalogHands,
    Lag,
    Links,
    OakDetections,
    OakFit,
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


def assert_quest_block(block, expected) -> None:
    """Compare the served Quest capture crop and factory matrices without rounding."""
    assert bool(block.pinholes) == expected["ok"]
    assert block.mode == expected["mode"]
    assert set(block.pinholes) == set(expected["pinholes"])
    for slot, pinhole in block.pinholes.items():
        old = expected["pinholes"][slot]
        assert pinhole.camera_id == old["camera_id"]
        assert (pinhole.capture_resolution.width, pinhole.capture_resolution.height) == (old["width"], old["height"])
        assert cy_shift_px(pinhole) == old["cy_shift_px"]
        np.testing.assert_allclose(
            intrinsic_matrix(pinhole), [[old["fx"], 0.0, old["cx"]], [0.0, old["fy"], old["cy"]], [0.0, 0.0, 1.0]], rtol=0.0, atol=0.0
        )
    for slot, transform in block.rig_T_cam.items():
        np.testing.assert_allclose(transform, expected["rig_T_cam"][slot], rtol=0.0, atol=1e-14)


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


def load_keypoint_reference(path: Path) -> KeypointReference:
    """Read exact detector streams; rounded timelines are for display only."""
    tables = reference_tables(path)
    metadata = tables["/reference"]
    phones = metadata["phones"][0].as_py()
    cameras = metadata["cameras"][0].as_py()
    times = np.asarray(tables["/reference/grid"]["time_s"].to_pylist(), dtype=np.float64).reshape(-1)
    views, pts, video_times = {}, {}, {}
    for phone, camera in zip(phones, cameras, strict=True):
        parent = f"{camera}/pinhole/pred/rtmw_persons"
        points, boxes = tables[f"{parent}/coco133_uv"], tables[f"{parent}/boxes"]
        corners = np.asarray(boxes["xyxy"].to_pylist(), dtype=np.float64).reshape(len(times), -1, 4)
        persons = corners.shape[1]
        views[phone] = ViewDetections(
            corners,
            np.asarray(boxes["confidence"].to_pylist(), dtype=np.float64),
            np.asarray(points["Points2D:positions"].to_pylist(), dtype=np.float64).reshape(len(times), persons, 133, 2),
            np.asarray(points["confidence"].to_pylist(), dtype=np.float64).reshape(len(times), persons, 133),
        )
        pts[phone] = np.asarray(points["pts_s"].to_pylist(), dtype=np.float64).reshape(-1)
        video_times[phone] = np.asarray(points["time_s"].to_pylist(), dtype=np.float64).reshape(-1)
    return KeypointReference(
        Keypoints2D(times, views, float(metadata["rate"][0].as_py()[0])), pts, video_times, int(metadata["schema_version"][0].as_py()[0])
    )


def load_keypoints2d(path: Path) -> Keypoints2D:
    """Load the frozen multi-person detector fixture used by triangulation."""
    return load_keypoint_reference(path).detections


def frozen_state(record: dict, quest) -> CalibSession:
    phones = {}
    for phone, source in record["exo"]["phones"].items():
        matrix = np.asarray(source["world_T_cam"])
        rotation = Rotation.from_matrix(matrix[:3, :3].T).as_rotvec()
        camera = Camera(np.r_[rotation, -matrix[:3, :3].T @ matrix[:3, 3], source["f"], source["cx"], source["cy"], 0.0, 0.0])
        episodes = {
            key: EpisodeCalibration(
                ep["n"], ep["median_px"], ep["ok"], ep["scope"], None if ep.get("world_T_cam") is None else np.asarray(ep["world_T_cam"])
            )
            for key, ep in source["episodes"].items()
        }
        phones[phone] = PhoneCalibration(True, camera, source["median_px"], source["p90_px"], source["n"], episodes)
    exo = SessionCalibration(record["session"], phones, record["exo"].get("bundle", {}).get("body_median_px"))
    return CalibSession(
        record["session"],
        exo,
        from_dict(OakFit, record["oak"]),
        quest,
        {name: np.asarray(record["oak_factory"][name]) for name in ("rgb_T_left", "rgb_T_right")},
    )


ENTITY = "/world/pred/rtmw_triangulated/coco133_xyz"


def dense(path: Path, entity: str, dimension: int):
    """Rows as (133, d) arrays: sparse served rows carry keypoint_ids, dense dataforge rows hold all 133 joints in order."""
    rows = {}
    for chunk in RrdReader(path).stream().filter(content=entity):
        batch = chunk.to_record_batch()
        if chunk.is_static or "video_time" not in batch.schema.names:
            continue
        kind = "Points3D" if dimension == 3 else "Points2D"
        positions_column = batch.column(f"{kind}:positions").to_pylist()
        ids_column = (
            batch.column(f"{kind}:keypoint_ids").to_pylist()
            if f"{kind}:keypoint_ids" in batch.schema.names
            else [None if row is None else list(range(133)) for row in positions_column]
        )
        for stamp, positions, ids in zip(batch.column("video_time").cast("int64").to_pylist(), positions_column, ids_column, strict=True):
            array = np.full((133, dimension), np.nan)
            if ids:
                array[np.asarray(ids).reshape(-1)] = np.asarray(positions).reshape(-1, dimension)
            rows[stamp] = array
    return np.asarray(sorted(rows), dtype=np.int64), np.stack([rows[t] for t in sorted(rows)])


def chunk_bytes(path):
    return {c.to_record_batch().schema.metadata[b"rerun:id"]: c.to_record_batch().serialize().to_pybytes() for c in RrdReader(path).stream()}


def calib_files(server, syncs, directory):
    """Write typed reference calibration without registering its chunks."""
    dataset = server.client().get_dataset(DATASET_NAME)
    record = ROOT / "calib/sessions" / f"{syncs[0].key.rsplit('__', 1)[0]}.json"
    if not record.is_file():
        pytest.skip(f"golden session record absent: {record}")
    segments = {sync.key: open_segment(dataset, sync.episode_key.segment) for sync in syncs}
    state = frozen_state(json.loads(record.read_text()), read_quest_block(segments[syncs[0].key]))
    outputs = [directory / f"selfcap__{sync.key}.rrd" for sync in syncs]
    for sync, output in zip(syncs, outputs, strict=True):
        write_calib_layer(output, state, read_session_episode(segments[sync.key], sync))
    return outputs


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


def load_oak_reference(path: Path) -> OakReference:
    """Read full-frame OAK hands and exact frame-planning metadata."""
    tables = reference_tables(path)
    metadata = tables["/reference"]
    entity = f"{metadata['camera'][0].as_py()[0]}/pinhole/pred/sapiens2/coco133_uv"
    rows = tables.get(entity)
    if rows is None:
        indices, times, aligned = np.empty(0, dtype=np.int64), np.empty(0), np.empty(0)
        xy, scores = np.empty((0, 133, 2)), np.empty((0, 133))
    else:
        indices = np.asarray(rows["idx"].to_pylist(), dtype=np.int64).reshape(-1)
        times = np.asarray(rows["time_s"].to_pylist(), dtype=np.float64).reshape(-1)
        aligned = np.asarray(rows["video_time_s"].to_pylist(), dtype=np.float64).reshape(-1)
        xy = np.asarray(rows["Points2D:positions"].to_pylist(), dtype=np.float64).reshape(-1, 133, 2)
        scores = np.asarray(rows["confidence"].to_pylist(), dtype=np.float64).reshape(-1, 133)
    return OakReference(
        indices,
        float(metadata["fps"][0].as_py()[0]),
        times,
        OakDetections(aligned, xy, scores),
        int(metadata["width"][0].as_py()[0]),
        int(metadata["height"][0].as_py()[0]),
    )


def load_oak_detections(path: Path) -> OakDetections:
    """Load a read-only OAK RRD reference; never a production input."""
    return load_oak_reference(path).detections
