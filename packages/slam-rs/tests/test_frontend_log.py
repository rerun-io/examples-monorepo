"""What the frontend's Rerun layer promises: colours, dump association, and what ``log`` writes.

The logging contract is checked against a recording, not against a mock. Each
test initialises the global recording :class:`slam_rs.frontend_log.FrontendLogger`
logs into (D03: the tools own no :class:`rerun.RecordingStream`), saves it to a
temp file, and reads the chunks back with
:class:`rerun.experimental.RrdReader` — the same rows a viewer would receive. So
an entity path, a ``video_time`` value or a trail that never reached the file
fails here.

The frames are the synthetic 200x200 textures of :mod:`conftest` rather than the
committed 960x960 fixtures: the fixtures are the Rust parity gate's, and four
framesets of them through the frontend cost more than this whole Python suite.
Reading the recording back is :mod:`conftest`'s :func:`read_rows`, which the
estimator's logging suite drives too; the aliases below are declared here
because ``tests`` is not on the typechecker's search path.
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple, TypeAlias

import numpy as np
import pytest
import rerun as rr
from fixture_types import FRAME_PERIOD_NS, FrontendFactory, Row, Rows, RowsReader, TextureFactory
from jaxtyping import Float32, Int64, UInt8
from numpy import ndarray

from slam_rs import _core
from slam_rs.apis.replay import SMOKE_SEGMENT
from slam_rs.catalog_feed import TIMELINE, LocalSegment, SegmentFeed, open_segment
from slam_rs.frontend_log import (
    CPP_COLOR,
    DEFAULT_DUMPS_DIR,
    SOURCE_FILE,
    STATS_ENTITY,
    TRAIL_LENGTH,
    FrontendLogger,
    camera_entity,
    dumps_source,
    read_cpp_dumps,
    track_colors,
)
from slam_rs.reference import ReferenceManifest, ReferenceSegment

OTHER_SEGMENT: str = "msd-index__MIO_others__MIO07_mapping_easy"
"""Another Index segment, whose ``video_time`` also starts at zero."""
OVERLAY: Float32[ndarray, "n_keypoints 2"] = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
"""The C++ keypoints every written dump carries, on both cameras."""



class ReplayResult(NamedTuple):
    """What one logged run gives back."""

    rows: Rows
    """Every non-static row of the recording, by entity path."""
    frames: list[_core.FlowFrame]
    """What the frontend produced, in frameset order."""


ReplayFactory: TypeAlias = Callable[[Path, int, str, Path | None], ReplayResult]
"""One logged run: where the ``.rrd`` goes, how many framesets, the id the replayed recording carries, the dumps."""


def write_dumps(directory: Path, segment_id: str, per_frameset: dict[int, list[Float32[ndarray, "n_keypoints 2"]]]) -> None:
    """Write C++ dumps in ``dump_flow.cpp``'s shape, with the segment they came from.

    Args:
        directory: Where the ``frame_XXX.json`` and the source file go.
        segment_id: Segment the dumps are to be attributed to.
        per_frameset: Frameset timestamp to one ``[x, y]`` array per camera.
    """
    directory.mkdir(parents=True, exist_ok=True)
    (directory / SOURCE_FILE).write_text(json.dumps({"segment_id": segment_id}))
    for index, (t_ns, cameras) in enumerate(sorted(per_frameset.items())):
        dump: dict[str, object] = {
            "frame": index,
            "t_ns": t_ns,
            "cameras": [
                {
                    "camera": camera,
                    "keypoints": [
                        {"id": slot, "x": float(point[0]), "y": float(point[1]), "linear": [1, 0, 0, 1], "response": 1}
                        for slot, point in enumerate(points)
                    ],
                }
                for camera, points in enumerate(cameras)
            ],
        }
        (directory / f"frame_{index:03d}.json").write_text(json.dumps(dump))


@pytest.fixture
def replay(frontend: FrontendFactory, texture: TextureFactory, read_rows: RowsReader) -> ReplayFactory:
    """Log ``framesets`` framesets of a drifting texture and read the recording back.

    Args:
        frontend: The two-camera frontend the framesets go through.
        texture: The scene each frameset is a shifted copy of.
        read_rows: Reads the recording back, from :mod:`conftest`.

    Returns:
        A function of the ``.rrd`` directory, the frameset count, the segment id
        the replayed recording carries, and where its C++ dumps come from, giving
        back the recording's rows and the frames that produced them.
    """

    def run(tmp_path: Path, framesets: int, segment_id: str, dumps_dir: Path | None) -> ReplayResult:
        flow: _core.OpticalFlow = frontend(2)
        logger: FrontendLogger = FrontendLogger(2, segment_id, dumps_dir)
        tmp_path.mkdir(parents=True, exist_ok=True)
        output: Path = tmp_path / "frontend.rrd"
        rr.init("slam-rs-frontend-log-test", recording_id=f"{segment_id}-{framesets}")
        rr.save(output)
        frames: list[_core.FlowFrame] = []
        for step in range(framesets):
            t_ns: int = step * FRAME_PERIOD_NS
            rr.set_time(TIMELINE, duration=np.timedelta64(t_ns, "ns"))
            frame: _core.FlowFrame = flow.process(t_ns, [texture(step, 0), texture(step, 1)])
            logger.log(frame, elapsed_ms=1.5 * (step + 1))
            frames.append(frame)
        rr.disconnect()
        return ReplayResult(rows=read_rows(output), frames=frames)

    return run


@pytest.fixture
def smoke_dumps(tmp_path: Path) -> Path:
    """Two framesets of two-camera C++ dumps, attributed to the smoke segment.

    Returns:
        The directory they were written to.
    """
    directory: Path = tmp_path / "dumps"
    write_dumps(directory, SMOKE_SEGMENT, {0: [OVERLAY, OVERLAY], FRAME_PERIOD_NS: [OVERLAY, OVERLAY]})
    return directory


def test_a_track_keeps_its_colour_and_neighbours_do_not_share_one() -> None:
    ids: Int64[ndarray, " n_tracks"] = np.arange(200, dtype=np.int64)
    colors: UInt8[ndarray, "n_tracks 3"] = track_colors(ids)
    assert colors.shape == (200, 3)
    assert colors.dtype == np.uint8
    # The colour is a pure function of the id, so a track that survives a frame
    # keeps it however the array is ordered.
    assert np.array_equal(track_colors(ids[::-1]), colors[::-1])
    # Hue only: every colour is saturated, so none of them is grey on grey imagery.
    assert np.all(colors.max(axis=1) == 255)
    assert np.all(colors.min(axis=1) == 0)
    # Neighbouring ids land far apart on the hue circle.
    assert np.all(np.abs(colors[1:].astype(np.int64) - colors[:-1].astype(np.int64)).sum(axis=1) > 30)


def test_no_track_is_ever_drawn_in_the_overlays_colour() -> None:
    """The palette walks five of the six hue ramps: the sixth starts at magenta."""
    colors: UInt8[ndarray, "n_tracks 3"] = track_colors(np.arange(5000, dtype=np.int64))
    assert not np.any(np.all(colors == np.array(CPP_COLOR, dtype=np.uint8), axis=1))


def test_an_empty_track_list_gives_an_empty_colour_list() -> None:
    assert track_colors(np.zeros(0, dtype=np.int64)).shape == (0, 3)


def test_the_committed_cpp_dumps_name_their_own_segment() -> None:
    """The dumps carry the smoke segment's id and the feed's own ``video_time``."""
    assert dumps_source(DEFAULT_DUMPS_DIR) == SMOKE_SEGMENT
    dumps: dict[int, list[Float32[ndarray, "n_keypoints 2"]]] = read_cpp_dumps(2, DEFAULT_DUMPS_DIR)
    assert len(dumps) == 8
    assert 0 in dumps and 18507000 in dumps, sorted(dumps)
    first: list[Float32[ndarray, "n_keypoints 2"]] = dumps[0]
    assert len(first) == 2, "the smoke segment is a two-camera rig"
    assert first[0].shape == (66, 2), "what dump_flow.cpp recorded for camera 0 of frameset 0"
    assert first[0].dtype == np.float32
    assert np.all((first[0] >= 0.0) & (first[0] < 960.0)), "keypoints lie inside the 960x960 frame"


def test_a_directory_without_dumps_leaves_the_overlay_empty(tmp_path: Path) -> None:
    for empty in (tmp_path, tmp_path / "does-not-exist"):
        assert dumps_source(empty) == ""
        assert read_cpp_dumps(2, empty) == {}


def test_dumps_with_no_source_file_are_refused(tmp_path: Path) -> None:
    """An unattributed dump could only be drawn on the wrong segment."""
    write_dumps(tmp_path, SMOKE_SEGMENT, {0: [np.zeros((1, 2), dtype=np.float32)]})
    (tmp_path / SOURCE_FILE).unlink()
    with pytest.raises(ValueError, match=SOURCE_FILE):
        dumps_source(tmp_path)


def test_dumps_from_another_rig_are_refused(tmp_path: Path) -> None:
    """A dump short of a camera would leave that camera's last overlay on screen."""
    write_dumps(tmp_path, SMOKE_SEGMENT, {0: [np.zeros((1, 2), dtype=np.float32)]})
    with pytest.raises(ValueError, match="1 cameras, the rig being replayed has 2"):
        read_cpp_dumps(2, tmp_path)


def test_a_dump_without_a_timestamp_is_refused(tmp_path: Path) -> None:
    """``t_ns`` is the whole association within a segment, so its absence is not a default."""
    write_dumps(tmp_path, SMOKE_SEGMENT, {0: [np.zeros((1, 2), dtype=np.float32)]})
    path: Path = tmp_path / "frame_000.json"
    dump: dict[str, object] = json.loads(path.read_text())
    del dump["t_ns"]
    path.write_text(json.dumps(dump))
    with pytest.raises(ValueError, match="t_ns"):
        read_cpp_dumps(1, tmp_path)


def test_a_dump_directory_with_no_source_segment_is_refused(tmp_path: Path) -> None:
    """An empty ``source.json`` names no segment, so nothing may be associated with it."""
    write_dumps(tmp_path, SMOKE_SEGMENT, {0: [np.zeros((1, 2), dtype=np.float32)]})
    (tmp_path / SOURCE_FILE).write_text("{}")
    with pytest.raises(ValueError, match="segment_id"):
        dumps_source(tmp_path)


def test_dumps_of_another_recording_do_not_refuse_the_rig_on_screen(smoke_dumps: Path) -> None:
    """A dump's camera count is a fact about the overlay, so it is checked only where one is drawn."""
    assert FrontendLogger(4, OTHER_SEGMENT, smoke_dumps).cpp_dumps == {}
    with pytest.raises(ValueError, match="2 cameras, the rig being replayed has 4"):
        FrontendLogger(4, SMOKE_SEGMENT, smoke_dumps)


def test_log_writes_the_dataset_tree_once_per_frameset(replay: ReplayFactory, tmp_path: Path) -> None:
    """Every entity the blueprint shows gets one row per frameset, on ``video_time``."""
    recorded: ReplayResult = replay(tmp_path, 3, OTHER_SEGMENT, tmp_path / "no-dumps")
    expected_times: list[int] = [step * FRAME_PERIOD_NS for step in range(3)]
    for camera in range(2):
        for leaf in ("keypoints", "trails", "cells"):
            entity: str = f"{camera_entity(camera)}/{leaf}"
            assert entity in recorded.rows, sorted(recorded.rows)
            assert [row.t_ns for row in recorded.rows[entity]] == expected_times
        for counter in ("num_tracks", "num_new"):
            entity = f"{STATS_ENTITY}/cam_{camera:02d}/{counter}"
            assert [row.t_ns for row in recorded.rows[entity]] == expected_times
    assert [row.t_ns for row in recorded.rows[f"{STATS_ENTITY}/frontend_ms"]] == expected_times
    # The keypoints logged are the ones the frame carries, in the frame's order.
    for step, frame in enumerate(recorded.frames):
        logged: Float32[ndarray, "n_tracks 2"] = np.array(
            recorded.rows[f"{camera_entity(0)}/keypoints"][step].values["Points2D:positions"], dtype=np.float32
        )
        assert np.allclose(logged, frame.positions(0))


def test_the_counters_report_each_cameras_own_numbers(replay: ReplayFactory, tmp_path: Path) -> None:
    recorded: ReplayResult = replay(tmp_path, 3, OTHER_SEGMENT, tmp_path / "no-dumps")
    for camera in range(2):
        tracks: list[Row] = recorded.rows[f"{STATS_ENTITY}/cam_{camera:02d}/num_tracks"]
        fresh: list[Row] = recorded.rows[f"{STATS_ENTITY}/cam_{camera:02d}/num_new"]
        assert [row.values["Scalars:scalars"] for row in tracks] == [[float(frame.num_tracks(camera))] for frame in recorded.frames]
        assert [row.values["Scalars:scalars"] for row in fresh] == [[float(frame.num_new(camera))] for frame in recorded.frames]
    assert [row.values["Scalars:scalars"] for row in recorded.rows[f"{STATS_ENTITY}/frontend_ms"]] == [[1.5], [3.0], [4.5]]


def test_trails_follow_a_track_by_id_and_stop_at_the_trail_length(replay: ReplayFactory, tmp_path: Path) -> None:
    """A strip is one id's history, so it survives the frame's order changing.

    Asserted as invariants rather than as a second copy of ``_log_trails``: a
    strip ends at that id's current position, it is the same id's previous strip
    with this position appended, and it stops growing at :data:`TRAIL_LENGTH`.
    Keying by slot would tie a strip to whatever id sits at that index, which
    drifts as tracks die, and no invariant here would hold.
    """
    framesets: int = TRAIL_LENGTH + 4
    recorded: ReplayResult = replay(tmp_path, framesets, OTHER_SEGMENT, tmp_path / "no-dumps")
    survived: dict[int, int] = {}
    previous: dict[int, tuple[tuple[float, float], ...]] = {}
    for step, frame in enumerate(recorded.frames):
        positions: Float32[ndarray, "n_tracks 2"] = frame.positions(0)
        live: dict[int, tuple[float, float]] = {
            identifier: (float(positions[slot, 0]), float(positions[slot, 1])) for slot, identifier in enumerate(frame.ids(0).tolist())
        }
        survived = {identifier: 1 + survived.get(identifier, 0) for identifier in live}
        strips: list = recorded.rows[f"{camera_entity(0)}/trails"][step].values["LineStrips2D:strips"]
        by_end: dict[tuple[float, float], tuple[tuple[float, float], ...]] = {
            (strip[-1][0], strip[-1][1]): tuple((point[0], point[1]) for point in strip) for strip in strips
        }
        assert len(by_end) == len(strips), f"frameset {step}: two strips end at the same pixel"
        current: dict[int, tuple[tuple[float, float], ...]] = {}
        for identifier, position in live.items():
            expected_length: int = min(survived[identifier], TRAIL_LENGTH)
            if expected_length == 1:
                # One position is a point, not a trail, so nothing is drawn yet.
                assert position not in by_end, f"frameset {step}: a strip for a track born on it"
                continue
            strip: tuple[tuple[float, float], ...] | None = by_end.get(position)
            assert strip is not None, f"frameset {step}: no strip ends at track {identifier}"
            assert len(strip) == expected_length, f"frameset {step}: track {identifier} carries {len(strip)} points"
            if identifier in previous:
                grown: tuple[tuple[float, float], ...] = (*previous[identifier], position)
                assert strip == grown[-TRAIL_LENGTH:], f"frameset {step}: track {identifier} is not its own history plus this position"
            current[identifier] = strip
        previous = current
    # The run is long enough that the cap is actually reached.
    last: list = recorded.rows[f"{camera_entity(0)}/trails"][-1].values["LineStrips2D:strips"]
    assert max(len(strip) for strip in last) == TRAIL_LENGTH


def test_the_overlay_is_drawn_on_the_recording_the_dumps_came_from(replay: ReplayFactory, smoke_dumps: Path, tmp_path: Path) -> None:
    """Read back out of the file: the recording whose own id is the dumps' gets them."""
    recorded: ReplayResult = replay(tmp_path / "smoke", 2, SMOKE_SEGMENT, smoke_dumps)
    for camera in range(2):
        entity: str = f"{camera_entity(camera)}/keypoints_cpp"
        assert entity in recorded.rows, sorted(recorded.rows)
        assert len(recorded.rows[entity]) == 2
        for row in recorded.rows[entity]:
            assert np.allclose(np.array(row.values["Points2D:positions"], dtype=np.float32), OVERLAY)
            assert row.values["Points2D:colors"] == [(CPP_COLOR[0] << 24) | (CPP_COLOR[1] << 16) | (CPP_COLOR[2] << 8) | 0xFF]


def test_no_other_recording_is_given_the_overlay(
    replay: ReplayFactory, smoke_dumps: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The review's finding: MIO10's dumps used to land on MIO07's first frame."""
    recorded: ReplayResult = replay(tmp_path / "other", 2, OTHER_SEGMENT, smoke_dumps)
    assert not [entity for entity in recorded.rows if entity.endswith("keypoints_cpp")], sorted(recorded.rows)
    # The frames themselves were still tracked and logged, and the run says why, once.
    assert len(recorded.rows[f"{camera_entity(0)}/keypoints"]) == 2
    assert capsys.readouterr().out == f"dumps are from {SMOKE_SEGMENT}, replaying {OTHER_SEGMENT}: no overlay\n"


@pytest.mark.slow
def test_the_overlay_follows_the_recording_and_not_the_filename(manifest: ReferenceManifest, tmp_path: Path) -> None:
    """The re-review's collision: a file named like the dumps' segment, holding another recording.

    ``--rrd`` used to switch the overlay off by the flag, so a path spelled like
    a segment id drew MIO10's keypoints over MIO07's pixels. The feed reads the
    id out of the recording, so what the file is called decides nothing.
    """
    smoke: ReferenceSegment = manifest.by_id(SMOKE_SEGMENT)
    other: ReferenceSegment = manifest.by_id(OTHER_SEGMENT)
    if not other.base_path.is_file():
        pytest.skip(f"{other.base_path} is not mounted on this host")
    spelled_like_the_dumps: Path = tmp_path / SMOKE_SEGMENT
    spelled_like_the_dumps.symlink_to(other.base_path)

    feed: SegmentFeed
    with open_segment(LocalSegment(base_rrd=spelled_like_the_dumps), smoke.imu) as feed:
        assert feed.segment_id == OTHER_SEGMENT
        assert FrontendLogger(len(feed.cameras), feed.segment_id).cpp_dumps == {}


def test_the_overlay_is_cleared_once_the_dumps_run_out_and_stays_cleared(replay: ReplayFactory, tmp_path: Path) -> None:
    """Latest-at would leave the last overlay on screen for the rest of the segment."""
    dumps_dir: Path = tmp_path / "dumps"
    write_dumps(dumps_dir, SMOKE_SEGMENT, {0: [OVERLAY, OVERLAY]})
    recorded: ReplayResult = replay(tmp_path, 4, SMOKE_SEGMENT, dumps_dir)
    for camera in range(2):
        logged: list[Row] = recorded.rows[f"{camera_entity(camera)}/keypoints_cpp"]
        # One drawn row and exactly one clearing row: the clear is not repeated
        # per frameset, and nothing is drawn after it.
        assert [row.t_ns for row in logged] == [0, FRAME_PERIOD_NS]
        assert len(logged[0].values["Points2D:positions"]) == len(OVERLAY)
        assert logged[1].values["Points2D:positions"] == []
