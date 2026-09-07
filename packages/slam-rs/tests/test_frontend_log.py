"""What the frontend's Rerun layer promises: colours, dump association, and what ``log`` writes.

The logging contract is checked against a recording, not against a mock. Each
test initialises the global recording :class:`slam_rs.frontend_log.FrontendLogger`
logs into (D03: the tools own no :class:`rerun.RecordingStream`), saves it to a
temp file, and reads the chunks back with
:class:`rerun.experimental.RrdReader` — the same rows a viewer would receive. So
an entity path, a ``video_time`` value or a trail that never reached the file
fails here.

The frames are the synthetic 200x200 textures of ``test_frontend_boundary``
rather than the committed 960x960 fixtures: the fixtures are the Rust parity
gate's, and four framesets of them through the frontend cost more than this whole
Python suite.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import rerun as rr
import rerun.experimental as rx
from jaxtyping import Float32, Int64, UInt8
from numpy import ndarray
from test_frontend_boundary import FRAME, frontend, texture

from slam_rs import _core
from slam_rs.frontend_log import (
    CPP_COLOR,
    DEFAULT_DUMPS_DIR,
    SOURCE_FILE,
    STATS_ENTITY,
    TRAIL_LENGTH,
    FrontendLogger,
    camera_entity,
    read_cpp_dumps,
    track_colors,
)

SMOKE_SEGMENT: str = "msd-index__MIO_others__MIO10_short_2_panorama"
"""The segment the committed dumps were recorded from."""
OTHER_SEGMENT: str = "msd-index__MIO_others__MIO07_mapping_easy"
"""Another Index segment, whose ``video_time`` also starts at zero."""
FRAME_INTERVAL_NS: int = 33_000_000
"""One 30 Hz frameset to the next."""

@dataclass(frozen=True, slots=True)
class Row:
    """One logged row of one entity, as it comes back out of the file."""

    t_ns: int
    """Where on ``video_time`` the row sits, in nanoseconds."""
    values: dict[str, list]
    """The components this row set, by their short name (``Points2D:positions`` and such)."""


Rows = dict[str, list[Row]]
"""Per entity path, its rows in ``video_time`` order."""


def read_rows(path: Path) -> Rows:
    """Read every non-static row of a recording, grouped by entity path.

    A component a row did not set comes back as a null and is dropped, so a
    missing key here means the row really did not carry that component.

    Args:
        path: An ``.rrd`` written by :func:`rerun.save`.

    Returns:
        Entity path to its rows, in ``video_time`` order.
    """
    rows: Rows = {}
    for chunk in rx.RrdReader(path).stream().collect().stream():
        if chunk.is_static:
            continue
        batch = chunk.to_record_batch()
        times: list = batch.column("video_time").to_pylist()
        components: dict[str, list] = {
            name: batch.column(name).to_pylist() for name in batch.schema.names if ":" in name and not name.startswith("rerun.")
        }
        for index, time in enumerate(times):
            values: dict[str, list] = {name: column[index] for name, column in components.items() if column[index] is not None}
            rows.setdefault(chunk.entity_path, []).append(Row(t_ns=int(np.timedelta64(time, "ns").astype(np.int64)), values=values))
    for entity in rows:
        rows[entity].sort(key=lambda row: row.t_ns)
    return rows


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


def replay(tmp_path: Path, framesets: int, segment_id: str, dumps_dir: Path | None) -> tuple[Rows, list[_core.FlowFrame]]:
    """Log ``framesets`` framesets of a drifting texture and read the recording back.

    Args:
        tmp_path: Where the ``.rrd`` is written.
        framesets: How many framesets to log, each shifted one pixel from the last.
        segment_id: Segment the logger is told it is replaying.
        dumps_dir: Where its C++ dumps come from, or None for the committed ones.

    Returns:
        The recording's rows, and the frames that produced them.
    """
    flow: _core.OpticalFlow = frontend(2)
    logger: FrontendLogger = FrontendLogger.create(2, segment_id, dumps_dir)
    tmp_path.mkdir(parents=True, exist_ok=True)
    output: Path = tmp_path / "frontend.rrd"
    rr.init("slam-rs-frontend-log-test", recording_id=f"{segment_id}-{framesets}")
    rr.save(output)
    frames: list[_core.FlowFrame] = []
    for step in range(framesets):
        t_ns: int = step * FRAME_INTERVAL_NS
        rr.set_time("video_time", duration=np.timedelta64(t_ns, "ns"))
        frame: _core.FlowFrame = flow.process(t_ns, [texture(step, 0), texture(step, 1)])
        logger.log(frame, elapsed_ms=1.5 * (step + 1))
        frames.append(frame)
    rr.disconnect()
    return read_rows(output), frames


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


def test_an_empty_track_list_gives_an_empty_colour_list() -> None:
    assert track_colors(np.zeros(0, dtype=np.int64)).shape == (0, 3)


def test_the_committed_cpp_dumps_are_keyed_by_their_own_segment() -> None:
    """The dumps carry the smoke segment's id and the feed's own ``video_time``."""
    dumps: dict[tuple[str, int], list[Float32[ndarray, "n_keypoints 2"]]] = read_cpp_dumps(DEFAULT_DUMPS_DIR)
    assert len(dumps) == 8
    assert {segment for segment, _ in dumps} == {SMOKE_SEGMENT}
    assert (SMOKE_SEGMENT, 0) in dumps and (SMOKE_SEGMENT, 18507000) in dumps, sorted(dumps)
    first: list[Float32[ndarray, "n_keypoints 2"]] = dumps[SMOKE_SEGMENT, 0]
    assert len(first) == 2, "the smoke segment is a two-camera rig"
    assert first[0].shape == (66, 2), "what dump_flow.cpp recorded for camera 0 of frameset 0"
    assert first[0].dtype == np.float32
    assert np.all((first[0] >= 0.0) & (first[0] < 960.0)), "keypoints lie inside the 960x960 frame"
    # And no other segment finds them, however well its clock lines up.
    assert (OTHER_SEGMENT, 0) not in dumps


def test_a_directory_without_dumps_leaves_the_overlay_empty(tmp_path: Path) -> None:
    assert read_cpp_dumps(tmp_path) == {}
    assert read_cpp_dumps(tmp_path / "does-not-exist") == {}


def test_dumps_with_no_source_file_are_refused(tmp_path: Path) -> None:
    """An unattributed dump could only be drawn on the wrong segment."""
    write_dumps(tmp_path, SMOKE_SEGMENT, {0: [np.zeros((1, 2), dtype=np.float32)]})
    (tmp_path / SOURCE_FILE).unlink()
    with pytest.raises(ValueError, match=SOURCE_FILE):
        read_cpp_dumps(tmp_path)


def test_log_writes_the_dataset_tree_once_per_frameset(tmp_path: Path) -> None:
    """Every entity the blueprint shows gets one row per frameset, on ``video_time``."""
    rows, frames = replay(tmp_path, 3, OTHER_SEGMENT, tmp_path / "no-dumps")
    expected_times: list[int] = [step * FRAME_INTERVAL_NS for step in range(3)]
    for camera in range(2):
        for leaf in ("keypoints", "trails", "cells"):
            entity: str = f"{camera_entity(camera)}/{leaf}"
            assert entity in rows, sorted(rows)
            assert [row.t_ns for row in rows[entity]] == expected_times
        for counter in ("num_tracks", "num_new"):
            entity = f"{STATS_ENTITY}/cam_{camera:02d}/{counter}"
            assert [row.t_ns for row in rows[entity]] == expected_times
    assert [row.t_ns for row in rows[f"{STATS_ENTITY}/frontend_ms"]] == expected_times
    # The keypoints logged are the ones the frame carries, in the frame's order.
    for step, frame in enumerate(frames):
        logged: Float32[ndarray, "n_tracks 2"] = np.array(rows[f"{camera_entity(0)}/keypoints"][step].values["Points2D:positions"], dtype=np.float32)
        assert np.allclose(logged, frame.positions(0))


def test_the_counters_report_each_cameras_own_numbers(tmp_path: Path) -> None:
    rows, frames = replay(tmp_path, 3, OTHER_SEGMENT, tmp_path / "no-dumps")
    for camera in range(2):
        tracks: list[Row] = rows[f"{STATS_ENTITY}/cam_{camera:02d}/num_tracks"]
        fresh: list[Row] = rows[f"{STATS_ENTITY}/cam_{camera:02d}/num_new"]
        assert [row.values["Scalars:scalars"] for row in tracks] == [[float(frame.num_tracks(camera))] for frame in frames]
        assert [row.values["Scalars:scalars"] for row in fresh] == [[float(frame.num_new(camera))] for frame in frames]
    assert [row.values["Scalars:scalars"] for row in rows[f"{STATS_ENTITY}/frontend_ms"]] == [[1.5], [3.0], [4.5]]


def test_trails_follow_a_track_by_id_and_stop_at_the_trail_length(tmp_path: Path) -> None:
    """A strip is one id's history, so it survives the frame's order changing.

    Keying by slot would tie a strip to whatever id happens to sit at that index,
    which drifts as tracks die: the expectation here is built per id from the
    frames themselves, so an index-keyed implementation cannot reproduce it.
    """
    framesets: int = TRAIL_LENGTH + 4
    rows, frames = replay(tmp_path, framesets, OTHER_SEGMENT, tmp_path / "no-dumps")
    history: dict[int, list[tuple[float, float]]] = {}
    for step, frame in enumerate(frames):
        positions: Float32[ndarray, "n_tracks 2"] = frame.positions(0)
        expected: set[tuple[tuple[float, float], ...]] = set()
        surviving: dict[int, list[tuple[float, float]]] = {}
        for slot, identifier in enumerate(frame.ids(0).tolist()):
            trail: list[tuple[float, float]] = history.get(identifier, [])[-(TRAIL_LENGTH - 1) :]
            trail.append((float(positions[slot, 0]), float(positions[slot, 1])))
            surviving[identifier] = trail
            if len(trail) > 1:
                expected.add(tuple(trail))
        history = surviving
        strips: list = rows[f"{camera_entity(0)}/trails"][step].values["LineStrips2D:strips"]
        logged: set[tuple[tuple[float, float], ...]] = {tuple((point[0], point[1]) for point in strip) for strip in strips}
        assert logged == expected, f"frameset {step}: {len(logged)} strips against {len(expected)} expected"
        assert all(len(strip) <= TRAIL_LENGTH for strip in strips)
    # The run is long enough that the cap is actually reached.
    last: list = rows[f"{camera_entity(0)}/trails"][-1].values["LineStrips2D:strips"]
    assert max(len(strip) for strip in last) == TRAIL_LENGTH


def test_the_overlay_is_drawn_only_on_the_segment_the_dumps_came_from(tmp_path: Path) -> None:
    """The review's finding: MIO10's dumps used to land on MIO07's first frame."""
    dumps_dir: Path = tmp_path / "dumps"
    overlay: Float32[ndarray, "n_keypoints 2"] = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    write_dumps(dumps_dir, SMOKE_SEGMENT, {0: [overlay, overlay], FRAME_INTERVAL_NS: [overlay, overlay]})

    smoke, _ = replay(tmp_path / "smoke", 2, SMOKE_SEGMENT, dumps_dir)
    for camera in range(2):
        entity: str = f"{camera_entity(camera)}/keypoints_cpp"
        assert entity in smoke, sorted(smoke)
        assert len(smoke[entity]) == 2
        for row in smoke[entity]:
            assert np.allclose(np.array(row.values["Points2D:positions"], dtype=np.float32), overlay)
            assert row.values["Points2D:colors"] == [(CPP_COLOR[0] << 24) | (CPP_COLOR[1] << 16) | (CPP_COLOR[2] << 8) | 0xFF]

    other, _ = replay(tmp_path / "other", 2, OTHER_SEGMENT, dumps_dir)
    assert not [entity for entity in other if entity.endswith("keypoints_cpp")], sorted(other)


def test_the_overlay_is_cleared_once_the_dumps_run_out_and_stays_cleared(tmp_path: Path) -> None:
    """Latest-at would leave the last overlay on screen for the rest of the segment."""
    dumps_dir: Path = tmp_path / "dumps"
    overlay: Float32[ndarray, "n_keypoints 2"] = np.array([[10.0, 20.0]], dtype=np.float32)
    write_dumps(dumps_dir, SMOKE_SEGMENT, {0: [overlay, overlay]})
    rows, _ = replay(tmp_path, 4, SMOKE_SEGMENT, dumps_dir)

    for camera in range(2):
        logged: list[Row] = rows[f"{camera_entity(camera)}/keypoints_cpp"]
        # One drawn row and exactly one clearing row: the clear is not repeated
        # per frameset, and nothing is drawn after it.
        assert [row.t_ns for row in logged] == [0, FRAME_INTERVAL_NS]
        assert len(logged[0].values["Points2D:positions"]) == 1
        assert logged[1].values["Points2D:positions"] == []


def test_the_frame_is_logged_at_the_calibrated_resolution(tmp_path: Path) -> None:
    """The keypoints are in the pixels of the frame that was tracked."""
    rows, frames = replay(tmp_path, 1, OTHER_SEGMENT, tmp_path / "no-dumps")
    logged: Row = rows[f"{camera_entity(0)}/keypoints"][0]
    positions: Float32[ndarray, "n_tracks 2"] = np.array(logged.values["Points2D:positions"], dtype=np.float32)
    assert len(positions) == frames[0].num_tracks(0)
    assert np.all((positions >= 0.0) & (positions < FRAME))
