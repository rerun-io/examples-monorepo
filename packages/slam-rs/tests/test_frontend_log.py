"""What the frontend's Rerun layer promises: colours, and what ``log`` writes.

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
from slam_rs.catalog_feed import TIMELINE
from slam_rs.frontend_log import (
    STATS_ENTITY,
    TRAIL_LENGTH,
    FrontendLogger,
    camera_entity,
    track_colors,
)

OTHER_SEGMENT: str = "msd-index__MIO_others__MIO07_mapping_easy"
"""Another Index segment, whose ``video_time`` also starts at zero."""



class ReplayResult(NamedTuple):
    """What one logged run gives back."""

    rows: Rows
    """Every non-static row of the recording, by entity path."""
    frames: list[_core.FlowFrame]
    """What the frontend produced, in frameset order."""


ReplayFactory: TypeAlias = Callable[[Path, int, str], ReplayResult]
"""One logged run: where the ``.rrd`` goes, how many framesets, the id the replayed recording carries."""




@pytest.fixture
def replay(frontend: FrontendFactory, texture: TextureFactory, read_rows: RowsReader) -> ReplayFactory:
    """Log ``framesets`` framesets of a drifting texture and read the recording back.

    Args:
        frontend: The two-camera frontend the framesets go through.
        texture: The scene each frameset is a shifted copy of.
        read_rows: Reads the recording back, from :mod:`conftest`.

    Returns:
        A function of the ``.rrd`` directory, the frameset count, the segment id
        the replayed recording carries, giving
        back the recording's rows and the frames that produced them.
    """

    def run(tmp_path: Path, framesets: int, segment_id: str) -> ReplayResult:
        flow: _core.OpticalFlow = frontend(2)
        logger: FrontendLogger = FrontendLogger(2, segment_id)
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










def test_log_writes_the_dataset_tree_once_per_frameset(replay: ReplayFactory, tmp_path: Path) -> None:
    """Every entity the blueprint shows gets one row per frameset, on ``video_time``."""
    recorded: ReplayResult = replay(tmp_path, 3, OTHER_SEGMENT)
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
    recorded: ReplayResult = replay(tmp_path, 3, OTHER_SEGMENT)
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
    recorded: ReplayResult = replay(tmp_path, framesets, OTHER_SEGMENT)
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
