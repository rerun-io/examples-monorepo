"""The shared writers: the time column, video remux timing, and the magnetometer node.

Every assertion reads the written rrd back through the public reader
(``RrdReader`` → ``ChunkStore`` → a datafusion view over one index), so these
test what a consumer sees rather than what the writer intended.

The remux tests run against a **checked-in** AV1 mp4 rather than an encode, so
they exercise ``log_video_stream`` on a machine with no GPU; proving the encoder
itself is ``test_encoding.py``'s job and stays NVENC-gated there. How the fixture
was made is in ``tests/fixtures/README.md``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
import rerun as rr
from beartype.roar import BeartypeException
from conftest import column_rows, read_back
from jaxtyping import Float64, Int64
from numpy import ndarray

from dataforge import schema
from dataforge.logging_toolkit import (
    VIDEO_KEYFRAME_COMPONENT,
    VIDEO_SAMPLE_COMPONENT,
    ImuChannel,
    VideoChunkKind,
    classify_video_chunk,
    log_magnetometer,
    log_trail_segments,
    log_video_stream,
    time_column,
)

CLIP: Path = Path(__file__).parent / "fixtures" / "av1_48f_192x160.mp4"
"""The checked-in AV1 clip every remux test reads; 48 samples, a keyframe every 12."""
NUM_FRAMES: int = 48
"""Samples in ``CLIP``, as its filename says."""
ENTITY: str = "/world/rig_00/cam_00/pinhole/video"


@pytest.fixture(scope="module")
def clip() -> Path:
    """The checked-in clip. A fixture, not a bare constant, so a missing file fails once."""
    assert CLIP.is_file(), f"{CLIP} is checked in; see tests/fixtures/README.md"
    return CLIP


def index_column(rrd: Path, index: str = schema.TIMELINE) -> Int64[ndarray, "n_rows"]:
    """Every value of one index in a saved rrd, ascending."""
    table: pa.Table = read_back(rrd).reader(index=index).to_arrow_table().sort_by(index)
    return table.column(index).combine_chunks().cast(pa.int64()).to_numpy()


def irregular_times_ns(count: int) -> Int64[ndarray, "n_samples"]:
    """Ascending 33 ms steps with per-sample jitter — a real device's capture clock."""
    jitter: Int64[ndarray, "n_samples"] = (np.arange(count, dtype=np.int64) * 7919 % 900_000) - 450_000
    return 1_700_000_000_000_000_000 + np.arange(count, dtype=np.int64) * 33_000_000 + jitter


# ── time_column ───────────────────────────────────────────────────────────


def test_the_time_column_view_carries_the_same_values_as_a_cast() -> None:
    """``view`` skips the copy ``astype`` makes; the column a consumer reads is unchanged."""
    times_ns: Int64[ndarray, "n_samples"] = irregular_times_ns(NUM_FRAMES)

    viewed: rr.TimeColumn = time_column(times_ns)
    cast: rr.TimeColumn = rr.TimeColumn(schema.TIMELINE, duration=times_ns.astype("timedelta64[ns]"))

    assert viewed.as_arrow_array() == cast.as_arrow_array()


def test_a_clock_that_is_not_int64_is_refused() -> None:
    """A float clock reinterpreted as nanoseconds would be silent nonsense."""
    with pytest.raises((AssertionError, BeartypeException)):
        time_column(irregular_times_ns(NUM_FRAMES).astype(np.float64))


# ── log_video_stream retiming ─────────────────────────────────────────────


def test_sample_count_excludes_the_keyframe_chunk(tmp_path: Path, clip: Path) -> None:
    """``Mp4Reader`` emits a trailing keyframe chunk that is indexed but is not a sample."""
    target: Path = tmp_path / "count.rrd"
    with rr.RecordingStream("dataforge", recording_id="count") as recording:
        recording.save(target)
        written: int = log_video_stream(recording, clip, ENTITY)
    assert written == NUM_FRAMES


def test_times_ns_replaces_every_sample_timestamp(tmp_path: Path, clip: Path) -> None:
    times_ns: Int64[ndarray, "n_samples"] = irregular_times_ns(NUM_FRAMES)
    target: Path = tmp_path / "retimed.rrd"
    with rr.RecordingStream("dataforge", recording_id="retimed") as recording:
        recording.save(target)
        written: int = log_video_stream(recording, clip, ENTITY, times_ns=times_ns)
    assert written == NUM_FRAMES
    assert np.array_equal(index_column(target), times_ns)


def test_shift_ns_still_offsets_the_container_pts(tmp_path: Path, clip: Path) -> None:
    shift_ns: int = 1_700_000_000_000_000_000
    plain: Path = tmp_path / "plain.rrd"
    shifted: Path = tmp_path / "shifted.rrd"
    for target, shift in ((plain, 0), (shifted, shift_ns)):
        with rr.RecordingStream("dataforge", recording_id=target.stem) as recording:
            recording.save(target)
            log_video_stream(recording, clip, ENTITY, shift_ns=shift)
    assert np.array_equal(index_column(shifted), index_column(plain) + shift_ns)


def test_times_ns_and_shift_ns_are_mutually_exclusive(tmp_path: Path, clip: Path) -> None:
    target: Path = tmp_path / "both.rrd"
    with rr.RecordingStream("dataforge", recording_id="both") as recording:
        recording.save(target)
        with pytest.raises(ValueError, match="mutually exclusive"):
            log_video_stream(recording, clip, ENTITY, shift_ns=5, times_ns=irregular_times_ns(NUM_FRAMES))


def test_too_few_timestamps_is_an_error(tmp_path: Path, clip: Path) -> None:
    target: Path = tmp_path / "short.rrd"
    with rr.RecordingStream("dataforge", recording_id="short") as recording:
        recording.save(target)
        # The tap raises mid-stream; a swallowed exception would show up as the trailing count check instead.
        with pytest.raises(ValueError, match="more samples than the 45 timestamps"):
            log_video_stream(recording, clip, ENTITY, times_ns=irregular_times_ns(NUM_FRAMES - 3))


def test_too_many_timestamps_is_an_error(tmp_path: Path, clip: Path) -> None:
    target: Path = tmp_path / "long.rrd"
    with rr.RecordingStream("dataforge", recording_id="long") as recording:
        recording.save(target)
        with pytest.raises(ValueError, match="holds 48 samples but 51 timestamps"):
            log_video_stream(recording, clip, ENTITY, times_ns=irregular_times_ns(NUM_FRAMES + 3))


# ── classify_video_chunk ──────────────────────────────────────────────────


def test_every_chunk_the_reader_emits_is_one_of_the_three_named_shapes(clip: Path) -> None:
    """The real reader's own output, classified: one codec chunk, then samples, then one keyframe chunk."""
    reader: rr.experimental.Mp4Reader = rr.experimental.Mp4Reader(
        clip, mode="stream", entity_path=ENTITY, timeline_name=schema.TIMELINE
    )
    kinds: list[VideoChunkKind] = [classify_video_chunk(chunk.to_record_batch()) for chunk in reader.stream()]

    assert kinds[0] == "codec", "the static codec chunk comes first and carries no index"
    assert kinds[-1] == "keyframe", "the keyframe chunk trails the samples it indexes into"
    assert set(kinds[1:-1]) == {"sample"}


def indexed_batch(component: str) -> pa.RecordBatch:
    """One ``video_time``-indexed chunk batch carrying ``component`` and nothing else.

    Built by hand rather than encoded: what is under test is the decision the tap
    makes about a chunk's components, and that needs no video at all.
    """
    schema_with_path: pa.Schema = pa.schema(
        [pa.field(schema.TIMELINE, pa.duration("ns")), pa.field(component, pa.int64())],
        metadata={"rerun:entity_path": ENTITY},
    )
    return pa.RecordBatch.from_arrays(
        [pa.array([0, 1], type=pa.duration("ns")), pa.array([7, 8], type=pa.int64())], schema=schema_with_path
    )


@pytest.mark.parametrize(
    ("component", "expected"), [(VIDEO_SAMPLE_COMPONENT, "sample"), (VIDEO_KEYFRAME_COMPONENT, "keyframe")]
)
def test_an_indexed_chunk_is_named_by_the_component_it_carries(component: str, expected: VideoChunkKind) -> None:
    assert classify_video_chunk(indexed_batch(component)) == expected


def test_an_indexed_chunk_of_an_unknown_shape_is_refused_by_entity_and_component() -> None:
    """A fourth chunk shape must fail loudly, not be retimed as if it were the keyframe chunk.

    Silently absorbing it is the failure this guards: the keyframe branch would
    look its rows up among the sample PTS and either raise something misleading
    or write plausible-looking wrong timestamps.
    """
    with pytest.raises(ValueError) as refusal:
        classify_video_chunk(indexed_batch("VideoStream:something_new"))

    message: str = str(refusal.value)
    assert ENTITY in message, "the reader names which entity's stream it could not classify"
    assert "VideoStream:something_new" in message, "and which components it saw instead"


# ── log_magnetometer ──────────────────────────────────────────────────────


RIG: int = 0
MAG: int = 0
HEADING_LENGTH_M: float = 0.15


def synthetic_field(count: int) -> ImuChannel:
    """A slowly rotating field of roughly constant magnitude, plus one dead sample."""
    angle: Float64[ndarray, "n_samples"] = np.linspace(0.0, 2.0 * np.pi, count)
    values_xyz: Float64[ndarray, "n_samples 3"] = np.stack(
        [300.0 * np.cos(angle), 300.0 * np.sin(angle), np.full(count, -40.0)], axis=1
    )
    values_xyz[3] = 0.0  # a dropout: zero field, whose direction is undefined
    times_ns: Int64[ndarray, "n_samples"] = 1_700_000_000_000_000_000 + np.arange(count, dtype=np.int64) * 20_000_000
    return ImuChannel(times_ns=times_ns, values_xyz=values_xyz)


def test_magnetometer_logs_field_and_heading(tmp_path: Path) -> None:
    field: ImuChannel = synthetic_field(50)
    target: Path = tmp_path / "mag.rrd"
    with rr.RecordingStream("dataforge", recording_id="mag") as recording:
        recording.save(target)
        log_magnetometer(recording, RIG, MAG, field=field, name="reverb-g2", unit="mG", heading_length_m=HEADING_LENGTH_M)

    scalars: pa.Table = column_rows(read_back(target), f"{schema.field_path(RIG, MAG)}:Scalars:scalars")
    assert scalars.num_rows == field.times_ns.size
    assert np.array_equal(scalars.column(schema.TIMELINE).combine_chunks().cast(pa.int64()).to_numpy(), field.times_ns)

    arrows: pa.Table = column_rows(read_back(target), f"{schema.heading_path(RIG, MAG)}:Arrows3D:vectors")
    # The zero-field sample has no direction, so it is not given an arrow.
    assert arrows.num_rows == field.times_ns.size - 1
    vectors: Float64[ndarray, "n_arrows 3"] = np.asarray(
        [row[0] for row in arrows.column(f"{schema.heading_path(RIG, MAG)}:Arrows3D:vectors").to_pylist()], dtype=np.float64
    )
    assert np.allclose(np.linalg.norm(vectors, axis=1), HEADING_LENGTH_M, atol=1e-5)


def test_magnetometer_node_carries_its_static_pose_and_metadata(tmp_path: Path) -> None:
    target: Path = tmp_path / "mag_static.rrd"
    with rr.RecordingStream("dataforge", recording_id="mag_static") as recording:
        recording.save(target)
        log_magnetometer(recording, RIG, MAG, field=synthetic_field(8), name="odyssey", unit=None)
    store: rr.experimental.ChunkStore = read_back(target)
    columns: set[str] = {str(column) for column in store.schema()}
    node: str = schema.mag_path(RIG, MAG)
    assert any(f"Column name: {node}:Transform3D:" in column for column in columns), "rig_T_mag is mandatory, like the IMU's"
    assert any(f"Column name: {node}:name" in column for column in columns)
    assert any(f"Column name: {node}:kind" in column for column in columns)
    # A None unit carries no value. AnyValues only *omits* the key while it is
    # untyped: once anything in the process logs a unit string the key becomes
    # typed and later Nones arrive as nulls, so assert on the value, not the column.
    table: pa.Table = store.reader(index=schema.TIMELINE).to_arrow_table()
    unit_column: str = f"{node}:unit"
    assert unit_column not in table.column_names or table.column(unit_column).null_count == table.num_rows


def test_empty_magnetometer_logs_only_the_static_node(tmp_path: Path) -> None:
    empty: ImuChannel = ImuChannel(times_ns=np.zeros(0, dtype=np.int64), values_xyz=np.zeros((0, 3), dtype=np.float64))
    target: Path = tmp_path / "mag_empty.rrd"
    with rr.RecordingStream("dataforge", recording_id="mag_empty") as recording:
        recording.save(target)
        log_magnetometer(recording, RIG, MAG, field=empty, name="none")
    store: rr.experimental.ChunkStore = read_back(target)
    columns: set[str] = {str(column) for column in store.schema()}
    assert any(f"{schema.mag_path(RIG, MAG)}:Transform3D:" in column for column in columns)
    assert not any(schema.field_path(RIG, MAG) in column for column in columns)
    assert not any(schema.heading_path(RIG, MAG) in column for column in columns)


# ── the motion trail ──────────────────────────────────────────────────────

TRAIL_ENTITY: str = "/world/runs/gt/trail"
"""Where a trail lands; the same path ``schema.trail_path`` builds."""
TRAIL_COLOR: tuple[int, int, int] = (255, 215, 90)
TRAIL_RADIUS_UI_POINTS: float = 3.0
"""A screen-space width, so the trail reads as a stroke at any zoom."""


def walk_positions(count: int) -> Float64[ndarray, "n_poses 3"]:
    """A path whose every step differs, so no two segments coincide."""
    step: Float64[ndarray, "n_poses"] = np.arange(count, dtype=np.float64)
    return np.column_stack([step * 0.1, np.sin(step * 0.3), np.cos(step * 0.2) * 0.5])


def logged_trail(tmp_path: Path, count: int) -> tuple[Path, Float64[ndarray, "n_poses 3"], Int64[ndarray, "n_poses"]]:
    """Write one trail into its own rrd and hand back the file and what went in."""
    positions_xyz: Float64[ndarray, "n_poses 3"] = walk_positions(count)
    times_ns: Int64[ndarray, "n_poses"] = np.arange(count, dtype=np.int64) * 1_000_000
    target: Path = tmp_path / "trail.rrd"
    with rr.RecordingStream("dataforge", recording_id="trail") as recording:
        recording.save(target)
        log_trail_segments(
            recording,
            TRAIL_ENTITY,
            times_ns=times_ns,
            translations_xyz=positions_xyz,
            color=TRAIL_COLOR,
            radius_ui_points=TRAIL_RADIUS_UI_POINTS,
        )
    return target, positions_xyz, times_ns


def trail_strips(rrd: Path) -> list[list[list[float]]]:
    """Every trail row's strips, index-sorted; one row per pose."""
    rows: pa.Table = column_rows(read_back(rrd), f"{TRAIL_ENTITY}:LineStrips3D:strips")
    return rows.column(1).to_pylist()


def test_a_trail_is_one_segment_per_pose_from_the_previous_position(tmp_path: Path) -> None:
    """The trail is drawn as strokes, not dots: each pose contributes the step that reached it.

    A metric-radius ``Points3D`` trail reads as a string of scattered balls,
    because a 2 cm sphere at a 1 kHz sample rate is wider than the gap between
    samples. One two-point strip per pose draws the same information as a line
    the eye follows.
    """
    count: int = 12
    target, positions_xyz, times_ns = logged_trail(tmp_path, count)

    strips: list[list[list[float]]] = trail_strips(target)

    assert len(strips) == count, "one row per pose, so the trail indexes exactly like the pose track"
    for pose in range(count):
        assert len(strips[pose]) == 1, "one strip per row"
        segment: Float64[ndarray, "2 3"] = np.asarray(strips[pose][0], dtype=np.float64)
        assert segment.shape == (2, 3), "a segment is two points"
        previous: int = max(pose - 1, 0)
        np.testing.assert_allclose(segment, positions_xyz[[previous, pose]], atol=1e-6)


def test_the_first_pose_gets_a_degenerate_segment_so_the_row_count_matches_the_pose_track(tmp_path: Path) -> None:
    """There is no step into the first pose, and skipping it would offset every row by one.

    The alternative — one fewer trail row than pose row — puts the trail's
    cursor-relative window a sample out of step with the rig it trails, which is
    worse than a zero-length strip the viewer draws as nothing.
    """
    target, positions_xyz, _ = logged_trail(tmp_path, 6)

    first: Float64[ndarray, "2 3"] = np.asarray(trail_strips(target)[0][0], dtype=np.float64)

    np.testing.assert_allclose(first[0], first[1], atol=1e-12)
    np.testing.assert_allclose(first[0], positions_xyz[0], atol=1e-6)


def test_a_trail_states_its_colour_and_a_ui_point_width_once_statically(tmp_path: Path) -> None:
    """One trail is one quantity, so the tint and the width are static, not per row.

    The width is in **ui points**, which Rerun carries as a negative radius: a
    metric radius would have to be re-picked per device (a headset's trail and a
    vehicle's are metres apart in scale), whereas a screen-space stroke reads the
    same at any zoom.
    """
    target, _, _ = logged_trail(tmp_path, 5)

    static: dict[str, list[object]] = read_back(target).reader(index=None, contents=TRAIL_ENTITY).to_arrow_table().to_pylist()[0]

    assert static[f"{TRAIL_ENTITY}:LineStrips3D:radii"] == [-TRAIL_RADIUS_UI_POINTS], "a ui-point radius is carried as its negation"
    colors: list[object] = static[f"{TRAIL_ENTITY}:LineStrips3D:colors"]
    assert len(colors) == 1, "one tint for the whole trail, not one per row"
    # Unpacked rather than round-tripped through the SDK, so the stored bytes are
    # checked against the literal the caller passed.
    packed: object = colors[0]
    assert isinstance(packed, int)
    assert ((packed >> 24) & 0xFF, (packed >> 16) & 0xFF, (packed >> 8) & 0xFF) == TRAIL_COLOR
    assert packed & 0xFF == 0xFF, "opaque"


def test_a_trail_of_one_pose_is_a_single_degenerate_segment(tmp_path: Path) -> None:
    """A one-pose sequence has no step at all; it must still produce one row."""
    target, positions_xyz, _ = logged_trail(tmp_path, 1)

    strips: list[list[list[float]]] = trail_strips(target)

    assert len(strips) == 1
    np.testing.assert_allclose(np.asarray(strips[0][0], dtype=np.float64), positions_xyz[[0, 0]], atol=1e-6)
