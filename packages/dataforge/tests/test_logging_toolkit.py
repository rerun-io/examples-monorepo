"""The shared writers: the time column, video remux timing, and the magnetometer node.

Every assertion reads the written rrd back through the public reader
(``RrdReader`` → ``ChunkStore`` → a datafusion view over one index), so these
test what a consumer sees rather than what the writer intended.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
import rerun as rr
from beartype.roar import BeartypeException
from conftest import column_rows, png_frame, read_back
from jaxtyping import Float64, Int64
from numpy import ndarray

from dataforge import schema
from dataforge.logging_toolkit import (
    FrameSource,
    ImuChannel,
    encode_frames_to_mp4,
    log_magnetometer,
    log_video_stream,
    time_column,
)

NUM_FRAMES: int = 24
WIDTH: int = 321
HEIGHT: int = 193
FPS: int = 30
ENTITY: str = "/world/rig_00/cam_00/pinhole/video"


def png_bytes() -> Iterator[bytes]:
    """A short synthetic clip as PNG bytes; every frame differs."""
    for index in range(NUM_FRAMES):
        yield png_frame(index, width=WIDTH, height=HEIGHT)


@pytest.fixture(scope="module")
def clip(tmp_path_factory, nvenc_ffmpeg: Path) -> Path:
    """One AV1 mp4 of ``NUM_FRAMES`` samples; skipped when the GPU encoder is absent."""
    output: Path = tmp_path_factory.mktemp("clip") / "clip.mp4"
    encode_frames_to_mp4(png_bytes(), output, source=FrameSource("png"), fps=FPS, gop=10, ffmpeg=nvenc_ffmpeg)
    return output


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
        with pytest.raises(ValueError, match="more samples than the 21 timestamps"):
            log_video_stream(recording, clip, ENTITY, times_ns=irregular_times_ns(NUM_FRAMES - 3))


def test_too_many_timestamps_is_an_error(tmp_path: Path, clip: Path) -> None:
    target: Path = tmp_path / "long.rrd"
    with rr.RecordingStream("dataforge", recording_id="long") as recording:
        recording.save(target)
        with pytest.raises(ValueError, match="holds 24 samples but 27 timestamps"):
            log_video_stream(recording, clip, ENTITY, times_ns=irregular_times_ns(NUM_FRAMES + 3))


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
