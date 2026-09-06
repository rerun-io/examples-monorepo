"""The shared writers: video remux timing, and the magnetometer node.

Every assertion reads the written rrd back through the public reader
(``RrdReader`` → ``ChunkStore`` → a datafusion view over one index), so these
test what a consumer sees rather than what the writer intended.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pytest
import rerun as rr
from jaxtyping import Float64, Int64, UInt8
from numpy import ndarray

from dataforge import schema
from dataforge.logging_toolkit import (
    FrameSource,
    encode_frames_to_mp4,
    log_video_stream,
    require_av1_nvenc,
    resolve_ffmpeg,
)

NUM_FRAMES: int = 24
WIDTH: int = 321
HEIGHT: int = 193
FPS: int = 30
ENTITY: str = "/world/rig_00/cam_00/pinhole/video"


def png_bytes() -> Iterator[bytes]:
    """A short synthetic clip as PNG bytes; every frame differs."""
    for index in range(NUM_FRAMES):
        frame: UInt8[ndarray, "193 321"] = np.tile(np.linspace(0, 255, WIDTH, dtype=np.uint8), (HEIGHT, 1))
        left: int = (index * 4) % (WIDTH - 16)
        frame[8:24, left : left + 16] = np.uint8(255 - (index * 7) % 256)
        success, buffer = cv2.imencode(".png", frame)
        assert success
        yield buffer.tobytes()


@pytest.fixture(scope="module")
def clip(tmp_path_factory) -> Path:
    """One AV1 mp4 of ``NUM_FRAMES`` samples; skipped when the GPU encoder is absent."""
    ffmpeg: Path = resolve_ffmpeg()
    try:
        require_av1_nvenc(ffmpeg)
    except RuntimeError as error:
        pytest.skip(f"no av1_nvenc: {error}")
    output: Path = tmp_path_factory.mktemp("clip") / "clip.mp4"
    encode_frames_to_mp4(png_bytes(), output, source=FrameSource("png"), fps=FPS, gop=10, ffmpeg=ffmpeg)
    return output


def index_column(rrd: Path, index: str = schema.TIMELINE) -> Int64[ndarray, "n_rows"]:
    """Every value of one index in a saved rrd, ascending."""
    store: rr.experimental.ChunkStore = rr.experimental.ChunkStore.from_chunks(rr.experimental.RrdReader(rrd).stream())
    table: pa.Table = store.reader(index=index).to_arrow_table().sort_by(index)
    return table.column(index).combine_chunks().cast(pa.int64()).to_numpy()


def irregular_times_ns(count: int) -> Int64[ndarray, "n_samples"]:
    """Ascending 33 ms steps with per-sample jitter — a real device's capture clock."""
    jitter: Int64[ndarray, "n_samples"] = (np.arange(count, dtype=np.int64) * 7919 % 900_000) - 450_000
    return 1_700_000_000_000_000_000 + np.arange(count, dtype=np.int64) * 33_000_000 + jitter


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
