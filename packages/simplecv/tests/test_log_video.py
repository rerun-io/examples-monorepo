"""Equivalence and bit-preservation tests for ``log_video``."""

from __future__ import annotations

import socket
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TypeAlias

import av
import numpy as np
import pytest
import rerun as rr
from jaxtyping import Int64, UInt8
from numpy import ndarray
from pyarrow import ChunkedArray

from simplecv.rerun_log_utils import (
    log_video,
    mux_h264_to_mp4,
    read_video_stream_from_rrd,
)

_FRAME_COUNT: int = 12
_WIDTH: int = 64
_HEIGHT: int = 64
_FPS: int = 30
VideoStreamData: TypeAlias = tuple[rr.VideoCodec, ChunkedArray, ChunkedArray]


def _frame_pixels(i: int) -> UInt8[ndarray, "h w 3"]:
    r: int = (i * 23) % 256
    g: int = (i * 71 + 40) % 256
    b: int = (i * 137 + 80) % 256
    frame: UInt8[ndarray, "h w 3"] = np.empty((_HEIGHT, _WIDTH, 3), dtype=np.uint8)
    frame[..., 0] = r
    frame[..., 1] = g
    frame[..., 2] = b
    return frame


@pytest.fixture(scope="session")
def synthetic_h264_mp4(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Deterministic H.264 MP4, no B-frames, lossless (CRF=0)."""
    out_path: Path = tmp_path_factory.mktemp("log_video") / "synthetic.mp4"
    container: av.container.OutputContainer = av.open(str(out_path), mode="w")
    stream: av.video.stream.VideoStream = container.add_stream(
        "h264", rate=_FPS, options={"preset": "ultrafast", "crf": "0"}
    )
    stream.width = _WIDTH
    stream.height = _HEIGHT
    stream.pix_fmt = "yuv420p"
    stream.max_b_frames = 0

    for i in range(_FRAME_COUNT):
        frame: av.VideoFrame = av.VideoFrame.from_ndarray(_frame_pixels(i), format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return out_path


@pytest.fixture(scope="session")
def synthetic_mpeg4_mp4(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Deterministic MPEG-4 Part 2 MP4, matching EPFL HoloLens codec behavior."""
    out_path: Path = tmp_path_factory.mktemp("log_video") / "synthetic-mpeg4.mp4"
    container: av.container.OutputContainer = av.open(str(out_path), mode="w")
    stream: av.video.stream.VideoStream = container.add_stream("mpeg4", rate=_FPS)
    stream.width = _WIDTH
    stream.height = _HEIGHT
    stream.pix_fmt = "yuv420p"
    stream.max_b_frames = 0

    for i in range(_FRAME_COUNT):
        frame: av.VideoFrame = av.VideoFrame.from_ndarray(_frame_pixels(i), format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return out_path


@pytest.fixture(scope="session")
def synthetic_hevc_mp4(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Deterministic HEVC MP4 large enough for the NVENC transcode path."""
    out_path: Path = tmp_path_factory.mktemp("log_video") / "synthetic-hevc.mp4"
    container: av.container.OutputContainer = av.open(str(out_path), mode="w")
    stream: av.video.stream.VideoStream = container.add_stream("libx265", rate=_FPS, options={"preset": "ultrafast"})
    stream.width = _WIDTH * 4
    stream.height = _HEIGHT * 4
    stream.pix_fmt = "yuv420p"
    for i in range(_FRAME_COUNT):
        big_rgb: UInt8[ndarray, "h w 3"] = np.tile(_frame_pixels(i), (4, 4, 1))
        for packet in stream.encode(av.VideoFrame.from_ndarray(big_rgb, format="rgb24")):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return out_path


def _decode_mp4(source: Path) -> list[UInt8[ndarray, "h w 3"]]:
    container: av.container.InputContainer = av.open(str(source), mode="r")
    frames: list[UInt8[ndarray, "h w 3"]] = []
    try:
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
    finally:
        container.close()
    return frames


def _log_to_tmp_rrd(
    mp4: Path,
    label: str,
    tmp_path: Path,
    *,
    entity: str,
    timeline: str,
) -> Path:
    rrd_path: Path = tmp_path / f"{label}.rrd"
    rec: rr.RecordingStream = rr.RecordingStream(
        application_id=f"test-log-video-{label}",
        recording_id=f"test-{label}",
    )
    rec.save(str(rrd_path))
    log_video(mp4, Path(entity), timeline=timeline, recording=rec)
    del rec  # finalizes on drop
    return rrd_path


def _can_open_local_socket() -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM):
            return True
    except OSError:
        return False


def test_log_video_timestamps_match(synthetic_h264_mp4: Path, tmp_path: Path) -> None:
    """Mp4Reader logging returns the source video frame timestamps."""
    rec_stream: rr.RecordingStream = rr.RecordingStream(
        application_id="test-log-video-stream",
        recording_id="ts-stream",
    )
    rec_stream.save(str(tmp_path / "stream.rrd"))

    entity: Path = Path("/video")
    ts_asset: Int64[ndarray, "num_frames"] = rr.AssetVideo(path=synthetic_h264_mp4).read_frame_timestamps_nanos()
    ts_stream: Int64[ndarray, "num_frames"] = log_video(synthetic_h264_mp4, entity, recording=rec_stream)

    assert len(ts_asset) == _FRAME_COUNT
    assert len(ts_stream) == _FRAME_COUNT
    np.testing.assert_array_equal(np.sort(ts_asset), np.sort(ts_stream))


def test_log_video_accepts_quicktime_brand(synthetic_h264_mp4: Path, tmp_path: Path) -> None:
    """iPhone .MOV files carry the ``qt`` brand (sniffed as video/quicktime, which rr.AssetVideo rejects)."""
    mov_path: Path = tmp_path / "clip.MOV"
    with av.open(str(synthetic_h264_mp4)) as src, av.open(str(mov_path), mode="w", format="mov") as dst:
        in_stream: av.video.stream.VideoStream = src.streams.video[0]
        out_stream: av.video.stream.VideoStream = dst.add_stream_from_template(in_stream)
        for packet in src.demux(in_stream):
            if packet.dts is None:
                continue
            packet.stream = out_stream
            dst.mux(packet)
    assert mov_path.read_bytes()[4:12] == b"ftypqt  "

    rec: rr.RecordingStream = rr.RecordingStream(application_id="test-log-video-mov", recording_id="mov")
    rec.save(str(tmp_path / "mov.rrd"))
    ts_stream: Int64[ndarray, "num_frames"] = log_video(mov_path, Path("/video"), recording=rec)
    assert len(ts_stream) == _FRAME_COUNT


def test_log_video_sends_chunks_incrementally(synthetic_h264_mp4: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The recording receives Mp4Reader's lazy iterator instead of a materialized chunk list."""
    received: list[rr.experimental.Chunk] = []

    def consume_lazily(_recording: rr.RecordingStream, chunks: Iterable[rr.experimental.Chunk]) -> None:
        assert not isinstance(chunks, list)
        chunk_iterator: Iterator[rr.experimental.Chunk] = iter(chunks)
        assert chunk_iterator is chunks
        received.extend(chunks)

    monkeypatch.setattr(rr.RecordingStream, "send_chunks", consume_lazily)
    rec: rr.RecordingStream = rr.RecordingStream(application_id="test-log-video-lazy", recording_id="lazy")

    timestamps_ns: Int64[ndarray, "num_frames"] = log_video(synthetic_h264_mp4, Path("/video"), recording=rec)

    assert received
    assert len(timestamps_ns) == _FRAME_COUNT


def test_log_video_reencodes_hevc_to_h264(synthetic_hevc_mp4: Path, tmp_path: Path) -> None:
    """Callers can request H.264 output for browser compatibility."""
    if not _can_open_local_socket():
        pytest.skip("Rerun RRD query server cannot open a local socket in this environment")
    rrd_path: Path = tmp_path / "hevc.rrd"
    rec: rr.RecordingStream = rr.RecordingStream(application_id="test-log-video-hevc", recording_id="hevc")
    rec.save(str(rrd_path))
    ts_stream: Int64[ndarray, "num_frames"] = log_video(
        synthetic_hevc_mp4,
        Path("/video"),
        timeline="video_time",
        recording=rec,
        output_codec=rr.VideoCodec.H264,
    )
    rec.flush()
    rec.disconnect()
    assert len(ts_stream) == _FRAME_COUNT
    stream_data: VideoStreamData = read_video_stream_from_rrd(str(rrd_path), "video", "video_time")
    codec: rr.VideoCodec = stream_data[0]
    assert codec == rr.VideoCodec.H264


def test_log_video_keeps_hevc_without_output_codec(synthetic_hevc_mp4: Path, tmp_path: Path) -> None:
    """The shared helper preserves the source codec unless its caller requests another one."""
    if not _can_open_local_socket():
        pytest.skip("Rerun RRD query server cannot open a local socket in this environment")
    rrd_path: Path = tmp_path / "hevc-preserved.rrd"
    rec: rr.RecordingStream = rr.RecordingStream(application_id="test-log-video-hevc-preserved", recording_id="hevc-preserved")
    rec.save(str(rrd_path))
    ts_stream: Int64[ndarray, "num_frames"] = log_video(
        synthetic_hevc_mp4,
        Path("/video"),
        timeline="video_time",
        recording=rec,
    )
    rec.flush()
    rec.disconnect()

    assert len(ts_stream) == _FRAME_COUNT
    stream_data: VideoStreamData = read_video_stream_from_rrd(str(rrd_path), "video", "video_time")
    codec: rr.VideoCodec = stream_data[0]
    assert codec == rr.VideoCodec.H265


def test_log_video_stream_requires_supported_codec(
    synthetic_mpeg4_mp4: Path,
    tmp_path: Path,
) -> None:
    rec: rr.RecordingStream = rr.RecordingStream(
        application_id="test-log-video-unsupported-codec",
        recording_id="unsupported-codec",
    )
    rec.save(str(tmp_path / "unsupported.rrd"))
    entity: str = "/video"

    with pytest.raises(RuntimeError, match='unsupported codec "mp4v"'):
        log_video(
            synthetic_mpeg4_mp4,
            Path(entity),
            timeline="video_time",
            recording=rec,
        )


def test_log_video_preserves_video_read_failure_context(tmp_path: Path) -> None:
    corrupt_mp4: Path = tmp_path / "corrupt.mp4"
    corrupt_mp4.write_bytes(b"not an mp4")
    rec: rr.RecordingStream = rr.RecordingStream(
        application_id="test-log-video-corrupt",
        recording_id="corrupt",
    )
    rec.save(str(tmp_path / "corrupt.rrd"))

    with pytest.raises(RuntimeError, match="Mp4Reader failed for") as exc_info:
        log_video(corrupt_mp4, Path("/video"), recording=rec)

    assert str(corrupt_mp4) in str(exc_info.value)
    assert exc_info.value.__cause__ is not None


def test_log_video_stream_roundtrip_pixels_match_source(
    synthetic_h264_mp4: Path,
    tmp_path: Path,
) -> None:
    """Lossless synthetic MP4 round-trips through Mp4Reader stream chunks."""
    if not _can_open_local_socket():
        pytest.skip("Rerun RRD query server cannot open a local socket in this environment")

    src_frames: list[UInt8[ndarray, "h w 3"]] = _decode_mp4(synthetic_h264_mp4)
    assert len(src_frames) == _FRAME_COUNT

    entity: str = "/video"
    timeline: str = "video_time"

    stream_rrd: Path = _log_to_tmp_rrd(
        synthetic_h264_mp4,
        "mp4-reader",
        tmp_path,
        entity=entity,
        timeline=timeline,
    )
    stream_data: VideoStreamData = read_video_stream_from_rrd(
        str(stream_rrd),
        entity.lstrip("/"),
        timeline,
    )
    codec: rr.VideoCodec = stream_data[0]
    times: ChunkedArray = stream_data[1]
    samples_chunked: ChunkedArray = stream_data[2]
    assert codec == rr.VideoCodec.H264
    stream_mp4: Path = tmp_path / "stream_remuxed.mp4"
    mux_h264_to_mp4(times, samples_chunked, str(stream_mp4))
    stream_frames: list[UInt8[ndarray, "h w 3"]] = _decode_mp4(stream_mp4)

    assert len(stream_frames) == _FRAME_COUNT
    for i in (0, _FRAME_COUNT // 2, _FRAME_COUNT - 1):
        np.testing.assert_array_equal(src_frames[i], stream_frames[i])
