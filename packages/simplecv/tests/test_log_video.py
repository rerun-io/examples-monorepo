"""Equivalence and bit-preservation tests for ``log_video``."""

from __future__ import annotations

import io
import socket
from pathlib import Path

import av
import numpy as np
import pytest
import rerun as rr
from jaxtyping import UInt8
from numpy import ndarray

from simplecv.rerun_log_utils import (
    _normalize_av1_sample_for_rerun,
    extract_asset_video_blob_fast,
    log_video,
    mux_h264_to_mp4,
    read_video_stream_from_rrd,
)

_FRAME_COUNT: int = 12
_WIDTH: int = 64
_HEIGHT: int = 64
_FPS: int = 30


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


def _decode_mp4(source: Path | bytes) -> list[UInt8[ndarray, "h w 3"]]:
    handle: io.BytesIO | str = io.BytesIO(source) if isinstance(source, bytes) else str(source)
    container: av.container.InputContainer = av.open(handle, mode="r")
    frames: list[UInt8[ndarray, "h w 3"]] = []
    try:
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
    finally:
        container.close()
    return frames


def _samples_chunked_to_bytes(samples_chunked) -> list[bytes]:
    out: list[bytes] = []
    for chunk in samples_chunked.iterchunks():
        for row in chunk.to_pylist():
            while isinstance(row, list) and len(row) > 0 and isinstance(row[0], list):
                row = row[0]
            if row is None:
                continue
            out.append(bytes(row))
    return out


def _direct_demux_bsf(mp4: Path) -> list[bytes]:
    """Demux ``mp4`` + ``h264_mp4toannexb`` directly — canonical bytes the
    RRD round-trip must match."""
    container: av.container.InputContainer = av.open(str(mp4), mode="r")
    in_stream: av.video.stream.VideoStream = container.streams.video[0]
    bsf: av.BitStreamFilterContext = av.BitStreamFilterContext("h264_mp4toannexb", in_stream)
    samples: list[bytes] = []
    try:
        for raw in container.demux(in_stream):
            if raw.pts is None or raw.dts is None:
                continue
            for f in bsf.filter(raw):
                if f.pts is None or f.dts is None:
                    continue
                samples.append(bytes(f))
    finally:
        container.close()
    return samples


def _log_to_tmp_rrd(
    mp4: Path,
    method: str,
    tmp_path: Path,
    *,
    entity: str,
    timeline: str,
) -> Path:
    rrd_path: Path = tmp_path / f"{method}.rrd"
    rec: rr.RecordingStream = rr.RecordingStream(
        application_id=f"test-log-video-{method}",
        recording_id=f"test-{method}",
    )
    rec.save(str(rrd_path))
    log_video(mp4, Path(entity), timeline=timeline, method=method, recording=rec)  # type: ignore[arg-type]
    del rec  # finalizes on drop
    return rrd_path


def _find_hocap_video(prefer: str = "hololens") -> Path | None:
    base: Path = Path("data/hocap/sample")
    if not base.exists():
        return None
    candidates: list[Path] = sorted(base.rglob("output.mp4"))
    if not candidates:
        return None
    preferred: list[Path] = [p for p in candidates if prefer in p.parent.name]
    return preferred[0] if preferred else candidates[0]


def _can_open_local_socket() -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM):
            return True
    except OSError:
        return False


def test_aria_gen2_rgb_av1_sequence_header_normalized_for_rerun() -> None:
    """Only the aria-gen2 RGB AV1 sequence header is rewritten for rerun."""
    frame_obu_prefix: bytes = bytes.fromhex("328cb30210011d810618")
    aria_rgb_keyframe_prefix: bytes = bytes.fromhex("0a0c00000062ea7ffbf804330080") + frame_obu_prefix

    normalized_prefix: bytes = _normalize_av1_sample_for_rerun(aria_rgb_keyframe_prefix)

    assert normalized_prefix == (bytes.fromhex("0a0c02000061753ffdfc02198040") + frame_obu_prefix)
    assert len(normalized_prefix) == len(aria_rgb_keyframe_prefix)

    slam_keyframe_prefix: bytes = bytes.fromhex("0a0b0000000ccbfeff80433008") + frame_obu_prefix
    assert _normalize_av1_sample_for_rerun(slam_keyframe_prefix) == slam_keyframe_prefix


def test_log_video_timestamps_match(synthetic_h264_mp4: Path, tmp_path: Path) -> None:
    """Both methods describe the same set of frame timestamps."""
    rec_asset: rr.RecordingStream = rr.RecordingStream(
        application_id="test-log-video-asset",
        recording_id="ts-asset",
    )
    rec_asset.save(str(tmp_path / "asset.rrd"))
    rec_stream: rr.RecordingStream = rr.RecordingStream(
        application_id="test-log-video-stream",
        recording_id="ts-stream",
    )
    rec_stream.save(str(tmp_path / "stream.rrd"))

    entity: Path = Path("/video")
    ts_asset: ndarray = log_video(synthetic_h264_mp4, entity, method="asset_video", recording=rec_asset)
    ts_stream: ndarray = log_video(synthetic_h264_mp4, entity, method="video_stream", recording=rec_stream)

    assert len(ts_asset) == _FRAME_COUNT
    assert len(ts_stream) == _FRAME_COUNT
    np.testing.assert_array_equal(np.sort(ts_asset), np.sort(ts_stream))


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

    with pytest.raises(ValueError, match="not supported by rr.VideoStream"):
        log_video(
            synthetic_mpeg4_mp4,
            Path(entity),
            timeline="video_time",
            method="video_stream",
            recording=rec,
        )


@pytest.mark.parametrize("video_fixture", ["synthetic", "hocap"])
def test_log_video_stream_bytes_are_bit_preserved(
    video_fixture: str,
    synthetic_h264_mp4: Path,
    tmp_path: Path,
) -> None:
    """RRD samples are byte-identical to direct ``demux + h264_mp4toannexb`` output.

    Tested on both a no-B-frame synthetic source and the B-frame hocap source
    (the latter requires DTS-ordered storage to be correct).
    """
    if not _can_open_local_socket():
        pytest.skip("Rerun RRD query server cannot open a local socket in this environment")

    if video_fixture == "synthetic":
        mp4: Path = synthetic_h264_mp4
    else:
        hocap: Path | None = _find_hocap_video()
        if hocap is None:
            pytest.skip("hocap sample not downloaded (run pixi _download-hocap-sample)")
        mp4 = hocap

    entity: str = "/video"
    timeline: str = "video_time"
    stream_rrd: Path = _log_to_tmp_rrd(
        mp4,
        "video_stream",
        tmp_path,
        entity=entity,
        timeline=timeline,
    )

    direct_samples = _direct_demux_bsf(mp4)
    _, _, rrd_samples_arr = read_video_stream_from_rrd(
        str(stream_rrd),
        entity.lstrip("/"),
        timeline,
    )
    rrd_samples: list[bytes] = _samples_chunked_to_bytes(rrd_samples_arr)

    assert len(direct_samples) == len(rrd_samples)
    mismatches: int = sum(1 for a, b in zip(direct_samples, rrd_samples, strict=False) if a != b)
    assert mismatches == 0, f"{mismatches}/{len(rrd_samples)} packets differ — VideoStream is no longer bit-preserving"


def test_log_video_rrd_roundtrip_pixels_match_source(
    synthetic_h264_mp4: Path,
    tmp_path: Path,
) -> None:
    """Lossless synthetic MP4 round-trips bit-identically through both archetypes."""
    if not _can_open_local_socket():
        pytest.skip("Rerun RRD query server cannot open a local socket in this environment")

    src_frames: list[UInt8[ndarray, "h w 3"]] = _decode_mp4(synthetic_h264_mp4)
    assert len(src_frames) == _FRAME_COUNT

    entity: str = "/video"
    timeline: str = "video_time"

    asset_rrd: Path = _log_to_tmp_rrd(
        synthetic_h264_mp4,
        "asset_video",
        tmp_path,
        entity=entity,
        timeline=timeline,
    )
    asset_blob: bytes = extract_asset_video_blob_fast(
        entity.lstrip("/"),
        timeline=timeline,
        rrd_path=asset_rrd,
    )
    asset_frames: list[UInt8[ndarray, "h w 3"]] = _decode_mp4(asset_blob)

    stream_rrd: Path = _log_to_tmp_rrd(
        synthetic_h264_mp4,
        "video_stream",
        tmp_path,
        entity=entity,
        timeline=timeline,
    )
    codec, times, samples_chunked = read_video_stream_from_rrd(
        str(stream_rrd),
        entity.lstrip("/"),
        timeline,
    )
    assert codec == rr.VideoCodec.H264
    stream_mp4: Path = tmp_path / "stream_remuxed.mp4"
    mux_h264_to_mp4(times, samples_chunked, str(stream_mp4))
    stream_frames: list[UInt8[ndarray, "h w 3"]] = _decode_mp4(stream_mp4)

    assert len(asset_frames) == _FRAME_COUNT
    assert len(stream_frames) == _FRAME_COUNT
    for i in (0, _FRAME_COUNT // 2, _FRAME_COUNT - 1):
        np.testing.assert_array_equal(src_frames[i], asset_frames[i])
        np.testing.assert_array_equal(src_frames[i], stream_frames[i])
