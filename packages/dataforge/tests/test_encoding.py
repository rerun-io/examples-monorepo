"""Pipe-fed AV1 encoder: ffmpeg discovery, the NVENC gate, and both frame sources.

Every encode here is real (there is no fake encoder worth trusting for "no
B-frames" and "the sample count survives"), so the tests skip when the resolved
ffmpeg has no ``av1_nvenc``.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import av
import cv2
import numpy as np
import pytest
from jaxtyping import UInt8
from numpy import ndarray

from dataforge.logging_toolkit import (
    FrameSource,
    encode_frames_to_mp4,
    encode_image_files_to_mp4,
    require_av1_nvenc,
    resolve_ffmpeg,
)

NUM_FRAMES: int = 48
"""Long enough to span two GOPs at the default gop=30, so keyframe cadence is observable."""
WIDTH: int = 321
"""Odd on purpose: the encoder pads to even dimensions, and NVENC has a minimum frame size."""
HEIGHT: int = 193
FPS: int = 24


def gray_frame(index: int) -> UInt8[ndarray, "193 321"]:
    """A horizontal gradient with a moving square, so no two frames are identical."""
    frame: UInt8[ndarray, "193 321"] = np.tile(np.linspace(0, 255, WIDTH, dtype=np.uint8), (HEIGHT, 1))
    left: int = (index * 4) % (WIDTH - 16)
    frame[8:24, left : left + 16] = np.uint8(255 - (index * 5) % 256)
    return frame


def gray_frames() -> Iterator[UInt8[ndarray, "193 321"]]:
    """The synthetic clip, frame by frame."""
    for index in range(NUM_FRAMES):
        yield gray_frame(index)


def png_bytes() -> Iterator[bytes]:
    """The same clip as encoded PNG bytes, the ``image2pipe`` input form."""
    for frame in gray_frames():
        success, buffer = cv2.imencode(".png", frame)
        assert success
        yield buffer.tobytes()


def raw_gray_bytes() -> Iterator[bytes]:
    """The same clip as raw ``gray8`` planes, the ``rawvideo`` input form."""
    for frame in gray_frames():
        yield frame.tobytes()


@pytest.fixture(scope="module")
def nvenc_ffmpeg() -> Path:
    """The resolved ffmpeg, or a skip when this machine cannot encode AV1 on the GPU."""
    ffmpeg: Path = resolve_ffmpeg()
    try:
        require_av1_nvenc(ffmpeg)
    except RuntimeError as error:
        pytest.skip(f"no av1_nvenc: {error}")
    return ffmpeg


def video_stream_facts(path: Path) -> tuple[str, int, list[int]]:
    """Canonical codec name, decoded frame count, and every packet pts, in stream order."""
    pts_values: list[int] = []
    decoded: int = 0
    with av.open(str(path)) as container:
        stream: av.video.stream.VideoStream = container.streams.video[0]
        # ``.name`` is the *decoder* PyAV picked (``libdav1d``); the canonical name is the codec.
        codec_name: str = stream.codec_context.codec.canonical_name
        for packet in container.demux(stream):
            if packet.pts is not None:
                pts_values.append(packet.pts)
            decoded += len(packet.decode())
    return codec_name, decoded, pts_values


# ── ffmpeg discovery and the NVENC gate ───────────────────────────────────


def test_resolve_ffmpeg_honors_the_env_var(tmp_path: Path, monkeypatch) -> None:
    override: Path = tmp_path / "my-ffmpeg"
    override.write_text("#!/bin/sh\n")
    monkeypatch.setenv("DATAFORGE_FFMPEG", str(override))
    assert resolve_ffmpeg() == override


def test_resolve_ffmpeg_names_the_env_var_when_nothing_is_found(monkeypatch) -> None:
    monkeypatch.delenv("DATAFORGE_FFMPEG", raising=False)
    monkeypatch.setenv("PATH", "")
    with pytest.raises(FileNotFoundError, match="DATAFORGE_FFMPEG"):
        resolve_ffmpeg()


def test_require_av1_nvenc_rejects_an_encoder_less_ffmpeg(tmp_path: Path) -> None:
    fake: Path = tmp_path / "ffmpeg"
    fake.write_text("#!/bin/sh\necho ' V..... libsvtav1  SVT-AV1 encoder (codec av1)'\n")
    fake.chmod(0o755)
    with pytest.raises(RuntimeError, match="DATAFORGE_FFMPEG"):
        require_av1_nvenc(fake)
    with pytest.raises(RuntimeError, match="conda-forge"):
        require_av1_nvenc(fake)


def test_encode_fails_before_spawning_when_the_encoder_is_missing(tmp_path: Path) -> None:
    fake: Path = tmp_path / "ffmpeg"
    fake.write_text("#!/bin/sh\necho ' V..... libsvtav1  SVT-AV1 encoder (codec av1)'\n")
    fake.chmod(0o755)
    output: Path = tmp_path / "out.mp4"
    with pytest.raises(RuntimeError, match="av1_nvenc"):
        encode_frames_to_mp4(png_bytes(), output, source=FrameSource("png"), fps=FPS, ffmpeg=fake)
    assert not output.exists()


# ── the two frame sources ─────────────────────────────────────────────────


def test_png_pipe_encodes_every_frame(tmp_path: Path, nvenc_ffmpeg: Path) -> None:
    output: Path = tmp_path / "png.mp4"
    fed: int = encode_frames_to_mp4(png_bytes(), output, source=FrameSource("png"), fps=FPS, ffmpeg=nvenc_ffmpeg)
    assert fed == NUM_FRAMES
    codec_name, decoded, pts_values = video_stream_facts(output)
    assert codec_name == "av1"
    assert decoded == NUM_FRAMES
    assert len(pts_values) == NUM_FRAMES


def test_gray8_rawvideo_encodes_every_frame(tmp_path: Path, nvenc_ffmpeg: Path) -> None:
    output: Path = tmp_path / "gray.mp4"
    source: FrameSource = FrameSource("gray8", width=WIDTH, height=HEIGHT)
    fed: int = encode_frames_to_mp4(raw_gray_bytes(), output, source=source, fps=FPS, ffmpeg=nvenc_ffmpeg)
    assert fed == NUM_FRAMES
    codec_name, decoded, _ = video_stream_facts(output)
    assert codec_name == "av1"
    assert decoded == NUM_FRAMES


def test_encoded_stream_has_no_reordered_samples(tmp_path: Path, nvenc_ffmpeg: Path) -> None:
    """rr.VideoStream rejects reordered samples, so ``-bf 0`` is mandatory."""
    output: Path = tmp_path / "monotonic.mp4"
    encode_frames_to_mp4(raw_gray_bytes(), output, source=FrameSource("gray8", width=WIDTH, height=HEIGHT), fps=FPS, ffmpeg=nvenc_ffmpeg)
    _, _, pts_values = video_stream_facts(output)
    assert len(pts_values) == NUM_FRAMES
    assert all(later > earlier for earlier, later in zip(pts_values, pts_values[1:], strict=False))


def test_gop_sets_the_keyframe_cadence(tmp_path: Path, nvenc_ffmpeg: Path) -> None:
    output: Path = tmp_path / "gop.mp4"
    encode_frames_to_mp4(
        raw_gray_bytes(), output, source=FrameSource("gray8", width=WIDTH, height=HEIGHT), fps=FPS, gop=16, ffmpeg=nvenc_ffmpeg
    )
    with av.open(str(output)) as container:
        keyframes: list[int] = [index for index, packet in enumerate(container.demux(container.streams.video[0])) if packet.is_keyframe]
    assert keyframes == [0, 16, 32]


def test_raw_sources_require_their_dimensions(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="width"):
        FrameSource("gray8")
    with pytest.raises(ValueError, match="height"):
        FrameSource("rgb24", width=WIDTH)


def test_encode_image_files_reads_each_png_from_disk(tmp_path: Path, nvenc_ffmpeg: Path) -> None:
    frame_dir: Path = tmp_path / "frames"
    frame_dir.mkdir()
    paths: list[Path] = []
    for index, blob in enumerate(png_bytes()):
        frame_path: Path = frame_dir / f"{index:05d}.png"
        frame_path.write_bytes(blob)
        paths.append(frame_path)
    output: Path = tmp_path / "files.mp4"
    fed: int = encode_image_files_to_mp4(paths, output, fps=FPS, ffmpeg=nvenc_ffmpeg)
    assert fed == NUM_FRAMES
    codec_name, decoded, _ = video_stream_facts(output)
    assert codec_name == "av1"
    assert decoded == NUM_FRAMES


def test_a_failing_encode_reports_ffmpeg_stderr(tmp_path: Path, nvenc_ffmpeg: Path) -> None:
    """Wrong raw dimensions make ffmpeg exit non-zero; its complaint must survive."""
    output: Path = tmp_path / "bad.mp4"
    source: FrameSource = FrameSource("gray8", width=WIDTH, height=HEIGHT)
    with pytest.raises(RuntimeError) as failure:
        encode_frames_to_mp4([b"\x00" * 7], output, source=source, fps=FPS, ffmpeg=nvenc_ffmpeg)
    assert "ffmpeg" in str(failure.value)


def test_env_var_ffmpeg_is_used_when_no_binary_is_passed(tmp_path: Path, monkeypatch, nvenc_ffmpeg: Path) -> None:
    monkeypatch.setenv("DATAFORGE_FFMPEG", str(nvenc_ffmpeg))
    output: Path = tmp_path / "env.mp4"
    assert encode_frames_to_mp4(png_bytes(), output, source=FrameSource("png"), fps=FPS) == NUM_FRAMES
    assert os.environ["DATAFORGE_FFMPEG"] == str(nvenc_ffmpeg)
