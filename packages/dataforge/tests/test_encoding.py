"""Pipe-fed AV1 encoder: ffmpeg discovery, the NVENC gate, and both frame sources.

Every encode here is real (there is no fake encoder worth trusting for "no
B-frames" and "the sample count survives"), so the tests skip when the resolved
ffmpeg has no ``av1_nvenc``.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

import av
import numpy as np
import pytest
from conftest import gray_frame, png_frame
from jaxtyping import UInt8
from numpy import ndarray

from dataforge.video_encoding import (
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
ROTATE_WIDTH: int = 192
"""Rotation-test frame width; non-square, so an odd quarter turn shows in the dimensions, and even, so the encoder's pad is a no-op."""
ROTATE_HEIGHT: int = 160
"""Rotation-test frame height, likewise even and different from the width."""
ROTATE_FRAMES: int = 8
"""Frames per rotation encode, so a per-frame check is not a one-sample check."""
MARK_PX: int = 24
"""Side of the corner marks that make the frame differ from every rotation of itself."""
ROTATION_ERROR_CEILING: float = 12.0
"""Mean absolute error a correctly rotated AV1 decode must stay under, in gray levels.

The right rotation measures ~0.35 and the wrong one ~84.8 on this frame, so the
two bounds are set an order of magnitude clear of both rather than at the
measurements: what is being checked is which rotation landed, not the encoder's
rate-distortion behaviour.
"""
WRONG_ROTATION_ERROR_FLOOR: float = 30.0
"""Mean absolute error the *wrong* rotation must exceed, so the ceiling above is a real check."""


def png_bytes() -> Iterator[bytes]:
    """The synthetic clip as encoded PNG bytes, the ``image2pipe`` input form."""
    for index in range(NUM_FRAMES):
        yield png_frame(index, width=WIDTH, height=HEIGHT)


def raw_gray_bytes() -> Iterator[bytes]:
    """The same clip as raw ``gray8`` planes, the ``rawvideo`` input form."""
    for index in range(NUM_FRAMES):
        yield gray_frame(index, width=WIDTH, height=HEIGHT).tobytes()


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
    # The message must name both the knob and a binary that works, or the reader
    # is left guessing which ffmpeg to point it at.
    with pytest.raises(RuntimeError, match="DATAFORGE_FFMPEG"):
        require_av1_nvenc(fake)
    with pytest.raises(RuntimeError, match="av1_nvenc"):
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


def marked_frame() -> UInt8[ndarray, "height width"]:
    """A frame that resembles no rotation of itself: a left-to-right ramp plus two opposite corner marks.

    The ramp breaks the horizontal mirror and the white/black corner pair breaks
    the half turn, so a decoded frame can be matched against ``np.rot90`` of this
    one *and* against the wrong rotations of it.
    """
    frame: UInt8[ndarray, "height width"] = np.tile(np.linspace(40, 200, ROTATE_WIDTH, dtype=np.uint8), (ROTATE_HEIGHT, 1))
    frame[:MARK_PX, :MARK_PX] = np.uint8(255)
    frame[-MARK_PX:, -MARK_PX:] = np.uint8(0)
    return frame


def decoded_gray_frames(path: Path) -> list[UInt8[ndarray, "height width"]]:
    """Every frame of an mp4 as a decoded 8-bit gray plane, in presentation order."""
    with av.open(str(path)) as container:
        return [np.asarray(frame.to_ndarray(format="gray"), dtype=np.uint8) for frame in container.decode(video=0)]


def mean_abs_error(left: UInt8[ndarray, "height width"], right: UInt8[ndarray, "height width"]) -> float:
    """Mean absolute per-pixel difference of two equally shaped gray planes, in gray levels."""
    return float(np.abs(left.astype(np.int32) - right.astype(np.int32)).mean())


@pytest.mark.parametrize("turns", [0, 1, 2, 3])
def test_a_rotated_encode_turns_every_frame_that_many_quarters_clockwise(tmp_path: Path, nvenc_ffmpeg: Path, turns: int) -> None:
    """``rotate_cw_quarter_turns=k`` must land exactly where ``np.rot90(frame, k=-k)`` does.

    ``np.rot90`` with a negative k is the clockwise direction, and it is the
    reference ``dataforge.basalt``'s calibration roll was derived against — so
    ffmpeg's ``transpose`` and the intrinsics remap agree on which way
    "clockwise" turns only if this passes.

    The decode is lossy, so the match is a mean-absolute-error bound rather than
    equality; the wrong rotation of the same shape is asserted to be far worse,
    which is what keeps that bound a real check rather than a loose one.
    """
    source: UInt8[ndarray, "height width"] = marked_frame()
    output: Path = tmp_path / f"cw{turns}.mp4"
    fed: int = encode_frames_to_mp4(
        [source.tobytes()] * ROTATE_FRAMES,
        output,
        source=FrameSource("gray8", width=ROTATE_WIDTH, height=ROTATE_HEIGHT),
        fps=FPS,
        rotate_cw_quarter_turns=turns,
        ffmpeg=nvenc_ffmpeg,
    )
    assert fed == ROTATE_FRAMES

    expected: UInt8[ndarray, "rotated_height rotated_width"] = np.ascontiguousarray(np.rot90(source, k=-turns))
    frames: list[UInt8[ndarray, "height width"]] = decoded_gray_frames(output)
    assert len(frames) == ROTATE_FRAMES
    for frame in frames:
        assert frame.shape == expected.shape, f"{turns} quarter turn(s) gave {frame.shape}, not {expected.shape}"
        error: float = mean_abs_error(frame, expected)
        assert error < ROTATION_ERROR_CEILING, f"{turns} quarter turn(s) did not land on np.rot90(k=-{turns}): MAE {error:.2f}"
        # The half turn is the only other rotation of this shape, and the ramp plus
        # the corner marks make it wildly different — so a frame that matched it
        # instead is what a flipped transpose direction would look like.
        wrong: float = mean_abs_error(frame, np.ascontiguousarray(np.rot90(expected, k=2)))
        assert wrong > WRONG_ROTATION_ERROR_FLOOR, f"the half turn away is only {wrong:.2f} off; the frame is too symmetric to check"


def test_an_odd_quarter_turn_swaps_the_encoded_dimensions(tmp_path: Path, nvenc_ffmpeg: Path) -> None:
    """The mp4 a rotated encode writes is H x W, so a consumer sizes its stream off the file."""
    dimensions: dict[int, tuple[int, int]] = {}
    for turns in range(4):
        output: Path = tmp_path / f"size{turns}.mp4"
        encode_frames_to_mp4(
            [marked_frame().tobytes()] * ROTATE_FRAMES,
            output,
            source=FrameSource("gray8", width=ROTATE_WIDTH, height=ROTATE_HEIGHT),
            fps=FPS,
            rotate_cw_quarter_turns=turns,
            ffmpeg=nvenc_ffmpeg,
        )
        with av.open(str(output)) as container:
            stream: av.video.stream.VideoStream = container.streams.video[0]
            dimensions[turns] = (stream.codec_context.width, stream.codec_context.height)
    assert dimensions == {
        0: (ROTATE_WIDTH, ROTATE_HEIGHT),
        1: (ROTATE_HEIGHT, ROTATE_WIDTH),
        2: (ROTATE_WIDTH, ROTATE_HEIGHT),
        3: (ROTATE_HEIGHT, ROTATE_WIDTH),
    }


def test_a_turn_count_that_is_not_a_quarter_turn_is_refused(tmp_path: Path) -> None:
    """Only 0..3 name a turn: 4 would be a silent no-op and -1 a direction the name does not state."""
    output: Path = tmp_path / "unrotatable.mp4"
    for turns in (-1, 4):
        with pytest.raises(ValueError, match="quarter turn"):
            encode_frames_to_mp4([b""], output, source=FrameSource("png"), fps=FPS, rotate_cw_quarter_turns=turns)


def test_a_failing_encode_reports_ffmpeg_stderr(tmp_path: Path, nvenc_ffmpeg: Path) -> None:
    """Wrong raw dimensions make ffmpeg exit non-zero; its complaint must survive."""
    output: Path = tmp_path / "bad.mp4"
    source: FrameSource = FrameSource("gray8", width=WIDTH, height=HEIGHT)
    with pytest.raises(RuntimeError) as failure:
        encode_frames_to_mp4([b"\x00" * 7], output, source=source, fps=FPS, ffmpeg=nvenc_ffmpeg)
    assert "ffmpeg" in str(failure.value)


def test_garbage_frames_report_ffmpeg_stderr(tmp_path: Path, nvenc_ffmpeg: Path) -> None:
    """ffmpeg rejects the first non-PNG frame and closes the pipe under us.

    Closing our end of a pipe whose reader is gone raises too, so this is the
    path where a careless ``finally`` would mask ffmpeg's complaint.
    """
    # Deliberately small: the frames sit in Python's write buffer, so the dead
    # reader is only discovered when close() flushes them.
    output: Path = tmp_path / "garbage.mp4"
    garbage: list[bytes] = [b"\x89not-a-png"] * 4
    with pytest.raises(RuntimeError) as failure:
        encode_frames_to_mp4(garbage, output, source=FrameSource("png"), fps=FPS, ffmpeg=nvenc_ffmpeg)
    message: str = str(failure.value)
    assert "ffmpeg exited" in message
    assert message.strip().splitlines()[1:], f"ffmpeg's own complaint is missing from: {message}"


def test_a_reader_that_dies_before_the_pipe_drains_still_reports_stderr(tmp_path: Path) -> None:
    """An encoder that exits without draining stdin breaks the pipe at close().

    Small payloads sit in Python's write buffer, so the EPIPE surfaces from
    ``close()`` rather than from ``write()``; an unguarded close would replace
    the RuntimeError carrying ffmpeg's stderr with a bare BrokenPipeError.
    """
    quitter: Path = tmp_path / "ffmpeg"
    quitter.write_text(
        '#!/bin/sh\ncase "$*" in *-encoders*) echo " V....D av1_nvenc  NVIDIA NVENC av1 encoder";; *) echo "died early" >&2; exit 3;; esac\n'
    )
    quitter.chmod(0o755)

    def slow_frames() -> Iterator[bytes]:
        """Outlive the child, so the buffered bytes meet a closed pipe at flush."""
        yield b"tiny"
        time.sleep(0.5)
        yield b"tiny"

    output: Path = tmp_path / "dead.mp4"
    with pytest.raises(RuntimeError, match="died early"):
        encode_frames_to_mp4(slow_frames(), output, source=FrameSource("png"), fps=FPS, ffmpeg=quitter)


def test_env_var_ffmpeg_is_used_when_no_binary_is_passed(tmp_path: Path, monkeypatch, nvenc_ffmpeg: Path) -> None:
    monkeypatch.setenv("DATAFORGE_FFMPEG", str(nvenc_ffmpeg))
    output: Path = tmp_path / "env.mp4"
    assert encode_frames_to_mp4(png_bytes(), output, source=FrameSource("png"), fps=FPS) == NUM_FRAMES
