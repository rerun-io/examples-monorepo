"""The pipe-fed AV1 encoder every dataforge converter shares, and the mp4 it produces.

Datasets that ship image sequences instead of video get their video here:
``encode_frames_to_mp4`` pipes PNG or raw frames straight into ffmpeg's stdin, so
a converter never materializes a decoded frame tree on disk. Two properties are
load-bearing for the Rerun side and are enforced rather than documented — the ban
on B-frames (``rr.VideoStream`` rejects reordered samples) and the sample-count
check against the finished container.

This module knows nothing about Rerun: it turns frames into an mp4 and counts
what landed. ``dataforge.logging_toolkit`` remuxes that mp4 into a recording, and
re-exports every name here so a converter has one import to make.
"""

from __future__ import annotations

import contextlib
import functools
import os
import shutil
import subprocess
import threading
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

import av

FrameKind: TypeAlias = Literal["png", "gray8", "rgb24"]
"""How one element of an encoder frame iterable is laid out."""

RAW_PIXEL_FORMATS: dict[FrameKind, str] = {"gray8": "gray", "rgb24": "rgb24"}
"""ffmpeg ``-pix_fmt`` name for each rawvideo frame kind."""

TRANSPOSE_FILTERS: dict[int, tuple[str, ...]] = {
    0: (),
    1: ("transpose=1",),
    2: ("transpose=1", "transpose=1"),
    3: ("transpose=2",),
}
"""``-vf`` stages that turn a frame that many quarters **clockwise**.

ffmpeg's ``transpose=1`` is the clockwise quarter and ``transpose=2`` the
counter-clockwise one, so three clockwise quarters go through one
counter-clockwise stage rather than three clockwise ones. There is no half-turn
transpose, hence the pair. Which way "clockwise" turns is pinned to
``np.rot90(frame, k=-1)`` by ``test_encoding``, because
``dataforge.basalt.rotate_camera_cw`` remaps a calibration against that same
reference and the two must agree.
"""
EVEN_DIMENSION_AND_PIXEL_FORMAT: str = "pad=ceil(iw/2)*2:ceil(ih/2)*2,format=yuv420p"
"""The ``-vf`` tail every encode ends with; a rotation stage goes *ahead* of it,
so the pad rounds up the dimensions the file actually carries."""


@dataclass(frozen=True, slots=True)
class FrameSource:
    """How the caller's frame iterable is laid out for ffmpeg's stdin."""

    kind: FrameKind
    """``"png"`` feeds encoded PNG bytes through ``image2pipe``; the raw kinds feed ``rawvideo`` planes."""
    width: int | None = None
    """Frame width in pixels; required for the raw kinds, which carry no header."""
    height: int | None = None
    """Frame height in pixels; required for the raw kinds, which carry no header."""

    def __post_init__(self) -> None:
        if self.kind == "png":
            return
        if self.width is None:
            raise ValueError(f"a {self.kind} source needs an explicit width: rawvideo frames carry no header")
        if self.height is None:
            raise ValueError(f"a {self.kind} source needs an explicit height: rawvideo frames carry no header")

    def input_args(self, *, fps: int) -> list[str]:
        """ffmpeg input-side arguments that describe this layout on ``pipe:0``."""
        if self.kind == "png":
            return ["-f", "image2pipe", "-framerate", str(fps), "-c:v", "png", "-i", "pipe:0"]
        return [
            "-f",
            "rawvideo",
            "-pix_fmt",
            RAW_PIXEL_FORMATS[self.kind],
            "-s",
            f"{self.width}x{self.height}",
            "-framerate",
            str(fps),
            "-i",
            "pipe:0",
        ]


def resolve_ffmpeg() -> Path:
    """Locate the ffmpeg to encode with: ``DATAFORGE_FFMPEG`` first, then ``PATH``."""
    override: str | None = os.environ.get("DATAFORGE_FFMPEG")
    if override:
        return Path(override)
    found: str | None = shutil.which("ffmpeg")
    if found is None:
        raise FileNotFoundError("no ffmpeg on PATH; set DATAFORGE_FFMPEG to an NVENC-capable binary")
    return Path(found)


@functools.lru_cache
def require_av1_nvenc(ffmpeg: Path) -> None:
    """Refuse an ffmpeg that cannot encode AV1 on the GPU, before any frame is read.

    Checked up front because the alternative failure is a software AV1 encode
    that takes hours on a full sequence and looks like a hang. Cached per binary
    path: a batch run asks once per camera and the answer cannot change under it.
    Only the *pass* is cached — ``lru_cache`` stores no entry for a call that
    raised, so a rejected binary is re-interrogated (and re-rejected) every time.

    Args:
        ffmpeg: Binary to interrogate with ``-encoders``.
    """
    listed: subprocess.CompletedProcess[str] = subprocess.run(
        [str(ffmpeg), "-hide_banner", "-encoders"], capture_output=True, text=True, check=False
    )
    if "av1_nvenc" in listed.stdout:
        return
    raise RuntimeError(f"{ffmpeg} lists no av1_nvenc encoder; point DATAFORGE_FFMPEG at an NVENC-capable ffmpeg")


def encode_frames_to_mp4(
    frames: Iterable[bytes],
    output: Path,
    *,
    source: FrameSource,
    fps: int,
    gop: int = 30,
    cq: int = 32,
    rotate_cw_quarter_turns: int = 0,
    ffmpeg: Path | None = None,
) -> int:
    """Encode an iterable of frames into an AV1 mp4 by piping them through ffmpeg.

    Nothing is written to disk but the mp4: a dataset that ships PNG or raw
    frames streams straight from its archive into ffmpeg's stdin. Two properties
    are load-bearing for the Rerun side:

    * **No B-frames** (``-bf 0``). ``rr.VideoStream`` rejects reordered samples,
      and ``Mp4Reader`` would otherwise have to re-encode the file it was just
      handed.
    * **Sample count is verified** against the mp4 after ffmpeg exits, so a
      short pipe (a truncated archive, a dead encoder) fails here rather than as
      a silent timestamp/sample misalignment in ``log_video_stream``.

    ffmpeg's stderr is drained by a thread while frames go into its stdin: both
    pipes are finite, so writing a large frame while stderr sits full deadlocks.

    Args:
        frames: One encoded PNG (``kind="png"``) or one raw plane per frame.
        output: mp4 to write; its parent directory must exist.
        source: Layout of the ``frames`` elements.
        fps: Nominal frame rate stamped into the container. Real per-sample
            timestamps are applied later by ``log_video_stream(times_ns=...)``.
        gop: Keyframe interval in frames.
        cq: NVENC constant-quality target; lower is bigger and better.
        rotate_cw_quarter_turns: Quarter turns **clockwise** to rotate every
            frame by before encoding, 0 to 3 — the same direction as
            ``np.rot90(frame, k=-turns)``. An odd count swaps the mp4's width
            and height, and a caller that also logs a calibration for these
            pixels must roll it the same way (``basalt.rotate_camera_cw``).
        ffmpeg: Binary to use; ``None`` resolves via ``resolve_ffmpeg()``.

    Returns:
        Number of frames fed into the encoder.

    Raises:
        ValueError: ``rotate_cw_quarter_turns`` is not one of 0, 1, 2, 3.
    """
    if rotate_cw_quarter_turns not in TRANSPOSE_FILTERS:
        raise ValueError(f"{rotate_cw_quarter_turns} is not a clockwise quarter turn count; it must be one of {sorted(TRANSPOSE_FILTERS)}")
    binary: Path = resolve_ffmpeg() if ffmpeg is None else ffmpeg
    require_av1_nvenc(binary)
    command: list[str] = [
        str(binary),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        *source.input_args(fps=fps),
        "-vf",
        ",".join([*TRANSPOSE_FILTERS[rotate_cw_quarter_turns], EVEN_DIMENSION_AND_PIXEL_FORMAT]),
        "-c:v",
        "av1_nvenc",
        "-preset",
        "p4",
        "-rc",
        "vbr",
        "-cq",
        str(cq),
        "-bf",
        "0",
        "-g",
        str(gop),
        "-movflags",
        "+faststart",
        str(output),
    ]
    complaints: list[bytes] = []
    fed: int = 0
    process: subprocess.Popen[bytes] = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    assert process.stdin is not None and process.stderr is not None
    drain: threading.Thread = threading.Thread(target=lambda: complaints.append(process.stderr.read()), daemon=True)  # pyrefly: ignore
    drain.start()
    try:
        for frame in frames:
            process.stdin.write(frame)
            fed += 1
    except BrokenPipeError:
        pass  # ffmpeg already died; its stderr below says why
    finally:
        # Closing flushes, so an encoder that already died would raise here and
        # mask the RuntimeError below that carries its stderr.
        with contextlib.suppress(BrokenPipeError):
            process.stdin.close()
        returncode: int = process.wait()
        drain.join()
    if returncode != 0:
        stderr_text: str = b"".join(complaints).decode(errors="replace").strip()
        raise RuntimeError(f"ffmpeg exited {returncode} while encoding {output.name} after {fed} frames:\n{stderr_text}")

    written: int = mp4_frame_count(output)
    if written != fed:
        raise ValueError(f"{output} holds {written} samples but {fed} frames were fed; the pipe lost data")
    return fed


def mp4_frame_count(path: Path) -> int:
    """Number of video samples in an mp4, from the container index."""
    with av.open(str(path)) as container:
        stream: av.video.stream.VideoStream = container.streams.video[0]
        if stream.frames:
            return stream.frames
        return sum(1 for packet in container.demux(stream) if packet.pts is not None)


def encode_image_files_to_mp4(paths: Sequence[Path], output: Path, *, fps: int, gop: int = 30, cq: int = 32, ffmpeg: Path | None = None) -> int:
    """Encode a PNG sequence already on disk, reading one file at a time.

    Args:
        paths: PNG files in presentation order.
        output: mp4 to write.
        fps: Nominal frame rate; see ``encode_frames_to_mp4``.
        gop: Keyframe interval in frames.
        cq: NVENC constant-quality target.
        ffmpeg: Binary to use; ``None`` resolves via ``resolve_ffmpeg()``.

    Returns:
        Number of frames fed into the encoder.
    """
    return encode_frames_to_mp4(
        (path.read_bytes() for path in paths), output, source=FrameSource("png"), fps=fps, gop=gop, cq=cq, ffmpeg=ffmpeg
    )
