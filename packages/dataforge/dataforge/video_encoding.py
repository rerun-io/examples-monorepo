"""Shared AV1 encoders for frame iterables and grayscale MP4 transcoding.

Datasets that ship image sequences instead of video get their video here:
``encode_frames_to_mp4`` pipes PNG, JPEG or raw frames straight into ffmpeg's stdin, so
a converter never materializes a decoded frame tree on disk. Two properties are
load-bearing for the Rerun side and are enforced rather than documented — the ban
on B-frames (``rr.VideoStream`` rejects reordered samples) and the sample-count
check against the finished container.
Datasets that ship JPEGs decoded on the CPU (HOT3D) feed planes from ``dataforge.jpeg``.

This module knows nothing about Rerun: it turns frames into an mp4 and counts
what landed. ``dataforge.logging_toolkit`` remuxes that mp4 into a recording, and
re-exports every name here so a converter has one import to make.
"""

from __future__ import annotations

import contextlib
import fcntl
import functools
import os
import shutil
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Literal, TypeAlias, TypeVar

import av

from dataforge import paths
from dataforge.timing import SequenceTimer

AV1_CQ: int = 36
"""Port-wide AV1 NVENC quality (binding brief). Selected on RTX 5090 with SHOW3D: CQ36 is
the smallest tested output above 40 dB median.

Keyboard / birdhouse (MB, dB, seconds): builtin 53.8/44.61/9.9,
117.5/44.77/15.0; CQ28 93.2/45.58/9.6, 188.7/45.43/12.1;
CQ32 60.1/44.19/9.5, 128.0/44.03/12.1; CQ36 37.2/43.05/9.5,
87.6/42.58/12.0. Source: 62.4 / 127.4 MB; 20 frames per camera.
"""
AV1_GOP: int = 60
"""Port-wide AV1 NVENC keyframe interval in frames (binding brief): one second at
SHOW3D's 60 fps, two at HO-Cap's 30 Hz."""
MAX_STORED_PIXELS: int = 3_200_000
"""Videos above this many pixels are stored at half size, repeatedly (Pablo's ~3 MP cap): MAMMA's 4K iPhones at 1920x1080 and
eval Singles' 4112x3008 at 2056x1504, Harmony4D's 3840x2160 GoPros at 1920x1080 (libjpeg's DCT scaling)."""


def halvings(width: int, height: int) -> int:
    """How many times a ``width`` x ``height`` video is halved to fit ``MAX_STORED_PIXELS``."""
    count: int = 0
    while (width >> count) * (height >> count) > MAX_STORED_PIXELS:
        count += 1
    return count


@contextlib.contextmanager
def work_dir(prefix: str) -> Iterator[Path]:
    """A private scratch directory under ``paths.work_root()``, removed on exit."""
    root: Path = paths.work_root()
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=prefix, dir=root) as work:
        yield Path(work)


def remux_prefix(source: Path, target: Path, count: int) -> None:
    """Copy the first N packets of the first video stream into an MP4 without decoding or encoding."""
    with av.open(str(source)) as container, av.open(str(target), "w") as output:
        stream = container.streams.video[0]
        destination = output.add_stream_from_template(stream, opaque=True)
        seen: int = 0
        for packet in container.demux(stream):
            if packet.pts is None:
                continue
            if seen >= count:
                break
            packet.stream = destination
            output.mux(packet)
            seen += 1
        if seen != count:
            raise ValueError(f"{source}: only {seen} packets, expected {count}")


T = TypeVar("T")


def encode_concurrently(jobs: Sequence[Callable[[], T]], *, workers: int | None = None) -> list[T]:  # noqa: UP047 - keep Python 3.10/3.11 syntax, as for UP046
    """Encode independent jobs up to the slot limit; return results in job order."""
    if not jobs:
        return []
    with ThreadPoolExecutor(max_workers=min(len(jobs), workers or nvenc_slot_count())) as pool:
        futures: list[Future[T]] = [pool.submit(job) for job in jobs]
        return [future.result() for future in futures]


@contextlib.contextmanager
def parallel_clips(jobs: list[tuple[Path, Callable[[], None]]], timer: SequenceTimer, *, workers: int = 3) -> Iterator[Iterator[Path]]:
    """Encode with the requested workers; yield clips in submission order while later jobs run.

    Each clip is deleted when the caller requests the next one, so at most one finished clip waits on disk
    for logging; a job may write no clip (nothing to encode). Cleanup waits for encoders and removes every
    leftover, including on failure.
    Transcode measures first submit to final encode completion, excluding logging.
    """
    finished: list[float] = []

    def encode(job: Callable[[], None]) -> None:
        try:
            job()
        finally:
            finished.append(perf_counter())

    started: float = perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures: list[Future[None]] = [executor.submit(encode, job) for _, job in jobs]

            def ready() -> Iterator[Path]:
                for (clip, _), future in zip(jobs, futures, strict=True):
                    future.result()
                    yield clip
                    clip.unlink(missing_ok=True)

            yield ready()
    finally:
        if finished:
            timer.add("transcode", max(finished) - started)
        for clip, _ in jobs:
            clip.unlink(missing_ok=True)


FrameKind: TypeAlias = Literal["hevc", "png", "jpeg", "gray8", "rgb24", "yuv420p", "yuv422p", "yuv444p"]
"""How one element of an encoder frame iterable is laid out."""

RAW_PIXEL_FORMATS: dict[FrameKind, str] = {"gray8": "gray", "rgb24": "rgb24", "yuv420p": "yuv420p", "yuv422p": "yuv422p", "yuv444p": "yuv444p"}
"""ffmpeg ``-pix_fmt`` name for each rawvideo frame kind."""

ENCODED_INPUTS: dict[FrameKind, tuple[str, ...]] = {
    "png": ("-f", "image2pipe", "-framerate", "{fps}", "-c:v", "png"),
    "jpeg": ("-f", "image2pipe", "-framerate", "{fps}", "-c:v", "mjpeg"),
    "hevc": ("-f", "hevc", "-r", "{fps}"),
}
"""ffmpeg input options for each self-describing frame kind (encoded images, a raw Annex-B stream)."""

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
    """``"png"`` / ``"jpeg"`` feed encoded image bytes through ``image2pipe``, ``"hevc"`` an Annex-B stream; the raw kinds feed ``rawvideo`` planes."""
    width: int | None = None
    """Frame width in pixels; required for the raw kinds, which carry no header."""
    height: int | None = None
    """Frame height in pixels; required for the raw kinds, which carry no header."""

    full_range: bool = False
    """JPEG colour planes require explicit full-to-limited range conversion."""

    def __post_init__(self) -> None:
        if self.kind in ENCODED_INPUTS:
            return
        if self.width is None:
            raise ValueError(f"a {self.kind} source needs an explicit width: rawvideo frames carry no header")
        if self.height is None:
            raise ValueError(f"a {self.kind} source needs an explicit height: rawvideo frames carry no header")

    def input_args(self, *, fps: int) -> list[str]:
        """ffmpeg input-side arguments that describe this layout on ``pipe:0``."""
        if self.kind in ENCODED_INPUTS:
            return [*(arg.format(fps=fps) for arg in ENCODED_INPUTS[self.kind]), "-i", "pipe:0"]
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


@functools.cache
def require_av1_nvenc(ffmpeg: Path) -> None:
    """Refuse an ffmpeg that cannot encode AV1 on the GPU, before any frame is read.

    Checked up front because the alternative failure is a software AV1 encode
    that takes hours on a full sequence and looks like a hang. Cached per binary
    path: a batch run asks once per camera and the answer cannot change under it.
    Only the *pass* is cached — ``cache`` stores no entry for a call that
    raised, so a rejected binary is re-interrogated (and re-rejected) every time.

    Args:
        ffmpeg: Binary to interrogate with ``-encoders``.
    """
    listed: subprocess.CompletedProcess[str] = subprocess.run([str(ffmpeg), "-hide_banner", "-encoders"], capture_output=True, text=True, check=False)
    if "av1_nvenc" in listed.stdout:
        return
    raise RuntimeError(f"{ffmpeg} lists no av1_nvenc encoder; point DATAFORGE_FFMPEG at an NVENC-capable ffmpeg")


NVENC_SLOT_DIR: Path = Path("/tmp/dataforge-nvenc")
"""Shared flock files; all dataforge processes on the machine use the same directory."""


def nvenc_slot_count() -> int:
    """Read the shared encoder capacity, rejecting invalid settings."""
    value: str = os.environ.get("DATAFORGE_NVENC_SLOTS", "6")
    try:
        slots: int = int(value)
    except ValueError:
        raise ValueError(f"DATAFORGE_NVENC_SLOTS must be a positive int, got {value!r}") from None
    if slots <= 0:
        raise ValueError(f"DATAFORGE_NVENC_SLOTS must be a positive int, got {value!r}")
    return slots


@contextlib.contextmanager
def _single_nvenc_slot(slots: int) -> Iterator[int]:
    """Wait for one locked descriptor for the child to inherit."""
    started: float = time.monotonic()
    warned: bool = False
    while True:
        for index in range(slots):
            with (NVENC_SLOT_DIR / f"slot-{index}.lock").open("a+b") as slot:
                try:
                    fcntl.flock(slot, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                try:
                    yield slot.fileno()
                finally:
                    fcntl.flock(slot, fcntl.LOCK_UN)
                return
        if not warned and time.monotonic() - started > 60.0:
            print("NVENC: waiting more than 60 s for a dataforge encoder slot", flush=True)
            warned = True
        time.sleep(0.2)


@contextlib.contextmanager
def nvenc_slot(count: int = 1) -> Iterator[tuple[int, ...]]:
    """Reserve all requested encoder slots before starting the child."""
    slots: int = nvenc_slot_count()
    if not 1 <= count <= slots:
        raise ValueError(f"requested {count} NVENC slots, capacity is {slots}")
    NVENC_SLOT_DIR.mkdir(parents=True, exist_ok=True)
    with contextlib.ExitStack() as stack:
        # Serialize acquisition so batches cannot each hold part of a reservation.
        with (NVENC_SLOT_DIR / "reservation.lock").open("a+b") as reservation:
            fcntl.flock(reservation, fcntl.LOCK_EX)
            fds: tuple[int, ...] = tuple(stack.enter_context(_single_nvenc_slot(slots)) for _ in range(count))
            fcntl.flock(reservation, fcntl.LOCK_UN)
        yield fds


def nvenc_session_failure(stderr: str) -> bool:
    """Driver session-limit messages that need an encode retry, never a decode fallback."""
    return any(message in stderr for message in ("OpenEncodeSessionEx failed", "incompatible client key"))


def run_nvenc(run: Callable[[tuple[int, ...]], subprocess.CompletedProcess[str]], *, slots: int = 1) -> subprocess.CompletedProcess[str]:
    """Run an encoder in a shared slot; on a session-limit failure release it, back off 2-32 s and retry."""
    for delay in (2.0, 4.0, 8.0, 16.0, 32.0):
        with nvenc_slot(slots) as fds:
            result: subprocess.CompletedProcess[str] = run(fds)
        if result.returncode == 0 or not nvenc_session_failure(result.stderr):
            return result
        time.sleep(delay)
    with nvenc_slot(slots) as fds:
        return run(fds)


def _nvenc_args(*, gop: int, cq: int) -> list[str]:
    """Shared output options for pipe and file inputs; never emit B-frames."""
    return [
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
    ]


CUDA_FILTERS: dict[Path, bool] = {}
CUDA_FILTER_LOCK = threading.Lock()


def has_scale_cuda(binary: Path) -> bool:
    """Whether this ffmpeg build can scale hardware frames; probe once across camera threads."""
    with CUDA_FILTER_LOCK:
        if binary not in CUDA_FILTERS:
            result: subprocess.CompletedProcess[str] = subprocess.run(
                [str(binary), "-hide_banner", "-filters"], capture_output=True, text=True, check=True
            )
            CUDA_FILTERS[binary] = any("scale_cuda" in line.split() for line in result.stdout.splitlines())
        return CUDA_FILTERS[binary]


def transcode_mp4(
    source: Path,
    output: Path,
    *,
    gop: int,
    cq: int,
    fps: int,
    frames: int,
    gray: bool,
    size: tuple[int, int] | None = None,
    crop: tuple[int, int, int, int] | None = None,
    keep: list[range] | None = None,
    decode: Literal["cpu", "cuda"] = "cpu",
) -> int:
    """Decode a file and encode AV1 directly in ffmpeg, checking sample count.

    frames is the exact expected output count; source timing is applied by the caller.
    keep, runs of evenly spaced source frames, keeps only those frames, before any other
    filter, for a stream whose real samples are padded to a higher frame rate.
    The input -r assigns nominal timestamps without dropping or duplicating frames.
    gray drops chroma on the CPU path; the CUDA path encodes the decoded planes as they are,
    which for a gray source carry neutral chroma, converted to 4:2:0. size (width, height) rescales on the GPU
    when scale_cuda is available; builds without it use CPU decode directly. Optional crop is (width, height, x, y), applied first, and always
    uses CPU decode. Other NVDEC failures fall back to CPU; session failures retry the same
    command and never trigger that fallback.
    """
    if frames <= 0:
        raise ValueError("frames must be positive")
    if crop is not None and (min(crop[:2]) <= 0 or min(crop[2:]) < 0):
        raise ValueError("crop requires positive dimensions and nonnegative offsets")
    runs: str = "+".join(f"between(n,{run.start},{run[-1]})*not(mod(n-{run.start},{run.step}))" for run in keep or [])
    select: list[str] = [f"select='{runs}'"] if keep is not None else []
    cpu_filters: list[str] = [
        *select,
        *([] if crop is None else ["crop=" + ":".join(str(value) for value in crop)]),
        *([] if size is None else [f"scale={size[0]}:{size[1]}"]),
        *(["format=gray"] if gray else []),
        EVEN_DIMENSION_AND_PIXEL_FORMAT,
    ]
    # NVENC's AV1 takes 4:2:0 only, as the CPU path's last filter ensures: a 4:4:4 source (MAMMA's HEVC) fails without this.
    # nv12 is what NVDEC hands over for 8-bit 4:2:0, so for such a source scale_cuda passes the frames through untouched.
    cuda_filters: list[str] = [*select, f"scale_cuda={'' if size is None else f'{size[0]}:{size[1]}:'}format=nv12"]
    binary: Path = resolve_ffmpeg()
    require_av1_nvenc(binary)
    def run(*, cuda: bool) -> subprocess.CompletedProcess[str]:
        filters: list[str] = cuda_filters if cuda else cpu_filters
        command: list[str] = [
            str(binary), "-hide_banner", "-loglevel", "error", "-y",
            *(["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"] if cuda else []),
            "-r", str(fps), "-i", str(source), "-map", "0:v:0", "-an",
            *(["-vf", ",".join(filters)] if filters else []),
            "-fps_mode", "passthrough", *_nvenc_args(gop=gop, cq=cq),
            "-frames:v", str(frames), str(output),
        ]
        return run_nvenc(lambda fds: subprocess.run(command, pass_fds=fds, capture_output=True, text=True, check=False))

    cuda: bool = decode == "cuda" and crop is None and has_scale_cuda(binary)
    result: subprocess.CompletedProcess[str] = run(cuda=cuda)
    if cuda and result.returncode and not nvenc_session_failure(result.stderr):
        tail: str = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "no stderr"
        print(f"{source}: NVDEC path failed ({tail}); CPU decode")
        result = run(cuda=False)
    if result.returncode:
        raise RuntimeError(f"ffmpeg exited {result.returncode} while transcoding {source}:\n{result.stderr[-4000:]}")
    written: int = mp4_frame_count(output)
    if written != frames:
        raise ValueError(f"{output} holds {written} samples, expected {frames}")
    return written


def transcode_crops(
    source: Path,
    outputs: list[Path],
    *,
    crops: list[tuple[int, int, int, int]],
    fps: int,
    frames: int,
    gop: int,
    cq: int,
) -> None:
    """Share one CPU decode across cropped gray outputs, batching at the NVENC slot limit.

    Four tiles decode once with four slots, twice with three. Each batch reserves
    all its encoder slots before ffmpeg starts; concurrent batches cannot deadlock
    by each holding part of a reservation. Output order and frame counts survive.
    """
    if not outputs or len(outputs) != len(crops) or frames <= 0:
        raise ValueError("positive frame count and one crop per output required")
    if any(min(crop[:2]) <= 0 or min(crop[2:]) < 0 for crop in crops):
        raise ValueError("crop requires positive dimensions and nonnegative offsets")
    slots: int = nvenc_slot_count()
    binary: Path = resolve_ffmpeg()
    require_av1_nvenc(binary)
    for start in range(0, len(outputs), slots):
        batch: list[Path] = outputs[start:start + slots]
        filters: list[str] = [f"[0:v]split={len(batch)}" + "".join(f"[tile{i}]" for i in range(len(batch)))]
        command: list[str] = [str(binary), "-hide_banner", "-loglevel", "error", "-y", "-filter_complex_threads", "1", "-r", str(fps), "-i", str(source)]
        for index, crop in enumerate(crops[start:start + slots]):
            filters.append(f"[tile{index}]crop=" + ":".join(str(value) for value in crop) + f",format=gray,{EVEN_DIMENSION_AND_PIXEL_FORMAT}[cam{index}]")
        command.extend(["-filter_complex", ";".join(filters)])
        for index, output in enumerate(batch):
            command.extend(["-map", f"[cam{index}]", "-an", "-fps_mode", "passthrough", *_nvenc_args(gop=gop, cq=cq), "-frames:v", str(frames), str(output)])
        result: subprocess.CompletedProcess[str] = run_nvenc(
            lambda fds, command=command: subprocess.run(command, pass_fds=fds, capture_output=True, text=True, check=False), slots=len(batch)
        )
        if result.returncode:
            raise RuntimeError(f"ffmpeg exited {result.returncode} while cropping {source}:\n{result.stderr[-4000:]}")
        for output in batch:
            written: int = mp4_frame_count(output)
            if written != frames:
                raise ValueError(f"{output} holds {written} samples, expected {frames}")


def encode_frames_to_mp4(
    frames: Iterable[bytes | memoryview],
    output: Path,
    *,
    source: FrameSource,
    fps: int,
    gop: int = 30,
    cq: int = 32,
    rotate_cw_quarter_turns: int = 0,
    ffmpeg: Path | None = None,
    filter_threads: int | None = None,
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
        frames: One encoded PNG/JPEG (``kind="png"``/``"jpeg"``) or all packed raw planes for one frame.
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
        filter_threads: CPU filter workers; cap when camera jobs also decode in parallel.

    Returns:
        Number of frames fed into the encoder.

    Raises:
        ValueError: ``rotate_cw_quarter_turns`` is not one of 0, 1, 2, 3.
    """
    if rotate_cw_quarter_turns not in TRANSPOSE_FILTERS:
        raise ValueError(f"{rotate_cw_quarter_turns} is not a clockwise quarter turn count; it must be one of {sorted(TRANSPOSE_FILTERS)}")
    if filter_threads is not None and filter_threads < 1:
        raise ValueError("filter_threads must be positive")
    binary: Path = resolve_ffmpeg() if ffmpeg is None else ffmpeg
    require_av1_nvenc(binary)
    command: list[str] = [
        str(binary),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        *(["-filter_threads", str(filter_threads)] if filter_threads is not None else []),
        *source.input_args(fps=fps),
        "-vf",
        ",".join(
            [
                *TRANSPOSE_FILTERS[rotate_cw_quarter_turns],
                *(["scale=in_range=full:out_range=limited"] if source.full_range else []),
                EVEN_DIMENSION_AND_PIXEL_FORMAT,
            ]
        ),
        *_nvenc_args(gop=gop, cq=cq),
        str(output),
    ]
    complaints: list[bytes] = []
    fed: int = 0
    # One slot, no retry: a pipe consumes its frames once, so a session refused by an outside process fails the sequence.
    with nvenc_slot() as fds:
        process: subprocess.Popen[bytes] = subprocess.Popen(command, pass_fds=fds, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
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


def video_size(path: Path) -> tuple[int, int]:
    """Width and height of an mp4's first video stream."""
    with av.open(str(path)) as container:
        stream: av.video.stream.VideoStream = container.streams.video[0]
        return stream.codec_context.width, stream.codec_context.height
