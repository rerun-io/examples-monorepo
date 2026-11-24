"""The Insta360 lens videos as pictures: where their encoded samples come from, and how they are decoded.

Two sources serve the same decoders (``LensSamples``): the ``.insv`` clip (``insta360_io.InsvRig``), which the base is
written from, and the base itself (``BaseLensVideos``), which the cube and panorama layers read beside the calibration
(``insta360.cameras_from_recording``), so their pixels and camera model come from one place.

The base stores each lens re-encoded to AV1 at most ``LENS_MAX_SIZE`` pixels wide: one halving of the X5's 3840x3840
lenses. It is an exception to dataforge's ~3 MP cap (``video_encoding.MAX_STORED_PIXELS``, which would halve 3840
twice, to 960): 1920 keeps the detail slam-rs tracks, the masks read and the cube faces sample, at a twentieth of the
camera's HEVC rate. A lens is halved by a box filter, and ``insta360_io.InsvReader(max_lens_size=...)`` logs each camera
for the halved image (``insta360_io.halved_camera``: pixel ``p`` lands at ``(p + 0.5) s - 0.5``, ``s = 2^-halvings``).

On CUDA a lens is decoded in simplecv's ``decode_runs`` (at most ``RUN_GOPS`` GOPs each): one NVDEC decoder reads each run front to back, so each sample is
decoded at most once at any stride; ``rebatch`` deals the pictures out ``batch`` at a time.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import av
import numpy as np
import rerun.chunk as rrc
import torch
from jaxtyping import Bool, Int64, UInt8
from numpy import ndarray
from simplecv.catalog_video_codec import SOFTWARE_DECODERS, CatalogCodecName, catalog_codec_name, wrap_mp4
from simplecv.catalog_video_decode import DecodeRun, decode_runs
from simplecv.video_io import bind_cuda_device
from torch import Tensor

from dataforge import schema
from dataforge.datasets.insta360_blueprint import RIG
from dataforge.logging_toolkit import VideoStreamSamples, read_video_stream

if TYPE_CHECKING:
    import torchcodec.decoders  # imported where a decoder is built: its import is slow

LENS_MAX_SIZE: int = 1920
"""Widest lens the base stores: lenses wider are halved until they fit (3840 -> 1920 on the X5)."""
RUN_GOPS: int = 8
"""GOPs one NVDEC decoder holds at most: on the X5's 3840 HEVC a run's samples are about 225 MB per lens, held once as
the muxed MP4 while the decoder reads it (the packed samples are freed when the decoder opens)."""
LENS_FPS: int = 30
"""Nominal rate of the muxed lens segments, the face videos and the panorama; ``log_video_stream`` stamps the real times."""


@runtime_checkable
class LensSamples(Protocol):
    """Where the lens decoders read encoded samples: the ``.insv`` clip (``insta360_io.InsvRig``) or the base (``BaseLensVideos``)."""

    @property
    def codec(self) -> CatalogCodecName:
        """``h264``, ``hevc`` or ``av1``."""
        ...

    def gops(self, lens: int) -> Int64[ndarray, "g 2"]:
        """``[start, stop)`` sample range of each GOP of one lens."""
        ...

    def read_packed(self, lens: int, start: int, stop: int) -> tuple[UInt8[ndarray, " n"], Int64[ndarray, " m"], Bool[ndarray, " k"]]:
        """Samples ``start .. stop - 1`` of one lens: their bytes end to end, the ``k + 1`` sample bounds, the keyframe flags."""
        ...


@dataclass(frozen=True, slots=True)
class BaseLensVideos:
    """The two lens videos of a base recording as encoded samples (``LensSamples``), held in memory, and their stamps."""

    codec: CatalogCodecName
    """The lens videos' codec."""
    times_ns: Int64[ndarray, " n"]
    """``video_time`` of each frameset, nanoseconds."""
    videos: tuple[VideoStreamSamples, VideoStreamSamples]
    """``cam_00``'s and ``cam_01``'s samples."""

    @property
    def num_frames(self) -> int:
        """Framesets in the base."""
        return self.times_ns.size

    def gops(self, lens: int) -> Int64[ndarray, "g 2"]:
        """``[start, stop)`` of each GOP of one lens, from its keyframe flags."""
        starts: Int64[ndarray, " g"] = np.flatnonzero(self.videos[lens].keyframes).astype(np.int64)
        return np.stack([starts, np.r_[starts[1:], self.num_frames]], axis=1)

    def read_packed(self, lens: int, start: int, stop: int) -> tuple[UInt8[ndarray, " n"], Int64[ndarray, " m"], Bool[ndarray, " k"]]:
        """Samples ``start .. stop - 1`` of one lens end to end, their bounds and keyframe flags."""
        chosen: list[bytes] = self.videos[lens].samples[start:stop]
        bounds: Int64[ndarray, " m"] = np.r_[0, np.cumsum([len(sample) for sample in chosen])].astype(np.int64)
        return np.frombuffer(b"".join(chosen), dtype=np.uint8), bounds, self.videos[lens].keyframes[start:stop].copy()


def read_base_lens_videos(base: Path) -> BaseLensVideos:
    """Both lens videos of a base recording (``cam_00``, ``cam_01``).

    Raises:
        ValueError: If a lens video does not start on a keyframe, or the two differ in stamps or codec.
    """
    store: rrc.LazyStore = rrc.RrdReader(base).store()
    videos: tuple[VideoStreamSamples, VideoStreamSamples] = (read_video_stream(store, schema.video_path(RIG, 0)), read_video_stream(store, schema.video_path(RIG, 1)))
    if any(not video.keyframes[0] for video in videos):
        raise ValueError(f"{base}: a lens video does not start on a keyframe")
    if videos[0].codec != videos[1].codec:
        raise ValueError(f"{base}: the lens videos have different codecs")
    if not np.array_equal(videos[0].times_ns, videos[1].times_ns):
        raise ValueError(f"{base}: the two lens videos are not on the same stamps")
    return BaseLensVideos(codec=catalog_codec_name(videos[0].codec), times_ns=videos[0].times_ns, videos=videos)


def _pictures(rig: LensSamples, lens: int, frames: int, *, threads: int) -> Iterator[av.VideoFrame]:
    """The pictures of the first ``frames`` samples of one lens, read one GOP at a time."""
    decoder = av.CodecContext.create(SOFTWARE_DECODERS[rig.codec], "r")
    if not isinstance(decoder, av.VideoCodecContext):
        raise TypeError(f"{rig.codec} is not a video codec")
    decoder.thread_count = threads
    decoder.thread_type = "NONE" if threads == 1 else "FRAME"
    for start, stop in rig.gops(lens).tolist():
        if start >= frames:
            break
        data, offsets, _ = rig.read_packed(lens, start, min(stop, frames))
        for first, last in zip(offsets[:-1].tolist(), offsets[1:].tolist(), strict=True):
            yield from decoder.decode(av.Packet(data[first:last].tobytes()))
    yield from decoder.decode(None)


def rgb_frames(rig: LensSamples, lens: int, indices: Int64[ndarray, " k"], *, threads: int) -> Iterator[UInt8[ndarray, "h w 3"]]:
    """The pictures ``indices`` (increasing) of one lens as ``rgb24``; every picture up to the last is decoded."""
    wanted: set[int] = {int(i) for i in indices}
    stop: int = int(indices[-1]) + 1
    decoded: int = 0
    for frame in _pictures(rig, lens, stop, threads=threads):
        if decoded in wanted:
            yield frame.to_ndarray(format="rgb24")
        decoded += 1
    if decoded < stop:
        raise ValueError(f"lens {lens} decoded {decoded} pictures of the {stop} asked for")


def rebatch(chunks: Iterable[UInt8[Tensor, "n 3 h w"]], batch: int) -> Iterator[UInt8[Tensor, "b 3 h w"]]:
    """The pictures of ``chunks``, in order, ``batch`` at a time (the last batch may be shorter)."""
    pending: list[UInt8[Tensor, "n 3 h w"]] = []
    held: int = 0
    for chunk in chunks:
        pending.append(chunk)
        held += chunk.shape[0]
        while held >= batch:
            joined: UInt8[Tensor, "n 3 h w"] = pending[0] if len(pending) == 1 else torch.cat(pending)
            yield joined[:batch]
            pending, held = ([joined[batch:]] if held > batch else []), held - batch
    if held:
        yield pending[0] if len(pending) == 1 else torch.cat(pending)


def run_decoder(rig: LensSamples, lens: int, run: DecodeRun, device: torch.device) -> torchcodec.decoders.VideoDecoder:
    """A torchcodec decoder over the samples of ``run``, muxed once into an in-memory MP4 (simplecv's ``wrap_mp4``).

    Raises:
        RuntimeError: If torchcodec decodes on the CPU (no NVDEC for the stream).
    """
    from torchcodec.decoders import VideoDecoder

    data, bounds, irap = rig.read_packed(lens, run.start, run.stop)
    samples: list[UInt8[ndarray, " n"]] = [data[a:b] for a, b in zip(bounds[:-1].tolist(), bounds[1:].tolist(), strict=True)]
    decoder = VideoDecoder(wrap_mp4(samples, irap.tolist(), LENS_FPS, codec=rig.codec), device=str(device), seek_mode="exact")
    if decoder.cpu_fallback:
        raise RuntimeError(f"NVDEC cannot decode lens {lens} ({decoder.cpu_fallback}); run the layer on the cpu device (--cube-device / --pano-device cpu)")
    return decoder


def lens_batches(rig: LensSamples, lens: int, indices: Int64[ndarray, " k"], device: torch.device, *, batch: int, threads: int) -> Iterator[UInt8[Tensor, "b 3 h w"]]:
    """The RGB pictures ``indices`` (increasing) of one lens on ``device``, ``batch`` at a time.

    ``cuda``: one NVDEC decoder per run of ``decode_runs``, read front to back, so each sample is decoded at most once;
    a batch may span two runs. ``cpu``: PyAV with ``threads`` frame threads, as torchcodec's CPU decoder runs about
    1.6x slower here (it seeks per picture; clip B).

    Raises:
        RuntimeError: If torchcodec decodes a ``cuda`` run on the CPU (no NVDEC for the stream).
    """
    if device.type == "cpu":
        frames: Iterator[UInt8[ndarray, "h w 3"]] = rgb_frames(rig, lens, indices, threads=threads)
        yield from rebatch((torch.from_numpy(frame[None]).permute(0, 3, 1, 2) for frame in frames), batch)
        return

    def chunks() -> Iterator[UInt8[Tensor, "n 3 h w"]]:
        for run in decode_runs(indices, rig.gops(lens)[:, 0], max_gops=RUN_GOPS):
            decoder = run_decoder(rig, lens, run, device)
            for offsets in itertools.batched((run.wanted - run.start).tolist(), batch):
                yield decoder.get_frames_at(list(offsets)).data
            del decoder  # its muxed samples go before the next run's are read

    yield from rebatch(chunks(), batch)


def lens_pictures(
    rig: LensSamples, indices: Int64[ndarray, " k"], device: torch.device, *, batch: int, threads: int
) -> Iterator[tuple[UInt8[Tensor, "b 3 h w"], UInt8[Tensor, "b 3 h w"]]]:
    """``lens_batches`` of both lenses, decoded side by side on two threads (CUDA bound on each), one batch ahead of the caller."""
    batches: list[Iterator[UInt8[Tensor, "b 3 h w"]]] = [lens_batches(rig, lens, indices, device, batch=batch, threads=threads) for lens in range(2)]
    with ThreadPoolExecutor(2, initializer=bind_cuda_device, initargs=(str(device),)) as pool:
        ahead: list[Future[UInt8[Tensor, "b 3 h w"] | None]] = [pool.submit(next, lens, None) for lens in batches]
        while True:
            lens0, lens1 = (future.result() for future in ahead)
            if lens0 is None or lens1 is None:
                return
            ahead = [pool.submit(next, lens, None) for lens in batches]
            yield lens0, lens1
