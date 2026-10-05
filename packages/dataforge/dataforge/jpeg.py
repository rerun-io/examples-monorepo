"""CPU JPEG decode to native planes for ``encode_frames_to_mp4``.

HOT3D ships every camera as JPEG records in its VRS. TurboJPEG decodes them to the
JPEG's own YCbCr planes, so ffmpeg receives the source chroma without an RGB round
trip. This lives apart from ``dataforge.video_encoding`` because PyTurboJPEG is a
dataforge-env dependency only: environments that import the shared encoders or
``dataforge.logging_toolkit`` (slam-rs registration) do not install it.
"""

import functools
from collections import deque
from collections.abc import Generator, Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor

from turbojpeg import TJCS_GRAY, TJSAMP_420, TJSAMP_422, TJSAMP_444, TJSAMP_GRAY, TJCS_YCbCr, TurboJPEG

from dataforge.video_encoding import FrameKind, FrameSource


@functools.cache
def _jpeg_decoder() -> TurboJPEG:
    """Initialize once across camera jobs; native handles remain per decode call."""
    return TurboJPEG()


def jpeg_frame_source(image: bytes) -> FrameSource:
    """Describe the JPEG's native chroma layout, refusing unsupported colour spaces."""
    header = _jpeg_decoder().decode_header(image)
    width: int = header[0]
    height: int = header[1]
    sampling: int = header[2]
    colorspace: int = header[3]
    kinds: dict[int, FrameKind] = {TJSAMP_GRAY: "gray8", TJSAMP_420: "yuv420p", TJSAMP_422: "yuv422p", TJSAMP_444: "yuv444p"}
    if sampling not in kinds or colorspace not in (TJCS_GRAY, TJCS_YCbCr):
        raise ValueError(f"unsupported JPEG sampling/colorspace {sampling}/{colorspace}")
    return FrameSource(kinds[sampling], width, height, full_range=sampling != TJSAMP_GRAY)


def decode_jpeg_frames(images: Iterable[bytes], *, source: FrameSource, workers: int = 8) -> Generator[bytes, None, None]:
    """Decode JPEGs to packed native planes with bounded, ordered look-ahead.

    At most twice the worker count is queued. Crop TurboJPEG's MCU padding and
    row strides before serializing each plane, including odd-sized images.
    Eight workers per camera means at most 24 decoders across parallel_clips.
    """
    if workers < 1:
        raise ValueError("JPEG decode workers must be positive")
    decoder: TurboJPEG = _jpeg_decoder()
    assert source.width is not None and source.height is not None
    width: int = source.width
    height: int = source.height
    divisors: dict[FrameKind, tuple[int, int]] = {"gray8": (1, 1), "yuv420p": (2, 2), "yuv422p": (2, 1), "yuv444p": (1, 1)}
    if source.kind not in divisors:
        raise ValueError(f"not a JPEG planar source: {source.kind}")
    dx, dy = divisors[source.kind]
    sizes: list[tuple[int, int]] = [(height, width)]
    if source.kind != "gray8":
        sizes.extend([((height + dy - 1) // dy, (width + dx - 1) // dx)] * 2)

    def decode(image: bytes) -> bytes:
        if jpeg_frame_source(image) != source:
            raise ValueError("JPEG layout changed within stream")
        # Loop locals stay unannotated: beartype would rebuild a jaxtyping checker per frame in the dev env.
        planes = decoder.decode_to_yuv_planes(image)
        return b"".join(plane[:h, :w].tobytes() for plane, (h, w) in zip(planes, sizes, strict=True))

    pending: deque[Future[bytes]] = deque()
    iterator: Iterator[bytes] = iter(images)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for _ in range(workers * 2):
            image: bytes | None = next(iterator, None)
            if image is None:
                break
            pending.append(executor.submit(decode, image))
        while pending:
            yield pending.popleft().result()
            image = next(iterator, None)
            if image is not None:
                pending.append(executor.submit(decode, image))
