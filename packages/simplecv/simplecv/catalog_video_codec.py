"""The catalog's ``VideoStream:codec`` component, and the MP4 wrapper its samples are decoded through.

Rerun stores the codec as a big-endian FourCC integer (``rr.VideoCodec``). Both
video paths in the workspace — simplecv's segment-wide NVDEC decoder and rerun's
``VideoFrameDecoder`` — accept the same three names, and both reach their
decoder through :func:`wrap_mp4`. This module is deliberately light: a CPU lane
that wants the muxer must not be made to install torchcodec and torchvision to
get it, which is why it lives here and not beside the GPU decoder that also
calls it.
"""

from collections.abc import Buffer, Sequence
from fractions import Fraction
from io import BytesIO
from typing import Literal, TypeAlias

import av
import rerun as rr

CatalogCodecName: TypeAlias = Literal["av1", "h264", "hevc"]
"""Codec names shared by PyAV muxing and ``VideoFrameDecoder`` (which aliases ``hevc`` to its H.265 decoder)."""

_CODEC_NAME: dict[rr.VideoCodec, CatalogCodecName] = {rr.VideoCodec.AV1: "av1", rr.VideoCodec.H264: "h264", rr.VideoCodec.H265: "hevc"}


def catalog_codec_name(fourcc: int) -> CatalogCodecName:
    """Map a ``VideoStream:codec`` FourCC integer to a decoder codec name.

    Raises:
        ValueError: If the FourCC is not a Rerun video codec, or one the workspace decoders cannot handle (VP8, VP9).
    """
    codec: rr.VideoCodec = rr.VideoCodec(int(fourcc))
    if codec not in _CODEC_NAME:
        raise ValueError(f"unsupported catalog video codec {codec.name} ({int(fourcc):#x})")
    return _CODEC_NAME[codec]


def wrap_mp4(samples: Sequence[Buffer], keyframes: list[bool], fps: int, codec: CatalogCodecName) -> bytes:
    """Mux pre-encoded samples into an in-memory MP4 with positional pts (no re-encode).

    ``add_mux_stream`` muxes without instantiating an encoder. The nominal 16x16
    stream dimensions are irrelevant: decoders read the real dimensions from the
    bitstream (parameter sets travel in-band).

    Args:
        samples: Encoded video samples in decode order; the first must be a
            keyframe. Anything with a buffer — ``bytes``, or a view into the
            column they were read from — because ``av.Packet`` copies into its
            own buffer either way, and a ``bytes`` copy on the way in is one
            full copy of a window's encoded bytes per camera for nothing.
        keyframes: Keyframe flag per sample.
        fps: Frame rate for the muxed track and its time base.
        codec: Codec name of the pre-encoded samples.

    Returns:
        The complete MP4 file as bytes.
    """
    buffer: BytesIO = BytesIO()
    # Pin the mp4 track timescale to fps: the muxer otherwise picks its own (15360)
    # without rescaling our positional pts, and the track then claims a ~0.1s
    # duration. Index-seeking decoders never notice; timestamp readers do.
    with av.open(buffer, "w", format="mp4", options={"video_track_timescale": str(fps)}) as container:
        stream = container.add_mux_stream(codec, rate=fps, width=16, height=16)
        stream.time_base = Fraction(1, fps)
        for sample_index, (sample, is_keyframe) in enumerate(zip(samples, keyframes, strict=True)):
            # PyAV's stub says `bytes`; the constructor takes any buffer and
            # copies into the packet's own (`av.Packet(memoryview)` is documented).
            packet: av.Packet = av.Packet(sample)  # pyrefly: ignore[bad-argument-type]
            packet.pts = packet.dts = sample_index
            packet.duration = 1
            packet.time_base = stream.time_base
            packet.stream = stream
            packet.is_keyframe = is_keyframe
            container.mux(packet)
    return buffer.getvalue()
