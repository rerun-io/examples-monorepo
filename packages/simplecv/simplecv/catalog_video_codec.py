"""The catalog's ``VideoStream:codec`` component, mapped to the codec names the decoders take.

Rerun stores the codec as a big-endian FourCC integer (``rr.VideoCodec``). Both
video paths in the workspace — simplecv's segment-wide NVDEC decoder and rerun's
``VideoFrameDecoder`` — accept the same three names.
"""

from typing import Literal, TypeAlias

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
