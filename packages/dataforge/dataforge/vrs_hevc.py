"""Gen2 all-intra H.265 access units, read through ``VrsFile.payloads`` like the JPEG reader.

Only the measured Gen2 layout is supported. Description validation keeps a
future layout change from silently moving the timestamp or image boundary.
The container itself (descriptions, record offsets, IMU and configuration
records) is ``dataforge.vrs.VrsFile``.
"""

import struct
from collections.abc import Iterator

import numpy as np
from jaxtyping import Int64
from numpy import ndarray

from dataforge.vrs import DATA_RECORD, DataLayout, ImageRecord, StreamDescription, VrsFile, read_layout

TIMESTAMP_OFFSET: int = 60
"""``capture_timestamp_ns`` (int64) in the Gen2 image DataLayout."""
KEY_FRAME_OFFSET: int = 108
"""``image_key_frame_index`` (uint32); 0 on every all-intra image."""
FOCUS_OFFSET: int = 112
"""``focus_distance_mm`` (double), the last fixed-size piece."""
FIXED_LAYOUT_SIZE: int = 120
"""Fixed-size part of the image DataLayout; the variable-size index follows it."""
VAR_INDEX_SIZE: int = 8
"""One (offset, length) uint32 pair for ``image_metadata``, the only variable piece."""
IMAGE_START: int = FIXED_LAYOUT_SIZE + VAR_INDEX_SIZE
"""First byte after the DataLayout index: the metadata bytes, then the access unit."""


def _layout_fields(document: str, where: str) -> dict[str, tuple[str, int]]:
    """DataLayout pieces by name as (type, offset)."""
    layout: DataLayout = read_layout(document, where)
    return {piece.name: (piece.type, piece.offset) for piece in layout.data_layout}


class VrsHevcReader:
    """Gen2 all-intra images with the measured 120-byte layout and metadata vector."""

    def __init__(self, vrs: VrsFile, stream_id: str) -> None:
        where: str = f"{vrs.path}/{stream_id}"
        self.vrs: VrsFile = vrs
        self.stream_id: str = stream_id
        self.description: StreamDescription = vrs.description(stream_id)
        if not self.description.record_formats:
            raise ValueError(f"{where}: missing image DataLayout")
        expected: dict[str, tuple[str, int]] = {
            "capture_timestamp_ns": ("DataPieceValue<int64_t>", TIMESTAMP_OFFSET),
            "image_key_frame_index": ("DataPieceValue<uint32_t>", KEY_FRAME_OFFSET),
            "focus_distance_mm": ("DataPieceValue<double>", FOCUS_OFFSET),
            "image_metadata": ("DataPieceVector<uint8_t>", -1),
        }
        for version, record_format in self.description.record_formats.items():
            if record_format != "data_layout+image/video/codec=H.265":
                raise ValueError(f"{where}: unsupported image format {record_format}")
            fields: dict[str, tuple[str, int]] = _layout_fields(self.description.layout_docs[version], where)
            if any(fields.get(name) != value for name, value in expected.items()):
                raise ValueError(f"{where}: unsupported Gen2 image DataLayout")
            if any(offset > FOCUS_OFFSET or (offset < 0 and name != "image_metadata") for name, (_, offset) in fields.items()):
                raise ValueError(f"{where}: unsupported Gen2 variable layout")

    def records(self, prefix_size: int | None = None) -> Iterator[bytes]:
        """Yield data payloads, or just a prefix."""
        return (payload for _, payload in self.vrs.payloads(self.stream_id, DATA_RECORD, dict.fromkeys(self.description.record_formats, prefix_size)))

    def capture_timestamps(self) -> Int64[ndarray, "n"]:
        """Every image's capture clock, reading only the DataLayout prefix up to the timestamp."""
        return np.fromiter(
            (struct.unpack_from("<q", payload, TIMESTAMP_OFFSET)[0] for payload in self.records(TIMESTAMP_OFFSET + 8)), dtype=np.int64
        )

    def images(self) -> Iterator[ImageRecord]:
        """Yield complete access units and int64 capture timestamps in file order."""
        where: str = f"{self.vrs.path}/{self.stream_id}"
        previous: int | None = None
        for payload in self.records():
            if len(payload) < IMAGE_START:
                raise ValueError(f"{where}: truncated image DataLayout")
            timestamp: int = struct.unpack_from("<q", payload, TIMESTAMP_OFFSET)[0]
            metadata_offset, metadata_length = struct.unpack_from("<II", payload, FIXED_LAYOUT_SIZE)
            if metadata_offset != 0 or len(payload) < IMAGE_START + metadata_length:
                raise ValueError(f"{where}: invalid variable-size DataLayout index")
            image: bytes = payload[IMAGE_START + metadata_length :]
            # Zero padding before the Annex-B start code is legal.
            if not image.startswith(b"\x00\x00") or not image.lstrip(b"\x00").startswith(b"\x01") or len(image) < 6:
                raise ValueError(f"{where}: invalid HEVC access unit")
            if struct.unpack_from("<I", payload, KEY_FRAME_OFFSET)[0] != 0:
                raise ValueError(f"{where}: expected all-intra HEVC")
            if previous is not None and timestamp <= previous:
                raise ValueError(f"{where}: capture timestamps must increase")
            previous = timestamp
            yield ImageRecord(timestamp, image)
