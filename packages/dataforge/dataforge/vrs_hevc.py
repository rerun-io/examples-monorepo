"""Gen2 all-intra H.265 access units; the existing JPEG reader is unchanged.

Only the measured Gen2 layout is supported. Description validation keeps a
future layout change from silently moving the timestamp or image boundary.
IMU streams are read through projectaria-tools (``aria.read_imu``), not here.
"""

import struct
from collections.abc import Collection, Iterator
from pathlib import Path

import numpy as np
import pyarrow as pa
from jaxtyping import Int64
from numpy import ndarray

from dataforge.vrs import (
    DATA_RECORD,
    RECORD_HEADER,
    UNCOMPRESSED,
    DataLayout,
    ImageRecord,
    RecordHeader,
    StreamDescription,
    read_descriptions,
    read_exact,
    read_layout,
    read_record_header,
)

CONFIGURATION_RECORD: int = 2
"""VRS ``Record::Type::CONFIGURATION`` (STATE is 1, DATA is 3)."""
ZSTD: int = 2
"""VRS ``CompressionType::Zstd``."""

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


class VrsFile:
    """One Gen2 VRS: every stream's description and record offsets, read once."""

    def __init__(self, path: Path) -> None:
        self.path: Path = path
        self.streams: dict[str, StreamDescription] = read_descriptions(path, headers=(b"cordVRS1", b"cordVRS2"))
        if not self.streams:
            raise ValueError(f"{path}: no streams in VRS description")
        first: StreamDescription = next(iter(self.streams.values()))
        self.file_tags: dict[str, str] = first.file_tags
        self.file_size: int = first.file_size
        # Header-only scan: data/configuration offsets per stream, never image bytes.
        self._offsets: dict[tuple[int, str], list[int]] = {}
        with path.open("rb") as stream:
            offset: int = first.first_record
            while offset < self.file_size:
                record: RecordHeader = read_record_header(stream, offset, self.file_size, path)
                if record.record_type in (DATA_RECORD, CONFIGURATION_RECORD):
                    self._offsets.setdefault((record.record_type, f"{record.recordable_type_id}-{record.instance_id}"), []).append(offset)
                offset += record.record_size

    def hevc(self, stream_id: str) -> "VrsHevcReader":
        """A validated image view over one camera stream."""
        return VrsHevcReader(self, stream_id)

    def payloads(self, stream_id: str, record_type: int, versions: Collection[int], prefix_size: int | None) -> Iterator[tuple[int, bytes]]:
        """Yield (format version, payload or its prefix), checking record boundaries and format versions."""
        where: str = f"{self.path}/{stream_id}"
        with self.path.open("rb") as stream:
            for offset in self._offsets.get((record_type, stream_id), []):
                record: RecordHeader = read_record_header(stream, offset, self.file_size, self.path)
                if record.compression not in (UNCOMPRESSED, ZSTD):
                    raise ValueError(f"{where}: unsupported compressed record")
                if record.format_version not in versions:
                    raise ValueError(f"{where}: unknown format version {record.format_version}")
                size: int = record.record_size - RECORD_HEADER.size
                if record.compression == ZSTD:
                    # Gen2 wraps its configuration records and a few tiny SLAM data records in
                    # zstd. This decompresses the record, never the H.265 image itself.
                    if not 0 < record.uncompressed_size <= 64 * 1024 * 1024:
                        raise ValueError(f"{where}: invalid uncompressed record size")
                    payload: bytes = pa.decompress(read_exact(stream, size), decompressed_size=record.uncompressed_size, codec="zstd", asbytes=True)
                    if len(payload) != record.uncompressed_size or (prefix_size is not None and len(payload) < prefix_size):
                        raise ValueError(f"{where}: truncated compressed data layout")
                    yield record.format_version, payload if prefix_size is None else payload[:prefix_size]
                else:
                    if prefix_size is not None:
                        if size < prefix_size:
                            raise ValueError(f"{where}: truncated data layout")
                        size = prefix_size
                    yield record.format_version, read_exact(stream, size)


class VrsHevcReader:
    """Gen2 all-intra images with the measured 120-byte layout and metadata vector."""

    def __init__(self, vrs: VrsFile, stream_id: str) -> None:
        where: str = f"{vrs.path}/{stream_id}"
        if stream_id not in vrs.streams:
            raise ValueError(f"{vrs.path}: stream {stream_id} absent from description")
        self.vrs: VrsFile = vrs
        self.stream_id: str = stream_id
        self.description: StreamDescription = vrs.streams[stream_id]
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
        return (payload for _, payload in self.vrs.payloads(self.stream_id, DATA_RECORD, self.description.record_formats, prefix_size))

    def image_size(self) -> tuple[int, int]:
        """Width and height from the stream's first configuration record.

        The factory calibration in the file tags can describe the full sensor
        (camera-rgb: 4032x3024) while the stream is downscaled (2560x1920).
        """
        where: str = f"{self.vrs.path}/{self.stream_id}"
        for version, payload in self.vrs.payloads(self.stream_id, CONFIGURATION_RECORD, self.description.configuration_docs, 28):
            fields: dict[str, tuple[str, int]] = _layout_fields(self.description.configuration_docs[version], where)
            if fields.get("image_width") != ("DataPieceValue<uint32_t>", 20) or fields.get("image_height") != ("DataPieceValue<uint32_t>", 24):
                raise ValueError(f"{where}: unsupported configuration DataLayout")
            width, height = struct.unpack_from("<II", payload, 20)
            return int(width), int(height)
        raise ValueError(f"{where}: no configuration record")

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
