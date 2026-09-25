"""Gen2 all-intra H.265 access units; the existing JPEG reader is unchanged.

Only the measured Gen2 layout is supported. Description validation keeps a
future layout change from silently moving the timestamp or image boundary.
"""

import struct
from collections.abc import Collection, Iterator
from functools import lru_cache
from pathlib import Path

import numpy as np
import pyarrow as pa
from jaxtyping import Float64, Int64
from numpy import ndarray

from dataforge.vrs import (
    _RECORD_HEADER,
    DATA_RECORD,
    UNCOMPRESSED,
    DataLayout,
    ImageRecord,
    RecordHeader,
    StreamDescription,
    _read_exact,
    read_description,
    read_layout,
    read_record_header,
)

CONFIGURATION_RECORD: int = 2
"""VRS ``Record::Type::CONFIGURATION`` (STATE is 1, DATA is 3)."""
ZSTD: int = 2
"""VRS ``CompressionType::Zstd``."""


@lru_cache(maxsize=1)
def _record_offsets(path: Path, first_record: int, file_size: int, _mtime_ns: int) -> dict[tuple[int, str], list[int]]:
    """Scan headers once per unchanged file; keep only data/configuration offsets, never image bytes.

    The mtime cache key prevents stale offsets after a source replacement. A
    one-file cache bounds memory and saves repeated full scans over NAS inputs.
    """
    offsets: dict[tuple[int, str], list[int]] = {}
    with path.open("rb") as stream:
        offset: int = first_record
        while offset < file_size:
            record: RecordHeader = read_record_header(stream, offset, file_size, path)
            if record.record_type in (DATA_RECORD, CONFIGURATION_RECORD):
                offsets.setdefault((record.record_type, f"{record.recordable_type_id}-{record.instance_id}"), []).append(offset)
            offset += record.record_size
    return offsets


def _layout_fields(document: str, where: str) -> dict[str, tuple[str, int]]:
    """DataLayout pieces by name as (type, offset)."""
    layout: DataLayout = read_layout(document, where)
    return {piece.name: (piece.type, piece.offset) for piece in layout.data_layout}


class VrsRecordReader:
    """Read a described stream without initializing an image decoder."""

    def __init__(self, path: Path, stream_id: str) -> None:
        self.path: Path = path
        self.stream_id: str = stream_id
        self.description: StreamDescription = read_description(path, stream_id, headers=(b"cordVRS1", b"cordVRS2"))

    def _payloads(self, record_type: int, versions: Collection[int], prefix_size: int | None) -> Iterator[tuple[int, bytes]]:
        """Yield (format version, payload or its prefix), checking record boundaries and format versions."""
        where: str = f"{self.path}/{self.stream_id}"
        description: StreamDescription = self.description
        with self.path.open("rb") as stream:
            offsets: dict[tuple[int, str], list[int]] = _record_offsets(
                self.path, description.first_record, description.file_size, self.path.stat().st_mtime_ns
            )
            for offset in offsets.get((record_type, self.stream_id), []):
                record: RecordHeader = read_record_header(stream, offset, description.file_size, self.path)
                if record.compression not in (UNCOMPRESSED, ZSTD):
                    raise ValueError(f"{where}: unsupported compressed record")
                if record.format_version not in versions:
                    raise ValueError(f"{where}: unknown format version {record.format_version}")
                size: int = record.record_size - _RECORD_HEADER.size
                if record.compression == ZSTD:
                    # Gen2 wraps its configuration records and a few tiny SLAM data records in
                    # zstd. This decompresses the record, never the H.265 image itself.
                    if not 0 < record.uncompressed_size <= 64 * 1024 * 1024:
                        raise ValueError(f"{where}: invalid uncompressed record size")
                    payload: bytes = pa.decompress(_read_exact(stream, size), decompressed_size=record.uncompressed_size, codec="zstd", asbytes=True)
                    if len(payload) != record.uncompressed_size or (prefix_size is not None and len(payload) < prefix_size):
                        raise ValueError(f"{where}: truncated compressed data layout")
                    yield record.format_version, payload if prefix_size is None else payload[:prefix_size]
                else:
                    if prefix_size is not None:
                        if size < prefix_size:
                            raise ValueError(f"{where}: truncated data layout")
                        size = prefix_size
                    yield record.format_version, _read_exact(stream, size)

    def records(self, prefix_size: int | None = None) -> Iterator[bytes]:
        """Yield data payloads, or just a prefix."""
        return (payload for _, payload in self._payloads(DATA_RECORD, self.description.record_formats, prefix_size))


class VrsHevcReader(VrsRecordReader):
    """Gen2 all-intra images with the measured 120-byte layout and metadata vector."""

    def __init__(self, path: Path, stream_id: str) -> None:
        super().__init__(path, stream_id)
        if not self.description.record_formats:
            raise ValueError(f"{path}/{stream_id}: missing image DataLayout")
        expected: dict[str, tuple[str, int]] = {
            "capture_timestamp_ns": ("DataPieceValue<int64_t>", 60),
            "image_key_frame_index": ("DataPieceValue<uint32_t>", 108),
            "focus_distance_mm": ("DataPieceValue<double>", 112),
            "image_metadata": ("DataPieceVector<uint8_t>", -1),
        }
        for version, record_format in self.description.record_formats.items():
            if record_format != "data_layout+image/video/codec=H.265":
                raise ValueError(f"{path}/{stream_id}: unsupported image format {record_format}")
            fields: dict[str, tuple[str, int]] = _layout_fields(self.description.layout_docs[version], f"{path}/{stream_id}")
            if any(fields.get(name) != value for name, value in expected.items()):
                raise ValueError(f"{path}/{stream_id}: unsupported Gen2 image DataLayout")
            if any(offset > 112 or (offset < 0 and name != "image_metadata") for name, (_, offset) in fields.items()):
                raise ValueError(f"{path}/{stream_id}: unsupported Gen2 variable layout")

    def image_size(self) -> tuple[int, int]:
        """Width and height from the stream's first configuration record.

        The factory calibration in the file tags can describe the full sensor
        (camera-rgb: 4032x3024) while the stream is downscaled (2560x1920).
        """
        where: str = f"{self.path}/{self.stream_id}"
        for version, payload in self._payloads(CONFIGURATION_RECORD, self.description.configuration_docs, 28):
            fields: dict[str, tuple[str, int]] = _layout_fields(self.description.configuration_docs[version], where)
            if fields.get("image_width") != ("DataPieceValue<uint32_t>", 20) or fields.get("image_height") != ("DataPieceValue<uint32_t>", 24):
                raise ValueError(f"{where}: unsupported configuration DataLayout")
            width, height = struct.unpack_from("<II", payload, 20)
            return int(width), int(height)
        raise ValueError(f"{where}: no configuration record")

    def capture_timestamps(self) -> Int64[ndarray, "n"]:
        """Every image's capture clock, reading only the fixed DataLayout prefix."""
        return np.fromiter((struct.unpack_from("<q", payload, 60)[0] for payload in self.records(124)), dtype=np.int64)

    def images(self) -> Iterator[ImageRecord]:
        """Yield complete access units and int64 capture timestamps in file order."""
        previous: int | None = None
        for payload in self.records():
            if len(payload) < 128:
                raise ValueError(f"{self.path}/{self.stream_id}: truncated image DataLayout")
            timestamp: int = struct.unpack_from("<q", payload, 60)[0]
            # The variable-size index holds one (offset, length) pair per variable piece;
            # image_metadata is the only one, so its bytes end at offset + length.
            metadata_offset, metadata_length = struct.unpack_from("<II", payload, 120)
            if metadata_offset != 0 or len(payload) < 128 + metadata_length:
                raise ValueError(f"{self.path}/{self.stream_id}: invalid variable-size DataLayout index")
            image: bytes = payload[128 + metadata_length :]
            # Zero padding before the Annex-B start code is legal.
            if not image.startswith(b"\x00\x00") or not image.lstrip(b"\x00").startswith(b"\x01") or len(image) < 6:
                raise ValueError(f"{self.path}/{self.stream_id}: invalid HEVC access unit")
            if struct.unpack_from("<I", payload, 108)[0] != 0:
                raise ValueError(f"{self.path}/{self.stream_id}: expected all-intra HEVC")
            if previous is not None and timestamp <= previous:
                raise ValueError(f"{self.path}/{self.stream_id}: capture timestamps must increase")
            previous = timestamp
            yield ImageRecord(timestamp, image)


class VrsImuReader(VrsRecordReader):
    """Uncompressed Gen2 motion records with native clocks and valid flags."""

    def __init__(self, path: Path, stream_id: str) -> None:
        super().__init__(path, stream_id)
        expected: dict[str, tuple[str, int]] = {
            "accelerometer_valid": ("DataPieceValue<Bool>", 0),
            "gyroscope_valid": ("DataPieceValue<Bool>", 1),
            "capture_timestamp_ns": ("DataPieceValue<int64_t>", 11),
            "accelerometer": ("DataPieceArray<float>", 43),
            "gyroscope": ("DataPieceArray<float>", 55),
        }
        if not self.description.record_formats:
            raise ValueError(f"{path}/{stream_id}: missing IMU data format")
        for version, record_format in self.description.record_formats.items():
            if record_format != "data_layout/size=79":
                raise ValueError(f"{path}/{stream_id}: unsupported IMU format {record_format}")
            fields: dict[str, tuple[str, int]] = _layout_fields(self.description.layout_docs[version], f"{path}/{stream_id}")
            if any(fields.get(name) != value for name, value in expected.items()):
                raise ValueError(f"{path}/{stream_id}: unsupported Gen2 IMU DataLayout")

    def samples(self, stop_ns: int | None) -> tuple[Int64[ndarray, "n"], Float64[ndarray, "n 3"], Float64[ndarray, "n 3"]]:
        """Capture clock, accelerometer and gyroscope up to ``stop_ns``; a sample flagged invalid is NaN."""
        stamps: list[int] = []
        accels: list[list[float]] = []
        gyros: list[list[float]] = []
        for payload in self.records(79):
            stamp: int = struct.unpack_from("<q", payload, 11)[0]
            if stop_ns is not None and stamp > stop_ns:
                break
            stamps.append(stamp)
            accels.append(list(struct.unpack_from("<fff", payload, 43)) if payload[0] else [float("nan")] * 3)
            gyros.append(list(struct.unpack_from("<fff", payload, 55)) if payload[1] else [float("nan")] * 3)
        times: Int64[ndarray, "n"] = np.asarray(stamps, dtype=np.int64)
        if np.any(np.diff(times) <= 0):
            raise ValueError(f"{self.path}/{self.stream_id}: unordered IMU timestamps")
        return times, np.asarray(accels, dtype=np.float64).reshape(-1, 3), np.asarray(gyros, dtype=np.float64).reshape(-1, 3)
