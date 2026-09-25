"""Gen2 all-intra H.265 access units; the existing JPEG reader is unchanged.

Only the measured Gen2 layout is supported. Description validation keeps a
future layout change from silently moving the timestamp or image boundary.
"""

import io
import json
import struct
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path

import pyarrow as pa
from serde import SerdeError
from serde.json import from_json

from dataforge.vrs import (
    _RECORD_HEADER,
    DATA_RECORD,
    DESCRIPTION_FORMAT_VERSION,
    DESCRIPTION_TYPE_ID,
    FILE_HEADER_SIZE,
    UNCOMPRESSED,
    DataLayout,
    ImageRecord,
    RecordHeader,
    _read_exact,
    _read_tags,
)

CONFIGURATION_RECORD: int = 2
"""VRS ``Record::Type::CONFIGURATION`` (STATE is 1, DATA is 3)."""
ZSTD: int = 2
"""VRS ``CompressionType::Zstd``."""


@lru_cache(maxsize=1)
def _data_offsets(path: Path, first_record: int, file_size: int, _mtime_ns: int) -> dict[str, list[int]]:
    """Scan headers once per unchanged file; keep only offsets, never image bytes.

    The mtime cache key prevents stale offsets after a source replacement. A
    one-file cache bounds memory and saves repeated full scans over NAS inputs.
    """
    offsets: dict[str, list[int]] = {}
    with path.open("rb") as stream:
        offset: int = first_record
        while offset < file_size:
            stream.seek(offset)
            record: RecordHeader = RecordHeader(*_RECORD_HEADER.unpack(_read_exact(stream, _RECORD_HEADER.size)))
            if record.record_size < _RECORD_HEADER.size or offset + record.record_size > file_size:
                raise ValueError(f"{path}: invalid VRS record size at {offset}")
            if record.record_type == DATA_RECORD:
                offsets.setdefault(f"{record.recordable_type_id}-{record.instance_id}", []).append(offset)
            offset += record.record_size
    return offsets


class VrsRecordReader:
    """Read a described stream without initializing an image decoder."""

    def __init__(self, path: Path, stream_id: str) -> None:
        self.path: Path = path
        self.stream_id: str = stream_id
        self._record_formats: dict[int, str] = {}
        self.layout_docs: dict[int, str] = {}
        self.configuration_docs: dict[int, str] = {}
        with path.open("rb") as stream:
            header: bytes = _read_exact(stream, FILE_HEADER_SIZE)
            if header[:8] != b"VisionRe" or header[72:80] not in (b"cordVRS1", b"cordVRS2"):
                raise ValueError(f"{path}: unsupported VRS file header")
            header_size, record_header_size = struct.unpack_from("<II", header, 16)
            if header_size != FILE_HEADER_SIZE or record_header_size != _RECORD_HEADER.size:
                raise ValueError(f"{path}: unsupported VRS header sizes {header_size}/{record_header_size}")
            description_offset: int = struct.unpack_from("<q", header, 32)[0]
            self._first_record: int = struct.unpack_from("<q", header, 40)[0]
            self._file_size: int = path.stat().st_size
            if not FILE_HEADER_SIZE <= description_offset < self._file_size or not FILE_HEADER_SIZE <= self._first_record <= self._file_size:
                raise ValueError(f"{path}: invalid VRS record offsets")
            stream.seek(description_offset)
            record: RecordHeader = RecordHeader(*_RECORD_HEADER.unpack(_read_exact(stream, _RECORD_HEADER.size)))
            if (
                record.recordable_type_id != DESCRIPTION_TYPE_ID
                or record.format_version != DESCRIPTION_FORMAT_VERSION
                or record.compression != UNCOMPRESSED
                or not _RECORD_HEADER.size <= record.record_size <= self._file_size - description_offset
            ):
                raise ValueError(f"{path}: unsupported VRS description record")
            description: io.BytesIO = io.BytesIO(_read_exact(stream, record.record_size - _RECORD_HEADER.size))
            stream_count: int = struct.unpack("<I", _read_exact(description, 4))[0]
            found: bool = False
            for _ in range(stream_count):
                type_id, instance_id = struct.unpack("<iH", _read_exact(description, 6))
                _read_tags(description)  # User tags precede the internal record-format tags.
                tags: dict[str, str] = _read_tags(description)
                if f"{type_id}-{instance_id}" == stream_id:
                    found = True
                    for key, value in tags.items():
                        if key.startswith("RF:Data:"):
                            version: int = int(key.removeprefix("RF:Data:"))
                            self._record_formats[version] = value
                            self.layout_docs[version] = tags.get(f"DL:Data:{version}:0", "")
                        elif key.startswith("RF:Configuration:"):
                            configuration: int = int(key.removeprefix("RF:Configuration:"))
                            self.configuration_docs[configuration] = tags.get(f"DL:Configuration:{configuration}:0", "")
            self.file_tags: dict[str, str] = _read_tags(description)
            if description.read(1):
                raise ValueError(f"{path}: trailing VRS description bytes")
            if not found:
                raise ValueError(f"{path}: stream {stream_id} absent from description")

    def records(self, prefix_size: int | None = None) -> Iterator[bytes]:
        """Yield data payloads, or just a prefix, checking record boundaries."""
        with self.path.open("rb") as stream:
            offsets: list[int] = _data_offsets(self.path, self._first_record, self._file_size, self.path.stat().st_mtime_ns).get(self.stream_id, [])
            for offset in offsets:
                stream.seek(offset)
                record: RecordHeader = RecordHeader(*_RECORD_HEADER.unpack(_read_exact(stream, _RECORD_HEADER.size)))
                if record.record_size < _RECORD_HEADER.size or offset + record.record_size > self._file_size:
                    raise ValueError(f"{self.path}: invalid VRS record size at {offset}")
                if record.compression not in (UNCOMPRESSED, ZSTD):
                    raise ValueError(f"{self.path}/{self.stream_id}: unsupported compressed data record")
                if record.format_version not in self._record_formats:
                    raise ValueError(f"{self.path}/{self.stream_id}: unknown data format version")
                size: int = record.record_size - _RECORD_HEADER.size
                if record.compression == ZSTD:
                    # A few tiny Gen2 SLAM records have a zstd envelope. This
                    # decompresses the record, never the H.265 image itself.
                    if not 0 < record.uncompressed_size <= 64 * 1024 * 1024:
                        raise ValueError(f"{self.path}/{self.stream_id}: invalid uncompressed record size")
                    payload: bytes = pa.decompress(_read_exact(stream, size), decompressed_size=record.uncompressed_size, codec="zstd", asbytes=True)
                    if len(payload) != record.uncompressed_size or (prefix_size is not None and len(payload) < prefix_size):
                        raise ValueError(f"{self.path}/{self.stream_id}: truncated compressed data layout")
                    yield payload if prefix_size is None else payload[:prefix_size]
                else:
                    if prefix_size is not None:
                        if size < prefix_size:
                            raise ValueError(f"{self.path}/{self.stream_id}: truncated data layout")
                        size = prefix_size
                    yield _read_exact(stream, size)


class VrsHevcReader(VrsRecordReader):
    """Gen2 all-intra images with the measured 120-byte layout and metadata vector."""

    def __init__(self, path: Path, stream_id: str) -> None:
        super().__init__(path, stream_id)
        self._versions: set[int] = set()
        for version, record_format in self._record_formats.items():
            if record_format != "data_layout+image/video/codec=H.265":
                raise ValueError(f"{path}/{stream_id}: unsupported image format {record_format}")
            try:
                layout: DataLayout = from_json(DataLayout, self.layout_docs[version])
            except (SerdeError, json.JSONDecodeError) as error:
                raise ValueError(f"{path}/{stream_id}: invalid DataLayout: {error}") from error
            fields: dict[str, tuple[str, int]] = {piece.name: (piece.type, piece.offset) for piece in layout.data_layout}
            expected: dict[str, tuple[str, int]] = {
                "capture_timestamp_ns": ("DataPieceValue<int64_t>", 60),
                "image_key_frame_index": ("DataPieceValue<uint32_t>", 108),
                "focus_distance_mm": ("DataPieceValue<double>", 112),
                "image_metadata": ("DataPieceVector<uint8_t>", -1),
            }
            if any(fields.get(name) != value for name, value in expected.items()):
                raise ValueError(f"{path}/{stream_id}: unsupported Gen2 image DataLayout")
            if any(piece.offset > 112 or (piece.offset < 0 and piece.name != "image_metadata") for piece in layout.data_layout):
                raise ValueError(f"{path}/{stream_id}: unsupported Gen2 variable layout")
            self._versions.add(version)
        if not self._versions:
            raise ValueError(f"{path}/{stream_id}: missing image DataLayout")

    def image_size(self) -> tuple[int, int]:
        """Width and height from the stream's first configuration record.

        The factory calibration in the file tags can describe the full sensor
        (camera-rgb: 4032x3024) while the stream is downscaled (2560x1920).
        """
        with self.path.open("rb") as stream:
            offset: int = self._first_record
            while offset < self._file_size:
                stream.seek(offset)
                record: RecordHeader = RecordHeader(*_RECORD_HEADER.unpack(_read_exact(stream, _RECORD_HEADER.size)))
                if record.record_size < _RECORD_HEADER.size or offset + record.record_size > self._file_size:
                    raise ValueError(f"{self.path}: invalid VRS record size at {offset}")
                offset += record.record_size
                if record.record_type != CONFIGURATION_RECORD or f"{record.recordable_type_id}-{record.instance_id}" != self.stream_id:
                    continue
                if record.compression not in (UNCOMPRESSED, ZSTD) or record.format_version not in self.configuration_docs:
                    raise ValueError(f"{self.path}/{self.stream_id}: unsupported configuration record")
                payload: bytes = _read_exact(stream, record.record_size - _RECORD_HEADER.size)
                if record.compression == ZSTD:
                    # Gen2 writes its configuration records zstd-wrapped.
                    if not 28 <= record.uncompressed_size <= 64 * 1024 * 1024:
                        raise ValueError(f"{self.path}/{self.stream_id}: invalid uncompressed configuration size")
                    payload = pa.decompress(payload, decompressed_size=record.uncompressed_size, codec="zstd", asbytes=True)
                if len(payload) < 28:
                    raise ValueError(f"{self.path}/{self.stream_id}: truncated configuration record")
                try:
                    layout: DataLayout = from_json(DataLayout, self.configuration_docs[record.format_version])
                except (SerdeError, json.JSONDecodeError) as error:
                    raise ValueError(f"{self.path}/{self.stream_id}: invalid configuration DataLayout: {error}") from error
                fields: dict[str, tuple[str, int]] = {piece.name: (piece.type, piece.offset) for piece in layout.data_layout}
                if fields.get("image_width") != ("DataPieceValue<uint32_t>", 20) or fields.get("image_height") != ("DataPieceValue<uint32_t>", 24):
                    raise ValueError(f"{self.path}/{self.stream_id}: unsupported configuration DataLayout")
                width, height = struct.unpack_from("<II", payload, 20)
                return int(width), int(height)
        raise ValueError(f"{self.path}/{self.stream_id}: no configuration record")

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
        if not self._record_formats:
            raise ValueError(f"{path}/{stream_id}: missing IMU data format")
        for version, record_format in self._record_formats.items():
            if record_format != "data_layout/size=79":
                raise ValueError(f"{path}/{stream_id}: unsupported IMU format {record_format}")
            try:
                layout: DataLayout = from_json(DataLayout, self.layout_docs[version])
            except (SerdeError, json.JSONDecodeError) as error:
                raise ValueError(f"{path}/{stream_id}: invalid IMU DataLayout: {error}") from error
            fields: dict[str, tuple[str, int]] = {piece.name: (piece.type, piece.offset) for piece in layout.data_layout}
            if any(fields.get(name) != value for name, value in expected.items()):
                raise ValueError(f"{path}/{stream_id}: unsupported Gen2 IMU DataLayout")
