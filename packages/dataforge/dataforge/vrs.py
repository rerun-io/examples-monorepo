"""VRS files without platform-specific VRS Python bindings: one header scan, then seeks to the records a reader asks for.

``VrsFile`` reads the description's stream tags and every record's offset once;
``VrsImageReader`` (JPEG), ``vrs_hevc.VrsHevcReader`` (H.265) and ``VrsFile.imu``
all read their records through ``VrsFile.payloads``. Unsupported formats and
record compressions fail closed; memory stays bounded to one record.
"""

import io
import re
import struct
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pyarrow as pa
from jaxtyping import Bool, Float32, Int64
from numpy import ndarray
from serde import serde

from dataforge.records import decode

FILE_HEADER_SIZE: int = 80
DESCRIPTION_TYPE_ID: int = 2
DESCRIPTION_FORMAT_VERSION: int = 2
CONFIGURATION_RECORD: int = 2
"""VRS ``Record::Type::CONFIGURATION`` (STATE is 1, DATA is 3)."""
DATA_RECORD: int = 3
UNCOMPRESSED: int = 0
CODECS: dict[int, str] = {1: "lz4", 2: "zstd"}
"""VRS CompressionType values; the recorder keeps a compressed record only when it is smaller."""


class RecordHeader(NamedTuple):
    """VRS FileFormat.h record fields in their packed order."""

    record_size: int
    previous_record_size: int
    recordable_type_id: int
    format_version: int
    timestamp: float
    instance_id: int
    record_type: int
    compression: int
    uncompressed_size: int


RECORD_HEADER: struct.Struct = struct.Struct("<IIiIdHBBI")


@serde
@dataclass(frozen=True, slots=True)
class LayoutPiece:
    """The subset of a VRS DataLayout piece needed to locate a timestamp."""

    name: str
    """Field name from the producer."""
    type: str
    """VRS DataPiece type, including its scalar width."""
    offset: int = -1
    """Fixed byte offset; variable pieces have no fixed offset."""
    size: int = 1
    """Element count of a ``DataPieceArray``; 1 for a single value."""


@serde
@dataclass(frozen=True, slots=True)
class DataLayout:
    """Description-record JSON schema; unrelated producer fields are allowed."""

    data_layout: list[LayoutPiece]
    """Fixed and variable fields in the first block."""


@dataclass(frozen=True, slots=True)
class ImageRecord:
    """One image and its capture clock, in file order."""

    capture_timestamp_ns: int
    """Device timestamp from the fixed DataLayout."""
    image: bytes
    """Encoded image block, without the record header or DataLayout."""


def read_exact(stream: io.BufferedReader | io.BytesIO, size: int) -> bytes:
    data: bytes = stream.read(size)
    if len(data) != size:
        raise ValueError(f"truncated VRS: expected {size} bytes, got {len(data)}")
    return data


def _read_tags(stream: io.BufferedReader | io.BytesIO) -> dict[str, str]:
    count: int = struct.unpack("<I", read_exact(stream, 4))[0]
    tags: dict[str, str] = {}
    for _ in range(count):
        pair: list[str] = []
        for _ in range(2):
            size: int = struct.unpack("<I", read_exact(stream, 4))[0]
            pair.append(read_exact(stream, size).decode("utf-8"))
        tags[pair[0]] = pair[1]
    return tags


def read_record_header(stream: io.BufferedReader, offset: int, file_size: int, path: Path) -> RecordHeader:
    """Seek to one record and check that it lies inside the file."""
    stream.seek(offset)
    record: RecordHeader = RecordHeader(*RECORD_HEADER.unpack(read_exact(stream, RECORD_HEADER.size)))
    if record.record_size < RECORD_HEADER.size or offset + record.record_size > file_size:
        raise ValueError(f"{path}: invalid VRS record size {record.record_size} at {offset}")
    return record


def read_layout(document: str, where: str) -> DataLayout:
    """Decode one description DataLayout, naming the stream on failure."""
    return decode(DataLayout, document, source=f"{where}: invalid DataLayout")


@dataclass(frozen=True, slots=True)
class StreamDescription:
    """One stream's record formats from the VRS description record."""

    first_record: int
    """Byte offset of the first record after the description."""
    file_size: int
    """File length when the description was read."""
    record_formats: dict[int, str]
    """``RF:Data:<version>`` record formats."""
    layout_docs: dict[int, str]
    """``DL:Data:<version>:0`` DataLayout JSON."""
    configuration_docs: dict[int, str]
    """``DL:Configuration:<version>:0`` DataLayout JSON."""
    file_tags: dict[str, str]
    """File-level tags that follow every stream entry."""


def read_descriptions(path: Path, *, headers: tuple[bytes, ...]) -> dict[str, StreamDescription]:
    """Validate the file header and read every stream's description tags in one pass.

    ``headers`` lists the accepted file-format magics (``cordVRS2``; Gen2 Aria
    also writes ``cordVRS1`` with a version-2 description).
    """
    with path.open("rb") as stream:
        header: bytes = read_exact(stream, FILE_HEADER_SIZE)
        if header[:8] != b"VisionRe" or header[72:80] not in headers:
            raise ValueError(f"{path}: unsupported VRS file header")
        header_size, record_header_size = struct.unpack_from("<II", header, 16)
        if header_size != FILE_HEADER_SIZE or record_header_size != RECORD_HEADER.size:
            raise ValueError(f"{path}: unsupported VRS header sizes {header_size}/{record_header_size}")
        description_offset: int = struct.unpack_from("<q", header, 32)[0]
        first_record: int = struct.unpack_from("<q", header, 40)[0]
        file_size: int = path.stat().st_size
        if not FILE_HEADER_SIZE <= description_offset < file_size or not FILE_HEADER_SIZE <= first_record <= file_size:
            raise ValueError(f"{path}: invalid VRS record offsets")
        stream.seek(description_offset)
        record: RecordHeader = RecordHeader(*RECORD_HEADER.unpack(read_exact(stream, RECORD_HEADER.size)))
        if (
            record.recordable_type_id != DESCRIPTION_TYPE_ID
            or record.format_version != DESCRIPTION_FORMAT_VERSION
            or record.compression != UNCOMPRESSED
            or not RECORD_HEADER.size <= record.record_size <= file_size - description_offset
        ):
            raise ValueError(f"{path}: unsupported VRS description record")
        description: io.BytesIO = io.BytesIO(read_exact(stream, record.record_size - RECORD_HEADER.size))
    stream_count: int = struct.unpack("<I", read_exact(description, 4))[0]
    formats: dict[str, tuple[dict[int, str], dict[int, str], dict[int, str]]] = {}
    for _ in range(stream_count):
        type_id, instance_id = struct.unpack("<iH", read_exact(description, 6))
        _read_tags(description)  # User tags precede the internal record-format tags.
        tags: dict[str, str] = _read_tags(description)
        record_formats: dict[int, str] = {}
        layout_docs: dict[int, str] = {}
        configuration_docs: dict[int, str] = {}
        for key, value in tags.items():
            if key.startswith("RF:Data:"):
                version: int = int(key.removeprefix("RF:Data:"))
                record_formats[version] = value
                layout_docs[version] = tags.get(f"DL:Data:{version}:0", "")
            elif key.startswith("RF:Configuration:"):
                configuration: int = int(key.removeprefix("RF:Configuration:"))
                configuration_docs[configuration] = tags.get(f"DL:Configuration:{configuration}:0", "")
        formats[f"{type_id}-{instance_id}"] = (record_formats, layout_docs, configuration_docs)
    file_tags: dict[str, str] = _read_tags(description)
    if description.read(1):
        raise ValueError(f"{path}: trailing VRS description bytes")
    return {stream_id: StreamDescription(first_record, file_size, *docs, file_tags) for stream_id, docs in formats.items()}


def census_images(images: Iterator[ImageRecord], times_ns: Int64[ndarray, "n"], source_count: int, *, preview: bool, where: str) -> Iterator[bytes]:
    """Check native timestamps and counts, reading only the selected prefix for previews."""
    records: Iterator[ImageRecord] = islice(images, len(times_ns)) if preview else images
    seen: int = 0
    for record in records:
        if seen < len(times_ns):
            if record.capture_timestamp_ns != times_ns[seen]:
                raise ValueError(f"{where}: capture timestamp mismatch at frame {seen}")
            yield record.image
        seen += 1
    expected: int = len(times_ns) if preview else source_count
    if seen != expected:
        raise ValueError(f"{where}: {seen} image records, expected {expected}")


@dataclass(frozen=True, slots=True)
class ImuRecords:
    """One motion stream's data records in file order, as the recorder wrote them."""

    capture_timestamp_ns: Int64[ndarray, "n"]
    """Device-clock capture time."""
    accel_valid: Bool[ndarray, "n"]
    """``accelerometer_valid``."""
    gyro_valid: Bool[ndarray, "n"]
    """``gyroscope_valid``."""
    accel_msec2: Float32[ndarray, "n 3"]
    """``accelerometer`` in m/s^2, raw (unrectified)."""
    gyro_radsec: Float32[ndarray, "n 3"]
    """``gyroscope`` in rad/s, raw (unrectified)."""


IMU_FIELDS: dict[str, tuple[str, int, str]] = {
    "capture_timestamp_ns": ("DataPieceValue<int64_t>", 1, "<i8"),
    "accelerometer_valid": ("DataPieceValue<Bool>", 1, "u1"),
    "gyroscope_valid": ("DataPieceValue<Bool>", 1, "u1"),
    "accelerometer": ("DataPieceArray<float>", 3, "<f4"),
    "gyroscope": ("DataPieceArray<float>", 3, "<f4"),
}
"""The motion DataLayout pieces an IMU reader needs: VRS type, element count, and numpy dtype."""


class VrsFile:
    """One VRS: every stream's description and record offsets, read once from the record headers.

    Accepts ``cordVRS2`` headers and Gen2 Aria's ``cordVRS1`` (which carries a
    version-2 description). Records may be uncompressed, lz4 or zstd.
    """

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

    def description(self, stream_id: str) -> StreamDescription:
        """One stream's description, naming the file when the stream is absent."""
        if stream_id not in self.streams:
            raise ValueError(f"{self.path}: stream {stream_id} absent from description")
        return self.streams[stream_id]

    def payloads(self, stream_id: str, record_type: int, prefix_sizes: Mapping[int, int | None]) -> Iterator[tuple[int, bytes]]:
        """Yield (format version, payload or its prefix) in file order, checking record boundaries and format versions.

        ``prefix_sizes`` maps every accepted format version to the prefix to read (its own DataLayout
        size, say), or to ``None`` for the whole payload.
        """
        where: str = f"{self.path}/{stream_id}"
        with self.path.open("rb") as stream:
            for offset in self._offsets.get((record_type, stream_id), []):
                record: RecordHeader = read_record_header(stream, offset, self.file_size, self.path)
                if record.compression != UNCOMPRESSED and record.compression not in CODECS:
                    raise ValueError(f"{where}: unsupported compressed record")
                if record.format_version not in prefix_sizes:
                    raise ValueError(f"{where}: unknown format version {record.format_version}")
                prefix_size: int | None = prefix_sizes[record.format_version]
                size: int = record.record_size - RECORD_HEADER.size
                if record.compression != UNCOMPRESSED:
                    # Aria wraps its configuration records and a few tiny data records. This
                    # decompresses one record, never an image codec's own stream.
                    if not 0 < record.uncompressed_size <= 64 * 1024 * 1024:
                        raise ValueError(f"{where}: invalid uncompressed record size")
                    payload: bytes = pa.decompress(
                        read_exact(stream, size), decompressed_size=record.uncompressed_size, codec=CODECS[record.compression], asbytes=True
                    )
                    if len(payload) != record.uncompressed_size or (prefix_size is not None and len(payload) < prefix_size):
                        raise ValueError(f"{where}: truncated compressed data layout")
                    yield record.format_version, payload if prefix_size is None else payload[:prefix_size]
                else:
                    if prefix_size is not None:
                        if size < prefix_size:
                            raise ValueError(f"{where}: truncated data layout")
                        size = prefix_size
                    yield record.format_version, read_exact(stream, size)

    def image_size(self, stream_id: str) -> tuple[int, int]:
        """Width and height from a camera stream's first configuration record.

        The factory calibration in the file tags can describe the full sensor
        (Gen2 camera-rgb: 4032x3024) while the stream is downscaled (2560x1920).
        """
        where: str = f"{self.path}/{stream_id}"
        description: StreamDescription = self.description(stream_id)
        for version, payload in self.payloads(stream_id, CONFIGURATION_RECORD, dict.fromkeys(description.configuration_docs)):
            pieces: dict[str, LayoutPiece] = {piece.name: piece for piece in read_layout(description.configuration_docs[version], where).data_layout}
            offsets: list[int] = []
            for name in ("image_width", "image_height"):
                piece: LayoutPiece | None = pieces.get(name)
                if piece is None or piece.type != "DataPieceValue<uint32_t>" or not 0 <= piece.offset <= len(payload) - 4:
                    raise ValueError(f"{where}: unsupported configuration DataLayout")
                offsets.append(piece.offset)
            return struct.unpack_from("<I", payload, offsets[0])[0], struct.unpack_from("<I", payload, offsets[1])[0]
        raise ValueError(f"{where}: no configuration record")

    def imu(self, stream_id: str) -> ImuRecords:
        """Every data record of one motion stream, located by its DataLayout pieces' names, types and offsets."""
        where: str = f"{self.path}/{stream_id}"
        description: StreamDescription = self.description(stream_id)
        dtypes: dict[int, np.dtype] = {}
        for version, record_format in description.record_formats.items():
            match: re.Match[str] | None = re.fullmatch(r"data_layout/size=(\d+)", record_format)
            if match is None:
                raise ValueError(f"{where}: unsupported motion format {record_format}")
            pieces: dict[str, LayoutPiece] = {piece.name: piece for piece in read_layout(description.layout_docs[version], where).data_layout}
            for name, (kind, count, _) in IMU_FIELDS.items():
                piece: LayoutPiece | None = pieces.get(name)
                if piece is None or piece.type != kind or piece.size != count or piece.offset < 0:
                    raise ValueError(f"{where}: unsupported motion DataLayout piece {name}")
            dtypes[version] = np.dtype(
                {
                    "names": list(IMU_FIELDS),
                    "formats": [(dtype, (count,)) if count > 1 else dtype for _, count, dtype in IMU_FIELDS.values()],
                    "offsets": [pieces[name].offset for name in IMU_FIELDS],
                    "itemsize": int(match[1]),
                }
            )
        # Each format version is read at its own DataLayout size, then every record is moved into one
        # packed dtype by field name, in file order.
        rows: list[tuple[int, bytes]] = list(self.payloads(stream_id, DATA_RECORD, {version: dtype.itemsize for version, dtype in dtypes.items()}))
        packed: np.dtype = np.dtype([(name, dtype, (count,)) if count > 1 else (name, dtype) for name, (_, count, dtype) in IMU_FIELDS.items()])
        records: ndarray = np.empty(len(rows), dtype=packed)
        versions: Int64[ndarray, "n"] = np.array([version for version, _ in rows], dtype=np.int64)
        for version, dtype in dtypes.items():
            records[versions == version] = np.frombuffer(b"".join(payload for row_version, payload in rows if row_version == version), dtype=dtype).astype(packed)
        return ImuRecords(
            capture_timestamp_ns=records["capture_timestamp_ns"].astype(np.int64),
            accel_valid=records["accelerometer_valid"] != 0,
            gyro_valid=records["gyroscope_valid"] != 0,
            accel_msec2=records["accelerometer"].astype(np.float32),
            gyro_radsec=records["gyroscope"].astype(np.float32),
        )


class VrsImageReader:
    """One JPEG camera stream of an open ``VrsFile``, its data formats validated.

    The image path accepts only a fixed DataLayout followed by JPEG, in
    uncompressed, lz4 or zstd records.
    """

    def __init__(self, vrs: VrsFile, stream_id: str) -> None:
        self.vrs: VrsFile = vrs
        self.stream_id: str = stream_id
        where: str = f"{vrs.path}/{stream_id}"
        description: StreamDescription = vrs.description(stream_id)
        self._layouts: dict[int, tuple[int, int]] = {}
        for version, record_format in description.record_formats.items():
            match: re.Match[str] | None = re.fullmatch(r"data_layout/size=(\d+)\+image/jpg", record_format)
            if match is None:
                raise ValueError(f"{where}: unsupported image format {record_format}")
            size: int = int(match[1])
            layout: DataLayout = read_layout(description.layout_docs[version], where)
            timestamps: list[LayoutPiece] = [piece for piece in layout.data_layout if piece.name == "capture_timestamp_ns"]
            if len(timestamps) != 1 or timestamps[0].type != "DataPieceValue<int64_t>" or not 0 <= timestamps[0].offset <= size - 8:
                raise ValueError(f"{where}: invalid capture_timestamp_ns layout")
            self._layouts[version] = (size, timestamps[0].offset)

    def images(self) -> Iterator[ImageRecord]:
        """Yield complete JPEG blocks with validated boundaries and timestamps."""
        where: str = f"{self.vrs.path}/{self.stream_id}"
        for version, payload in self.vrs.payloads(self.stream_id, DATA_RECORD, dict.fromkeys(self._layouts)):
            layout_size, timestamp_offset = self._layouts[version]
            if len(payload) < layout_size + 4:
                raise ValueError(f"{where}: truncated image payload")
            timestamp: int = struct.unpack_from("<q", payload, timestamp_offset)[0]
            image: bytes = payload[layout_size:]
            if not image.startswith(b"\xff\xd8") or not image.endswith(b"\xff\xd9"):
                raise ValueError(f"{where}: invalid JPEG boundaries")
            yield ImageRecord(timestamp, image)

    def capture_timestamps(self) -> Int64[ndarray, "n"]:
        """Every image's capture clock in file order, reading only the DataLayout of uncompressed records."""
        prefix_sizes: dict[int, int | None] = {version: size for version, (size, _) in self._layouts.items()}
        return np.fromiter(
            (
                struct.unpack_from("<q", payload, self._layouts[version][1])[0]
                for version, payload in self.vrs.payloads(self.stream_id, DATA_RECORD, prefix_sizes)
            ),
            dtype=np.int64,
        )
