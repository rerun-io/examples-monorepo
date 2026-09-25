"""Sequential VRS image blocks without platform-specific VRS Python bindings.

Read the VRS2 description's stream tags and fixed DataLayout before touching
image payloads. Unsupported formats and compressed image records fail closed;
other streams are skipped with seeks, so memory stays bounded to one record.
"""

import io
import json
import re
import struct
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import NamedTuple

from jaxtyping import Int64
from numpy import ndarray
from serde import SerdeError, serde
from serde.json import from_json

FILE_HEADER_SIZE: int = 80
DESCRIPTION_TYPE_ID: int = 2
DESCRIPTION_FORMAT_VERSION: int = 2
DATA_RECORD: int = 3
UNCOMPRESSED: int = 0


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


_RECORD_HEADER: struct.Struct = struct.Struct("<IIiIdHBBI")


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


def _read_exact(stream: io.BufferedReader | io.BytesIO, size: int) -> bytes:
    data: bytes = stream.read(size)
    if len(data) != size:
        raise ValueError(f"truncated VRS: expected {size} bytes, got {len(data)}")
    return data


def _read_tags(stream: io.BufferedReader | io.BytesIO) -> dict[str, str]:
    count: int = struct.unpack("<I", _read_exact(stream, 4))[0]
    tags: dict[str, str] = {}
    for _ in range(count):
        pair: list[str] = []
        for _ in range(2):
            size: int = struct.unpack("<I", _read_exact(stream, 4))[0]
            pair.append(_read_exact(stream, size).decode("utf-8"))
        tags[pair[0]] = pair[1]
    return tags


def read_record_header(stream: io.BufferedReader, offset: int, file_size: int, path: Path) -> RecordHeader:
    """Seek to one record and check that it lies inside the file."""
    stream.seek(offset)
    record: RecordHeader = RecordHeader(*_RECORD_HEADER.unpack(_read_exact(stream, _RECORD_HEADER.size)))
    if record.record_size < _RECORD_HEADER.size or offset + record.record_size > file_size:
        raise ValueError(f"{path}: invalid VRS record size {record.record_size} at {offset}")
    return record


def read_layout(document: str, where: str) -> DataLayout:
    """Decode one description DataLayout, naming the stream on failure."""
    try:
        return from_json(DataLayout, document)
    except (SerdeError, json.JSONDecodeError) as error:
        raise ValueError(f"{where}: invalid DataLayout: {error}") from error


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


def read_description(path: Path, stream_id: str, *, headers: tuple[bytes, ...]) -> StreamDescription:
    """Validate the file header and read one stream's description tags.

    ``headers`` lists the accepted file-format magics (``cordVRS2``; Gen2 Aria
    also writes ``cordVRS1`` with a version-2 description).
    """
    record_formats: dict[int, str] = {}
    layout_docs: dict[int, str] = {}
    configuration_docs: dict[int, str] = {}
    with path.open("rb") as stream:
        header: bytes = _read_exact(stream, FILE_HEADER_SIZE)
        if header[:8] != b"VisionRe" or header[72:80] not in headers:
            raise ValueError(f"{path}: unsupported VRS file header")
        header_size, record_header_size = struct.unpack_from("<II", header, 16)
        if header_size != FILE_HEADER_SIZE or record_header_size != _RECORD_HEADER.size:
            raise ValueError(f"{path}: unsupported VRS header sizes {header_size}/{record_header_size}")
        description_offset: int = struct.unpack_from("<q", header, 32)[0]
        first_record: int = struct.unpack_from("<q", header, 40)[0]
        file_size: int = path.stat().st_size
        if not FILE_HEADER_SIZE <= description_offset < file_size or not FILE_HEADER_SIZE <= first_record <= file_size:
            raise ValueError(f"{path}: invalid VRS record offsets")
        stream.seek(description_offset)
        record: RecordHeader = RecordHeader(*_RECORD_HEADER.unpack(_read_exact(stream, _RECORD_HEADER.size)))
        if (
            record.recordable_type_id != DESCRIPTION_TYPE_ID
            or record.format_version != DESCRIPTION_FORMAT_VERSION
            or record.compression != UNCOMPRESSED
            or not _RECORD_HEADER.size <= record.record_size <= file_size - description_offset
        ):
            raise ValueError(f"{path}: unsupported VRS description record")
        description: io.BytesIO = io.BytesIO(_read_exact(stream, record.record_size - _RECORD_HEADER.size))
    stream_count: int = struct.unpack("<I", _read_exact(description, 4))[0]
    found: bool = False
    for _ in range(stream_count):
        type_id, instance_id = struct.unpack("<iH", _read_exact(description, 6))
        _read_tags(description)  # User tags precede the internal record-format tags.
        tags: dict[str, str] = _read_tags(description)
        if f"{type_id}-{instance_id}" != stream_id:
            continue
        found = True
        for key, value in tags.items():
            if key.startswith("RF:Data:"):
                version: int = int(key.removeprefix("RF:Data:"))
                record_formats[version] = value
                layout_docs[version] = tags.get(f"DL:Data:{version}:0", "")
            elif key.startswith("RF:Configuration:"):
                configuration: int = int(key.removeprefix("RF:Configuration:"))
                configuration_docs[configuration] = tags.get(f"DL:Configuration:{configuration}:0", "")
    file_tags: dict[str, str] = _read_tags(description)
    if description.read(1):
        raise ValueError(f"{path}: trailing VRS description bytes")
    if not found:
        raise ValueError(f"{path}: stream {stream_id} absent from description")
    return StreamDescription(first_record, file_size, record_formats, layout_docs, configuration_docs, file_tags)


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


class VrsImageReader:
    """Open a separate sequential reader per camera and validate its data formats.

    Supports VRS2 headers and description version 2 (user tags followed by
    internal tags per stream). The image path accepts only a fixed DataLayout
    followed by JPEG, without record compression.
    """

    def __init__(self, path: Path, stream_id: str) -> None:
        self.path: Path = path
        self.stream_id: str = stream_id
        description: StreamDescription = read_description(path, stream_id, headers=(b"cordVRS2",))
        self._first_record: int = description.first_record
        self._file_size: int = description.file_size
        self._layouts: dict[int, tuple[int, int]] = {}
        for version, record_format in description.record_formats.items():
            match: re.Match[str] | None = re.fullmatch(r"data_layout/size=(\d+)\+image/jpg", record_format)
            if match is None:
                raise ValueError(f"{self.path}/{self.stream_id}: unsupported image format {record_format}")
            size: int = int(match[1])
            layout: DataLayout = read_layout(description.layout_docs[version], f"{self.path}/{self.stream_id}")
            timestamps: list[LayoutPiece] = [piece for piece in layout.data_layout if piece.name == "capture_timestamp_ns"]
            if len(timestamps) != 1 or timestamps[0].type != "DataPieceValue<int64_t>" or not 0 <= timestamps[0].offset <= size - 8:
                raise ValueError(f"{self.path}/{self.stream_id}: invalid capture_timestamp_ns layout")
            self._layouts[version] = (size, timestamps[0].offset)

    def images(self) -> Iterator[ImageRecord]:
        """Yield complete JPEG blocks with validated boundaries and timestamps."""
        with self.path.open("rb") as stream:
            offset: int = self._first_record
            while offset < self._file_size:
                record: RecordHeader = read_record_header(stream, offset, self._file_size, self.path)
                offset += record.record_size
                if record.record_type != DATA_RECORD or f"{record.recordable_type_id}-{record.instance_id}" != self.stream_id:
                    continue
                if record.compression != UNCOMPRESSED:
                    raise ValueError(f"{self.path}/{self.stream_id}: compressed image record ({record.compression})")
                if record.format_version not in self._layouts:
                    raise ValueError(f"{self.path}/{self.stream_id}: missing data format version {record.format_version}")
                layout_size, timestamp_offset = self._layouts[record.format_version]
                payload: bytes = _read_exact(stream, record.record_size - _RECORD_HEADER.size)
                if len(payload) < layout_size + 4:
                    raise ValueError(f"{self.path}/{self.stream_id}: truncated image payload")
                timestamp: int = struct.unpack_from("<q", payload, timestamp_offset)[0]
                image: bytes = payload[layout_size:]
                if not image.startswith(b"\xff\xd8") or not image.endswith(b"\xff\xd9"):
                    raise ValueError(f"{self.path}/{self.stream_id}: invalid JPEG boundaries")
                yield ImageRecord(timestamp, image)
