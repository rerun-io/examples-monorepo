"""Gen2 access units retain the integer capture clock and skip variable metadata."""

import json
import struct
from pathlib import Path

import pyarrow as pa
import pytest

from dataforge.video_encoding import FrameSource
from dataforge.vrs_hevc import VrsHevcReader


def hevc_vrs(*, metadata: bytes = b"", compression: int = 0) -> bytes:
    def tags(values: dict[str, str]) -> bytes:
        return struct.pack("<I", len(values)) + b"".join(
            struct.pack("<I", len(text.encode())) + text.encode() for pair in values.items() for text in pair
        )

    def record(payload: bytes, type_id: int, compressed: int = 0) -> bytes:
        size = len(payload)
        if compressed == 2:
            payload = pa.compress(payload, codec="zstd", asbytes=True)
        return struct.pack("<IIiIdHBBI", 32 + len(payload), 0, type_id, 2, 9.0, 1, 3, compressed, size if compressed else 0) + payload

    layout = {
        "data_layout": [
            {"name": "capture_timestamp_ns", "type": "DataPieceValue<int64_t>", "offset": 60},
            {"name": "image_key_frame_index", "type": "DataPieceValue<uint32_t>", "offset": 108},
            {"name": "focus_distance_mm", "type": "DataPieceValue<double>", "offset": 112},
            {"name": "image_metadata", "type": "DataPieceVector<uint8_t>", "index": 0},
        ]
    }
    description = record(
        struct.pack("<IiH", 1, 214, 1)
        + tags({})
        + tags({"RF:Data:2": "data_layout+image/video/codec=H.265", "DL:Data:2:0": json.dumps(layout)})
        + tags({}),
        2,
    )
    header = bytearray(80)
    header[:8] = b"VisionRe"
    header[72:80] = b"cordVRS2"
    struct.pack_into("<II", header, 16, 80, 32)
    struct.pack_into("<qq", header, 32, 80, 80 + len(description))
    records = []
    for stamp in (101, 33433434):
        fixed = bytearray(120)
        struct.pack_into("<q", fixed, 60, stamp)
        records.append(record(bytes(fixed) + struct.pack("<II", 0, len(metadata)) + metadata + b"\x00\x00\x00\x01\x26\x01test", 214, compression))
    return bytes(header) + description + b"".join(records)


def test_hevc_access_units_and_native_times(tmp_path: Path) -> None:
    path = tmp_path / "video.vrs"
    path.write_bytes(hevc_vrs(metadata=b"variable metadata"))
    records = list(VrsHevcReader(path, "214-1").images())
    assert [row.capture_timestamp_ns for row in records] == [101, 33433434]
    assert [row.image for row in records] == [b"\x00\x00\x00\x01\x26\x01test"] * 2
    assert FrameSource("hevc").input_args(fps=10) == ["-f", "hevc", "-r", "10", "-i", "pipe:0"]


def test_hevc_rejects_compressed_records(tmp_path: Path) -> None:
    path = tmp_path / "video.vrs"
    path.write_bytes(hevc_vrs(compression=1))
    with pytest.raises(ValueError, match="compressed"):
        list(VrsHevcReader(path, "214-1").images())


def test_gen2_vrs1_header_with_version2_description(tmp_path: Path) -> None:
    path = tmp_path / "video.vrs"
    raw = bytearray(hevc_vrs())
    raw[72:80] = b"cordVRS1"
    path.write_bytes(raw)
    reader = VrsHevcReader(path, "214-1")
    assert [row.capture_timestamp_ns for row in reader.images()] == [101, 33433434]


def test_zstd_wrapped_hevc_keeps_the_access_unit(tmp_path: Path) -> None:
    path = tmp_path / "video.vrs"
    path.write_bytes(hevc_vrs(compression=2))
    reader = VrsHevcReader(path, "214-1")
    assert [row.capture_timestamp_ns for row in reader.images()] == [101, 33433434]
    assert [struct.unpack_from("<q", data, 60)[0] for data in reader.records(124)] == [101, 33433434]
