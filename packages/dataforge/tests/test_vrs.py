"""Raw VRS image records retain their shipped order, format and device clock."""

import json
import struct
from pathlib import Path

import numpy as np
import pytest
from conftest import vrs_file, vrs_record

from dataforge.vrs import CONFIGURATION_RECORD, VrsFile, VrsImageReader


def synthetic_vrs(*, compression: int = 0, image_format: str = "jpg", timestamp_offset: int = 4) -> bytes:
    layout = json.dumps({"data_layout": [{"name": "capture_timestamp_ns", "type": "DataPieceValue<int64_t>", "offset": timestamp_offset}]})
    return vrs_file(
        {214: {"RF:Data:2": f"data_layout/size=12+image/{image_format}", "DL:Data:2:0": layout}},
        [
            vrs_record(b"ignored", type_id=1202, compression=2),
            *(
                vrs_record(b"pad!" + struct.pack("<q", stamp) + b"\xff\xd8test\xff\xd9", type_id=214, compression=compression if stamp == 200 else 0)
                for stamp in (100, 200)
            ),
        ],
        user_tags={"device": "test"},
    )


def test_vrs_image_order_layout_and_other_stream(tmp_path: Path) -> None:
    path = tmp_path / "tiny.vrs"
    path.write_bytes(synthetic_vrs())
    reader = VrsImageReader(VrsFile(path), "214-1")
    frames = list(reader.images())
    assert [frame.capture_timestamp_ns for frame in frames] == [100, 200]
    assert [frame.image for frame in frames] == [b"\xff\xd8test\xff\xd9"] * 2


@pytest.mark.parametrize("compression", [1, 2])
def test_vrs_compressed_image_decoded(tmp_path: Path, compression: int) -> None:
    # The recorder compresses a record when that saves space: P0015_179e1b84 has one zstd RGB record.
    path = tmp_path / "compressed.vrs"
    path.write_bytes(synthetic_vrs(compression=compression))
    frames = list(VrsImageReader(VrsFile(path), "214-1").images())
    assert [frame.capture_timestamp_ns for frame in frames] == [100, 200]
    assert [frame.image for frame in frames] == [b"\xff\xd8test\xff\xd9"] * 2


def test_vrs_unknown_compression_refused(tmp_path: Path) -> None:
    path = tmp_path / "compressed.vrs"
    path.write_bytes(synthetic_vrs(compression=3))
    with pytest.raises(ValueError, match="unsupported compressed record"):
        list(VrsImageReader(VrsFile(path), "214-1").images())


def test_vrs_unsupported_image_format_names_format(tmp_path: Path) -> None:
    path = tmp_path / "raw.vrs"
    path.write_bytes(synthetic_vrs(image_format="raw"))
    with pytest.raises(ValueError, match="data_layout/size=12\\+image/raw"):
        VrsImageReader(VrsFile(path), "214-1")


def test_vrs_invalid_layout_refused_at_open(tmp_path: Path) -> None:
    path = tmp_path / "layout.vrs"
    path.write_bytes(synthetic_vrs(timestamp_offset=8))
    with pytest.raises(ValueError, match=r"layout.vrs/214-1: invalid capture_timestamp_ns layout"):
        VrsImageReader(VrsFile(path), "214-1")


@pytest.mark.parametrize(
    "preview,timestamps,source_count,compression,error",
    [
        (True, [100], 2, 1, None),
        (False, [100, 200], 2, 0, None),
        (False, [100, 200], 3, 0, "2 image records, expected 3"),
        (True, [100, 200, 300], 3, 0, "2 image records, expected 3"),
        (True, [101], 2, 0, "capture timestamp mismatch at frame 0"),
        (False, [100, 200], 2, 2, None),
    ],
)
def test_census_images_checks_selected_census(
    tmp_path: Path, preview: bool, timestamps: list[int], source_count: int, compression: int, error: str | None
) -> None:
    import numpy as np

    from dataforge.vrs import census_images

    path = tmp_path / "camera.vrs"
    path.write_bytes(synthetic_vrs(compression=compression))
    images = census_images(
        VrsImageReader(VrsFile(path), "214-1").images(), np.array(timestamps, dtype=np.int64), source_count, preview=preview, where=f"{path}/214-1"
    )
    if error is not None:
        with pytest.raises(ValueError, match=error):
            list(images)
    else:
        assert list(images) == [b"\xff\xd8test\xff\xd9"] * len(timestamps)


@pytest.mark.parametrize(
    "mutation,match",
    [(lambda b: b[:-1], "record size"), (lambda b: b[:-2] + b"xx", "JPEG boundaries"), (lambda b: b"badmagic" + b[8:], "file header")],
)
def test_vrs_corrupt_input_refused(tmp_path: Path, mutation, match: str) -> None:
    path = tmp_path / "broken.vrs"
    path.write_bytes(mutation(synthetic_vrs()))
    with pytest.raises(ValueError, match=match):
        list(VrsImageReader(VrsFile(path), "214-1").images())


@pytest.mark.parametrize("compression", [0, 1, 2])
def test_capture_timestamps_walk_only_the_layout(tmp_path: Path, compression: int) -> None:
    path = tmp_path / "tiny.vrs"
    path.write_bytes(synthetic_vrs(compression=compression))
    assert VrsImageReader(VrsFile(path), "214-1").capture_timestamps().tolist() == [100, 200]


IMU_LAYOUT: str = json.dumps(
    {
        "data_layout": [
            {"name": "accelerometer_valid", "type": "DataPieceValue<Bool>", "offset": 0},
            {"name": "gyroscope_valid", "type": "DataPieceValue<Bool>", "offset": 1},
            {"name": "temperature_deg_c", "type": "DataPieceValue<double>", "offset": 3},
            {"name": "capture_timestamp_ns", "type": "DataPieceValue<int64_t>", "offset": 11},
            {"name": "accelerometer", "type": "DataPieceArray<float>", "offset": 43, "size": 3},
            {"name": "gyroscope", "type": "DataPieceArray<float>", "offset": 55, "size": 3},
        ]
    }
)
"""Aria's motion DataLayout (79 bytes; magnetometer and the other clocks left out)."""


def imu_record(stamp: int, accel: tuple[float, float, float], gyro: tuple[float, float, float], *, valid: bool = True, compression: int = 0) -> bytes:
    payload = bytearray(79)
    struct.pack_into("<??", payload, 0, valid, True)
    struct.pack_into("<q", payload, 11, stamp)
    struct.pack_into("<3f", payload, 43, *accel)
    struct.pack_into("<3f", payload, 55, *gyro)
    return vrs_record(bytes(payload), type_id=1202, compression=compression)


def imu_vrs(layout: str = IMU_LAYOUT) -> bytes:
    configuration = json.dumps({"data_layout": [{"name": "image_width", "type": "DataPieceValue<uint32_t>", "offset": 12}, {"name": "image_height", "type": "DataPieceValue<uint32_t>", "offset": 16}]})
    return vrs_file(
        {
            1202: {"RF:Data:2": "data_layout/size=79", "DL:Data:2:0": layout},
            214: {"RF:Configuration:2": "data_layout", "DL:Configuration:2:0": configuration},
        },
        [
            vrs_record(bytes(12) + struct.pack("<II", 1408, 704), type_id=214, record_type=CONFIGURATION_RECORD),
            imu_record(10, (0.1, 0.2, 9.8), (0.01, 0.02, 0.03)),
            imu_record(20, (0.3, 0.4, 9.7), (0.04, 0.05, 0.06), valid=False, compression=2),
            imu_record(30, (0.5, 0.6, 9.6), (0.07, 0.08, 0.09), compression=1),
        ],
    )


def test_imu_records_are_read_by_their_layout(tmp_path: Path) -> None:
    path = tmp_path / "imu.vrs"
    path.write_bytes(imu_vrs())
    records = VrsFile(path).imu("1202-1")
    assert records.capture_timestamp_ns.tolist() == [10, 20, 30]
    assert records.accel_valid.tolist() == [True, False, True] and records.gyro_valid.all()
    np.testing.assert_array_equal(records.accel_msec2, np.array([[0.1, 0.2, 9.8], [0.3, 0.4, 9.7], [0.5, 0.6, 9.6]], dtype=np.float32))
    np.testing.assert_array_equal(records.gyro_radsec[2], np.array([0.07, 0.08, 0.09], dtype=np.float32))


def test_imu_format_versions_are_read_at_their_own_layout_size(tmp_path: Path) -> None:
    # Version 3 is a shorter layout with its pieces elsewhere; each record is read by its own version.
    short_layout = json.dumps(
        {
            "data_layout": [
                {"name": "capture_timestamp_ns", "type": "DataPieceValue<int64_t>", "offset": 0},
                {"name": "accelerometer_valid", "type": "DataPieceValue<Bool>", "offset": 8},
                {"name": "gyroscope_valid", "type": "DataPieceValue<Bool>", "offset": 9},
                {"name": "accelerometer", "type": "DataPieceArray<float>", "offset": 10, "size": 3},
                {"name": "gyroscope", "type": "DataPieceArray<float>", "offset": 22, "size": 3},
            ]
        }
    )
    short = bytearray(34)
    struct.pack_into("<q??3f3f", short, 0, 20, False, True, 1.5, 2.5, 3.5, 0.25, 0.5, 0.75)
    path = tmp_path / "mixed.vrs"
    path.write_bytes(
        vrs_file(
            {1202: {"RF:Data:2": "data_layout/size=79", "DL:Data:2:0": IMU_LAYOUT, "RF:Data:3": "data_layout/size=34", "DL:Data:3:0": short_layout}},
            [
                imu_record(10, (0.1, 0.2, 9.8), (0.01, 0.02, 0.03)),
                vrs_record(bytes(short), type_id=1202, format_version=3),
                imu_record(30, (0.5, 0.6, 9.6), (0.07, 0.08, 0.09), compression=2),
                vrs_record(bytes(short), type_id=1202, format_version=3, compression=1),
            ],
        )
    )
    records = VrsFile(path).imu("1202-1")
    assert records.capture_timestamp_ns.tolist() == [10, 20, 30, 20]
    assert records.accel_valid.tolist() == [True, False, True, False] and records.gyro_valid.all()
    np.testing.assert_array_equal(records.accel_msec2[[1, 3]], np.array([[1.5, 2.5, 3.5]] * 2, dtype=np.float32))
    np.testing.assert_array_equal(records.gyro_radsec[[0, 1, 2]], np.array([[0.01, 0.02, 0.03], [0.25, 0.5, 0.75], [0.07, 0.08, 0.09]], dtype=np.float32))


def test_a_record_shorter_than_its_own_layout_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "short.vrs"
    path.write_bytes(vrs_file({1202: {"RF:Data:2": "data_layout/size=79", "DL:Data:2:0": IMU_LAYOUT}}, [vrs_record(bytes(40), type_id=1202)]))
    with pytest.raises(ValueError, match="truncated data layout"):
        VrsFile(path).imu("1202-1")


def test_imu_layout_without_a_piece_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "imu.vrs"
    layout = json.loads(IMU_LAYOUT)
    layout["data_layout"] = [piece for piece in layout["data_layout"] if piece["name"] != "gyroscope"]
    path.write_bytes(imu_vrs(json.dumps(layout)))
    with pytest.raises(ValueError, match="piece gyroscope"):
        VrsFile(path).imu("1202-1")


def test_image_size_comes_from_the_configuration_record(tmp_path: Path) -> None:
    path = tmp_path / "imu.vrs"
    path.write_bytes(imu_vrs())
    assert VrsFile(path).image_size("214-1") == (1408, 704)
    with pytest.raises(ValueError, match="stream 214-9 absent"):
        VrsFile(path).image_size("214-9")
