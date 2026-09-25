"""Raw VRS image records retain their shipped order, format and device clock."""

import json
import struct
from pathlib import Path
from typing import Literal

import pytest
from conftest import vrs_file, vrs_record

from dataforge.vrs import VrsImageReader


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
    reader = VrsImageReader(path, "214-1")
    frames = list(reader.images())
    assert [frame.capture_timestamp_ns for frame in frames] == [100, 200]
    assert [frame.image for frame in frames] == [b"\xff\xd8test\xff\xd9"] * 2


@pytest.mark.parametrize("compression", [1, 2])
def test_vrs_compressed_image_refused(tmp_path: Path, compression: int) -> None:
    path = tmp_path / "compressed.vrs"
    path.write_bytes(synthetic_vrs(compression=compression))
    with pytest.raises(ValueError, match="compressed image"):
        list(VrsImageReader(path, "214-1").images())


def test_vrs_unsupported_image_format_names_format(tmp_path: Path) -> None:
    path = tmp_path / "raw.vrs"
    path.write_bytes(synthetic_vrs(image_format="raw"))
    with pytest.raises(ValueError, match="data_layout/size=12\\+image/raw"):
        VrsImageReader(path, "214-1")


def test_vrs_invalid_layout_refused_at_open(tmp_path: Path) -> None:
    path = tmp_path / "layout.vrs"
    path.write_bytes(synthetic_vrs(timestamp_offset=8))
    with pytest.raises(ValueError, match=r"layout.vrs/214-1: invalid capture_timestamp_ns layout"):
        VrsImageReader(path, "214-1")


@pytest.mark.parametrize(
    "preview,timestamps,source_count,compression,error",
    [
        (True, [100], 2, 1, None),
        (False, [100, 200], 2, 0, None),
        (False, [100, 200], 3, 0, "2 image records, expected 3"),
        (True, [100, 200, 300], 3, 0, "2 image records, expected 3"),
        (True, [101], 2, 0, "capture timestamp mismatch at frame 0"),
        (False, [100, 200], 2, 1, "compressed image record"),
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
        VrsImageReader(path, "214-1").images(), np.array(timestamps, dtype=np.int64), source_count, preview=preview, where=f"{path}/214-1"
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
        list(VrsImageReader(path, "214-1").images())


@pytest.mark.integration
@pytest.mark.parametrize(
    "device,sequence,streams", [("aria", "P0001_4bf4e21a", ["214-1", "1201-1", "1201-2"]), ("quest3", "P0002_5a9cfa51", ["1201-1", "1201-2"])]
)
def test_raw_vrs_timestamps_match_provider(device: str, sequence: str, streams: list[Literal["214-1", "1201-1", "1201-2"]]) -> None:
    from projectaria_tools.core.stream_id import StreamId

    from dataforge import aria, paths

    path = paths.raw_root() / "hot3d" / device / sequence / "recording.vrs"
    if not path.is_file():
        pytest.skip(f"HOT3D VRS asset absent: {path}")
    provider = aria.open_vrs(path)
    for stream in streams:
        expected = aria.frame_timestamps_ns(provider, stream)
        count = 0
        for record, timestamp in zip(VrsImageReader(path, stream).images(), expected, strict=True):
            assert record.capture_timestamp_ns == timestamp
            count += 1
        assert count == provider.get_num_data(StreamId(stream))
