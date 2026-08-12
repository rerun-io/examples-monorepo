"""Behavior checks for reading CA-1M frame groups without image rotation."""

import io
import json
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image

from arkitscenes_download.ca1m.archive import Ca1mFrame, parse_archive


def _png_bytes(depth_hw: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(depth_hw).save(buffer, format="PNG")
    return buffer.getvalue()


def _add_bytes(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    member = tarfile.TarInfo(name)
    member.size = len(payload)
    archive.addfile(member, io.BytesIO(payload))


def test_archive_frames_roundtrip_in_timestamp_order(tmp_path: Path) -> None:
    """Grouped RT, K, and unchanged uint16 PNG payloads become sorted frames."""
    archive_path: Path = tmp_path / "ca1m-val-00000001.tar"
    expected_depth: dict[int, np.ndarray] = {
        2_000_000_000: np.asarray([[0, 1000, 2000], [3000, 4000, 5000]], dtype=np.uint16),
        1_000_000_000: np.asarray([[7, 8], [9, 10], [11, 12]], dtype=np.uint16),
    }
    expected_png: dict[int, bytes] = {stamp_ns: _png_bytes(depth_hw) for stamp_ns, depth_hw in expected_depth.items()}
    with tarfile.open(archive_path, mode="w") as archive:
        for frame_index, stamp_ns in enumerate(expected_depth):
            faro_from_camera_44: np.ndarray = np.eye(4, dtype=np.float64)
            faro_from_camera_44[:3, 3] = [float(frame_index), 2.0, 3.0]
            intrinsics_33: np.ndarray = np.asarray([[500.0, 0.0, 1.0], [0.0, 501.0, 0.5], [0.0, 0.0, 1.0]], dtype=np.float64)
            prefix: str = f"00000001/{stamp_ns}"
            _add_bytes(archive, f"{prefix}.gt/RT.json", json.dumps(faro_from_camera_44.tolist()).encode())
            _add_bytes(archive, f"{prefix}.gt/depth.png", expected_png[stamp_ns])
            _add_bytes(archive, f"{prefix}.gt/depth/K.json", json.dumps(intrinsics_33.tolist()).encode())
            _add_bytes(archive, f"{prefix}.wide/image.png", b"ignored")

    frames: list[Ca1mFrame] = parse_archive(archive_path, expected_video_id="00000001")

    assert [frame.timestamp_ns for frame in frames] == [1_000_000_000, 2_000_000_000]
    for frame in frames:
        assert frame.depth_png == expected_png[frame.timestamp_ns]
        with Image.open(io.BytesIO(frame.depth_png)) as decoded:
            decoded_depth_hw: np.ndarray = np.asarray(decoded)
        np.testing.assert_array_equal(decoded_depth_hw, expected_depth[frame.timestamp_ns])
        expected_hw: tuple[int, int] = expected_depth[frame.timestamp_ns].shape
        assert frame.resolution_wh == (expected_hw[1], expected_hw[0])
