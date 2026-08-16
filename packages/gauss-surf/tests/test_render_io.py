"""Behavior contracts for the full-grid camera manifest."""

import json
from pathlib import Path

import numpy as np
import pytest

from gauss_surf.render_io import RenderCamera, load_render_cameras


def test_renderer_loads_image_free_full_grid_camera_manifest(tmp_path: Path) -> None:
    cameras_path: Path = tmp_path / "cameras_all.json"
    world_from_camera_44: np.ndarray = np.eye(4, dtype=np.float32)
    world_from_camera_44[:3, 3] = np.asarray((2.0, 3.0, 4.0), dtype=np.float32)
    frames: list[dict[str, object]] = [
        {
            "stem": "wide_all_000000",
            "camera": "wide",
            "timestamp_ns": 100,
            "fl_x": 500.0,
            "fl_y": 501.0,
            "cx": 320.0,
            "cy": 240.0,
            "w": 640,
            "h": 480,
            "transform_matrix": world_from_camera_44.tolist(),
        },
        {
            "stem": "uw_all_000000",
            "camera": "uw",
            "timestamp_ns": 100,
            "fl_x": 400.0,
            "fl_y": 401.0,
            "cx": 319.0,
            "cy": 239.0,
            "w": 640,
            "h": 480,
            "transform_matrix": world_from_camera_44.tolist(),
        },
    ]
    cameras_path.write_text(
        json.dumps({"schema_version": 1, "camera_model": "OPENCV", "counts": {"wide": 1, "uw": 1, "total": 2}, "frames": frames}),
        encoding="utf-8",
    )

    cameras: list[RenderCamera] = load_render_cameras(cameras_path)

    assert [(camera.stem, camera.camera, camera.timestamp_ns) for camera in cameras] == [
        ("wide_all_000000", "wide", 100),
        ("uw_all_000000", "uw", 100),
    ]
    np.testing.assert_allclose(cameras[0].world_from_camera_34[:, 3], [2.0, 3.0, 4.0])


def test_renderer_rejects_camera_manifest_count_mismatch(tmp_path: Path) -> None:
    cameras_path: Path = tmp_path / "cameras_all.json"
    frame: dict[str, object] = {
        "stem": "wide_all_000000",
        "camera": "wide",
        "timestamp_ns": 0,
        "fl_x": 500.0,
        "fl_y": 501.0,
        "cx": 320.0,
        "cy": 240.0,
        "w": 640,
        "h": 480,
        "transform_matrix": np.eye(4, dtype=np.float32).tolist(),
    }
    cameras_path.write_text(
        json.dumps({"schema_version": 1, "camera_model": "OPENCV", "counts": {"wide": 2, "uw": 0, "total": 2}, "frames": [frame]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="manifest counts"):
        load_render_cameras(cameras_path)
