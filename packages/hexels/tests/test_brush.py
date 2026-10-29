"""Brush receives the upstream Hexels cameras, split, images and SfM initialization."""

import json
import struct
from collections.abc import Callable
from pathlib import Path
from typing import TypeAlias

import numpy as np
import pytest
from PIL import Image
from plyfile import PlyData, PlyElement

from hexels_bench.brush import opengl_world_from_camera, prepare_scene, target_size

SceneFactory: TypeAlias = Callable[..., tuple[Path, Path]]


def test_prepare_writes_contract_split_at_upstream_resolution(tmp_path: Path, protocol_scene: SceneFactory) -> None:
    source, contract = protocol_scene(tmp_path / "dtu/scan1", tmp_path / "contract.json", ["0000", "0008"])
    prepared = prepare_scene(source, contract, tmp_path / "prepared")
    output: Path = tmp_path / "prepared"
    test_frames: list[dict] = json.loads((output / "transforms_test.json").read_text())["frames"]
    train_frames: list[dict] = json.loads((output / "transforms_train.json").read_text())["frames"]
    assert [frame["file_path"] for frame in test_frames] == ["test/0000.png", "test/0008.png"]
    assert sorted(path.name for path in (output / "test").iterdir()) == ["0000.png", "0008.png"]
    assert len(train_frames) == 8
    assert (prepared.train_views, prepared.test_views, prepared.sfm_points) == (8, 2, 3)
    with Image.open(source / "images/0008.png") as original:
        expected = np.stack([np.asarray(band.resize((8, 6))) for band in original.split()[:3]], axis=-1)
    with Image.open(output / "test/0008.png") as written:
        assert written.mode == "RGB"
        np.testing.assert_array_equal(np.asarray(written), expected)
    assert test_frames[0]["camera_angle_x"] == 0.9 and test_frames[0]["camera_angle_y"] == 0.7
    assert "cx" not in test_frames[0], "upstream ignores the principal point; Brush must centre it too"


def test_target_size_uses_python_rounding_like_2dgs() -> None:
    assert target_size(17, 13, 2) == (8, 6)
    assert target_size(1554, 1162, 2) == (777, 581)
    assert target_size(1297, 840, 1) == (1297, 840)
    with pytest.raises(ValueError, match="resolution"):
        target_size(100, 100, 3)


def test_world_from_camera_matches_upstream_camera_center() -> None:
    """scan24 test view 0000 from the published contract: its center is upstream's own record."""
    world_view: np.ndarray = np.array(
        [
            [0.9702627062797546, -0.014742869883775711, -0.2416049987077713, 0.0],
            [0.007479910738766193, 0.9994929432868958, -0.0309509988874197, 0.0],
            [0.2419387847185135, 0.028223415836691856, 0.9698809385299683, 0.0],
            [-0.2518884837627411, -0.044171787798404694, 2.083570957183838, 1.0],
        ]
    )
    world_from_camera: np.ndarray = opengl_world_from_camera(world_view)
    np.testing.assert_allclose(world_from_camera[:3, 3], [0.7471479177474976, 0.11052209138870239, -1.9586275815963745], atol=1e-6)
    # OpenGL looks down -Z: the third column is the negated OpenCV forward axis (third row of world_view's rotation block).
    np.testing.assert_allclose(world_from_camera[:3, 2], -world_view[:3, 2], atol=1e-6)
    np.testing.assert_allclose(world_from_camera[:3, 1], -world_view[:3, 1], atol=1e-6)


def test_contract_split_must_follow_lexical_every_eighth_rule(tmp_path: Path, protocol_scene: SceneFactory) -> None:
    source, contract = protocol_scene(tmp_path / "dtu/scan1", tmp_path / "contract.json", ["0001", "0009"])
    with pytest.raises(ValueError, match="every 8th"):
        prepare_scene(source, contract, tmp_path / "prepared")
    assert not (tmp_path / "prepared").exists()


def test_image_size_must_match_contract(tmp_path: Path, protocol_scene: SceneFactory) -> None:
    source, contract = protocol_scene(tmp_path / "dtu/scan1", tmp_path / "contract.json", ["0000", "0008"], contract_size=(9, 7))
    with pytest.raises(ValueError, match="contract"):
        prepare_scene(source, contract, tmp_path / "prepared")


def test_init_cloud_keeps_every_sfm_point_with_unit_colors(tmp_path: Path, protocol_scene: SceneFactory) -> None:
    source, contract = protocol_scene(tmp_path / "dtu/scan1", tmp_path / "contract.json", ["0000", "0008"])
    prepare_scene(source, contract, tmp_path / "prepared")
    vertex = PlyData.read(tmp_path / "prepared/init.ply")["vertex"]
    np.testing.assert_allclose(np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1), [[0.5, -1.0, 2.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
    np.testing.assert_allclose(vertex["red"], [1.0, 0.0, 10 / 255], rtol=1e-6)
    np.testing.assert_allclose(vertex["blue"], [0.2, 0.0, 30 / 255], rtol=1e-6)


def test_binary_points_follow_colmap_layout_and_ply_wins(tmp_path: Path, protocol_scene: SceneFactory) -> None:
    source, contract = protocol_scene(tmp_path / "dtu/scan1", tmp_path / "contract.json", ["0000", "0008"])
    sparse: Path = source / "sparse/0"
    (sparse / "points3D.txt").unlink()
    track: bytes = struct.pack("<ii", 1, 0) * 2
    rows: bytes = b"".join(struct.pack("<Q3d3BdQ", i, x, 2.0 * x, 3.0, 9, 8, 7, 0.5, 2) + track for i, x in ((7, 1.0), (8, -1.0)))
    (sparse / "points3D.bin").write_bytes(struct.pack("<Q", 2) + rows)
    assert prepare_scene(source, contract, tmp_path / "from-bin").sfm_points == 2
    vertex = PlyData.read(tmp_path / "from-bin/init.ply")["vertex"]
    np.testing.assert_allclose(vertex["y"], [2.0, -2.0])
    np.testing.assert_allclose(vertex["green"], [8 / 255, 8 / 255], rtol=1e-6)
    # 2DGS reads sparse/0/points3D.ply first when it exists (its storePly writes uint8 colors on first load).
    cloud = np.array(
        [(4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 255, 0, 0)],
        dtype=[(n, "f4") for n in ("x", "y", "z", "nx", "ny", "nz")] + [(n, "u1") for n in ("red", "green", "blue")],
    )
    PlyData([PlyElement.describe(cloud, "vertex")]).write(str(sparse / "points3D.ply"))
    assert prepare_scene(source, contract, tmp_path / "from-ply").sfm_points == 1
    vertex = PlyData.read(tmp_path / "from-ply/init.ply")["vertex"]
    assert (float(vertex["z"][0]), float(vertex["red"][0])) == (6.0, 1.0)


def test_completed_preparation_is_reused(tmp_path: Path, protocol_scene: SceneFactory) -> None:
    source, contract = protocol_scene(tmp_path / "dtu/scan1", tmp_path / "contract.json", ["0000", "0008"])
    first = prepare_scene(source, contract, tmp_path / "prepared")
    (source / "images/0000.png").unlink()
    assert prepare_scene(source, contract, tmp_path / "prepared") == first
