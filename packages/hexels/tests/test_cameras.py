import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from serde.json import to_json

from hexels_bench.cameras import CameraContract, CamerasFile, cameras_file, load_target, write_cameras_file, write_targets
from hexels_bench.layout import read_json
from hexels_bench.png16 import read_png16


def upstream_digest(rows: list[tuple[str, np.ndarray]]) -> str:
    """Upstream's ordered target digest, written out independently of the module under test."""
    entries: list[list[str]] = []
    for name, rgb in rows:
        tensor: np.ndarray = np.ascontiguousarray((rgb.astype(np.float32) / np.float32(255.0)).transpose(2, 0, 1))
        digest = hashlib.sha256(tensor.dtype.str.encode() + json.dumps(list(tensor.shape), separators=(",", ":")).encode() + tensor.tobytes())
        entries.append([name, digest.hexdigest()])
    return hashlib.sha256(json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_contract(path: Path, names: list[str], size: tuple[int, int], digest: str) -> Path:
    camera = {"name": "", "width": size[0], "height": size[1], "fov_x": 0.5, "fov_y": 0.4,
              "world_view_transform": np.eye(4).tolist(), "full_proj_transform": np.eye(4).tolist(), "camera_center": [0.0, 0.0, 0.0]}  # fmt: skip
    contract = {
        "settings": {"source_path": "data/dtu/scanX", "images": "images", "resolution": 2, "eval": True, "llffhold": 8, "data_device": "cpu"},
        "training": {"count": 0, "ordered_target_tensors_sha256": "x"},
        "test": {"count": len(names), "ordered_names_sha256": "x", "ordered_geometry_sha256": "x", "ordered_target_tensors_sha256": digest},
        "geometry": {"training": [], "test": [{**camera, "name": name} for name in names]},
    }  # fmt: skip
    path.write_text(json.dumps(contract))
    return path


def test_rgba_target_is_not_premultiplied(tmp_path: Path) -> None:
    rgba: np.ndarray = np.random.default_rng(0).integers(0, 256, size=(12, 10, 4), dtype=np.uint8)
    rgba[..., 3] = 0  # fully transparent: a premultiplying resize would return black
    Image.fromarray(rgba, mode="RGBA").save(tmp_path / "a.png")
    Image.fromarray(np.ascontiguousarray(rgba[..., :3]), mode="RGB").save(tmp_path / "b.png")
    np.testing.assert_array_equal(load_target(tmp_path / "a.png", 5, 6), load_target(tmp_path / "b.png", 5, 6))
    assert load_target(tmp_path / "a.png", 5, 6).max() > 0


def test_targets_match_the_contract_digest(tmp_path: Path) -> None:
    rng: np.random.Generator = np.random.default_rng(1)
    scene: Path = tmp_path / "scene"
    (scene / "images").mkdir(parents=True)
    rows: list[tuple[str, np.ndarray]] = []
    for name in ("0000", "0008"):
        Image.fromarray(rng.integers(0, 256, size=(12, 10, 4), dtype=np.uint8), mode="RGBA").save(scene / "images" / f"{name}.png")
        rows.append((name, load_target(scene / "images" / f"{name}.png", 5, 6)))
    good = read_json(CameraContract, write_contract(tmp_path / "good.json", ["0000", "0008"], (5, 6), upstream_digest(rows)))
    written: list[Path] = write_targets(good, scene, tmp_path / "gt")
    assert [path.name for path in written] == ["0000.png", "0008.png"]
    np.testing.assert_array_equal(read_png16(written[0]), rows[0][1].astype(np.uint16) * 257)
    bad = read_json(CameraContract, write_contract(tmp_path / "bad.json", ["0000", "0008"], (5, 6), "0" * 64))
    with pytest.raises(ValueError, match="digest"):
        write_targets(bad, scene, tmp_path / "gt-bad")


def test_cameras_file_copies_contract_records(tmp_path: Path) -> None:
    contract = read_json(CameraContract, write_contract(tmp_path / "c.json", ["0000"], (5, 6), "x"))
    cameras: CamerasFile = cameras_file(contract, "dtu", "scanX", [0.0, 0.0, 0.0])
    write_cameras_file(tmp_path / "cameras.json", cameras)
    text: dict = json.loads((tmp_path / "cameras.json").read_text())
    assert text["format"] == "hexels-cameras-1"
    assert text["cameras"][0]["world_view_transform"] == np.eye(4).tolist()
    assert "full_proj_transform" not in text["cameras"][0]
    assert to_json(read_json(CamerasFile, tmp_path / "cameras.json")) == to_json(cameras)
