"""Shared synthetic inputs."""

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from hexels_bench.cameras import load_target, ordered_targets_sha256, target_sha256

NAMES: list[str] = [f"{index:04d}" for index in range(10)]


def write_protocol_scene(
    source: Path, contract_path: Path, test_names: list[str], *, image_size: tuple[int, int] = (17, 13), contract_size: tuple[int, int] = (8, 6)
) -> tuple[Path, Path]:
    """A 2DGS-style scene (RGBA images, COLMAP text points) and its upstream camera contract at resolution 2, with the
    target digests of the images at `contract_size` (the module's digest functions; `test_cameras` checks them
    against an independent copy of upstream's)."""
    (source / "images").mkdir(parents=True)
    (source / "sparse/0").mkdir(parents=True)
    rng: np.random.Generator = np.random.default_rng(0)
    for name in NAMES:
        rgba = rng.integers(0, 256, (image_size[1], image_size[0], 4), dtype=np.uint8)
        rgba[..., 3] = 0  # Upstream keeps the raw RGB; alpha must not darken it.
        Image.fromarray(rgba, "RGBA").save(source / "images" / f"{name}.png")
    (source / "sparse/0/points3D.txt").write_text(
        "# 3D point list\n1 0.5 -1.0 2.0 255 0 51 0.1 1 0\n2 0.0 0.0 1.0 0 255 0 0.2 2 1 3 4\n3 1.0 1.0 1.0 10 20 30 0.3 1 2\n"
    )

    def camera(name: str, index: int) -> dict:
        return {
            "name": name,
            "width": contract_size[0],
            "height": contract_size[1],
            "fov_x": 0.9,
            "fov_y": 0.7,
            "world_view_transform": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.1 * index, 0.0, 3.0, 1.0]],
            "camera_center": [-0.1 * index, 0.0, -3.0],
        }

    geometry: dict = {
        "training": [camera(name, i) for i, name in enumerate(NAMES) if name not in test_names],
        "test": [camera(name, i) for i, name in enumerate(NAMES) if name in test_names],
    }
    def digest(rows: list[dict]) -> str:
        targets: list[list[str]] = [[row["name"], target_sha256(load_target(source / "images" / f"{row['name']}.png", *contract_size))] for row in rows]
        return ordered_targets_sha256(targets)

    split = {
        key: {"count": len(rows), "ordered_names_sha256": "n", "ordered_geometry_sha256": "g", "ordered_target_tensors_sha256": digest(rows)}
        for key, rows in geometry.items()
    }
    contract: dict = {
        "settings": {"source_path": "data/dtu/scan1", "images": "images", "resolution": 2, "eval": True, "llffhold": 8, "data_device": "cpu"},
        "training": split["training"],
        "test": split["test"] | {"ordered_names": test_names},
        "geometry": geometry,
    }
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    contract_path.write_text(json.dumps(contract))
    return source, contract_path


@pytest.fixture
def protocol_scene() -> Callable[..., tuple[Path, Path]]:
    """Factory for a synthetic Hexels-protocol scene; see ``write_protocol_scene``."""
    return write_protocol_scene
