"""Ingest's depth PNGs (depth_png, packages/depth-png) decode bit-exactly in independent PNG readers."""

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("depth_png", reason="the ingest CLI needs depth_png from the ingest environment")
imagecodecs = pytest.importorskip("imagecodecs", reason="imagecodecs is the independent PNG decoder of these tests")

from arkitscenes_download.ingest.cli import _encode_depth_asset  # noqa: E402


def assert_readers_agree(encoded: bytes, depth_hw: np.ndarray) -> None:
    """PIL and imagecodecs both decode the blob to ``depth_hw``."""
    np.testing.assert_array_equal(np.asarray(Image.open(io.BytesIO(encoded))), depth_hw)
    np.testing.assert_array_equal(imagecodecs.png_decode(encoded), depth_hw)


@pytest.mark.parametrize("quarter_turns", [0, 1, 3])
def test_ingest_bakes_the_orientation_into_a_standard_png(tmp_path: Path, quarter_turns: int) -> None:
    rng: np.random.Generator = np.random.default_rng(47332195)
    depth_hw: np.ndarray = rng.integers(0, 65536, size=(192, 256), dtype=np.uint16)
    asset: Path = tmp_path / "47332195_1234.567.png"
    Image.fromarray(depth_hw).save(asset)
    assert_readers_agree(_encode_depth_asset((asset, quarter_turns)), np.rot90(depth_hw, quarter_turns))


@pytest.mark.parametrize("value", [0, 65535])
def test_constant_extreme_depths_roundtrip(tmp_path: Path, value: int) -> None:
    depth_hw: np.ndarray = np.full((192, 256), value, dtype=np.uint16)
    asset: Path = tmp_path / "47332195_1234.567.png"
    Image.fromarray(depth_hw).save(asset)
    assert_readers_agree(_encode_depth_asset((asset, 0)), depth_hw)
