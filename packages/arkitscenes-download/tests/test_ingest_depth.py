"""Behavior checks for depth PNG encoding."""

import io
import unittest

import imagecodecs
import numpy as np
from PIL import Image

from arkitscenes_download.ingest.depth import encode_depth_png


class DepthPngEncodingTest(unittest.TestCase):
    """Check that encoded depth pixels survive independent PNG decoders."""

    def assert_roundtrips(self, depth_hw: np.ndarray) -> None:
        """Assert bit-exact decoding through PIL and imagecodecs at both deflate levels."""
        for level in (1, 4):
            with self.subTest(level=level):
                encoded: bytes = encode_depth_png(depth_hw, level=level)
                pil_depth_hw: np.ndarray = np.asarray(Image.open(io.BytesIO(encoded)))
                imagecodecs_depth_hw: np.ndarray = imagecodecs.png_decode(encoded)
                np.testing.assert_array_equal(pil_depth_hw, depth_hw)
                np.testing.assert_array_equal(imagecodecs_depth_hw, depth_hw)

    def test_random_uint16_depth_roundtrips(self) -> None:
        """A random 192x256 uint16 depth image roundtrips bit-exactly."""
        rng: np.random.Generator = np.random.default_rng(47332195)
        depth_hw: np.ndarray = rng.integers(0, 65536, size=(192, 256), dtype=np.uint16)
        self.assert_roundtrips(depth_hw)

    def test_non_contiguous_rotated_depth_roundtrips(self) -> None:
        """A non-contiguous np.rot90 view roundtrips bit-exactly."""
        source_hw: np.ndarray = np.arange(192 * 256, dtype=np.uint16).reshape(192, 256)
        depth_hw: np.ndarray = np.rot90(source_hw)
        self.assertFalse(depth_hw.flags.c_contiguous)
        self.assert_roundtrips(depth_hw)

    def test_constant_extreme_depths_roundtrip(self) -> None:
        """All-zero and all-65535 depth images roundtrip bit-exactly."""
        for value in (0, 65535):
            with self.subTest(value=value):
                depth_hw: np.ndarray = np.full((192, 256), value, dtype=np.uint16)
                self.assert_roundtrips(depth_hw)


if __name__ == "__main__":
    unittest.main()
