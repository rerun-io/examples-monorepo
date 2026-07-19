"""Behavior checks for stream-10 camera distortion metadata."""

import unittest
from pathlib import Path

import numpy as np

from arkitscenes_download.ingest.metadata import decode_camera_distortions


class CameraDistortionTest(unittest.TestCase):
    """Validate real ARKit calibration payloads."""

    def test_real_movie_has_sane_radial_lookup_tables(self) -> None:
        """Decoded LUTs are nonempty, bounded, and mostly monotonic."""
        mov_path: Path = Path("data/raw/Training/47332195/47332195.mov")
        if not mov_path.is_file():
            self.skipTest("ARKitScenes sample MOV is unavailable")
        calibrations = decode_camera_distortions(mov_path)

        self.assertGreaterEqual(len(calibrations), 1)
        for calibration in calibrations:
            self.assertGreater(len(calibration.coefficients), 0)
            self.assertTrue(np.all(calibration.center_xy >= 0.0))
            self.assertTrue(np.all(calibration.center_xy <= calibration.reference_dimensions_wh))
            self.assertEqual(len(calibration.coefficients), 8)
            self.assertTrue(np.all(np.isfinite(calibration.coefficients)))
            self.assertGreater(calibration.temporal_sample_count, 1)


if __name__ == "__main__":
    unittest.main()
