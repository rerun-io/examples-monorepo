"""Behavior checks for correspondence-based camera clock recovery."""

import unittest

import numpy as np

from arkitscenes_download.ingest.clock import analyze_clock_correspondences


class ClockAlignmentTest(unittest.TestCase):
    """Reject alignments that a periodic nearest-frame grid can alias."""

    def test_one_frame_camera_offset_disagreement_is_rejected(self) -> None:
        """An ultrawide mapping shifted by one 60 Hz frame fails the primary gate."""
        pts = np.arange(120, dtype=np.float64) / 60.0
        wide = pts + 100.0
        ultrawide = pts + 100.0 + 1.0 / 60.0
        with self.assertRaisesRegex(ValueError, "disagree"):
            analyze_clock_correspondences(pts, wide, pts, ultrawide)


if __name__ == "__main__":
    unittest.main()
