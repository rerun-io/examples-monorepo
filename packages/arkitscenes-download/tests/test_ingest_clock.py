"""Behavior checks for correspondence-based camera clock recovery."""

import unittest

import numpy as np

from arkitscenes_download.ingest.clock import analyze_clock_correspondences, clamped_to_epoch, nearest_index, shared_epoch


class ClockAlignmentTest(unittest.TestCase):
    """Reject alignments that a periodic nearest-frame grid can alias."""

    def test_one_frame_camera_offset_disagreement_is_rejected(self) -> None:
        """An ultrawide mapping shifted by one 60 Hz frame fails the primary gate."""
        pts = np.arange(120, dtype=np.float64) / 60.0
        wide = pts + 100.0
        ultrawide = pts + 100.0 + 1.0 / 60.0
        with self.assertRaisesRegex(ValueError, "disagree"):
            analyze_clock_correspondences(pts, wide, pts, ultrawide)


class SharedTimelineTest(unittest.TestCase):
    """Every sequence consumer must derive the identical shared timeline."""

    def test_shared_epoch_is_first_instant_both_cameras_have_a_frame(self) -> None:
        """The later first frame wins and earlier leading frames are counted."""
        wide = np.array([9.9, 9.95, 10.0, 10.05], dtype=np.float64)
        ultrawide = np.array([10.0, 10.05], dtype=np.float64)
        epoch, wide_drop, ultrawide_drop = shared_epoch(wide, ultrawide)
        self.assertEqual(epoch, 10.0)
        self.assertEqual(wide_drop, 2)
        self.assertEqual(ultrawide_drop, 0)

    def test_clamped_to_epoch_snaps_only_a_near_epoch_first_sample(self) -> None:
        """A first sample within the clamp window snaps back; a late stream stays honest."""
        near = clamped_to_epoch(np.array([10.1, 10.2], dtype=np.float64), epoch=10.0)
        np.testing.assert_allclose(near, [10.0, 10.2])
        late = clamped_to_epoch(np.array([10.3, 10.4], dtype=np.float64), epoch=10.0)
        np.testing.assert_allclose(late, [10.3, 10.4])
        self.assertEqual(len(clamped_to_epoch(np.array([], dtype=np.float64), epoch=10.0)), 0)

    def test_nearest_index_picks_the_closer_neighbor(self) -> None:
        """Both insertion neighbors are considered, including the boundaries."""
        timestamps = np.array([1.0, 1.01, 1.02], dtype=np.float64)
        self.assertEqual(nearest_index(timestamps, 1.0119), 1)
        self.assertEqual(nearest_index(timestamps, 0.5), 0)
        self.assertEqual(nearest_index(timestamps, 2.0), 2)
        with self.assertRaisesRegex(ValueError, "empty"):
            nearest_index(np.array([], dtype=np.float64), 1.0)


if __name__ == "__main__":
    unittest.main()
