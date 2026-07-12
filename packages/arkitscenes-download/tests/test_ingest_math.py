"""Behavior checks for ingestion unit and orientation conventions."""

import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from arkitscenes_download.ingest.imu import gyro_degrees_to_radians
from arkitscenes_download.ingest.rig import (
    Intrinsics,
    bake_camera_orientation,
    measured_orientation_quarter_turns,
    orientation_ambiguity,
    rotate_intrinsics_sequence,
    rotate_pixels,
    sky_angles,
)


class IngestMathTest(unittest.TestCase):
    """Check observable numeric conventions used in the RRD."""

    def test_gyro_is_converted_to_radians_per_second(self) -> None:
        """A 180 degree/s sample becomes pi radians/s."""
        gyro_deg_s = np.asarray([[180.0, 0.0, -90.0]], dtype=np.float64)
        np.testing.assert_allclose(gyro_degrees_to_radians(gyro_deg_s), [[np.pi, 0.0, -np.pi / 2.0]])

    def test_sky_angle_uses_clockwise_image_rotation_convention(self) -> None:
        """Projected up, right, and down map to 0, +pi/2, and pi."""
        world_from_camera = Rotation.from_euler("xyx", [[-90.0, 0.0, 0.0], [0.0, -90.0, 0.0], [90.0, 0.0, 0.0]], degrees=True).as_quat()
        np.testing.assert_allclose(sky_angles(world_from_camera), [0.0, np.pi / 2.0, np.pi], atol=1e-12)

    def test_orientation_is_rounded_from_median_measured_gravity(self) -> None:
        """Stable measured angles select the matching np.rot90 quarter turn."""
        measured_angles = np.asarray([-1.60, -1.57, -1.54], dtype=np.float64)
        self.assertEqual(measured_orientation_quarter_turns(measured_angles), 3)

    def test_orientation_wraps_circularly_at_pi(self) -> None:
        """Angles straddling -pi/+pi select a half turn, not zero."""
        self.assertEqual(measured_orientation_quarter_turns(np.asarray([-3.13, 3.13], dtype=np.float64)), 2)

    def test_cardinal_orientation_is_far_from_ambiguity_boundary(self) -> None:
        """A tight distribution around pi is 45 degrees from a quarter-turn boundary."""
        ambiguous, _, boundary_distance = orientation_ambiguity(np.asarray([3.13, -3.13], dtype=np.float64))
        self.assertFalse(ambiguous)
        self.assertAlmostEqual(boundary_distance, np.pi / 4.0, places=2)

    def test_rotated_intrinsics_project_to_rotated_pixels_for_every_quarter_turn(self) -> None:
        """Each baked calibration maps a ray to the same pixel as np.rot90."""
        intrinsics = np.asarray([[700.0, 0.0, 311.25], [0.0, 710.0, 233.75], [0.0, 0.0, 1.0]])
        camera_point = np.asarray([0.31, -0.17, 2.4])
        pixel = (intrinsics @ camera_point)[:2] / camera_point[2]
        for quarter_turns in range(4):
            sequence = Intrinsics(np.asarray([0.0]), intrinsics[None, ...])
            rotated_sequence, _ = rotate_intrinsics_sequence(sequence, (640, 480), quarter_turns)
            rotated_intrinsics = rotated_sequence.matrices[0]
            rotated_point = bake_camera_orientation(Rotation.identity(), quarter_turns).apply(camera_point)
            projected = (rotated_intrinsics @ rotated_point)[:2] / rotated_point[2]
            np.testing.assert_allclose(projected, rotate_pixels(pixel, (640, 480), quarter_turns), atol=1e-12)

    def test_real_pose_projection_follows_the_baked_image_rotation(self) -> None:
        """A real trajectory pose and calibration preserve world-point alignment."""
        if not Path("data/raw/Training/47332195/lowres_wide.traj").is_file():
            self.skipTest("ARKitScenes sample sequence is unavailable")
        trajectory_row = np.loadtxt("data/raw/Training/47332195/lowres_wide.traj", max_rows=1)
        calibration = np.loadtxt(sorted(Path("data/raw/Training/47332195/lowres_wide_intrinsics").glob("*.pincam"))[0])
        camera_from_world = Rotation.from_rotvec(trajectory_row[1:4])
        translation = trajectory_row[4:7]
        intrinsics = np.asarray([[calibration[2], 0.0, calibration[4]], [0.0, calibration[3], calibration[5]], [0.0, 0.0, 1.0]])
        camera_point = np.asarray([0.13, -0.08, 1.7])
        world_point = camera_from_world.inv().apply(camera_point - translation)
        original = intrinsics @ (camera_from_world.apply(world_point) + translation)
        baked_pose = bake_camera_orientation(camera_from_world, 2)
        baked_translation = bake_camera_orientation(Rotation.identity(), 2).apply(translation)
        sequence = Intrinsics(np.asarray([0.0]), intrinsics[None, ...])
        baked_sequence, _ = rotate_intrinsics_sequence(sequence, (256, 192), 2)
        baked_intrinsics = baked_sequence.matrices[0]
        baked = baked_intrinsics @ (baked_pose.apply(world_point) + baked_translation)
        np.testing.assert_allclose(baked[:2] / baked[2], rotate_pixels(original[:2] / original[2], (256, 192), 2), atol=1e-10)


if __name__ == "__main__":
    unittest.main()
