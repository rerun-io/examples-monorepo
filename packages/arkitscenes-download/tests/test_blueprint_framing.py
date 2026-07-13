"""Behavior checks for the per-sequence 3D viewport framing."""

import unittest

import numpy as np
import rerun.blueprint as rrb
from simplecv.rerun_log_utils import mesh_bounding_geometry, orbit_eye_position

from arkitscenes_download.ingest.blueprint import EXTRA_ZOOM_OUT, FIT_DISTANCE_FACTOR, make_blueprint

FRAMING_FACTOR: float = FIT_DISTANCE_FACTOR * EXTRA_ZOOM_OUT


class OrbitEyePositionTest(unittest.TestCase):
    """The framed eye sits on the viewing direction at the padded fit distance."""

    def test_distance_is_fit_factor_times_margin(self) -> None:
        """Eye distance from the look target is fit factor x bounding radius x extra zoom-out."""
        center: np.ndarray = np.array([1.0, -2.0, 0.5])
        radius: float = 3.0
        position: np.ndarray = orbit_eye_position(center, radius, FRAMING_FACTOR)
        distance: float = float(np.linalg.norm(position - center))
        self.assertAlmostEqual(distance, FRAMING_FACTOR * radius, places=9)

    def test_direction_is_elevated_and_scale_invariant(self) -> None:
        """The eye looks down from above (+Z offset) along the same direction at any scale."""
        center: np.ndarray = np.zeros(3)
        small: np.ndarray = orbit_eye_position(center, 1.0, FRAMING_FACTOR)
        large: np.ndarray = orbit_eye_position(center, 10.0, FRAMING_FACTOR)
        self.assertGreater(small[2], 0.0)
        np.testing.assert_allclose(small / np.linalg.norm(small), large / np.linalg.norm(large), atol=1e-12)


class MeshBoundingGeometryTest(unittest.TestCase):
    """Bounding geometry frames every vertex around the box center."""

    def test_center_and_radius_cover_all_vertices(self) -> None:
        """The center is the AABB midpoint and the radius reaches the farthest vertex."""
        vertices: np.ndarray = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 4.0, 6.0]])
        center, radius = mesh_bounding_geometry(vertices)
        np.testing.assert_allclose(center, [1.0, 2.0, 3.0])
        distances: np.ndarray = np.linalg.norm(vertices - center, axis=1)
        self.assertAlmostEqual(radius, float(distances.max()), places=12)
        self.assertTrue(np.all(distances <= radius + 1e-12))


class MakeBlueprintTest(unittest.TestCase):
    """Blueprint construction succeeds with and without framing geometry."""

    def test_generic_and_framed_blueprints_build(self) -> None:
        """Both the dataset-default and the per-sequence framed layout construct."""
        self.assertIsInstance(make_blueprint(portrait=True), rrb.Blueprint)
        framed: rrb.Blueprint = make_blueprint(portrait=False, mesh_center_xyz=np.array([1.0, 2.0, 3.0]), bounding_radius_m=2.5)
        self.assertIsInstance(framed, rrb.Blueprint)


if __name__ == "__main__":
    unittest.main()
