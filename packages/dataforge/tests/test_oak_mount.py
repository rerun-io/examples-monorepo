from __future__ import annotations

import numpy as np

from dataforge.oak_mount import compose_track, quaternion_matrices


def test_compose_track_hangs_the_oak_on_the_quest() -> None:
    # Quest at (1, 2, 3) yawed 90 deg about +y; the mount puts the OAK 5 cm along the Quest's +x.
    half = np.sqrt(0.5)
    mount = np.eye(4)
    mount[:3, 3] = [0.05, 0.0, 0.0]
    world_T_oak = compose_track(np.array([[1.0, 2.0, 3.0]]), np.array([[0.0, half, 0.0, half]]), mount)
    np.testing.assert_allclose(world_T_oak[0, :3, 3], [1.0, 2.0, 3.0 - 0.05], atol=1e-12)


def test_quaternion_matrices_match_identity_and_normalise() -> None:
    np.testing.assert_allclose(quaternion_matrices(np.array([[0.0, 0.0, 0.0, 2.0]]))[0], np.eye(3))
