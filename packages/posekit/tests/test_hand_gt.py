"""Ground-truth reading helpers that need no catalog."""

import numpy as np

from posekit.eval.hand_gt import rig_rotation

QUARTER_TURN_ABOUT_Z = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def test_a_quaternion_rig_rotation_is_read_as_xyzw() -> None:
    half = np.sqrt(0.5)
    np.testing.assert_allclose(rig_rotation([0.0, 0.0, half, half], "quaternion"), QUARTER_TURN_ABOUT_Z, atol=1e-12)


def test_a_mat3x3_rig_rotation_is_read_column_major_as_rerun_stores_it() -> None:
    columns = [0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0]  # first column (0, 1, 0): x turns to y
    np.testing.assert_allclose(rig_rotation(columns, "mat3x3"), QUARTER_TURN_ABOUT_Z, atol=1e-12)
