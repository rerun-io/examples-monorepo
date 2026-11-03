import numpy as np

from selfcap_soma.upper_body import mesh_selection, visible_joint_mask


def test_rest_waist_cut_drops_crossing_faces_and_keeps_indices_fixed() -> None:
    rest = np.array([[0.0, 0.1, 0.0], [1.0, 0.1, 0.0], [0.0, -0.02, 1.0], [0.0, -0.1, 0.0], [1.0, -0.1, 0.0]], dtype=np.float32)
    faces = np.array([[0, 1, 2], [0, 2, 3], [2, 3, 4]], dtype=np.int32)
    selected = mesh_selection(rest, faces, -0.05)
    np.testing.assert_array_equal(selected.vertices, [0, 1, 2])
    np.testing.assert_array_equal(selected.faces, [[0, 1, 2]])


def test_visible_joints_keep_hips_but_remove_leg_descendants() -> None:
    names = ("root", "Hips", "LeftLeg", "LeftShin", "LeftHand")
    parents = np.array([-1, 0, 1, 2, 1], dtype=np.int64)
    np.testing.assert_array_equal(visible_joint_mask(names, parents), [True, True, True, False, True])


def test_retained_boundary_is_flattened_in_rest_pose_only() -> None:
    rest = np.array([[0., 0.1, 0.], [1., 0.1, 0.], [0., 0.2, 1.], [0., -0.1, 0.]], dtype=np.float32)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    selected = mesh_selection(rest, faces, 0.0)
    np.testing.assert_allclose(selected.rest_offsets[:, 1], [-0.1, 0., -0.2, 0.])
