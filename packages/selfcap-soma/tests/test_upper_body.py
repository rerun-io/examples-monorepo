import numpy as np

from selfcap_soma.upper_body import mesh_selection, visible_joint_mask


def test_crop_removes_faces_touching_lower_body_and_preserves_hip_joints() -> None:
    names = ("root", "Hips", "LeftLeg", "LeftShin", "LeftHand")
    parents = np.array([-1, 0, 1, 2, 1], dtype=np.int64)
    weights = np.array(
        [[0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.3, 0.7, 0.0, 0.0], [0.0, 0.4, 0.3, 0.3, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0]]
    )
    faces = np.array([[0, 1, 4], [0, 1, 2], [1, 3, 4]], dtype=np.int32)
    selected = mesh_selection(weights, names, parents, faces)
    np.testing.assert_array_equal(selected.vertices, [0, 1, 4])
    np.testing.assert_array_equal(selected.faces, [[0, 1, 2]])
    np.testing.assert_array_equal(visible_joint_mask(names, parents), [True, True, True, False, True])
