import numpy as np

from selfcap_soma.evaluation import apply_body_offsets, heldout_blocks, learn_body_offsets


def test_alternating_two_second_blocks_include_target_without_boundary_leak():
    np.testing.assert_array_equal(heldout_blocks(np.array([0.3, 1.99, 2.0, 3.99, 4.144, 6.0])), [True, True, False, False, True, False])


def test_offsets_use_only_training_rows_and_rotate_with_body():
    prediction = np.zeros((3, 1, 3))
    target = np.array([[[0.03, 0.0, 0.0]], [[0.03, 0.0, 0.0]], [[100.0, 100.0, 100.0]]])
    rotations = np.tile(np.eye(3), (3, 1, 1))
    rotations[2] = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    offsets = learn_body_offsets(prediction, target, rotations, np.array([True, True, False]))
    np.testing.assert_allclose(offsets, [[0.03, 0.0, 0.0]])
    corrected = apply_body_offsets(prediction, rotations, offsets)
    np.testing.assert_allclose(corrected[2], [[0.0, 0.03, 0.0]], atol=1e-9)
