import numpy as np

from selfcap_soma.observations import quest_sample_times


def test_video_time_maps_to_delayed_raw_quest_hand_sample():
    head, hand = quest_sample_times(np.array([4.144]), offset_ms=120.0)
    np.testing.assert_allclose(head, [4.024])
    np.testing.assert_allclose(hand, [4.072])


def test_coco_mapping_skips_soma_metacarpals_and_missing_face():
    from selfcap_soma.observations import coco_joint_names

    names = coco_joint_names()
    assert names[5] == "LeftArm"
    assert names[9] == "LeftHand"
    assert names[91] == "LeftHand"
    assert names[92] == "LeftHandThumb1"
    assert names[95] == "LeftHandThumbEnd"
    assert names[96] == "LeftHandIndex2"
    assert names[99] == "LeftHandIndexEnd"
    assert names[117] == "RightHandIndex2"
    assert set(names).isdisjoint(range(23, 91))


def test_overlay_maps_only_body_and_hands_not_head_or_root():
    from selfcap_soma.observations import overlay_points

    names = ("Hips", "Head", "LeftArm", "LeftHand", "LeftHandThumb1")
    points = np.arange(15, dtype=np.float32).reshape(1, 5, 3)
    result = overlay_points(points, names)
    np.testing.assert_array_equal(result[0, 5], points[0, 2])
    np.testing.assert_array_equal(result[0, 91], points[0, 3])
    np.testing.assert_array_equal(result[0, 92], points[0, 4])
    assert np.isnan(result[0, :5]).all()
    assert np.isnan(result[0, 17:91]).all()


def test_head_samples_outside_coverage_are_missing_not_extrapolated():
    from selfcap_soma.catalog import interpolate_head

    positions, rotations = interpolate_head(
        np.array([0.0, 0.02]),
        np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 3.0]]),
        np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]),
        np.array([-0.01, 0.01, 0.03]),
    )
    assert np.isnan(positions[[0, 2]]).all()
    np.testing.assert_allclose(positions[1], [2.0, 2.0, 3.0])
    np.testing.assert_allclose(rotations[1], np.eye(3))


def test_head_attachment_offset_rotates_with_head_not_world():
    import torch

    from selfcap_soma.losses import attached_eye

    heads = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    rotations = np.array([[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]], dtype=np.float32)
    point = attached_eye(torch.from_numpy(heads), torch.from_numpy(rotations), torch.tensor([0.1, 0.0, 0.0]))
    np.testing.assert_allclose(point, [[1.0, 2.1, 3.0]])


def test_upper_body_overlay_keeps_hips_but_omits_knees_and_ankles():
    from selfcap_soma.observations import overlay_points

    names = ("LeftLeg", "LeftShin", "LeftFoot", "RightLeg", "RightShin", "RightFoot", "LeftHand")
    points = np.ones((1, len(names), 3), dtype=np.float32)
    result = overlay_points(points, names)
    assert np.isfinite(result[:, [11, 12, 91]]).all()
    assert np.isnan(result[:, 13:91]).all()
