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
