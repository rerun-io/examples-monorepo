import torch

from handtrack.data.rerun_source import teacher_unsure_sides
from handtrack.labels.validity import HandLabel

A, P, X = int(HandLabel.ABSENT), int(HandLabel.PRESENT), int(HandLabel.PARTIAL)
ALL: tuple[tuple[int, ...], ...] = ((0, 1, 2),)


def test_a_side_seen_in_another_camera_is_unknown_where_the_teacher_missed_it() -> None:
    # frame 0: right hand seen by camera 0 only; frame 1: no camera sees the right hand
    labels = torch.tensor([[[P, P], [P, A], [A, A]], [[P, A], [A, A], [A, A]]])
    valid = torch.tensor([[True, True, False], [True, True, True]])
    unsure = teacher_unsure_sides(labels, valid, ALL)
    assert unsure[0].tolist() == [[False, False], [False, True], [False, False]]  # camera 2 is not labelled at all
    assert unsure[1].tolist() == [[False, False], [True, False], [True, False]]  # left seen in camera 0: unknown elsewhere
    assert not unsure[..., 1][1].any()  # right absent everywhere stays absent


def test_cameras_outside_the_view_group_keep_their_absent_labels() -> None:
    labels = torch.tensor([[[P, P], [A, A], [A, A]]])
    unsure = teacher_unsure_sides(labels, torch.ones((1, 3), dtype=torch.bool), ((0, 1), (2,)))
    assert unsure[0].tolist() == [[False, False], [True, True], [False, False]]


def test_partial_hands_never_count_as_seen() -> None:
    labels = torch.tensor([[[X, A], [A, A]]])
    assert not teacher_unsure_sides(labels, torch.ones((1, 2), dtype=torch.bool), ((0, 1),)).any()
