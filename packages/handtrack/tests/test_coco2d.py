import math

import pytest
import torch
from jaxtyping import Bool, Float32
from torch import Tensor

from handtrack.geometry.letterbox import letterbox_for
from handtrack.labels.coco2d import COCO_SLOTS, coco133_hands
from handtrack.labels.validity import HandLabel

LEFT_SLOTS: list[int] = sorted(COCO_SLOTS)
"""The 20 COCO-133 left-hand slots that carry one of our landmarks (right hand: slot + 21)."""


def _frame(left: list[tuple[float, float]], right: list[tuple[float, float]]) -> tuple[Float32[Tensor, '1 133 2'], Float32[Tensor, '1 133']]:
    """One image: the given points fill the first slots of each hand; every other slot is NaN with confidence 0."""
    uv: Float32[Tensor, '1 133 2'] = torch.full((1, 133, 2), math.nan)
    confidence: Float32[Tensor, '1 133'] = torch.zeros((1, 133))
    for offset, points in ((0, left), (21, right)):
        for slot, point in zip(LEFT_SLOTS, points, strict=False):
            uv[0, slot + offset] = torch.tensor(point)
            confidence[0, slot + offset] = 1.0
    return uv, confidence


def _ring(cx: float, cy: float, radius: float, count: int = 20) -> list[tuple[float, float]]:
    return [(cx + radius * math.cos(2 * math.pi * i / count), cy + radius * math.sin(2 * math.pi * i / count)) for i in range(count)]


BOTH: Bool[Tensor, '1 2'] = torch.tensor([[True, True]])


def test_a_fully_visible_hand_is_present_with_its_enclosing_circle_and_a_missing_hand_is_absent() -> None:
    uv, confidence = _frame(_ring(100.0, 200.0, 10.0), [])
    hands = coco133_hands(uv, confidence, BOTH, letterbox_for(640, 480))
    assert hands.image_valid.tolist() == [True]
    assert hands.hand_label.tolist() == [[int(HandLabel.PRESENT), int(HandLabel.ABSENT)]]
    torch.testing.assert_close(hands.circles[0, 0], torch.tensor([100.0, 200.0, 10.0]), atol=0.05, rtol=0.0)
    assert bool(hands.circles[0, 1].isnan().all())


@pytest.mark.parametrize(('visible', 'label'), [(16, HandLabel.PRESENT), (15, HandLabel.PARTIAL), (1, HandLabel.PARTIAL)])
def test_sixteen_of_twenty_points_make_a_hand_present(visible: int, label: HandLabel) -> None:
    uv, confidence = _frame([], _ring(300.0, 300.0, 20.0)[:visible])
    assert coco133_hands(uv, confidence, BOTH, letterbox_for(640, 480)).hand_label.tolist() == [[int(HandLabel.ABSENT), int(label)]]


def test_points_with_zero_confidence_or_outside_the_image_do_not_count() -> None:
    uv, confidence = _frame(_ring(100.0, 200.0, 10.0), [])
    confidence[0, LEFT_SLOTS[0]] = 0.0
    uv[0, LEFT_SLOTS[1]] = torch.tensor([-5.0, 200.0])
    uv[0, LEFT_SLOTS[2]] = torch.tensor([100.0, 480.0])
    hands = coco133_hands(uv, confidence, BOTH, letterbox_for(640, 480))
    assert hands.hand_label.tolist() == [[int(HandLabel.PRESENT), int(HandLabel.ABSENT)]]
    assert int(hands.in_front[0, 0].sum()) == 17


def test_an_untrusted_hand_drops_the_image() -> None:
    uv, confidence = _frame(_ring(100.0, 200.0, 10.0), _ring(400.0, 200.0, 10.0))
    hands = coco133_hands(uv, confidence, torch.tensor([[True, False]]), letterbox_for(640, 480))
    assert hands.image_valid.tolist() == [False]


def test_points_go_through_the_letterbox_and_keep_slot_order() -> None:
    uv, confidence = _frame([(960.0, 540.0)], [(0.0, 0.0)])
    hands = coco133_hands(uv, confidence, BOTH, letterbox_for(1920, 1080))
    # COCO slot 91 is our landmark 5 for both hands; GSI's letterbox: (u + 0.5) / 3 - 0.5, plus 60 px on y.
    torch.testing.assert_close(hands.net_xy[0, 0, 5], torch.tensor([960.5 / 3.0 - 0.5, 540.5 / 3.0 - 0.5 + 60.0]))
    torch.testing.assert_close(hands.net_xy[0, 1, 5], torch.tensor([0.5 / 3.0 - 0.5, 0.5 / 3.0 - 0.5 + 60.0]))
    assert bool(hands.net_xy[0, :, 20].isnan().all()), 'COCO-133 has no palm centre (landmark 20)'
