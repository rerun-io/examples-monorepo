"""DetNet-F as a posekit hand detector: the letterboxed input and the boxes mapped back into the source frame."""

import torch

from posekit.models.detnet import detnet_boxes, detnet_input


def net_circle(cx: float, cy: float, r: float) -> tuple[torch.Tensor, torch.Tensor]:
    """One frame whose left slot holds a net-pixel circle and whose right slot is empty."""
    center = torch.tensor([[[cx / 640, cy / 480], [0.0, 0.0]]])
    radius = torch.tensor([[r / 640, 0.0]])
    return center, radius


def test_a_robocap_frame_fills_the_width_with_bars_above_and_below() -> None:
    frames = torch.full((1, 1080, 1920, 3), 200, dtype=torch.uint8)
    net, fit = detnet_input(frames, quarter_turns=0)
    assert net.shape == (1, 1, 480, 640)
    assert (fit.scale, fit.pad_x, fit.pad_y) == (1.0 / 3.0, 0, 60)
    assert float(net[0, 0, 60:420].min()) > 0.7 and float(net[0, 0, :60].max()) == 0.0


def test_a_box_maps_back_to_source_pixels() -> None:
    # Net centre (320, 240), r 30: source ((320.5) * 3 - 0.5, (180.5) * 3 - 0.5) = (961, 541), half side 90.
    _, fit = detnet_input(torch.zeros((1, 1080, 1920, 3), dtype=torch.uint8), quarter_turns=0)
    center, radius = net_circle(320, 240, 30)
    found = detnet_boxes(center, radius, torch.tensor([[3.0, -3.0]]), fit, threshold=0.5)
    assert found.sides == ("left",)
    torch.testing.assert_close(found.boxes.xyxy, torch.tensor([[871.0, 451.0, 1051.0, 631.0]]))
    assert found.boxes.frame_indices.tolist() == [0]


def test_a_turned_camera_turns_its_boxes_back() -> None:
    # Turned 180 degrees: the same net box sits at (1919 - 961, 1079 - 541) = (958, 538) in the stored frame.
    _, fit = detnet_input(torch.zeros((1, 1080, 1920, 3), dtype=torch.uint8), quarter_turns=2)
    center, radius = net_circle(320, 240, 30)
    found = detnet_boxes(center, radius, torch.tensor([[3.0, -3.0]]), fit, threshold=0.5)
    torch.testing.assert_close(found.boxes.xyxy, torch.tensor([[868.0, 448.0, 1048.0, 628.0]]))
