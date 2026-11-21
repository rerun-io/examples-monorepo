"""detnet_compare: net-frame detections map back to stored pixels, and only positive slots become boxes."""

import numpy as np
import pytest
import torch


def test_a_turned_cameras_detection_lands_on_its_stored_pixel() -> None:
    from handtrack.apis.detnet_compare import stored_boxes
    from handtrack.geometry.letterbox import letterbox_for
    from handtrack.models.detnet import DetNetOutput

    letterbox = letterbox_for(1920, 1080, 2)
    stored_xy = torch.tensor([[[300.0, 200.0], [1500.0, 900.0]]])
    net_xy = letterbox.to_net(stored_xy)
    output = DetNetOutput(center=net_xy / torch.tensor([640.0, 480.0]), radius=torch.tensor([[30.0, 15.0]]) / 640.0,
                          presence_logit=torch.tensor([[3.0, -3.0]]))
    boxes = stored_boxes(output, letterbox)
    np.testing.assert_allclose(boxes.centers_px, stored_xy.numpy(), atol=1e-3)
    np.testing.assert_allclose(boxes.half_px, [[90.0, 45.0]], rtol=1e-6)  # net radius / (1/3 scale)
    assert boxes.probability[0, 0] > 0.9 and boxes.probability[0, 1] < 0.1


def test_box_columns_keep_positive_slots_and_give_empty_frames_an_empty_row() -> None:
    from handtrack.apis.detnet_compare import CameraBoxes, box_rows

    boxes = CameraBoxes(centers_px=np.zeros((3, 2, 2), dtype=np.float32), half_px=np.ones((3, 2), dtype=np.float32),
                        probability=np.array([[0.9, 0.2], [0.1, 0.3], [0.6, 0.7]], dtype=np.float32))
    rows = box_rows(boxes, "mix1")
    assert rows.lengths.tolist() == [1, 0, 2]
    assert rows.labels == ["mix1 L 0.90", "mix1 L 0.60", "mix1 R 0.70"]
    assert rows.class_ids.tolist() == [0, 0, 1]


def test_config_needs_one_convention_per_model() -> None:
    from pathlib import Path

    from handtrack.apis.detnet_compare import DetNetCompareConfig

    with pytest.raises(ValueError):
        DetNetCompareConfig(segments=("s",), names=("a", "b"), weights=(Path("a"), Path("b")), inputs=("upright",), out_dir=Path("/tmp"))
