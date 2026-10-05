"""Truth-box diagnostics preserve edge centres and only floor the radius."""

import torch

from handtrack.labels.crops import apply_affine
from handtrack.oracle import GroundTruthViews, KeyNetOnTruthBoxes
from handtrack.tracker import CropRequest, KeypointEstimate


def test_truth_boxes_keep_centres_at_and_beyond_image_edges() -> None:
    circles = torch.tensor([[[[2.0, 120.0, 1.0], [-5.0, 3.0, 10.0]]]])
    truth = GroundTruthViews(
        torch.zeros((1, 1, 2, 21, 2)), torch.zeros((1, 1, 2, 21, 3)),
        torch.ones((1, 1, 2, 21), dtype=torch.bool), circles, torch.ones((1, 1, 2), dtype=torch.bool),
    )
    requests: list[CropRequest] = []

    def keynet(images: torch.Tensor, frame: int, request: CropRequest) -> KeypointEstimate:
        requests.append(request)
        return KeypointEstimate(torch.zeros((2, 21, 2)), torch.zeros((2, 21)), torch.ones(2), torch.ones((2, 21)))

    estimator = KeyNetOnTruthBoxes(truth, keynet)
    estimator(torch.zeros((1, 480, 640), dtype=torch.uint8), 0,
              CropRequest(torch.tensor([0, 0]), torch.tensor([0, 1]), torch.eye(3).expand(2, 3, 3), torch.ones((2, 63))))
    received = requests[0]
    centres = apply_affine(received.crop_from_net, circles[0, 0, :, None, :2])
    torch.testing.assert_close(centres, torch.full((2, 1, 2), 47.5))
    torch.testing.assert_close(received.crop_from_net[:, 0, 0].abs(), torch.tensor([5.0, 4.0]))
    assert not received.keypoint_input.any()
