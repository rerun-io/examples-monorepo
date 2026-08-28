"""UniDepthV2's head predicts error (log(raw) == |pred - gt|); the predictor maps it to a confidence in (0, 1]."""

import torch

from monopriors.models.relative_depth.unidepth import predicted_error_to_confidence
from monopriors.rr_logging_utils import CONFIDENCE_THRESHOLD


def test_mapping_is_monotone_bounded_and_thresholds_at_one_metre() -> None:
    raw = torch.tensor([[[[0.5, 1.0, torch.e, 280.0]]]])  # log: negative (clamped to 0), 0, 1 metre, ~5.6 metres
    conf = predicted_error_to_confidence(raw)
    assert conf.shape == raw.shape
    assert torch.allclose(conf[0, 0, 0, :2], torch.tensor([1.0, 1.0]))  # no predicted error -> fully confident
    assert torch.isclose(conf[0, 0, 0, 2], torch.tensor(CONFIDENCE_THRESHOLD))  # exactly one metre sits on the mask threshold
    assert 0.0 < conf[0, 0, 0, 3] < CONFIDENCE_THRESHOLD  # large predicted error -> not confident
    assert torch.all(conf[0, 0, 0, :-1] >= conf[0, 0, 0, 1:])  # monotone decreasing in raw
