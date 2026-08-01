import numpy as np
import torch

from monopriors.models.relative_depth.depth_anything_v1 import DepthAnythingV1Predictor


def test_set_model_device_moves_in_place() -> None:
    """A device switch moves the single existing model; no second copy is ever created."""
    predictor: DepthAnythingV1Predictor = DepthAnythingV1Predictor(device="cpu")
    model_before: torch.nn.Module = predictor.model
    assert predictor.model is predictor.pipe.model

    predictor.set_model_device("cpu")
    assert predictor.model is model_before
    assert predictor.model is predictor.pipe.model
    assert predictor.pipe.device == torch.device("cpu")

    rgb: np.ndarray = np.zeros((70, 70, 3), dtype=np.uint8)
    prediction = predictor(rgb, None)
    assert np.isfinite(prediction.depth).all()
