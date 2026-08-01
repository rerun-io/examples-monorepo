import torch

from monopriors.models.relative_depth.depth_anything_v1 import DepthAnythingV1Predictor


def test_set_model_device_keeps_model_and_pipe_in_sync() -> None:
    """After a device switch the base contract must expose the active pipeline's module."""
    predictor: DepthAnythingV1Predictor = DepthAnythingV1Predictor(device="cpu")
    assert predictor.model is predictor.pipe.model

    predictor.set_model_device("cpu")
    assert predictor.model is predictor.pipe.model
    assert isinstance(predictor.model, torch.nn.Module)
