import numpy as np
import pytest
from beartype.roar import BeartypeException

torch = pytest.importorskip("torch", reason="Multiview predictions require PyTorch")
pytest.importorskip("simplecv", reason="Multiview predictions require SimpleCV")

from monopriors.models.multiview.multiview_model import (  # noqa: E402
    MultiviewModelPredictions,
    MultiviewPred,
    generate_multiview_pred,
)


def _predictions(batch: int, num_cams: int, h: int = 4, w: int = 6) -> MultiviewModelPredictions:
    intrinsic = np.tile(np.array([[5.0, 0.0, 3.0], [0.0, 5.0, 2.0], [0.0, 0.0, 1.0]], dtype=np.float32), (batch, num_cams, 1, 1))
    cam_T_world = np.tile(np.eye(3, 4, dtype=np.float32), (batch, num_cams, 1, 1))
    return MultiviewModelPredictions(
        depth=np.ones((batch, num_cams, h, w, 1), dtype=np.float32),
        depth_conf=np.ones((batch, num_cams, h, w), dtype=np.float32),
        intrinsic=intrinsic,
        cam_T_world_b34=cam_T_world,
    )


def test_predictions_without_batch_axis_are_rejected() -> None:
    """A squeezed (num_cams, ...) instance used to pass as *batch and lose its camera axis later."""
    with pytest.raises(BeartypeException):
        MultiviewModelPredictions(
            depth=np.ones((1, 4, 6, 1), dtype=np.float32),
            depth_conf=np.ones((1, 4, 6), dtype=np.float32),
            intrinsic=np.ones((1, 3, 3), dtype=np.float32),
            cam_T_world_b34=np.ones((1, 3, 4), dtype=np.float32),
        )


def test_generate_multiview_pred_rejects_more_than_one_scene() -> None:
    with pytest.raises(ValueError, match="batch of one scene"):
        generate_multiview_pred(_predictions(batch=2, num_cams=1), torch.zeros((1, 3, 4, 6)), [np.zeros((4, 6, 3), dtype=np.uint8)])


def test_generate_multiview_pred_keeps_single_camera() -> None:
    preds: list[MultiviewPred] = generate_multiview_pred(
        _predictions(batch=1, num_cams=1), torch.zeros((1, 3, 4, 6)), [np.zeros((4, 6, 3), dtype=np.uint8)]
    )

    assert len(preds) == 1
    assert preds[0].depth_map.shape == (4, 6)
