import numpy as np
import pytest
from beartype import beartype
from beartype.roar import BeartypeCallHintParamViolation

from mast3r_slam.image_types import RgbNormalized
from mast3r_slam.image_utils import resize_img


@beartype
def _accept_rgb_normalized(rgb: RgbNormalized) -> RgbNormalized:
    return rgb


def test_rgb_normalized_accepts_unit_float_rgb() -> None:
    rgb = np.linspace(0.0, 1.0, 3 * 8 * 8, dtype=np.float32).reshape(8, 8, 3)

    accepted = _accept_rgb_normalized(rgb)

    assert accepted is rgb


@pytest.mark.parametrize(
    "rgb",
    [
        np.full((8, 8, 3), -0.01, dtype=np.float32),
        np.full((8, 8, 3), 1.01, dtype=np.float32),
        np.zeros((8, 8, 4), dtype=np.float32),
        np.zeros((8, 8, 3), dtype=np.uint8),
    ],
)
def test_rgb_normalized_rejects_invalid_arrays(rgb: np.ndarray) -> None:
    with pytest.raises(BeartypeCallHintParamViolation):
        _accept_rgb_normalized(rgb)


def test_resize_img_accepts_rgb_normalized() -> None:
    rgb = np.ones((32, 48, 3), dtype=np.float32) * 0.5

    resized = resize_img(rgb, size=224)

    assert resized.rgb_tensor.shape[1] == 3
    assert resized.rgb_uint8.shape[-1] == 3
