"""The window-only WiLoR crop equals the crop of the fully blurred frame."""

import cv2
import numpy as np
import pytest
from jaxtyping import Float32, UInt8
from numpy import ndarray
from posekit.models.wilor import CROP_SIZE
from posekit.models.wilor_net.geometry import generate_image_patch_cv2
from posekit.predictions import HandSide
from skimage.filters import gaussian

from mv_api.wilor_refine import hand_patch


def full_frame_patch(rgb_hw3: UInt8[ndarray, "h w 3"], xyxy: Float32[ndarray, "4"], side: HandSide, rescale_factor: float) -> ndarray:
    """The crop as the refiner built it before: float copy and blur of the whole frame."""
    center: Float32[ndarray, "2"] = (xyxy[2:4] + xyxy[0:2]) / 2.0
    box_size: float = float((rescale_factor * (xyxy[2:4] - xyxy[0:2])).max())
    image: ndarray = rgb_hw3.copy().astype(np.float32)
    downsampling: float = box_size / CROP_SIZE / 2.0
    if downsampling > 1.1:
        image = gaussian(image, sigma=(downsampling - 1) / 2, channel_axis=2, preserve_range=True)
    patch, _ = generate_image_patch_cv2(
        image, int(center[0]), int(center[1]), int(box_size), int(box_size), CROP_SIZE, CROP_SIZE, side == "left", 1.0, 0, border_mode=cv2.BORDER_CONSTANT
    )
    return patch


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize(
    "xyxy",
    [
        (900.0, 400.0, 1100.0, 640.0),  # blurred, inside the frame
        (1500.0, 50.0, 1915.0, 470.0),  # strong blur, cut by the top and right borders
        (5.0, 900.0, 300.0, 1078.0),  # cut by the left and bottom borders
        (600.5, 300.2, 700.7, 380.9),  # small box, no blur
    ],
)
def test_window_patch_equals_full_frame_patch(side: HandSide, xyxy: tuple[float, float, float, float]) -> None:
    rng: np.random.Generator = np.random.default_rng(3)
    coarse: ndarray = rng.integers(0, 256, size=(34, 60, 3)).astype(np.float32)
    smooth: ndarray = cv2.resize(coarse, (1920, 1080), interpolation=cv2.INTER_CUBIC)
    frame: UInt8[ndarray, "1080 1920 3"] = np.clip(smooth + rng.normal(0.0, 12.0, smooth.shape), 0, 255).astype(np.uint8)
    box: Float32[ndarray, "4"] = np.asarray(xyxy, dtype=np.float32)
    ours: ndarray = hand_patch(frame, box, side, 2.5)
    reference: ndarray = full_frame_patch(frame, box, side, 2.5)
    assert ours.dtype == reference.dtype
    np.testing.assert_array_equal(ours, reference)
