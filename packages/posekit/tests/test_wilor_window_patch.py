"""WiLoR's torch-path crop blurs only the window it samples and equals the crop of the fully blurred frame."""

import cv2
import numpy as np
import pytest

pytest.importorskip("skimage", reason="WiLoR reference blur requires the posekit-hands environment")
pytest.importorskip("trtkit", reason="posekit.models.wilor needs the posekit-hands Pixi environment (posekit-wilor feature)")

from jaxtyping import UInt8
from numpy import ndarray
from skimage.filters import gaussian

from posekit.models.wilor import CROP_SIZE, reference_hand_patch
from posekit.models.wilor_net.geometry import generate_image_patch_cv2


def full_frame_patch(image: UInt8[ndarray, "h w 3"], center_x: np.float32, center_y: np.float32, box_size: np.float32, flip: bool) -> ndarray:
    """The crop as ``_run_torch`` built it before: copy and blur the whole frame, then warp."""
    source: ndarray = image.copy()
    downsampling = box_size * 1.0 / CROP_SIZE / 2.0
    if downsampling > 1.1:
        source = gaussian(source, sigma=(downsampling - 1) / 2, channel_axis=2, preserve_range=True)
    patch, _ = generate_image_patch_cv2(
        source, center_x, center_y, box_size, box_size, CROP_SIZE, CROP_SIZE, flip, 1.0, 0, border_mode=cv2.BORDER_CONSTANT
    )
    return patch


@pytest.mark.parametrize("flip", [False, True])
@pytest.mark.parametrize(
    "xyxy",
    [
        (900.3, 400.7, 1100.1, 640.9),  # blurred, inside the frame
        (1500.6, 50.2, 1915.4, 470.8),  # strong blur, cut by the top and right borders
        (5.5, 900.25, 300.75, 1078.5),  # cut by the left and bottom borders
        (600.5, 300.2, 700.7, 380.9),  # small box, no blur (uint8 crop)
        (1201.37, 611.93, 1288.41, 702.17),  # just below the blur threshold (uint8 crop)
        (143.63232, 928.7538, 637.6597, 1000.1986),  # large box past the bottom: a shifted origin rounded differently here
    ],
)
def test_window_patch_equals_full_frame_patch(flip: bool, xyxy: tuple[float, float, float, float]) -> None:
    rng: np.random.Generator = np.random.default_rng(7)
    coarse: ndarray = rng.integers(0, 256, size=(34, 60, 3)).astype(np.float32)
    smooth: ndarray = cv2.resize(coarse, (1920, 1080), interpolation=cv2.INTER_CUBIC)
    frame: UInt8[ndarray, "1080 1920 3"] = np.clip(smooth + rng.normal(0.0, 12.0, smooth.shape), 0, 255).astype(np.uint8)
    boxes: ndarray = np.asarray([xyxy], dtype=np.float32)
    # The same float32 arithmetic as WilorHandPose3d.__call__ (rescale_factor 2.5).
    centers: ndarray = (boxes[:, 2:4] + boxes[:, 0:2]) / 2.0
    box_sizes: ndarray = (2.5 * (boxes[:, 2:4] - boxes[:, 0:2])).max(axis=1)
    ours: ndarray = reference_hand_patch(frame, centers[0, 0], centers[0, 1], box_sizes[0], flip)
    reference: ndarray = full_frame_patch(frame, centers[0, 0], centers[0, 1], box_sizes[0], flip)
    assert ours.dtype == reference.dtype
    np.testing.assert_array_equal(ours, reference)
