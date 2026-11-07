"""video_to_frame_transform places a source-video frame exactly over the MASt3R frame."""

from typing import Literal

import numpy as np
import pytest
from jaxtyping import Float32, UInt8
from numpy import ndarray

from mast3r_slam.image_utils import ResizedImage, VideoToFrame, resize_img, video_to_frame_transform


@pytest.mark.parametrize(
    ("video_w", "video_h", "size"),
    [
        (1920, 1080, 512),  # 16:9, no crop
        (1000, 750, 512),  # resize rounding plus a crop on both axes
        (720, 1280, 512),  # portrait
        (600, 600, 512),  # square: forced to 4:3
        (1280, 720, 224),  # center crop to a square
        (750, 1000, 224),
    ],
)
def test_transform_matches_resize_img(video_w: int, video_h: int, size: Literal[224, 512]) -> None:
    """A marker block in the video lands where resize_img puts it in the frame."""
    xform: VideoToFrame = video_to_frame_transform(video_w, video_h, size)
    resized: ResizedImage = resize_img(np.zeros((video_h, video_w, 3), dtype=np.float32), size)
    frame_h: int = int(resized.true_shape[0, 0])
    frame_w: int = int(resized.true_shape[0, 1])

    # The video's full extent maps to the resized image, so the frame is its sub-rectangle.
    assert xform.translation[0] <= 0.0 and xform.translation[1] <= 0.0
    np.testing.assert_allclose(
        np.array(xform.scale) * (video_w, video_h) + np.array(xform.translation) * 2,
        (frame_w, frame_h),
        atol=1.0,
    )

    # A bright square in the video at the frame's center: its centroid in the frame must match.
    marker_x0: int = video_w // 2 - video_w // 20
    marker_y0: int = video_h // 2 - video_h // 20
    video: Float32[ndarray, "h w 3"] = np.zeros((video_h, video_w, 3), dtype=np.float32)
    video[marker_y0 : marker_y0 + video_h // 10, marker_x0 : marker_x0 + video_w // 10] = 1.0
    frame: UInt8[ndarray, "fh fw 3"] = resize_img(video, size).rgb_uint8
    ys, xs = np.nonzero(frame[..., 0] > 127)
    marker_center_video: Float32[ndarray, "2"] = np.array(
        [marker_x0 + video_w // 20, marker_y0 + video_h // 20], dtype=np.float32
    )
    expected: Float32[ndarray, "2"] = np.array(xform.scale) * marker_center_video + np.array(xform.translation)
    # Pixel centers sit at +0.5 in Rerun's image space; the block centroid uses integer indices.
    np.testing.assert_allclose((xs.mean() + 0.5, ys.mean() + 0.5), expected, atol=0.75)


def test_downsample_divides_the_transform() -> None:
    """img_downsample strides the cropped frame, so both scale and translation shrink by it."""
    full: VideoToFrame = video_to_frame_transform(1000, 750, 512)
    half: VideoToFrame = video_to_frame_transform(1000, 750, 512, downsample=2)
    np.testing.assert_allclose(half.scale, np.array(full.scale) / 2)
    np.testing.assert_allclose(half.translation, np.array(full.translation) / 2)
