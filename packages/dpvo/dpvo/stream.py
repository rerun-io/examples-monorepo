"""Frame reader utilities for feeding images to the DPVO pipeline.

Provides two multiprocessing-friendly generator functions that load frames
from either a directory of images (:func:`image_stream`) or a video file
(:func:`video_stream`), pair them with camera intrinsics, and push
``(timestamp, image, intrinsics)`` tuples into a :class:`multiprocessing.Queue`.

The sentinel tuple ``(-1, last_image, last_intrinsics)`` signals end-of-stream.

Images are cropped so that both height and width are divisible by 16,
as required by the stride-4 feature extractor (which itself has a stride-2
convolution followed by stride-2 pooling).
"""

from itertools import chain
from multiprocessing.queues import Queue
from pathlib import Path

import cv2
import numpy as np
from jaxtyping import Float64, UInt8
from numpy import ndarray
from simplecv.video_io import VideoReader


def load_calib(calib: str) -> tuple[Float64[ndarray, "3 3"], Float64[ndarray, "n"]]:
    """Load camera calibration from a space-delimited text file.

    Expected file format: ``fx fy cx cy [k1 k2 ...]`` on a single line.
    The first four values are used to construct a 3x3 intrinsic matrix K;
    the full line (including optional distortion coefficients) is also
    returned for downstream use.

    Args:
        calib: Path to the calibration text file.

    Returns:
        A 2-tuple of:
        - ``K``: 3x3 camera intrinsic matrix.
        - ``calib_data``: Raw calibration values as a 1-D array.
    """
    calib_data: Float64[ndarray, "n"] = np.loadtxt(calib, delimiter=" ")
    fx: float = float(calib_data[0])
    fy: float = float(calib_data[1])
    cx: float = float(calib_data[2])
    cy: float = float(calib_data[3])

    K: Float64[ndarray, "3 3"] = np.eye(3)
    K[0, 0] = fx
    K[0, 2] = cx
    K[1, 1] = fy
    K[1, 2] = cy
    return K, calib_data


def image_stream(
    queue: Queue, imagedir: str, calib: str | None, stride: int, skip: int = 0
) -> None:
    """Read frames from a directory of images and push them to a queue.

    Images are sorted lexicographically by filename.  The ``skip`` and
    ``stride`` parameters allow subsampling: ``skip`` initial frames are
    discarded, then every ``stride``-th frame is loaded.

    Each item pushed to the queue is a 3-tuple
    ``(timestamp: int, image: ndarray[h, w, 3], intrinsics: ndarray[4] | None)``.
    A sentinel ``(t=-1, ...)`` signals end-of-stream.

    Args:
        queue: Multiprocessing queue to push ``(t, image, intrinsics)``
            tuples into.
        imagedir: Directory containing image files (png, jpeg, jpg).
        calib: Path to calibration file, or ``None`` if intrinsics are
            unavailable (in which case ``intrinsics`` is set to ``None``).
        stride: Frame stride -- only every ``stride``-th image is loaded.
        skip: Number of initial frames to skip before striding.
    """
    calib_data: Float64[ndarray, "n"] | None = None
    fx: float = 0.0
    fy: float = 0.0
    cx: float = 0.0
    cy: float = 0.0

    if calib is not None:
        K: Float64[ndarray, "3 3"]
        K, calib_data = load_calib(calib)
        fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])

    img_exts: list[str] = ["*.png", "*.jpeg", "*.jpg"]
    image_list: list[Path] = sorted(chain.from_iterable(Path(imagedir).glob(e) for e in img_exts))[
        skip::stride
    ]

    image: UInt8[ndarray, "h w 3"]
    intrinsics: Float64[ndarray, "4"] | None = None

    for t, imfile in enumerate(image_list):
        image = cv2.imread(str(imfile))

        if calib_data is not None:  # noqa: SIM108
            intrinsics = np.array([fx, fy, cx, cy])
        else:
            intrinsics = None

        h: int
        w: int
        h, w, _ = image.shape
        # Crop to make dimensions divisible by 16 (required by feature extractor)
        image = image[: h - h % 16, : w - w % 16]

        queue.put((t, image, intrinsics))

    # Sentinel: t=-1 signals end of stream
    queue.put((-1, image, intrinsics))


def video_stream(
    queue: Queue, imagedir: str, calib: str | None, stride: int, skip: int = 0
) -> None:
    """Read frames from a video file and push them to a queue.

    Uses ``simplecv.video_io.VideoReader`` to decode frames.  Frames are downscaled by
    0.5x (both axes) to reduce memory and compute requirements.  The
    intrinsics are correspondingly scaled by 0.5.

    Each item pushed to the queue is a 3-tuple
    ``(timestamp: int, image: ndarray[h, w, 3], intrinsics: ndarray[4] | None)``.
    A sentinel ``(t=-1, ...)`` signals end-of-stream.

    Args:
        queue: Multiprocessing queue to push ``(t, image, intrinsics)``
            tuples into.
        imagedir: Path to the video file (despite the parameter name).
        calib: Path to calibration file, or ``None``.
        stride: Frame stride -- only every ``stride``-th decoded frame is
            kept.
        skip: Number of initial frames to skip before striding.
    """
    calib_data: Float64[ndarray, "n"] | None = None
    fx: float = 0.0
    fy: float = 0.0
    cx: float = 0.0
    cy: float = 0.0

    if calib is not None:
        K: Float64[ndarray, "3 3"]
        K, calib_data = load_calib(calib)
        fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])

    video_reader: VideoReader = VideoReader(Path(imagedir))

    t: int = 0

    image: UInt8[ndarray, "h w 3"] | None = None
    intrinsics: Float64[ndarray, "4"] | None = None

    # Skip initial frames
    for _ in range(skip):
        image = video_reader.read()

    while True:
        # Read and discard (stride - 1) frames, keeping only the stride-th
        for _ in range(stride):
            image = video_reader.read()
            if image is None:
                break

        if image is None:
            break

        # if len(calib) > 4:
        #     image = cv2.undistort(image, K, calib[4:])

        # Downscale by 2x to reduce compute
        image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        h: int
        w: int
        h, w, _ = image.shape
        # Crop to make dimensions divisible by 16
        image = image[: h - h % 16, : w - w % 16]

        if calib_data is not None:  # noqa: SIM108
            # Scale intrinsics by 0.5 to match the downscaled image
            intrinsics = np.array([fx * 0.5, fy * 0.5, cx * 0.5, cy * 0.5])
        else:
            intrinsics = None
        queue.put((t, image, intrinsics))

        t += 1

    # Sentinel: t=-1 signals end of stream
    queue.put((-1, image, intrinsics))
