from itertools import chain
from multiprocessing import Queue
from pathlib import Path

import cv2
import mmcv
import numpy as np
from jaxtyping import Float64, UInt8
from numpy import ndarray


def load_calib(calib: str) -> tuple[Float64[ndarray, "3 3"], Float64[ndarray, "n"]]:
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
    """image generator"""

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

        if calib_data is not None:
            intrinsics = np.array([fx, fy, cx, cy])
        else:
            intrinsics = None

        h: int
        w: int
        h, w, _ = image.shape
        image = image[: h - h % 16, : w - w % 16]

        queue.put((t, image, intrinsics))

    queue.put((-1, image, intrinsics))


def video_stream(
    queue: Queue, imagedir: str, calib: str | None, stride: int, skip: int = 0
) -> None:
    """video generator"""
    calib_data: Float64[ndarray, "n"] | None = None
    fx: float = 0.0
    fy: float = 0.0
    cx: float = 0.0
    cy: float = 0.0

    if calib is not None:
        K: Float64[ndarray, "3 3"]
        K, calib_data = load_calib(calib)
        fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])

    video_reader: mmcv.VideoReader = mmcv.VideoReader(imagedir)

    t: int = 0

    image: UInt8[ndarray, "h w 3"] | None = None
    intrinsics: Float64[ndarray, "4"] | None = None

    for _ in range(skip):
        image = video_reader.read()

    while True:
        # Capture frame-by-frame
        for _ in range(stride):
            image = video_reader.read()
            if image is None:
                break

        if image is None:
            break

        # if len(calib) > 4:
        #     image = cv2.undistort(image, K, calib[4:])

        image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        h: int
        w: int
        h, w, _ = image.shape
        image = image[: h - h % 16, : w - w % 16]

        if calib_data is not None:
            intrinsics = np.array([fx * 0.5, fy * 0.5, cx * 0.5, cy * 0.5])
        else:
            intrinsics = None
        queue.put((t, image, intrinsics))

        t += 1

    queue.put((-1, image, intrinsics))
