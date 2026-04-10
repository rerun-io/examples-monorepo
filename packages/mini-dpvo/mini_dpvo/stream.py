import cv2
import numpy as np
from pathlib import Path
from itertools import chain
from multiprocessing import Queue
import mmcv


def load_calib(calib: str) -> np.ndarray:
    calib = np.loadtxt(calib, delimiter=" ")
    fx, fy, cx, cy = calib[:4]

    K = np.eye(3)
    K[0, 0] = fx
    K[0, 2] = cx
    K[1, 1] = fy
    K[1, 2] = cy
    return K, calib


def image_stream(
    queue: Queue, imagedir: str, calib: str | None, stride: int, skip: int = 0
) -> None:
    """image generator"""

    if calib is not None:
        K, calib = load_calib(calib)
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    img_exts = ["*.png", "*.jpeg", "*.jpg"]
    image_list = sorted(chain.from_iterable(Path(imagedir).glob(e) for e in img_exts))[
        skip::stride
    ]

    for t, imfile in enumerate(image_list):
        image = cv2.imread(str(imfile))

        if calib is not None:
            intrinsics = np.array([fx, fy, cx, cy])
        else:
            intrinsics = None

        h, w, _ = image.shape
        image = image[: h - h % 16, : w - w % 16]

        queue.put((t, image, intrinsics))

    queue.put((-1, image, intrinsics))


def video_stream(
    queue: Queue, imagedir: str, calib: str | None, stride: int, skip: int = 0
) -> None:
    """video generator"""
    if calib is not None:
        K, calib = load_calib(calib)
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    video_reader = mmcv.VideoReader(imagedir)

    t = 0

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
        h, w, _ = image.shape
        image = image[: h - h % 16, : w - w % 16]

        if calib is not None:
            intrinsics = np.array([fx * 0.5, fy * 0.5, cx * 0.5, cy * 0.5])
        else:
            intrinsics = None
        queue.put((t, image, intrinsics))

        t += 1

    queue.put((-1, image, intrinsics))
