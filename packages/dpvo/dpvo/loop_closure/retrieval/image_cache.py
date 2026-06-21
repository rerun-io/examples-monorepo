"""Efficient image cache using temporary directory + JPEG compression.

Stores images to disk asynchronously via a multiprocessing pool to avoid
blocking the SLAM thread.  Used by the classical loop closure module to
load triplets of frames for keypoint triangulation.
"""

import os
from multiprocessing import Pool
from multiprocessing.pool import Pool as PoolType
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import torch
from einops import parse_shape
from jaxtyping import UInt8
from torch import Tensor

IMEXT = ".jpeg"
JPEG_QUALITY = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
BLANK = np.zeros((500, 500, 3), dtype=np.uint8)


class ImageCache:
    """Disk-backed image cache with async JPEG writes."""

    def __init__(self) -> None:
        self.image_buffer: dict[int, np.ndarray] = {}
        self.tmpdir: TemporaryDirectory = TemporaryDirectory()
        self.stored_indices: np.ndarray = np.zeros(100000, dtype=bool)
        self.writer_pool: PoolType = Pool(processes=1)
        self.write_result = self.writer_pool.apply_async(cv2.imwrite, [f"{self.tmpdir.name}/warmup{IMEXT}", BLANK, JPEG_QUALITY])
        self._wait()

    def __call__(self, image: UInt8[np.ndarray, "h w 3"], n: int) -> None:
        assert isinstance(image, np.ndarray)
        assert image.dtype == np.uint8
        assert parse_shape(image, "_ _ RGB") == dict(RGB=3)
        self.image_buffer[n] = image

    def _wait(self) -> None:
        self.write_result.wait()

    def _write_image(self, i: int) -> None:
        img = self.image_buffer.pop(i)
        filepath = f"{self.tmpdir.name}/{i:08d}{IMEXT}"
        assert not os.path.exists(filepath)
        self._wait()
        self.write_result = self.writer_pool.apply_async(cv2.imwrite, [filepath, img, JPEG_QUALITY])

    def load_frames(self, idxs: list[int], device: str | torch.device = "cuda") -> Tensor:
        self._wait()
        assert np.all(self.stored_indices[idxs])
        frame_list = [f"{self.tmpdir.name}/{i:08d}{IMEXT}" for i in idxs]
        assert all(map(os.path.exists, frame_list))
        image_list: list[UInt8[np.ndarray, "h w 3"]] = []
        for frame_path in frame_list:
            image: UInt8[np.ndarray, "h w 3"] | None = cv2.imread(frame_path)
            if image is None:
                raise FileNotFoundError(f"Failed to load cached image: {frame_path}")
            image_list.append(image)

        image_batch: UInt8[np.ndarray, "n h w 3"] = np.stack(image_list, axis=0)
        frames: UInt8[Tensor, "n 3 h w"] = torch.from_numpy(image_batch).permute(0, 3, 1, 2).contiguous()
        return frames.to(device=device)

    def keyframe(self, k: int) -> None:
        tmp = dict(self.image_buffer)
        self.image_buffer.clear()
        for n, v in tmp.items():
            if n != k:
                key = (n - 1) if (n > k) else n
                self.image_buffer[key] = v

    def save_up_to(self, c: int) -> None:
        for n in list(self.image_buffer):
            if n <= c:
                assert not self.stored_indices[n]
                self._write_image(n)
                self.stored_indices[n] = True

    def __enter__(self) -> "ImageCache":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._wait()
        self.tmpdir.cleanup()
        self.writer_pool.close()
