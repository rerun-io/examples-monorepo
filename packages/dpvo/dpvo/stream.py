"""Frame sources for the DPVO pipeline.

Both sources yield :class:`Frame` tuples whose ``bgr_3hw`` is already on the tracker's
device (CUDA by default):

* :func:`video_frames` decodes a video on that device (NVDEC on CUDA, simplecv
  ``TorchCodecVideoReader``), halves it with an area resize there and never copies a frame
  to the host.
* :func:`image_dir_frames` reads an image directory in a background process
  (:class:`FrameReader`) and keeps the host copy for image logging.

Frames are cropped so that both height and width are divisible by 16,
as required by the stride-4 feature extractor (which itself has a stride-2
convolution followed by stride-2 pooling).
"""

import os
from collections.abc import Iterator
from itertools import chain
from multiprocessing import Process
from multiprocessing import Queue as MpQueue
from multiprocessing.queues import Queue
from pathlib import Path
from types import TracebackType
from typing import NamedTuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float64, UInt8
from numpy import ndarray
from simplecv.video_io import TorchCodecVideoReader
from torch import Tensor
from torchcodec.decoders import VideoDecoder, VideoStreamMetadata

from dpvo.utils import DEFAULT_DEVICE

VIDEO_SCALE: float = 0.5
"""Video frames are halved before DPVO to reduce compute; calibration files describe the native video."""


class Frame(NamedTuple):
    """One frame handed to DPVO."""

    t: int
    """Index among the processed (strided) frames."""
    source_index: int
    """Index in the source video or sorted image list."""
    bgr_3hw: UInt8[Tensor, "3 h w"]
    """BGR image on the tracker's device, cropped to multiples of 16."""
    bgr_hw3: UInt8[ndarray, "h w 3"] | None
    """Host copy for image logging; ``None`` for video, which is logged once as a VideoStream."""


def crop16(height: int, width: int) -> tuple[int, int]:
    """Largest ``(height, width)`` divisible by 16 that fits the input."""
    return height - height % 16, width - width % 16


def video_frames(video_path: Path, stride: int, skip: int = 0, *, device: torch.device = DEFAULT_DEVICE) -> Iterator[Frame]:
    """Decode a video on ``device`` and yield every ``stride``-th frame after ``skip``.

    Keeps the last frame of each complete stride group (source index
    ``skip + i * stride + stride - 1``), halves it with an area resize and
    crops it to multiples of 16. Frames stay on ``device``. TorchCodec decodes on CUDA or CPU only, so an
    MPS (or CPU) tracker gets CPU-decoded frames moved to its device.
    """
    decode_device: torch.device = device if device.type == "cuda" else torch.device("cpu")
    reader: TorchCodecVideoReader = TorchCodecVideoReader(video_path, device=decode_device)
    half_hw: tuple[int, int] = (round(reader.source_height * VIDEO_SCALE), round(reader.source_width * VIDEO_SCALE))
    crop_h, crop_w = crop16(*half_hw)
    try:
        t: int = 0
        source_index: int = -1
        while (rgb_3hw := reader.read()) is not None:
            source_index += 1
            if source_index < skip or (source_index - skip) % stride != stride - 1:
                continue
            half: Tensor = F.interpolate(rgb_3hw[None].to(device).float(), size=half_hw, mode="area")[0]
            bgr_3hw: UInt8[Tensor, "3 h w"] = half.round().to(torch.uint8)[[2, 1, 0], :crop_h, :crop_w].contiguous()
            yield Frame(t, source_index, bgr_3hw, None)
            t += 1
    finally:
        reader.close()


def image_dir_frames(imagedir: str, stride: int, skip: int = 0, *, device: torch.device = DEFAULT_DEVICE) -> Iterator[Frame]:
    """Yield every ``stride``-th image of a directory after ``skip``, read in a background process, on ``device``."""
    with FrameReader(imagedir, stride, skip) as queue:
        while True:
            t, bgr_hw3 = queue.get()
            if t < 0:
                return
            yield Frame(t, skip + t * stride, torch.from_numpy(bgr_hw3).permute(2, 0, 1).to(device), bgr_hw3)


def _video_metadata(video_path: Path) -> VideoStreamMetadata:
    """Container metadata of the video's best stream, as TorchCodec decodes it."""
    return VideoDecoder(video_path, seek_mode="approximate").metadata


def video_hw(video_path: Path) -> tuple[int, int]:
    """Native ``(height, width)`` of a video's frames."""
    metadata: VideoStreamMetadata = _video_metadata(video_path)
    return int(metadata.height or 0), int(metadata.width or 0)


def num_frames(video_or_image_dir: str, stride: int, skip: int) -> int:
    """Number of frames a source yields after ``skip`` and ``stride``."""
    if os.path.isdir(video_or_image_dir):
        total: int = len(
            [name for name in os.listdir(video_or_image_dir) if os.path.isfile(os.path.join(video_or_image_dir, name))]
        )
    else:
        total = int(_video_metadata(Path(video_or_image_dir)).num_frames or 0)
    return max(0, (total - skip) // stride)


class FrameReader:
    """Context manager that runs :func:`image_stream` in a background process.

    Automatically kills and joins the reader process on exit — whether
    the block completes normally, raises an exception, or the generator
    consumer stops early.  The reader is a daemon process, so it also
    dies if the parent is killed.

    Usage::

        with FrameReader(imagedir, stride) as queue:
            while True:
                t, image = queue.get()
                if t < 0:
                    break
                ...
    """

    def __init__(self, imagedir: str, stride: int, skip: int = 0) -> None:
        self.imagedir = imagedir
        self.stride = stride
        self.skip = skip
        self._queue: Queue = MpQueue(maxsize=8)
        self._process: Process | None = None

    def __enter__(self) -> Queue:
        self._process = Process(target=image_stream, args=(self._queue, self.imagedir, self.stride, self.skip))
        self._process.daemon = True
        self._process.start()
        return self._queue

    def __exit__(self, _exc_type: type[BaseException] | None, _exc_val: BaseException | None, _exc_tb: TracebackType | None) -> None:
        if self._process is not None:
            if self._process.is_alive():
                self._process.kill()
            self._process.join(timeout=5)
            self._process = None


def load_calib(calib: str) -> tuple[Float64[ndarray, "3 3"], Float64[ndarray, "n"]]:
    """Load camera calibration from a space-delimited text file.

    Expected file format: ``fx fy cx cy [k1 k2 ...]`` on a single line.
    The first four values are used to construct a 3×3 intrinsic matrix K;
    the full line (including optional distortion coefficients) is also
    returned for downstream use.

    Args:
        calib: Path to the calibration text file.

    Returns:
        A 2-tuple of:
        - ``K``: 3×3 camera intrinsic matrix.
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


def image_stream(queue: Queue, imagedir: str, stride: int, skip: int = 0) -> None:
    """Read frames from a directory of images and push them to a queue.

    Images are sorted lexicographically by filename.  The ``skip`` and
    ``stride`` parameters allow subsampling: ``skip`` initial frames are
    discarded, then every ``stride``-th frame is loaded.

    Each item pushed to the queue is ``(timestamp: int, image: ndarray[h, w, 3])``.
    A sentinel ``(-1, None)`` signals end-of-stream.

    Args:
        queue: Multiprocessing queue to push ``(t, image)`` tuples into.
        imagedir: Directory containing image files (png, jpeg, jpg).
        stride: Frame stride -- only every ``stride``-th image is loaded.
        skip: Number of initial frames to skip before striding.
    """
    img_exts: list[str] = ["*.png", "*.jpeg", "*.jpg"]
    image_list: list[Path] = sorted(chain.from_iterable(Path(imagedir).glob(e) for e in img_exts))[skip::stride]

    for t, imfile in enumerate(image_list):
        image: UInt8[ndarray, "h w 3"] | None = cv2.imread(str(imfile))
        assert image is not None, f"Failed to read image: {imfile}"
        crop_h, crop_w = crop16(image.shape[0], image.shape[1])
        queue.put((t, image[:crop_h, :crop_w]))

    # Sentinel: t=-1 signals end of stream
    queue.put((-1, None))
