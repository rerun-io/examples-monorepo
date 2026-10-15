"""Local inputs as dagkit Sources: a video file, a directory of images or a live camera, as one-view ``MultiViewFrames`` on the CPU.

Each also gives its frame size without reading a frame (``resolution_wh``), for buffers that are sized before the run's first frame."""

import time
from collections.abc import Iterator, Mapping
from pathlib import Path

import cv2
import numpy as np
import torch
from dagkit.node import Event, PortType
from jaxtyping import UInt8
from natsort import natsorted
from numpy import ndarray
from PIL import ExifTags, Image
from torch import Tensor

from simplecv.types.frames import MultiViewFrames

IMAGE_SUFFIXES: tuple[str, ...] = (".png", ".jpg", ".jpeg")


def _frames(t_ns: int, index: int, view: str, rgb: UInt8[Tensor, "3 h w"]) -> Event:
    return Event(t_ns, "frames", MultiViewFrames(t_ns, index, (view,), rgb[None], torch.tensor([t_ns])))


class VideoFileSource:
    """Decode a video file.

    Every ``stride``-th frame (torchcodec, CPU), whole strides only, on the stream's presentation time.
    """

    clock: str = "video_time"
    outputs: Mapping[str, PortType] = {"frames": MultiViewFrames}

    def __init__(self, path: Path, *, stride: int = 1, max_frames: int | None = None, view: str = "video") -> None:
        self.path, self.stride, self.max_frames, self.view = path, stride, max_frames, view

    def resolution_wh(self) -> tuple[int, int]:
        """From the stream's header."""
        from torchcodec.decoders import VideoDecoder

        metadata = VideoDecoder(str(self.path), device="cpu", seek_mode="approximate").metadata
        assert metadata.width and metadata.height, f"{self.path}: the stream header has no frame size"
        return metadata.width, metadata.height

    def events(self) -> Iterator[Event]:
        from torchcodec.decoders import VideoDecoder

        decoder = VideoDecoder(str(self.path), device="cpu", seek_mode="exact")
        count: int = int(decoder.metadata.num_frames or 0) // self.stride
        for i in range(count if self.max_frames is None else min(count, self.max_frames)):
            frame = decoder.get_frame_at(i * self.stride)
            yield _frames(round(frame.pts_seconds * 1e9), i * self.stride, self.view, frame.data)

    def close(self) -> None:
        pass


class ImageDirSource:
    """Read the images of a directory.

    Natural sort order, every ``stride``-th, timed at ``fps``.
    """

    clock: str = "video_time"
    outputs: Mapping[str, PortType] = {"frames": MultiViewFrames}

    def __init__(self, directory: Path, *, stride: int = 1, max_frames: int | None = None, fps: float = 30.0, view: str = "images") -> None:
        self.paths: list[Path] = natsorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)[::stride][:max_frames]
        if not self.paths:
            raise ValueError(f"{directory}: no {'/'.join(IMAGE_SUFFIXES)} images")
        self.stride, self.fps, self.view = stride, fps, view

    def resolution_wh(self) -> tuple[int, int]:
        """The first image's, from its header, turned by its EXIF orientation as ``cv2.imread`` turns the pixels."""
        with Image.open(self.paths[0]) as image:
            width, height = image.size
            turned: bool = image.getexif().get(ExifTags.Base.Orientation, 1) in (5, 6, 7, 8)
        return (height, width) if turned else (width, height)

    def events(self) -> Iterator[Event]:
        for i, path in enumerate(self.paths):
            bgr: UInt8[ndarray, "h w 3"] | None = cv2.imread(str(path))
            if bgr is None:
                raise ValueError(f"{path}: not a readable image")
            rgb: UInt8[ndarray, "h w 3"] = np.ascontiguousarray(bgr[..., ::-1])
            yield _frames(round(i * self.stride / self.fps * 1e9), i * self.stride, self.view, torch.from_numpy(rgb).permute(2, 0, 1))

    def close(self) -> None:
        pass


class WebcamSource:
    """Read frames from a live camera.

    OpenCV camera ``device`` on a monotonic clock that starts at the first frame (feed it through a DropOldest edge and
    run the graph realtime)."""

    clock: str = "video_time"
    outputs: Mapping[str, PortType] = {"frames": MultiViewFrames}

    def __init__(self, device: int = 0, max_frames: int | None = None) -> None:
        self.device, self.max_frames = device, max_frames

    def resolution_wh(self) -> tuple[int, int]:
        """From the camera's capture properties (it opens, no frame is read)."""
        capture = cv2.VideoCapture(self.device)
        width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        capture.release()
        if width * height == 0:
            raise ValueError(f"camera {self.device}: no frame size")
        return width, height

    def events(self) -> Iterator[Event]:
        capture = cv2.VideoCapture(self.device)
        start: int | None = None
        try:
            index: int = 0
            while self.max_frames is None or index < self.max_frames:
                ok, bgr = capture.read()
                if not ok:
                    raise ValueError(f"camera {self.device}: no frame")
                start = start if start is not None else time.monotonic_ns()
                rgb: UInt8[ndarray, "h w 3"] = np.ascontiguousarray(bgr[..., ::-1])
                yield _frames(time.monotonic_ns() - start, index, "webcam", torch.from_numpy(rgb).permute(2, 0, 1))
                index += 1
        finally:
            capture.release()

    def close(self) -> None:
        pass
