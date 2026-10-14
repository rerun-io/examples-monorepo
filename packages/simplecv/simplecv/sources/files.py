"""Local inputs as dagkit Sources: a video file or a directory of images, as one-view ``MultiViewFrames`` on the CPU."""

from collections.abc import Iterator, Mapping
from pathlib import Path

import cv2
import numpy as np
import torch
from dagkit.node import Event, PortType
from jaxtyping import UInt8
from natsort import natsorted
from numpy import ndarray
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

    def events(self) -> Iterator[Event]:
        for i, path in enumerate(self.paths):
            bgr: UInt8[ndarray, "h w 3"] | None = cv2.imread(str(path))
            if bgr is None:
                raise ValueError(f"{path}: not a readable image")
            rgb: UInt8[ndarray, "h w 3"] = np.ascontiguousarray(bgr[..., ::-1])
            yield _frames(round(i * self.stride / self.fps * 1e9), i * self.stride, self.view, torch.from_numpy(rgb).permute(2, 0, 1))

    def close(self) -> None:
        pass
