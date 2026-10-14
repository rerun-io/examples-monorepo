"""A live camera as a dagkit Source (feed it through a DropOldest edge and run the graph realtime)."""

import time
from collections.abc import Iterator, Mapping

import cv2
import numpy as np
import torch
from dagkit.node import Event, PortType
from jaxtyping import UInt8
from numpy import ndarray
from simplecv.types.frames import MultiViewFrames


class Webcam:
    """Read frames from a live camera.

    OpenCV camera ``device`` as one-view ``MultiViewFrames`` on a monotonic clock that starts at the first frame."""

    clock: str = "video_time"
    outputs: Mapping[str, PortType] = {"frames": MultiViewFrames}

    def __init__(self, device: int = 0, max_frames: int | None = None) -> None:
        self.device, self.max_frames = device, max_frames

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
                t_ns: int = time.monotonic_ns() - start
                rgb: UInt8[ndarray, "h w 3"] = np.ascontiguousarray(bgr[..., ::-1])
                yield Event(t_ns, "frames", MultiViewFrames(t_ns, index, ("webcam",), torch.from_numpy(rgb).permute(2, 0, 1)[None], torch.tensor([t_ns])))
                index += 1
        finally:
            capture.release()

    def close(self) -> None:
        pass
