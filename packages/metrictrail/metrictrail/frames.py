"""MetricTrail inputs: a video, a frame directory or a VSLAM-LAB sequence, and their frames for logging.

Frame order is ViPE's: ``FrameDirStream`` reads every image file of a directory in sorted path order, so source frame
``i`` is the ``i``-th sorted file; a video's source frame ``i`` is its ``i``-th decoded frame.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

import cv2
import numpy as np
from jaxtyping import UInt8
from numpy import ndarray

FRAME_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")
"""Image extensions ViPE's ``FrameDirStream`` reads (either case)."""
VIDEO_EXTENSIONS: tuple[str, ...] = (".mp4", ".mov", ".avi", ".mkv", ".webm")
"""Containers the ``video`` preset hands to ViPE's ``raw_mp4_stream``."""

InputKind: TypeAlias = Literal["video", "frame_dir", "vslamlab_sequence"]
"""What a MetricTrail input path is."""
DisplayRaster: TypeAlias = Literal["unchanged", "cover_crop", "resize", "pipeline_default"]
"""How an input frame was put on the tracking raster (the preset's ``raster_policy``)."""


def list_frames(frame_dir: Path) -> list[str]:
    """Image file names of a frame directory in ViPE's ``FrameDirStream`` order (sorted paths).

    Raises:
        ValueError: If the directory holds no image file.
    """
    paths: set[Path] = {path for path in frame_dir.iterdir() if path.suffix.lower() in FRAME_EXTENSIONS and path.is_file()}
    if not paths:
        raise ValueError(f"no frames in {frame_dir}")
    return [path.name for path in sorted(paths)]


def input_kind(path: Path) -> InputKind:
    """Classify an input path.

    A VSLAM-LAB sequence is a directory with ``rgb.csv`` and ``rgb_0/``; any other directory is a frame directory;
    a file with a video extension is a video.

    Raises:
        ValueError: If the path is neither.
    """
    if path.is_dir():
        return "vslamlab_sequence" if (path / "rgb.csv").is_file() and (path / "rgb_0").is_dir() else "frame_dir"
    if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
        return "video"
    raise ValueError(f"{path} is not a frame directory, a VSLAM-LAB sequence or a video ({', '.join(VIDEO_EXTENSIONS)})")


def link_sequence_frames(rgb_dir: Path, target_dir: Path) -> list[str]:
    """Symlink every frame of a VSLAM-LAB ``rgb_0/`` into ``target_dir`` under its own name; return the names in ViPE order.

    ``target_dir``'s name is what ViPE names the outputs and seeds DPVO with, so the caller names it after upstream's
    eval case. Existing links to the same files are kept.

    Raises:
        FileExistsError: If ``target_dir`` holds a file that is not a link to the same frame.
    """
    names: list[str] = list_frames(rgb_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        link: Path = target_dir / name
        source: Path = (rgb_dir / name).resolve()
        if link.is_symlink() and link.resolve() == source:
            continue
        if link.exists() or link.is_symlink():
            raise FileExistsError(f"{link} exists and does not link {source}")
        link.symlink_to(source)
    return names


def read_rgb(path: Path) -> UInt8[ndarray, "h w 3"]:
    """One image file as RGB uint8 (OpenCV decode, as ViPE's frame-directory stream).

    Raises:
        ValueError: If OpenCV cannot decode the file.
    """
    image: ndarray | None = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"could not read frame {path}")
    return np.ascontiguousarray(image[:, :, ::-1])


class VideoFrames:
    """Source frame ``i`` of a video, decoded sequentially with OpenCV (re-opened on a backwards seek)."""

    def __init__(self, path: Path) -> None:
        self.path: Path = path
        self._capture: cv2.VideoCapture | None = None
        self._next: int = 0

    def __call__(self, index: int) -> UInt8[ndarray, "h w 3"]:
        if self._capture is None or index < self._next:
            if self._capture is not None:
                self._capture.release()
            self._capture = cv2.VideoCapture(str(self.path))
            self._next = 0
        while self._next < index:
            if not self._capture.grab():
                raise IndexError(f"{self.path} ends before frame {index}")
            self._next += 1
        ok: bool
        frame: ndarray
        ok, frame = self._capture.read()
        if not ok:
            raise IndexError(f"{self.path} ends before frame {index}")
        self._next += 1
        return np.ascontiguousarray(frame[:, :, ::-1])


FrameReader: TypeAlias = Callable[[int], UInt8[ndarray, "h w 3"]]
"""``read(source_frame) -> RGB``."""


@dataclass(frozen=True, slots=True)
class FrameSource:
    """The frames a run read, addressed by source frame index."""

    path: Path
    """Video file or frame directory."""
    kind: Literal["video", "frame_dir"]
    """Which of the two."""
    names: list[str]
    """Frame file names in ViPE order (empty for a video)."""

    def reader(self) -> FrameReader:
        """A reader of source frame ``i`` as RGB uint8."""
        if self.kind == "video":
            return VideoFrames(self.path)
        names: list[str] = self.names
        directory: Path = self.path
        return lambda index: read_rgb(directory / names[index])


def open_frames(path: Path) -> FrameSource:
    """Open a video or a frame directory as a :class:`FrameSource`."""
    if path.is_file():
        return FrameSource(path=path, kind="video", names=[])
    return FrameSource(path=path, kind="frame_dir", names=list_frames(path))


def fit_for_display(image: UInt8[ndarray, "h w 3"], raster_hw: tuple[int, int], policy: DisplayRaster) -> UInt8[ndarray, "H W 3"]:
    """An input frame on the tracking raster, for display next to the depth (not the tracker's exact resampler).

    ``resize`` stretches; ``cover_crop`` and ``pipeline_default`` cover-fit and take the integer centre crop (the
    geometry of upstream's ``cover_crop``); ``unchanged`` frames are returned as they are when already on the raster
    and stretched otherwise. Resampling is OpenCV ``INTER_AREA``, so pixels may differ from what DPVO saw by a few
    levels; the live capture (:mod:`metrictrail.capture`) logs DPVO's own image instead.
    """
    height, width = raster_hw
    if image.shape[:2] == (height, width):
        return image
    if policy in ("cover_crop", "pipeline_default"):
        scale: float = max(width / image.shape[1], height / image.shape[0])
        resized_hw: tuple[int, int] = (max(height, round(image.shape[0] * scale)), max(width, round(image.shape[1] * scale)))
        resized: UInt8[ndarray, "a b 3"] = cv2.resize(image, (resized_hw[1], resized_hw[0]), interpolation=cv2.INTER_AREA)
        top: int = (resized_hw[0] - height) // 2
        left: int = (resized_hw[1] - width) // 2
        return np.ascontiguousarray(resized[top : top + height, left : left + width])
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
