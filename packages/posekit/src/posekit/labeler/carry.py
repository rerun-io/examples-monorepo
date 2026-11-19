"""Visual single-object tracking that carries a reviewed hand box between EgoHand detections (MEgATrack section 4.2).

``VisualCarrier`` is the ``BoxCarrier`` of ``posekit.labeler.tracker``: a *run* is one OpenCV tracker initialised on
one camera's frame at a grid time with a box, then updated on EVERY following frame of that camera (30 fps, not only
the 5 fps grid frames: a hand moves too far in 200 ms, measured on s44 where grid-only tracking lost a hand lifting a
phone and every-frame tracking kept it). The run loses the box when the tracker reports failure, when the camera has
no frame at a grid time, or when less than ``MIN_INSIDE`` of the box lies inside the image; it stays lost. Carried
boxes are clipped to the image.

Runs are keyed by (cam, start time, start box) and kept (LRU), so a label write that recomputes the machine track
reuses them; a run only depends on its key and the decoded frames, so the same key always gives the same boxes.
Frames are decoded by ``frame_range`` in one forward pass from the frame a run needs to its last possible grid time
(``CARRY_STEPS`` ahead), downscaled by ``TRACK_SCALE``, and kept in a small LRU shared by the runs of the segment.
"""

from __future__ import annotations

import hashlib
import threading
import urllib.request
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, TypeAlias, runtime_checkable

import cv2
from jaxtyping import UInt8
from numpy import ndarray

from posekit.labeler.schema import Box
from posekit.labeler.tracker import CARRY_STEPS

VisualTrackerKind: TypeAlias = Literal["off", "nano", "csrt"]
"""OpenCV tracker that carries reviewed boxes (``off`` = chains only hold, as before).

Measured on RoboCap at half size (2026-11-08, 100 starts on s12/s13/s15) against EgoHand re-detections over 10 grid
steps: ``nano`` (NanoTrack v2, two small ONNX files) 2.7 ms per update and the most steps carried (8.1-9.0 of 10);
``csrt`` (no model file) ~10 ms per update, 6.7-8.6 steps. KCF loses the box too often (33-88 %); ViT trails both.
"""
OPENCV_THREADS: int = 4
"""OpenCV threads for a server that tracks: a NanoTrack update takes 1.6 ms on 4 threads, 2.3 ms on the default 20
(oversubscribed), 3.6 ms on 1 (GB10, 960x540)."""
TRACK_SCALE: float = 0.5
"""Frames are tracked at this fraction of their native size (1920x1080 -> 960x540)."""
MIN_INSIDE: float = 0.5
"""Share of a carried box's area that must lie inside the image; below it the hand has left the view."""
MIN_SIDE_PX: float = 4.0
"""Smallest side of a carried box, native px."""

NANOTRACK_URL: str = (
    "https://raw.githubusercontent.com/HonglinChu/SiamTrackers/8fe6791fecf13b95e6f19770af8963a9fc98759b/NanoTrack/models/nanotrackv2/{name}.onnx"
)
"""NanoTrack v2 ONNX files, pinned to one commit (the files OpenCV's tracker sample uses)."""
NANOTRACK_SHA256: dict[str, str] = {
    "nanotrack_backbone_sim": "530bdd0cd00f19afab79a863e71ba71e3312395a5dc9151af675082bdaaa2fc4",
    "nanotrack_head_sim": "0d8c0637be849f092cc7236cae02e55c8b9455ebe37ba50601d6115db4247cd9",
}
MODEL_DIR: Path = Path.home() / ".cache" / "posekit" / "nanotrack"
"""Where the NanoTrack files are downloaded to (once)."""

Rect: TypeAlias = tuple[int, int, int, int]
"""OpenCV rectangle ``x y w h`` in tracking px."""
FrameKey: TypeAlias = tuple[int, int]
"""(cam, frame index)."""
RunKey: TypeAlias = tuple[int, int, Box]
"""(cam, start grid time, start box)."""


@runtime_checkable
class CvTracker(Protocol):
    """An OpenCV single-object tracker (``cv2.TrackerCSRT`` and friends)."""

    def init(self, image: UInt8[ndarray, "h w 3"], bounding_box: Sequence[int], /) -> None:
        """Start tracking ``bounding_box`` (``x y w h``) on ``image``."""
        ...

    def update(self, image: UInt8[ndarray, "h w 3"], /) -> tuple[bool, Sequence[int]]:
        """Track to ``image``: (found, ``x y w h``)."""
        ...


@runtime_checkable
class SequentialFrames(Protocol):
    """Every decoded frame of one segment's cameras; ``posekit.labeler.frames.FrameSource`` satisfies it."""

    def sample_index(self, cam: int, t_ns: int) -> int:
        """The frame of ``cam`` shown at grid time ``t_ns``; KeyError when there is none."""
        ...

    def frame_count(self, cam: int) -> int:
        """Frames in ``cam``'s video."""
        ...

    def frame_range(self, cam: int, first: int, stop: int) -> UInt8[ndarray, "b h w 3"]:
        """Native RGB frames ``first..stop-1`` of ``cam``, in one forward pass."""
        ...


def nanotrack_file(name: str, model_dir: Path = MODEL_DIR) -> Path:
    """One NanoTrack ONNX file, downloaded on first use and checked against its SHA-256.

    Raises:
        ValueError: If the download does not match the pinned checksum.
    """
    path: Path = model_dir / f"{name}.onnx"
    if path.exists():
        return path
    data: bytes = urllib.request.urlopen(NANOTRACK_URL.format(name=name), timeout=60).read()
    if hashlib.sha256(data).hexdigest() != NANOTRACK_SHA256[name]:
        raise ValueError(f"{NANOTRACK_URL.format(name=name)}: SHA-256 does not match the pinned file")
    model_dir.mkdir(parents=True, exist_ok=True)
    partial: Path = path.with_suffix(".part")
    partial.write_bytes(data)
    partial.replace(path)
    return path


def opencv_tracker(kind: VisualTrackerKind) -> Callable[[], CvTracker]:
    """A factory of fresh OpenCV trackers of ``kind`` (``nano`` fetches its model files here).

    Raises:
        ValueError: For ``off``.
    """
    if kind == "csrt":
        return cv2.TrackerCSRT.create
    if kind == "nano":
        params: cv2.TrackerNano.Params = cv2.TrackerNano.Params()
        params.backbone = str(nanotrack_file("nanotrack_backbone_sim"))
        params.neckhead = str(nanotrack_file("nanotrack_head_sim"))
        return lambda: cv2.TrackerNano.create(params)
    raise ValueError(f"no OpenCV tracker for {kind!r}")


@dataclass(slots=True)
class Run:
    """One tracker carrying one box forward from its start frame."""

    tracker: CvTracker
    """The OpenCV tracker, advanced to ``frame``."""
    start: int
    """Grid index of the start frame."""
    frame: int
    """Index of the last frame the tracker has seen."""
    box: Box
    """Its box there, native px."""
    boxes: list[Box]
    """``boxes[k - 1]``: the box at grid index ``start + k``, native px."""
    lost: bool
    """The tracker lost the box after the last entry of ``boxes``."""


def clip_carried(rect: Sequence[float], scale: float, width: int, height: int) -> Box | None:
    """A tracker rectangle (tracking px) as a native box clipped to the image; None when the hand left the view.

    Args:
        rect: ``x y w h`` in tracking px.
        scale: Tracking px per native px.
        width: Native image width, px.
        height: Native image height, px.
    """
    x0: float = rect[0] / scale
    y0: float = rect[1] / scale
    x1: float = (rect[0] + rect[2]) / scale
    y1: float = (rect[1] + rect[3]) / scale
    area: float = max(x1 - x0, 0.0) * max(y1 - y0, 0.0)
    cx0, cy0, cx1, cy1 = max(x0, 0.0), max(y0, 0.0), min(x1, float(width)), min(y1, float(height))
    if area <= 0.0 or cx1 - cx0 < MIN_SIDE_PX or cy1 - cy0 < MIN_SIDE_PX:
        return None
    if (cx1 - cx0) * (cy1 - cy0) < MIN_INSIDE * area:
        return None
    return Box(cx0, cy0, cx1, cy1)


class VisualCarrier:
    """Carries reviewed boxes of one segment with a visual tracker; see the module docstring."""

    def __init__(
        self,
        frames: SequentialFrames,
        times_ns: list[int],
        make_tracker: Callable[[], CvTracker],
        *,
        scale: float = TRACK_SCALE,
        cached_frames: int = 160,
        cached_runs: int = 256,
    ) -> None:
        """Bind the carrier to a segment's frames and grid.

        Args:
            frames: The segment's decoded grid frames.
            times_ns: The 5 fps grid, ascending, ns.
            make_tracker: Makes a fresh tracker per run (``opencv_tracker``).
            scale: Tracking px per native px.
            cached_frames: Downscaled frames kept for the runs (~1.5 MB each at 960x540; two 2-second runs ~ 130).
            cached_runs: Runs kept (each holds one tracker and its boxes).
        """
        self.frames: SequentialFrames = frames
        self.times_ns: list[int] = list(times_ns)
        self.make_tracker: Callable[[], CvTracker] = make_tracker
        self.scale: float = scale
        self.cached_frames: int = cached_frames
        self.cached_runs: int = cached_runs
        self._index: dict[int, int] = {t: i for i, t in enumerate(self.times_ns)}
        self._frames: OrderedDict[FrameKey, UInt8[ndarray, "h w 3"]] = OrderedDict()
        self._sizes: dict[int, tuple[int, int]] = {}
        """Native (width, height) per camera, from its first decoded frame."""
        self._runs: OrderedDict[RunKey, Run] = OrderedDict()
        self._lock = threading.Lock()

    def carry(self, cam: int, start_ns: int, start_box: Box, t_ns: int) -> Box | None:
        """The box at ``t_ns`` of the run (cam, start_ns, start_box); None once the tracker has lost it.

        Raises:
            ValueError: If a time is not on the grid or ``t_ns`` is not after ``start_ns``.
        """
        start: int | None = self._index.get(start_ns)
        target: int | None = self._index.get(t_ns)
        if start is None or target is None or target <= start:
            raise ValueError(f"carry from {start_ns} to {t_ns}: both must be grid times, the second after the first")
        with self._lock:
            key: RunKey = (cam, start_ns, start_box)
            run: Run | None = self._runs.get(key)
            if run is None:
                run = self._start(cam, start, start_box)
                self._runs[key] = run
                while len(self._runs) > self.cached_runs:
                    self._runs.popitem(last=False)
            else:
                self._runs.move_to_end(key)
            while len(run.boxes) < target - start and not run.lost:
                self._step(cam, run)
            return run.boxes[target - start - 1] if target - start <= len(run.boxes) else None

    def _start(self, cam: int, start: int, box: Box) -> Run:
        """A run initialised on the start frame (lost at once when the camera has no frame there)."""
        tracker: CvTracker = self.make_tracker()
        try:
            first: int = self.frames.sample_index(cam, self.times_ns[start])
        except KeyError:
            return Run(tracker, start, -1, box, [], True)
        rect: Rect = (
            round(box.x0 * self.scale),
            round(box.y0 * self.scale),
            max(1, round((box.x1 - box.x0) * self.scale)),
            max(1, round((box.y1 - box.y0) * self.scale)),
        )
        tracker.init(self._frame(cam, first, start), rect)
        return Run(tracker, start, first, box, [], False)

    def _step(self, cam: int, run: Run) -> None:
        """Advance the run to its next grid time, through every frame between."""
        try:
            target: int = self.frames.sample_index(cam, self.times_ns[run.start + len(run.boxes) + 1])
        except KeyError:
            run.lost = True
            return
        for index in range(run.frame + 1, target + 1):
            found, rect = run.tracker.update(self._frame(cam, index, run.start))
            width, height = self._sizes[cam]
            box: Box | None = clip_carried([float(v) for v in rect], self.scale, width, height) if found else None
            if box is None:
                run.lost = True
                return
            run.frame = index
            run.box = box
        run.boxes.append(run.box)

    def _frame(self, cam: int, index: int, start: int) -> UInt8[ndarray, "h w 3"]:
        """The downscaled BGR frame ``index`` of ``cam``; a miss decodes on to the run's last possible grid time."""
        key: FrameKey = (cam, index)
        cached: UInt8[ndarray, "h w 3"] | None = self._frames.get(key)
        if cached is not None:
            self._frames.move_to_end(key)
            return cached
        last_grid: int = min(start + CARRY_STEPS, len(self.times_ns) - 1)
        try:
            stop: int = self.frames.sample_index(cam, self.times_ns[last_grid]) + 1
        except KeyError:
            stop = index + 1
        stop = min(max(stop, index + 1), self.frames.frame_count(cam))
        while stop > index + 1 and (cam, stop - 1) in self._frames:
            stop -= 1
        decoded: UInt8[ndarray, "b h w 3"] = self.frames.frame_range(cam, index, stop)
        self._sizes.setdefault(cam, (decoded.shape[2], decoded.shape[1]))
        size: tuple[int, int] = (round(decoded.shape[2] * self.scale), round(decoded.shape[1] * self.scale))
        for offset, rgb in enumerate(decoded):
            self._frames[(cam, index + offset)] = cv2.cvtColor(cv2.resize(rgb, size, interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR)
        small: UInt8[ndarray, "h w 3"] = self._frames[key]
        while len(self._frames) > self.cached_frames:
            self._frames.popitem(last=False)
        return small
