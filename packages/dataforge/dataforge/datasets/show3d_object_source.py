"""Typed object poses and the headset inputs used for geometric census checks."""

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import NamedTuple

import numpy as np
from jaxtyping import Float64
from numpy import ndarray
from serde import serde

from dataforge.datasets.show3d_calibration import validate_transform
from dataforge.datasets.show3d_source import DEFAULT_CONFIDENCE, FrameClock, FrameInfo, read_json


@serde
@dataclass(frozen=True, slots=True)
class ObjectFrame(FrameInfo):
    """One object record, including empty pose arrays when confidence is zero."""

    R: list[list[float]]
    """World-from-object rotation, or an empty list."""
    t: list[list[float]]
    """Object centre in world millimetres, or an empty list."""
    confidence: float
    """Positive only where an object pose exists."""

    @property
    def world_T_object(self) -> Float64[ndarray, "4 4"] | None:
        """World-from-object transform in source millimetres, if posed."""
        if not self.posed:
            return None
        transform: Float64[ndarray, "4 4"] = np.eye(4)
        transform[:3, :3] = self.R
        transform[:3, 3] = np.asarray(self.t)[:, 0]
        return transform

    @property
    def posed(self) -> bool:
        """Whether the source reports a pose."""
        return self.confidence > 0.0

    @property
    def trusted(self) -> bool:
        """Posed and above the Hub's default threshold; the one place that rule lives for objects."""
        return self.confidence > DEFAULT_CONFIDENCE

    def __post_init__(self) -> None:
        if not isfinite(self.confidence) or self.confidence < 0.0:
            raise ValueError("object confidence must be finite and nonnegative")
        if self.posed:
            if np.shape(self.R) != (3, 3) or np.shape(self.t) != (3, 1):
                raise ValueError("posed object requires 3x3 R and 3x1 t")
            transform: Float64[ndarray, "4 4"] | None = self.world_T_object
            assert transform is not None
            validate_transform(transform)


class ObjectTrack(NamedTuple):
    """Object records aligned to the base clock, plus the source clock offset they carry."""

    frames: list[ObjectFrame]
    clock_offset_s: float
    """Constant ``object timestamp - frame_info timestamp`` in seconds; 0.0 when the clocks agree."""


CLOCK_TOLERANCE_S: float = 1e-6
"""Timestamps closer than this are the same instant; an offset must be constant to this tolerance."""


def read_object_frames(path: Path, clock: FrameClock) -> ObjectTrack:
    """Validate the full census, then align records to the base clock by index and frame id.

    Some releases stamp ``object_pose.json`` on a different clock origin than
    ``frame_info.json`` (a constant offset on every frame, thousands of seconds in
    ``keyboard_fix-sticky-key_910a``). ``index`` is upstream's join key, so a constant
    offset is tolerated and reported; a varying one is a corrupt file.
    """
    frames: dict[str, ObjectFrame] = read_json(path, dict[str, ObjectFrame])
    if len(frames) != clock.info.num_frames:
        raise ValueError(f"{path}: object census disagrees with scene census")
    for key, frame in frames.items():
        if key != str(frame.index):
            raise ValueError(f"{path}: key {key} disagrees with object index {frame.index}")
    selected: list[ObjectFrame] = []
    offsets: list[float] = []
    for base in clock.frames:
        frame: ObjectFrame | None = frames.get(str(base.index))
        if frame is None or frame.agt_frame_id != base.agt_frame_id:
            raise ValueError(f"{path}: object frame {base.index} disagrees with base sidecars")
        selected.append(frame)
        offsets.append(frame.timestamp - base.timestamp)
    spread: float = max(offsets) - min(offsets)
    if spread > CLOCK_TOLERANCE_S:
        raise ValueError(f"{path}: object timestamps drift against frame_info by {spread:.6g} s; the offset must be constant")
    offset: float = float(np.median(offsets))
    return ObjectTrack(selected, 0.0 if abs(offset) <= CLOCK_TOLERANCE_S else offset)
