"""The tracker's per-segment output: arrays in one ``.npz`` with fixed keys, metadata in a JSON sidecar.

Shapes: f frames, 4 cameras, 2 hands (slot 0 = left, 1 = right). Pixels are the camera's own image
(pixel-centre convention), 3D points world metres.

A box is the hand box itself: the square that encloses the hand circle (DetNet's circle, or the smallest
enclosing circle of the extrapolated pose's keypoints in that camera). The KeyNet crop is that box
enlarged by 20% about its centre (``labels.crops.BOX_ENLARGE``).

The DetNet-alone run (``kind = "detnet_alone"``) uses the same record: DetNet ran on every frame and camera,
``presence`` holds DetNet's presence probability per camera and hand, ``box`` and ``box_source = DETNET``
are set where it reported a hand (presence > 0.5), and the tracker fields are empty.
"""

from dataclasses import dataclass, fields
from enum import IntEnum
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
from jaxtyping import Bool, Float32, Int8, Int64
from numpy import ndarray
from serde import serde
from serde.json import from_json, to_json

HandMode: TypeAlias = Literal["known", "unknown"]
"""Known hand: the subject's profile model; unknown: the generic model scaled by the calibrated ϕ."""
TrackKind: TypeAlias = Literal["tracker", "detnet_alone"]
DetectorSource: TypeAlias = Literal["detnet", "oracle"]
"""``oracle``: ground-truth circles with presence 1 where the hand is present (>= 17 keypoints inside)."""
KeypointSource: TypeAlias = Literal["keynet", "oracle", "keynet_gt_boxes"]
"""``oracle``: projected ground-truth keypoints plus Gaussian noise instead of KeyNet; ``keynet_gt_boxes``: KeyNet on the
ground-truth crop of each requested view instead of the tracker's crop (a diagnostic that removes crop drift)."""


class BoxSource(IntEnum):
    """Where a camera's hand box came from on a frame."""

    NONE = 0
    DETNET = 1
    TRACKED = 2


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class TrackMetadata:
    """What produced a ``SegmentTrack``."""

    segment: str
    """Catalog segment id (the recording id every layer of the segment shares)."""
    detnet_sha256: str
    """sha256 of the DetNet weights file; ``oracle`` or ``random`` when no checkpoint was used."""
    keynet_sha256: str
    """sha256 of the KeyNet weights file; ``oracle`` or ``random`` when no checkpoint was used."""
    hand_mode: HandMode
    hand_scale: float
    """ϕ: the subject's size relative to the generic hand model (the profile's for a known hand, the calibrated one otherwise)."""
    timings_s: dict[str, float]
    """Wall-clock seconds per stage."""
    dataset: str = "dataforge-umetrack"
    kind: TrackKind = "tracker"
    detector: DetectorSource = "detnet"
    keypoints: KeypointSource = "keynet"


@dataclass(frozen=True, slots=True)
class SegmentTrack:
    """One segment's tracker output."""

    meta: TrackMetadata
    video_time_ns: Int64[ndarray, "f"]
    frame_index: Int64[ndarray, "f"]
    """Row of the segment's ``video_time`` timeline (0-based)."""
    tracked: Bool[ndarray, "f 2"]
    """The hand has a fitted pose on this frame."""
    rotation: Float32[ndarray, "f 2 3 3"]
    """World-from-wrist rotation, NaN when untracked."""
    translation: Float32[ndarray, "f 2 3"]
    """World-from-wrist translation, metres, NaN when untracked."""
    joint_angles: Float32[ndarray, "f 2 22"]
    """Radians, NaN when untracked."""
    landmarks: Float32[ndarray, "f 2 21 3"]
    """World metres, NaN when untracked."""
    box: Float32[ndarray, "f 4 2 4"]
    """(x0, y0, x1, y1) camera pixels, NaN when none."""
    box_source: Int8[ndarray, "f 4 2"]
    """``BoxSource`` values."""
    keypoints_2d: Float32[ndarray, "f 4 2 21 2"]
    """KeyNet's keypoints in camera pixels, NaN where KeyNet did not run."""
    presence: Float32[ndarray, "f 4 2"]
    """KeyNet presence, NaN where KeyNet did not run (DetNet's presence in a DetNet-alone record)."""
    detnet_camera: Int8[ndarray, "f"]
    """Camera DetNet ran on, -1 when it did not run."""
    detnet_presence: Float32[ndarray, "f 2"]
    """DetNet's presence per hand slot on ``detnet_camera``, NaN when it did not run."""
    fit_energy: Float32[ndarray, "f 2"]
    """The fit's final energy (pixels²), NaN when untracked."""


ARRAY_KEYS: tuple[str, ...] = tuple(field.name for field in fields(SegmentTrack) if field.name != "meta")
"""The npz keys, one per array field."""


def track_paths(directory: Path, segment: str) -> tuple[Path, Path]:
    """The npz and its JSON sidecar for a segment."""
    return directory / f"{segment}.npz", directory / f"{segment}.json"


def save_track(track: SegmentTrack, directory: Path) -> Path:
    """Write ``<segment>.npz`` and ``<segment>.json`` into ``directory``; returns the npz path."""
    directory.mkdir(parents=True, exist_ok=True)
    npz, sidecar = track_paths(directory, track.meta.segment)
    np.savez(npz, **{name: getattr(track, name) for name in ARRAY_KEYS})
    sidecar.write_text(to_json(track.meta))
    return npz


def load_track(npz: Path) -> SegmentTrack:
    """Read a ``SegmentTrack`` written by ``save_track``."""
    meta: TrackMetadata = from_json(TrackMetadata, npz.with_suffix(".json").read_text())
    with np.load(npz) as arrays:
        return SegmentTrack(meta, **{name: arrays[name] for name in ARRAY_KEYS})
