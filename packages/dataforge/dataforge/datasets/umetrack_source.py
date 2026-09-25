"""Typed UmeTrack labels, container clock and rigid camera geometry."""

import json
import re
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from jaxtyping import Bool, Float32, Float64, Int64
from numpy import ndarray
from serde import SerdeError, serde
from serde.json import from_json
from simplecv.camera_parameters import KannalaBrandtDistortion
from simplecv.umetrack_temp.generic_hand_model_numpy import HandModelNumpy


@serde
@dataclass(frozen=True, slots=True)
class Camera:
    """Source FishEye62 intrinsics; synthetic p3/p4 retain their source names and are not projected."""

    ImageSizeX: int
    """Tile width."""
    ImageSizeY: int
    """Tile height."""
    fx: float
    """Horizontal focal length in pixels."""
    fy: float
    """Vertical focal length in pixels."""
    cx: float
    """Horizontal principal point."""
    cy: float
    """Vertical principal point."""
    DistortionModel: str
    """Source model name."""
    k1: float
    """First radial coefficient."""
    k2: float
    """Second radial coefficient."""
    k3: float
    """Third radial coefficient."""
    k4: float
    """Fourth radial coefficient."""
    p1: float
    """First tangential coefficient."""
    p2: float
    """Second tangential coefficient."""
    k5: float | None = None
    """Real fifth radial coefficient."""
    k6: float | None = None
    """Real sixth radial coefficient."""
    p3: float | None = None
    """Synthetic fifth radial coefficient, as named in the source."""
    p4: float | None = None
    """Synthetic sixth radial coefficient, as named in the source."""

    def __post_init__(self) -> None:
        if self.DistortionModel != "FishEye62" or min(self.ImageSizeX, self.ImageSizeY) <= 0:
            raise ValueError("expected positive-size FishEye62 camera")

        real: bool = self.k5 is not None and self.k6 is not None and self.p3 is None and self.p4 is None
        synthetic: bool = self.p3 is not None and self.p4 is not None and self.k5 is None and self.k6 is None
        if not (real or synthetic):
            raise ValueError("expected exactly one complete radial pair: k5/k6 or p3/p4")

    def lens(self) -> KannalaBrandtDistortion:
        """The projection model the images follow; synthetic p3/p4 are not part of it.

        UmeTrack_data issue #4 reads synthetic p3/p4 as k5/k6, but values such as 56.7
        and -70.6 fold the polynomial: projected hands then miss the rendered hands by
        hundreds of pixels, while k5 = k6 = 0 lands on them in every camera (checked on
        two synthetic recordings). p3/p4 stay source data at the pinhole.
        """
        return KannalaBrandtDistortion(
            k1=self.k1, k2=self.k2, k3=self.k3, k4=self.k4, k5=self.k5 or 0.0, k6=self.k6 or 0.0, p1=self.p1, p2=self.p2
        )


@serde
@dataclass(frozen=True, slots=True)
class Labels:
    """Whole source label record; lengths and tracking markers checked at entry."""

    cameras: list[Camera]
    """Four cameras in tile order."""
    camera_angles: Float64[ndarray, "4"]
    """Crop roll angles in degrees, not camera extrinsics."""
    hand_model: HandModelNumpy
    """Subject profile in millimetres."""
    joint_angles: Float32[ndarray, "n 2 22"]
    """Joint angles in radians."""
    wrist_transforms: Float32[ndarray, "n 2 4 4"]
    """World-from-wrist matrices in millimetres; zero means absent."""
    hand_confidences: Float32[ndarray, "n 2"]
    """Shipped binary hand confidence."""
    camera_to_world_transforms: Float64[ndarray, "n 4 4 4"]
    """World-from-camera matrices in millimetres; all-zero frame means untracked."""

    def __post_init__(self) -> None:
        count: int = len(self.joint_angles)
        if (
            len(self.cameras) != 4
            or count == 0
            or any(len(a) != count for a in (self.wrist_transforms, self.hand_confidences, self.camera_to_world_transforms))
        ):
            raise ValueError("expected four cameras and equal nonempty label row counts")
        if any(not np.isfinite(a).all() for a in (self.joint_angles, self.wrist_transforms, self.hand_confidences, self.camera_to_world_transforms)):
            raise ValueError("non-finite source label")
        if not np.isin(self.hand_confidences, [0.0, 1.0]).all():
            raise ValueError("expected binary hand confidence")
        absent: Bool[ndarray, "n 2"] = np.all(self.wrist_transforms == 0, axis=(2, 3))
        if not np.array_equal(absent, self.hand_confidences == 0):
            raise ValueError("zero wrist must match confidence zero")
        missing: Bool[ndarray, "n 4"] = np.all(self.camera_to_world_transforms == 0, axis=(2, 3))
        if np.any(missing != missing[:, :1]) or np.any(self.hand_confidences[missing[:, 0]] > 0):
            raise ValueError("camera dropout must affect all cameras and both hands")


@dataclass(frozen=True, slots=True)
class SequenceData:
    """Labels and the exact presentation clock, with metric camera geometry."""

    source: Path
    """Source label path."""
    labels: Labels
    """Validated label record cut to the selected frame count."""
    source_num_frames: int
    """Full source frame count before selection."""
    profile_text: str
    """Verbatim hand_model sub-object."""
    fps: int
    """Constant integer frame rate measured from packet PTS."""
    times_ns: Int64[ndarray, "n"]
    """Selected presentation times in nanoseconds."""
    frame_indices: Int64[ndarray, "n"]
    """Selected source frame indices."""
    tracked: Bool[ndarray, "n"]
    """Rig tracking mask."""
    world_T_rig: Float64[ndarray, "n 4 4"]
    """Cam0 world pose in metres; NaN on every missing row."""
    rig_T_cam: Float64[ndarray, "4 4 4"]
    """Static extrinsics from the first tracked source frame."""
    headset_up: Float64[ndarray, "3"]
    """Normalized mean roll-corrected image-up over the whole recording."""
    headset_up_spread_deg: float
    """Largest angle from a camera's mean roll-corrected up to headset_up."""

    @property
    def count(self) -> int:
        return len(self.times_ns)

    def crop(self, camera: int) -> tuple[int, int, int, int]:
        """ffmpeg width, height, x, y for one horizontal source tile."""
        source: Camera = self.labels.cameras[camera]
        return source.ImageSizeX, source.ImageSizeY, camera * source.ImageSizeX, 0


def read_sequence(source: Path, frame_limit: int | None = None) -> SequenceData:
    """Read JSON once and inspect compressed packet timestamps without decoding video."""
    text: str = source.read_text()
    try:
        labels: Labels = from_json(Labels, text)
    except (SerdeError, json.JSONDecodeError) as error:
        raise ValueError(f"{source}: {error}") from error
    match: re.Match[str] | None = re.search(r'"hand_model"\s*:\s*', text)
    if match is None:
        raise ValueError(f"{source}: missing hand_model")
    # Decode only to find the exact end; keep the source substring, not a re-serialization.
    _, end = json.JSONDecoder().raw_decode(text, match.end())
    profile: str = text[match.end() : end]
    with av.open(str(source.with_suffix(".mp4"))) as container:
        stream: av.video.stream.VideoStream = container.streams.video[0]
        if stream.time_base is None:
            raise ValueError(f"{source}: missing video time base")
        pts: list[int] = sorted(packet.pts for packet in container.demux(stream) if packet.pts is not None)
        if len(pts) != len(labels.joint_angles) or len(pts) < 2 or pts[0] != 0:
            raise ValueError(f"{source}: expected zero-origin PTS matching label rows")
        duration: int = pts[1] - pts[0]
        if duration <= 0 or any(b - a != duration for a, b in zip(pts, pts[1:], strict=False)):
            raise ValueError(f"{source}: nonconstant presentation clock")
        rate: Fraction = 1 / (duration * stream.time_base)
        if rate.denominator != 1:
            raise ValueError(f"{source}: noninteger frame rate {rate}")
        fps: int = int(rate)
        for camera in labels.cameras:
            if (camera.ImageSizeX * 4, camera.ImageSizeY) != (stream.width, stream.height):
                raise ValueError(f"{source}: stacked video dimensions disagree with labels")
        count: int = min(frame_limit or len(pts), len(pts))
        times_ns: Int64[ndarray, "n"] = np.array([round(p * stream.time_base * 1_000_000_000) for p in pts[:count]], dtype=np.int64)
    transforms: Float64[ndarray, "n 4 4 4"] = labels.camera_to_world_transforms
    tracked: Bool[ndarray, "n"] = np.any(transforms != 0, axis=(1, 2, 3))
    if not np.any(tracked):
        raise ValueError(f"{source}: no tracked frame for rig extrinsics")
    first: Float64[ndarray, "4 4 4"] = transforms[np.flatnonzero(tracked)[0]]
    relative: Float64[ndarray, "4 4 4"] = np.linalg.inv(first[0]) @ first
    relative[:, :3, 3] *= 0.001
    world: Float64[ndarray, "n 4 4"] = transforms[:count, 0].copy()
    world[:, :3, 3] *= 0.001
    world[~tracked[:count]] = np.nan
    roll: Float64[ndarray, "4"] = np.deg2rad(labels.camera_angles)
    rotation: Float64[ndarray, "n 4 3 3"] = transforms[tracked, :, :3, :3]
    camera_up: Float64[ndarray, "4 3"] = (
        -np.cos(roll)[None, :, None] * rotation[:, :, :, 1] + np.sin(roll)[None, :, None] * rotation[:, :, :, 0]
    ).mean(axis=0)
    up: Float64[ndarray, "3"] = camera_up.mean(axis=0)
    up /= np.linalg.norm(up)
    camera_up /= np.linalg.norm(camera_up, axis=1, keepdims=True)
    spread_deg: float = float(np.rad2deg(np.arccos(np.clip(camera_up @ up, -1.0, 1.0))).max())
    source_num_frames: int = len(labels.joint_angles)
    labels = replace(
        labels,
        joint_angles=labels.joint_angles[:count],
        wrist_transforms=labels.wrist_transforms[:count],
        hand_confidences=labels.hand_confidences[:count],
        camera_to_world_transforms=labels.camera_to_world_transforms[:count],
    )
    return SequenceData(
        source, labels, source_num_frames, profile, fps, times_ns, np.arange(count, dtype=np.int64), tracked[:count], world, relative, up, spread_deg
    )
