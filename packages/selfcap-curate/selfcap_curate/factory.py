"""Factory transforms and catalog camera-50 borrowing policy."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from dataforge.datasets.selfcap_raw import FactoryCalibration, FactoryExtrinsic, Intrinsic
from jaxtyping import Float64
from numpy import ndarray
from rerun.catalog import DatasetEntry
from serde import serde

from selfcap_curate.catalog_io import MissingSegmentError, Segment, open_segment, read_factories
from selfcap_curate.device_tables import read_camera50_table
from selfcap_curate.records import EpisodeKey


def transform(rotation: Float64[ndarray, "3 3"], translation: Float64[ndarray, "3"]) -> Float64[ndarray, "4 4"]:
    """Construct a rigid homogeneous transform from typed factory/solver arrays."""
    matrix: Float64[ndarray, "4 4"] = np.eye(4)
    matrix[:3, :3], matrix[:3, 3] = rotation, translation
    return matrix


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class QuestBlock:
    pinholes: dict[str, Intrinsic]  # Slot-indexed pinholes; matrix applies the December cy−160 crop correction.
    rig_T_cam: dict[str, Float64[ndarray, "4 4"]]  # Left-rig to right-camera factory transform, or borrowed camera-50 transform.
    mode: Literal["healthy", "faulty-left-borrowed-cam50", "faulty-unresolved", "unknown-layout"]  # Explicit headset layout/fallback status.


def optical_targets(quest: QuestBlock) -> dict[str, Float64[ndarray, "4 4"]]:
    """Map physical left/right RGB targets to resolved transforms, not video slot numbers."""
    if quest.mode == "healthy":
        return {"quest0": np.eye(4), "quest1": quest.rig_T_cam["1"]}
    if quest.mode == "faulty-left-borrowed-cam50":
        return {"quest0": quest.rig_T_cam["1"]}
    return {}


def quest_block(
    calibration: FactoryCalibration, *, camera50: FactoryExtrinsic | None = None, rig_T_device: Float64[ndarray, "4 4"] | None = None
) -> QuestBlock:
    """Preserve camera-50/right-slot fallback with explicit same-headset borrowing.

    The caller supplies the measured rig/device lever arm; the base carries only
    the left-eye track, so a two-eye midpoint must not be invented from it.
    """
    pinholes = {
        str(slot): c
        for slot, name in enumerate(("left", "right"))
        for c in calibration.intrinsics
        if c.positional_layout == name and c.lens_intrinsics.available
    }
    ext = {e.type: e for e in calibration.extrinsics}
    if "device_to_left" in ext and "device_to_right" in ext and len(pinholes) == 2:
        left, right = ext["device_to_left"], ext["device_to_right"]
        return QuestBlock(pinholes, {"1": np.linalg.inv(transform(left.R.T, left.T)) @ transform(right.R.T, right.T)}, "healthy")
    if "1" in pinholes and pinholes["1"].camera_id == "50":
        if camera50 is None or rig_T_device is None or camera50.to != "50" or camera50.type != "device_to_left":
            return QuestBlock(pinholes, {}, "faulty-unresolved")
        return QuestBlock(pinholes, {"1": rig_T_device @ transform(camera50.R.T, camera50.T)}, "faulty-left-borrowed-cam50")
    return QuestBlock(pinholes, {}, "unknown-layout")


def oak_stereo(calibration: FactoryCalibration) -> dict[str, Float64[ndarray, "4 4"]]:
    """Return RGB-relative stereo transforms, converting OAK centimetres to metres."""
    ext = {e.type: e for e in calibration.extrinsics}
    if not {"left_to_right", "left_to_rgb"} <= ext.keys():
        return {}
    right, rgb = ext["left_to_right"], ext["left_to_rgb"]
    rgb_T_left = np.linalg.inv(transform(rgb.R, rgb.T * 0.01))
    return {"rgb_T_left": rgb_T_left, "rgb_T_right": rgb_T_left @ transform(right.R, right.T * 0.01)}


def read_quest_block(segment: Segment, calibration: FactoryCalibration | None = None, *, donors: DatasetEntry | None = None) -> QuestBlock:
    """Resolve camera-50 borrowing automatically from strict segment-key provenance; donors default to the segment's dataset."""
    calibration = calibration if calibration is not None else read_factories(segment)["quest"]
    block = quest_block(calibration)
    if block.mode != "faulty-unresolved":
        return block
    session = EpisodeKey.parse(segment.id).session
    borrowed = read_camera50_table().sessions.get(session)
    if borrowed is None:
        return block
    try:
        donor: Segment = open_segment(donors if donors is not None else segment.dataset, borrowed.healthy_segment)
    except MissingSegmentError as error:
        raise ValueError(f"session {session} borrows camera 50 from {borrowed.healthy_segment}; register that segment") from error
    healthy = read_factories(donor)
    candidates = [value for value in healthy.values() if value.device_id == calibration.device_id == borrowed.device_id]
    if len(candidates) != 1 or quest_block(candidates[0]).mode != "healthy":
        raise ValueError(f"{segment.id}: borrowed camera-50 source must be a healthy segment of {calibration.device_id}")
    extrinsic = next((e for e in candidates[0].extrinsics if e.type == "device_to_left" and e.to == "50"), None)
    if extrinsic is None:
        raise ValueError(f"{borrowed.healthy_segment}: missing device_to_left camera 50")
    return quest_block(calibration, camera50=extrinsic, rig_T_device=transform(np.eye(3), borrowed.rig_T_device_m))


def cy_shift_px(intrinsic: Intrinsic) -> float:
    """Correct a same-width centred vertical crop in curation, leaving ingest raw."""
    sensor, capture = intrinsic.sensor_info.pixel_array_size, intrinsic.capture_resolution
    return (sensor.height - capture.height) / 2.0 if sensor.width == capture.width and sensor.height > capture.height else 0.0


def intrinsic_matrix(intrinsic: Intrinsic) -> Float64[ndarray, "3 3"]:
    """Build the factory pinhole with curation's centred-crop correction."""
    lens = intrinsic.lens_intrinsics
    assert lens.principal_point_y is not None
    return np.array(
        [
            [lens.focal_length_x, 0.0, lens.principal_point_x],
            [0.0, lens.focal_length_y, float(lens.principal_point_y) - cy_shift_px(intrinsic)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def oak_centre(calibration: FactoryCalibration) -> Intrinsic:
    """Select the OAK RGB factory slot."""
    return next(c for c in calibration.intrinsics if c.positional_layout == "center")
