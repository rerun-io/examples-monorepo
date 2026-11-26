"""TUM-VI's raw ASL sequence layout, read through pyserde and ``dataforge.euroc``: the door TUM-VI comes in by.

Every ``dataset-<name>_512_16`` directory of the TUM Visual-Inertial benchmark holds::

    mav0/cam0/data.csv, mav0/cam0/data/<ns>.png    512x512 16-bit grayscale, 20 Hz
    mav0/cam1/...                                  the right camera, hardware-synchronized with cam0
    mav0/imu0/data.csv                             200 Hz gyro (rad/s) + accel (m/s^2)
    mav0/mocap0/data.csv                           world_T_imu0 from the motion-capture room, ~120 Hz
    dso/camchain.yaml                              Kalibr: equidistant (KB4) intrinsics and T_cam_imu per camera
    dso/imu_config.yaml                            Kalibr IMU noise model (the inflated values, as the file states)

The csvs are EuRoC-style, so ``dataforge.euroc`` decodes them. The two yaml files are records read whole, so each is a
pyserde record; unknown keys are allowed (their schema is not ours to police). Kalibr states ``T_cam_imu``; the rig
frame is the IMU frame, so a camera's rig pose is its inverse (basalt's ``T_imu_cam``).

mocap0 is not continuous: the motion-capture system covers one room, so the corridor, magistrale and slides runs have
ground truth at their start and their end only. ``mocap_spans`` splits mocap0 at every gap over 1 s: the exits from
the room, and also the short occlusions inside it (room1 has 2 spans, room2 6, room3 5), so a span count is not a
count of room visits.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from jaxtyping import Bool, Float64, Int64
from numpy import ndarray
from scipy.spatial.transform import Rotation
from serde import SerdeError, coerce, field, serde
from serde.yaml import from_yaml
from yaml import YAMLError

from dataforge.basalt import BasaltPose, CalibratedCamera, Kb4Intrinsics
from dataforge.euroc import CameraRow

NUM_CAMERAS: int = 2
"""TUM-VI is one stereo pair, ``cam0`` (left) and ``cam1``."""
MOCAP_GAP_NS: int = 1_000_000_000
"""A pause in mocap0 longer than this starts a new span (samples run at ~120 Hz). It splits the exits from the room and
also the in-room occlusions, which stay under 3.5 s: a span is a stretch of mocap without a gap over 1 s, not a
visit to the room. The registered ``gt`` layers carry spans by this rule."""


def row_major_4x4(rows: list[list[float]]) -> Float64[ndarray, "4 4"]:
    """A Kalibr 4x4 (a list of four rows) as an array; anything but 4x4 is refused."""
    matrix: Float64[ndarray, "r c"] = np.array(rows, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"expected a 4x4 transform, got shape {matrix.shape}")
    return matrix


@serde(type_check=coerce)
@dataclass(frozen=True, slots=True)
class KalibrCamera:
    """One ``cam<N>`` block of camchain.yaml.

    ``dataforge.robocap_kalibr.KalibrCamWithExtrinsic`` reads the same block, but takes any model
    string and plain lists. This class is the strict door: it refuses another projection or distortion model and a
    wrong-length vector when the file is read, not later in ``kb4_camera``.
    """

    cam_T_imu: Float64[ndarray, "4 4"] = field(rename="T_cam_imu", deserializer=row_major_4x4)
    """Kalibr's ``T_cam_imu``: maps a point in the IMU frame into this camera's frame."""
    camera_model: Literal["pinhole"]
    """Projection; TUM-VI ships ``pinhole`` with an equidistant distortion, which is Kannala-Brandt (KB4)."""
    distortion_model: Literal["equidistant"]
    """Distortion family; ``equidistant`` is KB4's four radial terms."""
    distortion_coeffs: tuple[float, float, float, float]
    """k1 k2 k3 k4."""
    intrinsics: tuple[float, float, float, float]
    """fx fy cx cy, pixels."""
    resolution: tuple[int, int]
    """Width, height in pixels."""


@serde(type_check=coerce)
@dataclass(frozen=True, slots=True)
class Camchain:
    """A whole ``dso/camchain.yaml``: the stereo pair."""

    cam0: KalibrCamera
    """The left camera."""
    cam1: KalibrCamera
    """The right camera."""


@serde(type_check=coerce)
@dataclass(frozen=True, slots=True)
class ImuNoise:
    """A whole ``dso/imu_config.yaml``: Kalibr's continuous-time IMU noise model and the sample rate.

    ``dataforge.datasets.robocap._KalibrImu`` reads the same five keys, but it is private to that converter and makes
    every key optional (RoboCap's vendor files may omit some), so it would accept an imu_config without a noise term.
    Here every key is required; sharing one strict class would change RoboCap's door, which this converter does not own.
    """

    accelerometer_noise_density: float
    """m/s^2/sqrt(Hz)."""
    accelerometer_random_walk: float
    """m/s^3/sqrt(Hz)."""
    gyroscope_noise_density: float
    """rad/s/sqrt(Hz)."""
    gyroscope_random_walk: float
    """rad/s^2/sqrt(Hz)."""
    update_rate: float
    """Nominal sample rate, Hz."""


def kb4_camera(index: int, camera: KalibrCamera) -> CalibratedCamera:
    """One Kalibr camera as dataforge's validated record: KB4 intrinsics, its pose in the rig (IMU) frame."""
    imu_T_cam: Float64[ndarray, "4 4"] = np.linalg.inv(camera.cam_T_imu)
    quaternion_xyzw: Float64[ndarray, "4"] = Rotation.from_matrix(imu_T_cam[:3, :3]).as_quat()
    fx, fy, cx, cy = camera.intrinsics
    k1, k2, k3, k4 = camera.distortion_coeffs
    return CalibratedCamera(
        index=index,
        rig_pose=BasaltPose(
            px=float(imu_T_cam[0, 3]),
            py=float(imu_T_cam[1, 3]),
            pz=float(imu_T_cam[2, 3]),
            qx=float(quaternion_xyzw[0]),
            qy=float(quaternion_xyzw[1]),
            qz=float(quaternion_xyzw[2]),
            qw=float(quaternion_xyzw[3]),
        ),
        resolution=camera.resolution,
        model=Kb4Intrinsics(fx=fx, fy=fy, cx=cx, cy=cy, k1=k1, k2=k2, k3=k3, k4=k4),
    )


def read_camchain(path: Path) -> tuple[CalibratedCamera, ...]:
    """``dso/camchain.yaml`` as the two cameras, in ``cam0``, ``cam1`` order.

    Raises:
        ValueError: The file is not a two-camera equidistant Kalibr camchain.
    """
    try:
        chain: Camchain = from_yaml(Camchain, path.read_text())
    except (SerdeError, YAMLError) as error:
        raise ValueError(f"{path}: not a TUM-VI camchain: {error}") from error
    return (kb4_camera(0, chain.cam0), kb4_camera(1, chain.cam1))


def read_imu_noise(path: Path) -> ImuNoise:
    """``dso/imu_config.yaml`` as the IMU noise model.

    Raises:
        ValueError: A noise term or the rate is missing or not a number.
    """
    try:
        return from_yaml(ImuNoise, path.read_text())
    except (SerdeError, YAMLError) as error:
        raise ValueError(f"{path}: not a TUM-VI imu_config: {error}") from error


@dataclass(frozen=True, slots=True)
class StereoFrames:
    """The stereo pairs of one sequence: the frames both cameras took at the same stamp, and what was left over."""

    times_ns: Int64[ndarray, "n_pairs"]
    """Pair stamps on the dataset's own clock, ascending."""
    filenames: tuple[tuple[str, ...], tuple[str, ...]]
    """Each camera's PNG names under ``mav0/cam<N>/data/``, one per pair, in ``times_ns`` order."""
    unpaired: tuple[int, int]
    """Frames of cam0, cam1 whose stamp the other camera does not have; they are dropped."""


def pair_stereo(cam0: list[CameraRow], cam1: list[CameraRow]) -> StereoFrames:
    """Join the two cameras' indexes on the stamp.

    TUM-VI's cameras are hardware-synchronized and write identical stamps, so a frame is paired by exact stamp
    equality; a frame only one camera has (magistrale4's cam0 holds one) is dropped and counted, never matched to a
    neighbour.

    Raises:
        ValueError: A camera index is not strictly increasing, or no stamp is shared.
    """
    by_camera: list[dict[int, str]] = []
    for index, rows in enumerate((cam0, cam1)):
        stamps: Int64[ndarray, "n_rows"] = np.array([row.timestamp_ns for row in rows], dtype=np.int64)
        if stamps.size > 1 and not bool(np.all(np.diff(stamps) > 0)):
            raise ValueError(f"cam{index}/data.csv stamps are not strictly increasing")
        by_camera.append({row.timestamp_ns: row.filename for row in rows})
    shared: list[int] = sorted(by_camera[0].keys() & by_camera[1].keys())
    if not shared:
        raise ValueError("cam0 and cam1 share no stamp, so the sequence has no stereo pair")
    return StereoFrames(
        times_ns=np.array(shared, dtype=np.int64),
        filenames=(tuple(by_camera[0][stamp] for stamp in shared), tuple(by_camera[1][stamp] for stamp in shared)),
        unpaired=(len(by_camera[0]) - len(shared), len(by_camera[1]) - len(shared)),
    )


@dataclass(frozen=True, slots=True)
class MocapSpan:
    """One stretch of mocap0 without a pause longer than ``MOCAP_GAP_NS``."""

    start_ns: int
    """First sample's stamp."""
    stop_ns: int
    """Last sample's stamp."""
    num_poses: int
    """Samples in the span."""


def mocap_spans(times_ns: Int64[ndarray, "n_poses"]) -> tuple[MocapSpan, ...]:
    """Split mocap0's stamps at every pause over ``MOCAP_GAP_NS`` (exits from the room and in-room occlusions alike)."""
    if times_ns.size == 0:
        return ()
    breaks: Bool[ndarray, "n_gaps"] = np.diff(times_ns) > MOCAP_GAP_NS
    starts: Int64[ndarray, "n_spans"] = np.concatenate([[0], np.flatnonzero(breaks) + 1]).astype(np.int64)
    stops: Int64[ndarray, "n_spans"] = np.concatenate([starts[1:], [times_ns.size]]).astype(np.int64)
    return tuple(
        MocapSpan(start_ns=int(times_ns[start]), stop_ns=int(times_ns[stop - 1]), num_poses=int(stop - start))
        for start, stop in zip(starts, stops, strict=True)
    )
