"""The synthetic rig the boundary and logging suites both drive, as fixtures.

One 200x200 kb4 camera per index with every distortion coefficient zero, the
Index device's frozen noise model, and one blocky-noise scene that can be shifted
by whole pixels: enough to detect on, track through and refuse the wrong thing,
and small enough that the whole Python suite stays inside a few seconds.

Factories rather than values, because a test picks the camera count, the baseline
and the shift. Session-scoped, because a factory holds no per-test state and a
``@given`` test may not request a function-scoped fixture — each call still hands
back a fresh calibration or frontend, so a test may mutate what it is given.
Fixtures rather than a module the tests import from each other: pytest injects
these, so no test module has to be on another one's import path.
"""

from collections.abc import Callable
from typing import TypeAlias

import numpy as np
import pytest
from jaxtyping import Float64, UInt8
from numpy import ndarray

from slam_rs import _core
from slam_rs.catalog_feed import CameraCalib, ImuCalib

FRAME: int = 200
"""Synthetic frame size: four whole 50-pixel detection cells per side."""

CameraFactory: TypeAlias = Callable[[int, float], CameraCalib]
"""One camera of the synthetic rig, by rig index and baseline in metres."""
RigFactory: TypeAlias = Callable[[int], _core.Calibration]
"""A calibration for a rig of the given camera count."""
FrontendFactory: TypeAlias = Callable[[int], _core.OpticalFlow]
"""A frontend on a rig of the given camera count."""
PipelineFactory: TypeAlias = Callable[[int], _core.Vio]
"""The whole pipeline on a rig of the given camera count."""
TextureFactory: TypeAlias = Callable[[int, int], UInt8[ndarray, "h w"]]
"""The synthetic scene, shifted by whole pixels in x and y."""


@pytest.fixture(scope="session")
def camera() -> CameraFactory:
    """A 200x200 pinhole-like camera: kb4 with every coefficient zero."""

    def build(index: int, baseline_m: float) -> CameraCalib:
        imu_T_cam: Float64[ndarray, "4 4"] = np.eye(4, dtype=np.float64)
        imu_T_cam[0, 3] = baseline_m
        return CameraCalib(
            index=index,
            width=FRAME,
            height=FRAME,
            frequency_hz=30.0,
            fx=100.0,
            fy=100.0,
            cx=FRAME / 2,
            cy=FRAME / 2,
            model="kb4",
            distortion=np.zeros(4, dtype=np.float64),
            distortion_valid_radius=None,
            imu_T_cam=imu_T_cam,
            image_rotation_cw_deg=0,
        )

    return build


@pytest.fixture(scope="session")
def imu() -> ImuCalib:
    """The Index device's frozen noise model, which no test here is sensitive to."""
    return ImuCalib(
        frequency_hz=1000.0,
        gyro_noise_std=0.000282,
        accel_noise_std=0.016,
        gyro_bias_std=0.0001,
        accel_bias_std=0.001,
        cam_time_offset_ns=0,
        imu_T_body=np.eye(4, dtype=np.float64),
    )


@pytest.fixture(scope="session")
def rig(camera: CameraFactory, imu: ImuCalib) -> RigFactory:
    """A calibration for ``camera_count`` identical cameras, 10 cm apart."""

    def build(camera_count: int) -> _core.Calibration:
        return _core.Calibration.from_catalog([camera(index, 0.1 * index) for index in range(camera_count)], imu)

    return build


@pytest.fixture(scope="session")
def frontend(rig: RigFactory) -> FrontendFactory:
    """A frontend on a rig of ``camera_count`` identical cameras."""

    def build(camera_count: int) -> _core.OpticalFlow:
        return _core.OpticalFlow(rig(camera_count), _core.VioConfig())

    return build


@pytest.fixture(scope="session")
def pipeline(rig: RigFactory) -> PipelineFactory:
    """The whole pipeline on a rig of ``camera_count`` identical cameras.

    Two cameras at least: the epipolar filter is a hard precondition
    (``optical_flow.h:210``), so a one-camera rig is a refusal to assert on, not
    a fixture to build from.
    """

    def build(camera_count: int) -> _core.Vio:
        return _core.Vio(rig(camera_count), _core.VioConfig())

    return build


@pytest.fixture(scope="session")
def texture() -> TextureFactory:
    """One fixed blocky-noise scene, shifted by whole pixels.

    Blocky **noise**, not a lattice: a repeating pattern gives every corner the
    same score, and basalt's suppression is strictly-greater-than, so a tie kills
    both sides and a perfectly regular scene detects almost nothing.
    """

    def build(shift_x: int, shift_y: int) -> UInt8[ndarray, "h w"]:
        blocks: UInt8[ndarray, "b b"] = np.random.default_rng(20250907).integers(0, 256, (FRAME // 2, FRAME // 2), dtype=np.uint8)
        image: UInt8[ndarray, "h w"] = np.repeat(np.repeat(blocks, 2, axis=0), 2, axis=1)
        return np.ascontiguousarray(np.roll(image, (shift_y, shift_x), axis=(0, 1)))

    return build
