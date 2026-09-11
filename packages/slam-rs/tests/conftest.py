"""What the boundary and logging suites share: the synthetic rig, and the recording reader.

One 200x200 kb4 camera per index with every distortion coefficient zero, the
Index device's frozen noise model, and one blocky-noise scene that can be shifted
by whole pixels: enough to detect on, track through and refuse the wrong thing,
and small enough that the whole Python suite stays inside a few seconds.

Factories rather than values, because a test picks the camera count, the baseline
and the shift. Session-scoped, because a factory holds no per-test state and a
``@given`` test may not request a function-scoped fixture — each call still hands
back a fresh calibration or frontend, so a test may mutate what it is given.
Fixtures rather than a module the tests import from each other: pytest injects
these, so no test module has to be on another one's import path. That is also
where :func:`read_rows` belongs — the two logging suites check their rungs
against a real recording, and one reader means one account of what a row is,
and :func:`manifest` — five suites read the frozen reference set and parsing it
once a session is both cheaper and one account of what "the manifest" means.
"""

from pathlib import Path

import numpy as np
import pytest
import rerun.experimental as rx
from fixture_types import FRAME, CameraFactory, FrontendFactory, PipelineFactory, RigFactory, Row, Rows, RowsReader, TextureFactory
from jaxtyping import Float64, UInt8
from numpy import ndarray

from slam_rs import _core
from slam_rs.catalog_feed import TIMELINE, CameraCalib, ImuCalib
from slam_rs.reference import ReferenceManifest, load_manifest


@pytest.fixture(scope="session")
def manifest() -> ReferenceManifest:
    """The frozen reference set, parsed once for the whole session."""
    return load_manifest()


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
            fx=100.0,
            fy=100.0,
            cx=FRAME / 2,
            cy=FRAME / 2,
            model="kb4",
            distortion=np.zeros(4, dtype=np.float64),
            distortion_valid_radius=None,
            imu_T_cam=imu_T_cam,
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
def read_rows() -> RowsReader:
    """Read every non-static row of a recording, grouped by entity path.

    A component a row did not set comes back as a null and is dropped, so a
    missing key means the row really did not carry that component.

    Returns:
        A function of an ``.rrd`` written by :func:`rerun.save`, giving each
        entity path's rows in ``video_time`` order.
    """

    def read(path: Path) -> Rows:
        rows: Rows = {}
        for chunk in rx.RrdReader(path).stream().collect().stream():
            if chunk.is_static:
                continue
            batch = chunk.to_record_batch()
            if TIMELINE not in batch.schema.names:
                # A row written before the caller set a cursor sits on no timeline.
                continue
            times: list = batch.column(TIMELINE).to_pylist()
            components: dict[str, list] = {
                name: batch.column(name).to_pylist() for name in batch.schema.names if ":" in name and not name.startswith("rerun.")
            }
            for index, time in enumerate(times):
                values: dict[str, list] = {name: column[index] for name, column in components.items() if column[index] is not None}
                rows.setdefault(chunk.entity_path, []).append(Row(t_ns=int(np.timedelta64(time, "ns").astype(np.int64)), values=values))
        for entity in rows:
            rows[entity].sort(key=lambda row: row.t_ns)
        return rows

    return read


@pytest.fixture(scope="session")
def texture() -> TextureFactory:
    """One fixed blocky-noise scene, shifted by whole pixels.

    Blocky **noise**, not a lattice: a repeating pattern gives every corner the
    same score, and the detector's suppression is strictly-greater-than, so a tie kills
    both sides and a perfectly regular scene detects almost nothing.
    """

    def build(shift_x: int, shift_y: int) -> UInt8[ndarray, "h w"]:
        blocks: UInt8[ndarray, "b b"] = np.random.default_rng(20250907).integers(0, 256, (FRAME // 2, FRAME // 2), dtype=np.uint8)
        image: UInt8[ndarray, "h w"] = np.repeat(np.repeat(blocks, 2, axis=0), 2, axis=1)
        return np.ascontiguousarray(np.roll(image, (shift_y, shift_x), axis=(0, 1)))

    return build
