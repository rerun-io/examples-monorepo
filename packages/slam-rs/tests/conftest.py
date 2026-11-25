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
Runtime settings and benchmark definitions are separate session-scoped fixtures.
Tests request benchmark definitions only when they evaluate regression cases.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import os
import socket
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import numpy as np
import pytest
import rerun.chunk as rrc
from jaxtyping import Float64, UInt8
from numpy import ndarray

# Package imports (fixture_types imports slam_rs) stay inside the fixtures: pytest-beartype instruments slam_rs only
# if this file has not imported it.
if TYPE_CHECKING:
    from fixture_types import CameraFactory, FrontendFactory, Mp4Factory, PipelineFactory, RigFactory, Rows, RowsReader, TextureFactory

    from slam_rs import _core
    from slam_rs.catalog_feed import CameraCalib, ImuCalib
    from slam_rs.config import SlamConfig
    from slam_rs.reference import Benchmarks

# beartype runs under pytest only; code that cannot run under it reads this (test_frameset_matching.py checks it).
os.environ.setdefault("PIXI_DEV_MODE", "1")


def pytest_configure() -> None:
    """Stop the session with the fix when the Rust core is not built.

    ``_core.so`` is a gitignored output of the ``slam-rs-build`` task (pixi install does not run cargo), so a fresh
    worktree has none and every test would fail at import. The check looks for the file next to the package without
    importing slam_rs, so pytest-beartype still instruments every slam_rs module.
    """
    spec = importlib.util.find_spec("slam_rs")
    locations: list[str] = list(spec.submodule_search_locations or []) if spec is not None else []
    if not any((Path(location) / f"_core{suffix}").is_file() for location in locations for suffix in importlib.machinery.EXTENSION_SUFFIXES):
        pytest.exit("slam_rs._core is not built: run `pixi run -e slam-rs-dev slam-rs-build` (pixi install does not build it)", returncode=4)


@pytest.fixture(autouse=True)
def unit_tests_stay_off_the_gpu(request: pytest.FixtureRequest) -> Iterator[None]:
    """Fail an unmarked (unit) test that selected a non-CPU backend.

    The core is built with the GPU frontend and every tool defaults to ``auto``,
    so a unit test that forgets ``backend="cpu"`` would probe the device. The
    process's probe counter must not move during a unit test.
    """
    from slam_rs import _core

    if request.node.get_closest_marker("integration") or request.node.get_closest_marker("golden"):
        yield
        return
    before: int = _core.device_probes_run()
    yield
    assert _core.device_probes_run() == before, (
        f"{request.node.nodeid} probed the GPU: unit tests must pass backend='cpu' (or stub the selector); mark it integration if it needs the device"
    )


@pytest.fixture(scope="session")
def settings() -> SlamConfig:
    """Runtime settings, parsed once for the whole session."""
    from slam_rs.config import load_slam_config

    return load_slam_config()


@pytest.fixture
def live_catalog(settings: SlamConfig) -> None:
    """Skip when the catalog server the integration/golden tests read from is not reachable.

    The recordings are an asset like any other: a missing server is a skip that names it, not a failure.
    """
    url = urlparse(settings.catalog_url.replace("rerun+http://", "http://", 1))
    try:
        with socket.create_connection((url.hostname or "127.0.0.1", url.port or 80), timeout=2.0):
            pass
    except OSError as error:
        pytest.skip(f"catalog server {settings.catalog_url} is not reachable ({error}); start it with `pixi run -e slam-rs slam-rs-catalog-up`")


@pytest.fixture(scope="session")
def benchmarks(settings: SlamConfig) -> Benchmarks:
    """Frozen regression cases, loaded only by evaluation tests."""
    from slam_rs.reference import load_benchmarks

    return load_benchmarks(settings)


@pytest.fixture(scope="session")
def camera() -> CameraFactory:
    """A 200x200 pinhole-like camera: kb4 with every coefficient zero."""
    from fixture_types import FRAME

    from slam_rs.catalog_feed import CameraCalib

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
            readout_s=None,
            imu_T_cam=imu_T_cam,
        )

    return build


@pytest.fixture(scope="session")
def imu() -> ImuCalib:
    """The Index device's frozen noise model, which no test here is sensitive to."""
    from slam_rs.catalog_feed import ImuCalib

    return ImuCalib(
        frequency_hz=1000.0,
        gyro_noise_std=0.000282,
        accel_noise_std=0.016,
        gyro_bias_std=0.0001,
        accel_bias_std=0.001,
        cam_time_offset_ns=0,
        gravity_ms2=None,
        imu_T_body=np.eye(4, dtype=np.float64),
    )


@pytest.fixture(scope="session")
def rig(camera: CameraFactory, imu: ImuCalib) -> RigFactory:
    """A calibration for ``camera_count`` identical cameras, 10 cm apart."""
    from slam_rs import _core

    def build(camera_count: int) -> _core.Calibration:
        return _core.Calibration.from_catalog([camera(index, 0.1 * index) for index in range(camera_count)], imu)

    return build


@pytest.fixture(scope="session")
def frontend(rig: RigFactory) -> FrontendFactory:
    """A frontend on a rig of ``camera_count`` identical cameras."""
    from slam_rs import _core

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
    from slam_rs import _core

    def build(camera_count: int) -> _core.Vio:
        return _core.Vio(rig(camera_count), _core.VioConfig(), backend="cpu")

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
    from fixture_types import Row

    from slam_rs.catalog_feed import TIMELINE

    def read(path: Path) -> Rows:
        rows: Rows = {}
        for chunk in rrc.RrdReader(path).stream().collect().stream():
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
    from fixture_types import FRAME

    def build(shift_x: int, shift_y: int) -> UInt8[ndarray, "h w"]:
        blocks: UInt8[ndarray, "b b"] = np.random.default_rng(20250907).integers(0, 256, (FRAME // 2, FRAME // 2), dtype=np.uint8)
        image: UInt8[ndarray, "h w"] = np.repeat(np.repeat(blocks, 2, axis=0), 2, axis=1)
        return np.ascontiguousarray(np.roll(image, (shift_y, shift_x), axis=(0, 1)))

    return build


@pytest.fixture(scope="session")
def synthetic_mp4() -> Mp4Factory:
    """One RGB picture, encoded ``frames`` times into an in-memory yuv420p H.264 MP4 (even sides only)."""
    from io import BytesIO

    import av
    from av.container import OutputContainer
    from av.video.stream import VideoStream

    def build(picture: UInt8[ndarray, "h w 3"], frames: int) -> bytes:
        buffer: BytesIO = BytesIO()
        container: OutputContainer
        with av.open(buffer, mode="w", format="mp4") as container:
            stream: VideoStream = container.add_stream("libx264", rate=30)
            stream.width, stream.height, stream.pix_fmt = picture.shape[1], picture.shape[0], "yuv420p"
            stream.options = {"crf": "0"}
            for _ in range(frames):
                for packet in stream.encode(av.VideoFrame.from_ndarray(picture, format="rgb24")):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        return buffer.getvalue()

    return build


@pytest.fixture(scope="session")
def robocap_imu() -> ImuCalib:
    """Cap A's factory noise model, used only as a test expectation."""
    from slam_rs.catalog_feed import ImuCalib

    return ImuCalib(frequency_hz=200.0, gyro_noise_std=0.0007300442812547, accel_noise_std=0.005955224218014,
                    gyro_bias_std=3.445397083168e-05, accel_bias_std=0.0001963150489218, cam_time_offset_ns=14_902_432,
                    gravity_ms2=None, imu_T_body=np.eye(4, dtype=np.float64))
