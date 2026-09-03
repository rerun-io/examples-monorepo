"""Camera-adapter contract tests for LAMP tracking and lifting."""

import importlib.util
import sys
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import torch
from jaxtyping import Float32, Float64
from numpy import ndarray
from simplecv.camera_parameters import Extrinsics, Fisheye62Parameters, Intrinsics, KannalaBrandtDistortion, PinholeParameters

from lamptrack.cameras import RigCamera, gravity_aligned_world_transform
from lamptrack.third_party.lamp.models.model_utils import pinhole_unproject

REFERENCE_DIR = Path(__file__).parent / "reference_data" / "lamp"


def _load_reference_module(name: str, filename: str) -> ModuleType:
    """Load one pristine upstream source file under its original module name."""
    path = REFERENCE_DIR / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load upstream fixture {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _upstream_gravity_transform(gravity_world: Float64[ndarray, "3"]) -> Float32[ndarray, "4 4"]:
    """Run pristine ``MpsLoader._compute_T_gravityWorld_world`` on a provider stub."""
    for package_name in ("lamp", "lamp.core", "lamp.io"):
        package = ModuleType(package_name)
        package.__path__ = [str(REFERENCE_DIR)]
        sys.modules[package_name] = package
    _load_reference_module("lamp.core.se3", "upstream_core_se3.py")
    _load_reference_module("lamp.core.types", "upstream_core_types.py")
    projectaria_tools = ModuleType("projectaria_tools")
    core = ModuleType("projectaria_tools.core")
    data_provider = ModuleType("projectaria_tools.core.data_provider")
    calibration = ModuleType("projectaria_tools.core.calibration")
    mps = ModuleType("projectaria_tools.core.mps")
    sensor_data = ModuleType("projectaria_tools.core.sensor_data")
    stream_id = ModuleType("projectaria_tools.core.stream_id")

    class DeviceVersion(Enum):
        """Names imported by the pristine module but unused by this test seam."""

        GEN1 = "gen1"
        GEN2 = "gen2"

    class ImportedTypeStub:
        """Stand in for Aria types outside the tested gravity seam."""

        AFTER = "after"

    core.data_provider = data_provider
    calibration.DeviceVersion = DeviceVersion
    mps.MpsDataPaths = ImportedTypeStub
    mps.MpsDataProvider = ImportedTypeStub
    sensor_data.TimeDomain = ImportedTypeStub
    sensor_data.TimeQueryOptions = ImportedTypeStub
    stream_id.StreamId = ImportedTypeStub
    for module in (projectaria_tools, core, data_provider, calibration, mps, sensor_data, stream_id):
        sys.modules[module.__name__] = module
    sensor_io = _load_reference_module("lamp.io.sensor_io", "upstream_io_sensor_io.py")

    class ProviderStub:
        """Return one pose carrying the requested gravity vector."""

        def get_closed_loop_pose(self, _timestamp_ns: int, _query: object) -> SimpleNamespace:
            return SimpleNamespace(gravity_world=gravity_world)

    loader = object.__new__(sensor_io.MpsLoader)
    loader._provider = ProviderStub()
    return loader._compute_T_gravityWorld_world()


def _intrinsics() -> Intrinsics:
    """Robocap-like 1920x1080 calibration."""
    return Intrinsics.from_focal_principal_point(
        camera_conventions="RDF",
        fl_x=636.4,
        fl_y=634.7,
        cx=956.2,
        cy=525.4,
        width=1920,
        height=1080,
    )


def _identity_extrinsics() -> Extrinsics:
    """Identity rig-to-camera pose."""
    return Extrinsics(cam_R_world=np.eye(3, dtype=np.float64), cam_t_world=np.zeros(3, dtype=np.float64))


def _points_camera() -> Float64[ndarray, "n 3"]:
    """Well-conditioned camera-space points spread across the image."""
    return np.array(
        [[0.0, 0.0, 1.0], [0.2, -0.1, 1.0], [-0.35, 0.22, 1.3], [0.5, 0.3, 2.0]],
        dtype=np.float64,
    )


def test_pinhole_projection_round_trip() -> None:
    """Pinhole pixels unproject to rays that project back to the same pixels."""
    camera = RigCamera(PinholeParameters(name="pinhole", extrinsics=_identity_extrinsics(), intrinsics=_intrinsics()))
    pixels: Float32[ndarray, "n 2"] = camera.project(_points_camera())
    rays: Float32[ndarray, "n 3"] = camera.unproject(pixels)
    assert np.allclose(camera.project(rays), pixels, atol=1e-5)
    assert np.array_equal(camera.to_virtual_pinhole(pixels), pixels)


def test_kb4_projection_round_trip_on_robocap_calibration() -> None:
    """Robocap KB4 pixels survive project/unproject within 1e-3 pixels."""
    camera = RigCamera(
        Fisheye62Parameters(
            name="left_front",
            extrinsics=_identity_extrinsics(),
            intrinsics=_intrinsics(),
            distortion=KannalaBrandtDistortion(k1=0.0617, k2=-0.0211, k3=0.0372, k4=-0.0135),
        )
    )
    pixels: Float32[ndarray, "n 2"] = camera.project(_points_camera())
    rays: Float32[ndarray, "n 3"] = camera.unproject(pixels)
    assert np.allclose(camera.project(rays), pixels, atol=1e-3)


def test_virtual_pinhole_lifter_rays_equal_kb4_unprojection() -> None:
    """Undistorting keypoints and passing four intrinsics preserves KB4 rays."""
    camera = RigCamera(
        Fisheye62Parameters(
            name="left_front",
            extrinsics=_identity_extrinsics(),
            intrinsics=_intrinsics(),
            distortion=KannalaBrandtDistortion(k1=0.0617, k2=-0.0211, k3=0.0372, k4=-0.0135),
        )
    )
    distorted: Float32[ndarray, "n 2"] = camera.project(_points_camera())
    virtual: Float32[ndarray, "n 2"] = camera.to_virtual_pinhole(distorted)
    params: Float32[ndarray, "4"] = camera.lifter_params()
    lifter_rays: torch.Tensor = pinhole_unproject(
        torch.from_numpy(virtual)[None],
        torch.from_numpy(params)[None],
    )[0]
    lifter_rays = torch.nn.functional.normalize(lifter_rays, dim=-1)
    assert np.allclose(lifter_rays.numpy(), camera.unproject(distorted), atol=1e-6)


def test_gravity_alignment_matches_pristine_mps_loader() -> None:
    """The package helper preserves upstream axis choice and handedness."""
    gravity_vectors = (
        np.array([0.0, 0.0, -9.81], dtype=np.float64),
        np.array([1.2, -2.3, -9.4], dtype=np.float64),
        np.array([-9.81, 1e-4, 2e-4], dtype=np.float64),
    )
    for gravity_world in gravity_vectors:
        expected = _upstream_gravity_transform(gravity_world)
        actual = gravity_aligned_world_transform(gravity_world)
        assert np.array_equal(actual, expected)


def test_kb4_virtual_pinhole_accepts_an_empty_keypoint_set() -> None:
    """A camera with no valid keypoints preserves the empty batch shape."""
    camera = RigCamera(
        Fisheye62Parameters(
            name="left_front",
            extrinsics=_identity_extrinsics(),
            intrinsics=_intrinsics(),
            distortion=KannalaBrandtDistortion(k1=0.0617, k2=-0.0211, k3=0.0372, k4=-0.0135),
        )
    )

    virtual = camera.to_virtual_pinhole(np.empty((0, 2), dtype=np.float32))

    assert virtual.shape == (0, 2)
    assert virtual.dtype == np.float32
