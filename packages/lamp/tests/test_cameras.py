"""Camera-adapter contract tests for LAMP tracking and lifting."""

import numpy as np
import torch
from jaxtyping import Float32, Float64
from numpy import ndarray
from simplecv.camera_parameters import Extrinsics, Fisheye62Parameters, Intrinsics, KannalaBrandtDistortion, PinholeParameters

from lamptrack.cameras import RigCamera
from lamptrack.third_party.lamp.models.model_utils import pinhole_unproject


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
