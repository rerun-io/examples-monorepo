"""Camera-frame unit rays for pinhole and Kannala-Brandt rig cameras."""

import cv2
import numpy as np
import pytest
from jaxtyping import Float32, Float64
from numpy import ndarray
from simplecv.camera_parameters import Extrinsics, Fisheye62Parameters, Intrinsics, KannalaBrandtDistortion, PinholeParameters, rescale_intri

from monopriors.models.rig_depth.rays import camera_type, unit_rays


def _identity_extrinsics() -> Extrinsics:
    return Extrinsics(cam_R_world=np.eye(3, dtype=np.float64), cam_t_world=np.zeros(3, dtype=np.float64))


def _robocap_fisheye() -> Fisheye62Parameters:
    return Fisheye62Parameters(
        name="left_front",
        extrinsics=_identity_extrinsics(),
        intrinsics=Intrinsics.from_focal_principal_point(
            camera_conventions="RDF",
            fl_x=636.4,
            fl_y=634.7,
            cx=956.2,
            cy=525.4,
            width=1920,
            height=1080,
        ),
        distortion=KannalaBrandtDistortion(k1=0.0617, k2=-0.0211, k3=0.0372, k4=-0.0135),
    )


def test_pinhole_rays_match_inverse_intrinsics_at_pixel_centres() -> None:
    K_33: Float64[ndarray, "3 3"] = np.array([[420.0, 0.0, 4.2], [0.0, 415.0, 2.7], [0.0, 0.0, 1.0]], dtype=np.float64)
    camera: PinholeParameters = PinholeParameters(
        name="pinhole",
        extrinsics=_identity_extrinsics(),
        intrinsics=Intrinsics.from_k_matrix(camera_conventions="RDF", k_matrix=K_33, width=8, height=6),
    )
    rays_hw3: Float32[ndarray, "6 8 3"] = unit_rays(camera)

    u_hw: Float64[ndarray, "6 8"]
    v_hw: Float64[ndarray, "6 8"]
    u_hw, v_hw = np.meshgrid(np.arange(8, dtype=np.float64) + 0.5, np.arange(6, dtype=np.float64) + 0.5, indexing="xy")
    pixels_3n: Float64[ndarray, "3 n"] = np.stack((u_hw, v_hw, np.ones((6, 8), dtype=np.float64))).reshape(3, -1)
    expected_n3: Float64[ndarray, "n 3"] = (np.linalg.inv(K_33) @ pixels_3n).T
    expected_n3 /= np.linalg.norm(expected_n3, axis=1, keepdims=True)
    np.testing.assert_allclose(rays_hw3.reshape(-1, 3), expected_n3, rtol=1e-6, atol=1e-7)
    assert rays_hw3.dtype == np.float32
    assert camera_type(camera) == 1


@pytest.mark.parametrize(("width", "height"), [(1920, 1080), (896, 504)])
def test_robocap_kb4_rays_project_back_to_pixel_centres(width: int, height: int) -> None:
    original: Fisheye62Parameters = _robocap_fisheye()
    camera: Fisheye62Parameters = Fisheye62Parameters(
        name=original.name,
        extrinsics=original.extrinsics,
        intrinsics=rescale_intri(original.intrinsics, target_width=width, target_height=height),
        distortion=original.distortion,
    )
    rays_hw3: Float32[ndarray, "h w 3"] = unit_rays(camera)
    assert rays_hw3.shape == (height, width, 3) and rays_hw3.dtype == np.float32
    assert camera_type(camera) == 0

    row_step: int = max(1, height // 24)
    col_step: int = max(1, width // 32)
    sampled_n13: Float64[ndarray, "n 1 3"] = rays_hw3[::row_step, ::col_step].reshape(-1, 1, 3).astype(np.float64)
    expected_n12: Float64[ndarray, "n 1 2"] = np.stack(
        np.meshgrid(
            np.arange(0, width, col_step, dtype=np.float64) + 0.5,
            np.arange(0, height, row_step, dtype=np.float64) + 0.5,
            indexing="xy",
        ),
        axis=-1,
    ).reshape(-1, 1, 2)
    valid_n: np.ndarray = np.linalg.norm(sampled_n13[:, 0], axis=1) > 1e-3
    zero_3: Float64[ndarray, "3"] = np.zeros(3, dtype=np.float64)
    K_33: Float64[ndarray, "3 3"] = np.asarray(camera.intrinsics.k_matrix, dtype=np.float64)
    assert camera.distortion is not None
    distortion_4: Float64[ndarray, "4"] = np.array(
        [camera.distortion.k1, camera.distortion.k2, camera.distortion.k3, camera.distortion.k4], dtype=np.float64
    )
    projected_result: tuple[Float64[ndarray, "n 1 2"], Float64[ndarray, "2n 15"]] = cv2.fisheye.projectPoints(
        sampled_n13[valid_n], zero_3, zero_3, K_33, distortion_4
    )
    np.testing.assert_allclose(projected_result[0], expected_n12[valid_n], rtol=0.0, atol=1e-3)
