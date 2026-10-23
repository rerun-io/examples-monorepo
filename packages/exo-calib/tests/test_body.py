"""The SMPL-X fit (integration: needs ``data/body_models/smplx/SMPLX_NEUTRAL.npz``, which the first fit downloads after ``hf auth login``)."""

import numpy as np
import pytest
from conftest import project_points, ring_rig
from jaxtyping import Float32, Float64
from numpy import ndarray
from simplecv.data.skeleton.coco_133 import FACE_IDX

pytest.importorskip("smplx", reason="the exo-calib env carries smplx")
import torch  # noqa: E402
from simplecv.ops.smplx.smplx_coco133 import smplx_joints_to_coco133_torch  # noqa: E402
from simplecv.ops.smplx.smplx_torch import SmplxLayerTorch  # noqa: E402

from exo_calib.ops.body import BODY_MODEL_DIR, SmplxFitter, fit_body  # noqa: E402

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def model():
    if not (BODY_MODEL_DIR / "smplx" / "SMPLX_NEUTRAL.npz").exists():
        pytest.skip("SMPLX_NEUTRAL.npz is missing: run the App once after hf auth login (simplecv downloads it)")
    return SmplxFitter("cpu").model


def test_a_body_seen_by_six_cameras_comes_back_within_a_centimetre_and_its_height_within_two_percent(model) -> None:
    rng = np.random.default_rng(5)
    betas: Float32[ndarray, " 16"] = np.zeros(16, np.float32)
    betas[:3] = (1.2, -0.6, 0.4)  # taller than the average adult
    pose: Float32[ndarray, "3 165"] = np.zeros((3, 165), np.float32)
    pose[:, 0] = np.pi / 2  # SMPL-X's up (+y) to the rig's up (+z)
    pose[:, 3:66] = rng.normal(0.0, 0.15, (3, 63))
    transl: Float32[ndarray, "3 3"] = rng.normal(0.0, 0.2, (3, 3)).astype(np.float32)
    truth: Float64[ndarray, "3 133 3"] = coco_joints(model, betas, pose, transl)
    intrinsics: Float64[ndarray, "6 3 3"] = np.repeat(np.array([[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]])[None], 6, axis=0)
    cam_T_world: Float64[ndarray, "6 4 4"] = ring_rig(6, 3.5)
    kp_xy: Float64[ndarray, "6 3 133 2"] = np.stack([project_points(frame, intrinsics, cam_T_world) for frame in truth], axis=1)
    conf: Float64[ndarray, "6 3 133"] = np.full((6, 3, 133), 0.9)
    conf[..., FACE_IDX] = 0.0  # not fitted

    fit = fit_body(model, kp_xy, conf, intrinsics, cam_T_world)

    body = np.setdiff1d(np.arange(133), FACE_IDX)
    assert np.linalg.norm(coco_joints(model, fit.betas, fit.pose, fit.transl) - truth, axis=-1)[:, body].mean() < 0.01
    assert height(model, fit.betas) == pytest.approx(height(model, betas), rel=0.02)


def coco_joints(model: SmplxLayerTorch, betas: Float32[ndarray, " b"], pose: Float32[ndarray, "t 165"], transl: Float32[ndarray, "t 3"]) -> Float64[ndarray, "t 133 3"]:
    """The COCO-133 joints of SMPL-X bodies of one shape."""
    with torch.no_grad():
        _, joints = model(torch.as_tensor(pose), torch.as_tensor(transl), betas=torch.as_tensor(betas)[None])
    return smplx_joints_to_coco133_torch(joints).numpy().astype(np.float64)


def height(model, betas: Float32[ndarray, " 16"]) -> float:
    """Standing height (metres) of the rest mesh of a shape."""
    with torch.no_grad():
        vertices, _ = model(torch.zeros((1, 165)), torch.zeros((1, 3)), betas=torch.as_tensor(betas)[None])
    return float(vertices[0, :, 1].max() - vertices[0, :, 1].min())
