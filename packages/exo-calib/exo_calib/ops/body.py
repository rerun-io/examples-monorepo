"""The person's SMPL-X body, fitted to the multi-view COCO-133 keypoints through a calibrated rig (cameras fixed).

One shape (``betas``) for the segment and one pose and root translation per frameset: first to the triangulated joints in 3D
(a basin), then to the keypoints in every view (robust reprojection), with priors that keep the shape near the average adult
and the pose near rest. The model and the COCO-133 readout are simplecv's (``SmplxLayerTorch``, ``smplx_joints_to_coco133_torch``).
"""

from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
import torch
from jaxtyping import Bool, Float32, Float64, Int64
from numpy import ndarray
from simplecv.ops.smplx.smplx_coco133 import smplx_joints_to_coco133_torch
from simplecv.ops.smplx.smplx_torch import SMPLX_BODY, SMPLX_LHAND, SMPLX_RHAND, SmplxLayerTorch
from simplecv.ops.triangulate import triangulate_weighted
from simplecv.ops.umeyama import umeyama_alignment
from torch import Tensor

from exo_calib.ops.confidence import drop_face_confidences

BODY_MODEL_DIR: Path = Path(__file__).resolve().parents[2] / "data" / "body_models"
"""Holds ``smplx/SMPLX_NEUTRAL.npz`` (simplecv's ``SmplxLayerTorch`` downloads it on first use: ``hf auth login``)."""
NUM_BETAS: int = 16
TORSO: tuple[int, ...] = (5, 6, 11, 12)
"""Shoulders and hips: the joints that place the body before it is posed."""
HANDS: slice = slice(SMPLX_LHAND.start, SMPLX_RHAND.stop)
FIT_3D_STEPS: int = 150
FIT_2D_STEPS: int = 300
LEARNING_RATE: float = 0.03
ROBUST_PX: float = 20.0
"""Geman-McClure scale of the reprojection residual: a keypoint farther off than this counts less and less."""
ROBUST_M: float = 0.1
"""Geman-McClure scale of the 3D residual to the triangulated joints."""
SHAPE_PRIOR: float = 0.1
POSE_PRIOR: float = 0.01


class BodyFit(NamedTuple):
    betas: Float32[ndarray, " b"]
    pose: Float32[ndarray, "t 165"]
    transl: Float32[ndarray, "t 3"]
    vertices: Float32[ndarray, "t n 3"]


class SmplxFitter:
    """The body node's resource: the neutral SMPL-X model (16 shape coefficients, MANO mean hands at zero pose), fitted with
    ``fit_body``."""

    def __init__(self, device: str) -> None:
        self.model: SmplxLayerTorch = SmplxLayerTorch(betas=np.zeros(NUM_BETAS, np.float32), model_root_dir=BODY_MODEL_DIR).to(device)

    @property
    def faces(self) -> Int64[ndarray, "f 3"]:
        return self.model.faces

    def __call__(
        self, kp_xy: Float64[ndarray, "v t 133 2"], conf: Float64[ndarray, "v t 133"], intrinsics: Float64[ndarray, "v 3 3"], cam_T_world: Float64[ndarray, "v 4 4"]
    ) -> "BodyFit":
        return fit_body(self.model, kp_xy, conf, intrinsics, cam_T_world)


def _geman_mcclure(residual: Tensor, scale: float) -> Tensor:
    squared = (residual**2).sum(dim=-1)
    return squared * scale**2 / (squared + scale**2)


def fit_body(
    model: SmplxLayerTorch,
    kp_xy: Float64[ndarray, "v t 133 2"],
    conf: Float64[ndarray, "v t 133"],
    intrinsics: Float64[ndarray, "v 3 3"],
    cam_T_world: Float64[ndarray, "v 4 4"],
) -> BodyFit:
    """Fit one SMPL-X body to every frameset's keypoints (pinhole pixels of a calibrated rig; ``conf`` 0 = unseen, face ignored).

    Returns:
        The shape, and each frameset's pose, translation and mesh.
    """
    n_views, n_frames = kp_xy.shape[:2]
    conf = drop_face_confidences(conf)
    projection: Float64[ndarray, "v 3 4"] = intrinsics @ cam_T_world[:, :3]
    triangulated, valid = triangulate_weighted(
        torch.from_numpy(kp_xy.reshape(n_views, -1, 2)), torch.from_numpy(conf.reshape(n_views, -1) ** 2), torch.from_numpy(projection)
    )
    points: Float64[ndarray, "t 133 3"] = triangulated.numpy().reshape(n_frames, 133, 3)
    seen: Bool[ndarray, "t 133"] = valid.numpy().reshape(n_frames, 133)

    # Place the rest body (the fit's start: zero shape, zero pose) on each frameset's torso, as smplx poses it: about its root
    # joint, then translated. A frameset without three torso joints keeps the last one's orientation.
    with torch.no_grad():
        _, rest_joints = model(torch.zeros((1, 165), device=model.device), torch.zeros((1, 3), device=model.device), betas=torch.zeros((1, NUM_BETAS), device=model.device))
    rest: Float64[ndarray, "133 3"] = smplx_joints_to_coco133_torch(rest_joints)[0].cpu().numpy().astype(np.float64)
    root: Float64[ndarray, "3"] = rest_joints[0, 0].cpu().numpy().astype(np.float64)
    is_torso: Bool[ndarray, " 133"] = np.isin(np.arange(133), TORSO)
    pose0: Float32[ndarray, "t 165"] = np.zeros((n_frames, 165), np.float32)
    transl0: Float32[ndarray, "t 3"] = np.zeros((n_frames, 3), np.float32)
    rotation: Float64[ndarray, "3 3"] = np.eye(3)
    for frame, anchor in enumerate(seen):
        if (torso := anchor & is_torso).sum() >= 3:
            rotation = umeyama_alignment(rest[torso], points[frame, torso]).dst_R_src
        offset = points[frame, anchor].mean(axis=0) - rotation @ rest[anchor].mean(axis=0) if anchor.any() else np.zeros(3)
        pose0[frame, :3] = cv2.Rodrigues(rotation)[0][:, 0]
        transl0[frame] = offset + rotation @ root - root

    def tensor(array: ndarray, grad: bool) -> Tensor:
        return torch.tensor(array, dtype=torch.float32, device=model.device, requires_grad=grad)

    betas, pose, transl = tensor(np.zeros((1, NUM_BETAS)), True), tensor(pose0, True), tensor(transl0, True)
    target, target_seen = tensor(np.nan_to_num(points), False), tensor(seen.astype(np.float32), False)
    xy, weight, P = tensor(np.nan_to_num(kp_xy), False), tensor(conf, False), tensor(projection, False)

    def to_triangulated(joints: Tensor) -> Tensor:
        return (target_seen * _geman_mcclure(joints - target, ROBUST_M)).sum() / target_seen.sum().clamp(min=1.0) / ROBUST_M**2

    def to_keypoints(joints: Tensor) -> Tensor:
        uvw = torch.einsum("vij,tnj->vtni", P, torch.cat((joints, torch.ones_like(joints[..., :1])), dim=-1))
        return (weight * _geman_mcclure(uvw[..., :2] / uvw[..., 2:].clamp(min=1e-6) - xy, ROBUST_PX)).sum() / weight.sum().clamp(min=1e-9)

    # First a basin from the triangulated joints with the hands held (3D hand joints are the noisiest), then every view.
    for steps, data_term, hands_free in ((FIT_3D_STEPS, to_triangulated, False), (FIT_2D_STEPS, to_keypoints, True)):
        optimizer = torch.optim.Adam([betas, pose, transl], lr=LEARNING_RATE)
        for _ in range(steps):
            optimizer.zero_grad()
            _, smplx_joints = model(pose, transl, betas=betas)
            prior = SHAPE_PRIOR * (betas**2).sum() + POSE_PRIOR * ((pose[:, SMPLX_BODY] ** 2).mean() + (pose[:, HANDS] ** 2).mean())
            (data_term(smplx_joints_to_coco133_torch(smplx_joints)) + prior).backward()
            if not hands_free:
                assert pose.grad is not None
                pose.grad[:, HANDS] = 0.0
            optimizer.step()

    with torch.no_grad():
        vertices, _ = model(pose, transl, betas=betas)
    return BodyFit(betas[0].detach().cpu().numpy(), pose.detach().cpu().numpy(), transl.detach().cpu().numpy(), vertices.cpu().numpy())
