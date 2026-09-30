"""The hand pose θ (6-DoF wrist + UmeTrack joint angles), its extrapolation, and its skinned landmarks.

The UmeTrack model is a left hand; a right hand is skinned by mirroring the wrist frame's x axis
(simplecv's ``wrist_for_hand``). Landmarks come back in ``LANDMARK`` order, in metres.
"""

from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

import torch
from dataforge.umetrack_hands import HandProfile
from jaxtyping import Float32
from serde import Untagged
from serde.json import from_json
from simplecv.umetrack_temp.generic_hand_model_numpy import HandModelNumpy
from simplecv.umetrack_temp.generic_hand_model_torch import HandModelTorch, hand_model_numpy_to_tensor, skin_landmarks, skin_mesh
from torch import Tensor


class Side(IntEnum):
    """Hand slot: 0 is the left hand, 1 the right, as in DetNet's outputs and UmeTrack's hand index."""

    LEFT = 0
    RIGHT = 1


@dataclass(frozen=True, slots=True)
class HandPose:
    """θ for a batch of hands: world-from-wrist rotation and translation, and the 22 stored joint angles.

    Skinning reads joint angles 0-19 only, so the pose has 6 + 20 = 26 degrees of freedom.
    """

    rotation: Float32[Tensor, "*batch 3 3"]
    translation: Float32[Tensor, "*batch 3"]
    """Metres."""
    joint_angles: Float32[Tensor, "*batch 22"]
    """Radians."""

    def world_from_wrist_mm(self, side: Side) -> Float32[Tensor, "*batch 4 4"]:
        """The skinning transform: millimetres, with the right hand's x axis mirrored."""
        transform: Float32[Tensor, "*batch 4 4"] = torch.zeros((*self.translation.shape[:-1], 4, 4), dtype=self.translation.dtype, device=self.translation.device)
        transform[..., :3, :3] = self.rotation
        transform[..., :3, 3] = self.translation * 1000.0
        transform[..., 3, 3] = 1.0
        if side == Side.RIGHT:
            transform[..., :, 0] *= -1
        return transform


def extrapolate(previous: HandPose, before_previous: HandPose, gain: float = 1.0, max_step_m: float | None = None) -> HandPose:
    """θ̂ = θ(t−1) + gain·(θ(t−1) − θ(t−2)); gain 1 is the constant-velocity guess 2θ(t−1) − θ(t−2), with the rotation extrapolated
    on SO(3): R̂ = D·R(t−1), D = R(t−1)·R(t−2)ᵀ. Another gain scales D towards the identity (I + gain·(D − I), projected back onto
    SO(3); exact to first order in the per-frame rotation). ``max_step_m`` clamps the length of the wrist's translation step."""
    delta: Float32[Tensor, "*batch 3 3"] = previous.rotation @ before_previous.rotation.transpose(-1, -2)
    if gain != 1.0:
        eye: Float32[Tensor, "3 3"] = torch.eye(3, dtype=delta.dtype, device=delta.device)
        u, _, vt = torch.linalg.svd(eye + gain * (delta - eye))
        flip: Float32[Tensor, "*batch 3"] = torch.ones_like(delta[..., 0])
        flip[..., -1] = torch.linalg.det(u @ vt)
        delta = (u * flip[..., None, :]) @ vt
    step: Float32[Tensor, "*batch 3"] = gain * (previous.translation - before_previous.translation)
    if max_step_m is not None:
        step = step * torch.clamp(max_step_m / step.norm(dim=-1, keepdim=True).clamp_min(1e-9), max=1.0)
    return HandPose(
        rotation=delta @ previous.rotation,
        translation=previous.translation + step,
        joint_angles=previous.joint_angles + gain * (previous.joint_angles - before_previous.joint_angles),
    )


def landmarks(model: HandModelTorch, pose: HandPose, side: Side) -> Float32[Tensor, "*batch 21 3"]:
    """The 21 skinned landmarks in world metres."""
    return skin_landmarks(model, pose.joint_angles, pose.world_from_wrist_mm(side)) / 1000.0


def mesh_vertices(model: HandModelTorch, pose: HandPose, side: Side) -> Float32[Tensor, "*batch v 3"]:
    """The skinned mesh in world metres."""
    return skin_mesh(model, pose.joint_angles, pose.world_from_wrist_mm(side)) / 1000.0


def hand_model_numpy_from_profile(text: str) -> HandModelNumpy:
    """The subject's hand model from the ``/world/gt/hands/profile`` JSON document.

    SHOW3D wraps the model in a ``hand_model`` envelope; UmeTrack stores the bare model.
    """
    document: HandProfile | HandModelNumpy = from_json(Untagged(HandProfile | HandModelNumpy), text)
    return document.hand_model if isinstance(document, HandProfile) else document


def hand_model_from_profile(text: str) -> HandModelTorch:
    """``hand_model_numpy_from_profile`` as torch tensors."""
    return hand_model_numpy_to_tensor(hand_model_numpy_from_profile(text))


GENERIC_HAND_MODEL: Path = Path(__file__).resolve().parents[1] / "assets" / "generic_hand_model.json"
"""UmeTrack's generic hand model (``dataset/generic_hand_model.json`` upstream): the unknown-hand model before scale calibration."""


def generic_hand_model() -> HandModelTorch:
    """The generic hand model as torch tensors."""
    return hand_model_numpy_to_tensor(from_json(HandModelNumpy, GENERIC_HAND_MODEL.read_text()))
