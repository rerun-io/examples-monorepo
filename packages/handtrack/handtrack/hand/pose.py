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
            transform[..., :, 0] = -transform[..., :, 0]
        return transform


def extrapolate(previous: HandPose, before_previous: HandPose) -> HandPose:
    """Constant-velocity guess θ̂ = 2θ(t−1) − θ(t−2), with the rotation extrapolated on SO(3): R̂ = R(t−1)·R(t−2)ᵀ·R(t−1)."""
    rotation: Float32[Tensor, "*batch 3 3"] = previous.rotation @ before_previous.rotation.transpose(-1, -2) @ previous.rotation
    return HandPose(
        rotation=rotation,
        translation=2 * previous.translation - before_previous.translation,
        joint_angles=2 * previous.joint_angles - before_previous.joint_angles,
    )


def landmarks(model: HandModelTorch, pose: HandPose, side: Side) -> Float32[Tensor, "*batch 21 3"]:
    """The 21 skinned landmarks in world metres."""
    return skin_landmarks(model, pose.joint_angles, pose.world_from_wrist_mm(side)) / 1000.0


def mesh_vertices(model: HandModelTorch, pose: HandPose, side: Side) -> Float32[Tensor, "*batch v 3"]:
    """The skinned mesh in world metres."""
    return skin_mesh(model, pose.joint_angles, pose.world_from_wrist_mm(side)) / 1000.0


def hand_model_from_profile(text: str) -> HandModelTorch:
    """The subject's hand model from the ``/world/gt/hands/profile`` JSON document.

    SHOW3D wraps the model in a ``hand_model`` envelope; UmeTrack stores the bare model.
    """
    model: HandModelNumpy = from_json(HandProfile, text).hand_model if '"hand_model"' in text[:64] else from_json(HandModelNumpy, text)
    return hand_model_numpy_to_tensor(model)


GENERIC_HAND_MODEL: Path = Path(__file__).resolve().parents[1] / "assets" / "generic_hand_model.json"
"""UmeTrack's generic hand model (``dataset/generic_hand_model.json`` upstream): the unknown-hand model before scale calibration."""


def generic_hand_model() -> HandModelTorch:
    return hand_model_numpy_to_tensor(from_json(HandModelNumpy, GENERIC_HAND_MODEL.read_text()))
