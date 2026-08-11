"""Catalog camera geometry and causal source-view selection."""

import math
from collections import deque
from dataclasses import dataclass

import torch
from jaxtyping import Float, Int
from torch import Tensor

KEYFRAME_DISTANCE: float = 0.1
OPTIMAL_TRANSLATION: float = 0.15
OPTIMAL_ROTATION: float = 0.0
MAX_KEYFRAMES: int = 30
NUM_SOURCE_VIEWS: int = 7

RIG_QUATERNION: str = "/world/rig_00:Transform3D:quaternion"
RIG_TRANSLATION: str = "/world/rig_00:Transform3D:translation"
CAM_QUATERNION: str = "/world/rig_00/cam_00:Transform3D:quaternion"
CAM_TRANSLATION: str = "/world/rig_00/cam_00:Transform3D:translation"
IMAGE_FROM_CAMERA: str = "/world/rig_00/cam_00/pinhole:Pinhole:image_from_camera"
RESOLUTION: str = "/world/rig_00/cam_00/pinhole:Pinhole:resolution"


@dataclass(frozen=True, slots=True)
class Keyframe:
    """One retained posed and preprocessed video frame."""

    frame_index: int
    """Zero-based video frame index within the segment."""
    world_T_cam_44: Float[Tensor, "4 4"]
    """CPU camera-to-world transform with shape ``4 4``."""
    cam_T_world_44: Float[Tensor, "4 4"]
    """CPU world-to-camera transform with shape ``4 4``."""
    K_s1_44: Float[Tensor, "4 4"]
    """CPU homogeneous s1 matching-scale intrinsics."""
    image_3hw: Float[Tensor, "c=3 h=384 w=512"]
    """Preprocessed RGB image."""


class KeyframeBuffer:
    """Retain moved camera frames and rank them as depth-model sources."""

    def __init__(self) -> None:
        """Create an empty 30-frame keyframe buffer."""
        self._keyframes: deque[Keyframe] = deque(maxlen=MAX_KEYFRAMES)

    def __len__(self) -> int:
        """Return the number of retained keyframes."""
        return len(self._keyframes)

    def add(self, keyframe: Keyframe) -> bool:
        """Retain a frame when it moved more than 0.1 from the last keyframe.

        Args:
            keyframe: Candidate frame; the buffer takes ownership of its tensors.

        Returns:
            Whether the frame was retained.
        """
        if (
            self._keyframes
            and pose_distance(self._keyframes[-1].world_T_cam_44, keyframe.world_T_cam_44).combined
            <= KEYFRAME_DISTANCE
        ):
            return False
        self._keyframes.append(keyframe)
        return True

    def select_sources(self, world_T_cam_44: Float[Tensor, "4 4"]) -> list[Keyframe]:
        """Select up to seven keyframes nearest the desired stereo baseline.

        Selection is control flow: the scores must land on CPU to pick Python
        objects, so this math runs on CPU tensors by design (GPU would force a
        sync per frame for a 30-row computation).

        Args:
            world_T_cam_44: Current CPU camera-to-world transform with shape ``4 4``.

        Returns:
            Retained frames selected by the depth model's asymmetric baseline penalty.
        """
        if not self._keyframes:
            return []
        poses_n44: Float[Tensor, "n 4 4"] = torch.stack([keyframe.world_T_cam_44 for keyframe in self._keyframes])
        relative_n44: Float[Tensor, "n 4 4"] = torch.linalg.inv(world_T_cam_44)[None] @ poses_n44
        rotation_trace_n: Float[Tensor, "n"] = relative_n44[:, :3, :3].diagonal(dim1=1, dim2=2).sum(-1).clamp(max=3.0)
        rotation_n: Float[Tensor, "n"] = torch.sqrt(2.0 * (1.0 - rotation_trace_n / 3.0))
        translation_n: Float[Tensor, "n"] = torch.linalg.norm(relative_n44[:, :3, 3], dim=1)
        rotation_penalty_n: Float[Tensor, "n"] = (rotation_n - OPTIMAL_ROTATION).abs() ** 2.0
        translation_delta_n: Float[Tensor, "n"] = translation_n - OPTIMAL_TRANSLATION
        translation_penalty_n: Float[Tensor, "n"] = (
            torch.where(translation_delta_n < 0.0, 5.0, 1.0) * translation_delta_n.abs() ** 2.0
        )
        penalties_n: Float[Tensor, "n"] = rotation_penalty_n + translation_penalty_n
        num_selected: int = min(NUM_SOURCE_VIEWS, len(self._keyframes))
        selected_positions: Int[Tensor, "k"] = torch.topk(penalties_n, num_selected, largest=False).indices
        return [self._keyframes[int(position)] for position in selected_positions]


@dataclass(frozen=True, slots=True)
class PoseDistance:
    """Translation, rotation, and combined distance between two poses."""

    combined: float
    """Root sum of squared translation and rotation measures."""
    rotation: float
    """Trace-based unitless rotation measure."""
    translation: float
    """Translation distance in meters."""


def pose_distance(
    reference_world_T_cam_44: Float[Tensor, "4 4"], measurement_world_T_cam_44: Float[Tensor, "4 4"]
) -> PoseDistance:
    """Measure the relative translation and rotation of two camera poses.

    Args:
        reference_world_T_cam_44: Reference CPU camera-to-world transform with shape ``4 4``.
        measurement_world_T_cam_44: Measured CPU camera-to-world transform with shape ``4 4``.

    Returns:
        Translation, trace-based rotation, and root-sum-square combined distance.
    """
    relative_44: Float[Tensor, "4 4"] = torch.linalg.inv(reference_world_T_cam_44) @ measurement_world_T_cam_44
    rotation_trace: float = min(3.0, float(relative_44[:3, :3].diagonal().sum()))
    rotation: float = math.sqrt(2.0 * (1.0 - rotation_trace / 3.0))
    translation: float = float(torch.linalg.norm(relative_44[:3, 3]))
    combined: float = math.sqrt(translation**2 + rotation**2)
    return PoseDistance(combined=combined, rotation=rotation, translation=translation)
