"""The pose fit's inputs: one hand in one frame, seen in one or two camera views.

The tracker builds these from KeyNet's outputs after undoing the crop, the mirror and the letterbox, so the
keypoints are in each camera's own image pixels. ``observe`` builds the exact observation of a known pose,
which is what a perfect KeyNet would return; tests and ground-truth experiments use it.
"""

from dataclasses import dataclass

import torch
from jaxtyping import Bool, Float32
from simplecv.umetrack_temp.generic_hand_model_torch import LANDMARK, HandModelTorch
from torch import Tensor

from handtrack.geometry.camera import CameraRig, in_front, inside_image, project, world_to_cameras
from handtrack.hand.pose import HandPose, Side, landmarks
from handtrack.labels.keypoint_input import relative_distances

MAX_VIEWS: int = 2
"""KeyNet runs on at most two views per hand (§5.1)."""
DIST_REFERENCE: int = int(LANDMARK.WRIST_JOINT)
"""p_0 of E_dist, the keypoint the relative distances are measured from; the paper does not name it."""


@dataclass(frozen=True, slots=True)
class ViewObservation:
    """One camera's evidence for one hand."""

    camera: CameraRig
    """The rig of this one camera (``len(camera.names) == 1``)."""
    world_from_rig: Float32[Tensor, "4 4"]
    keypoints_px: Float32[Tensor, "21 2"]
    """The 21 keypoints in ``LANDMARK`` order, in the camera's own image pixels (pixel-centre convention)."""
    weights: Float32[Tensor, "21"]
    """Per-keypoint weight on E_2D and E_dist; 0 means not observed."""
    d_rel_mm: Float32[Tensor, "21"]
    """Relative distance d_rel,i = (d_i − d̄) / ϕ to this camera's centre, in millimetres of the generic hand."""

    def __post_init__(self) -> None:
        if len(self.camera.names) != 1:
            raise ValueError(f"a view holds one camera, got {len(self.camera.names)}: {self.camera.names}")
        for name, finite in (
            ("keypoints_px", torch.isfinite(self.keypoints_px).all(dim=-1)),
            ("d_rel_mm", torch.isfinite(self.d_rel_mm)),
        ):
            invalid: Bool[Tensor, "21"] = (self.weights > 0) & ~finite
            if bool(invalid.any()):
                keypoint: int = int(torch.nonzero(invalid)[0, 0])
                raise ValueError(f"{name} is non-finite at observed keypoint {keypoint}")


@dataclass(frozen=True, slots=True)
class HandObservation:
    """The fit's input for one hand in one frame: one or two views."""

    side: Side
    views: tuple[ViewObservation, ...]

    def __post_init__(self) -> None:
        if not 1 <= len(self.views) <= MAX_VIEWS:
            raise ValueError(f"a hand is observed in 1 to {MAX_VIEWS} views, got {len(self.views)}")
        if len({view.camera.fisheye62 is None for view in self.views}) != 1:
            raise ValueError("the views of one hand mix pinhole and Fisheye62 cameras")


def observe(
    model: HandModelTorch, pose: HandPose, side: Side, phi: float, views: tuple[tuple[CameraRig, Float32[Tensor, "4 4"]], ...]
) -> HandObservation:
    """The exact observation of an unbatched ``pose``: projected keypoints, weight 1 where a keypoint is in front and inside the image.

    Args:
        model: The hand model that ``pose`` is skinned with.
        pose: θ, unbatched.
        side: Which hand ``pose`` is.
        phi: ϕ of ``model``, the scale that d_rel is divided by.
        views: ``(camera, world_from_rig)`` per view; each camera is a one-camera rig.
    """
    points_world: Float32[Tensor, "21 3"] = landmarks(model, pose, side)
    observations: list[ViewObservation] = []
    for camera, world_from_rig in views:
        points_cam: Float32[Tensor, "21 3"] = world_to_cameras(camera, world_from_rig, points_world)[0]
        pixels: Float32[Tensor, "21 2"] = project(camera, points_cam[None])[0]
        seen: Bool[Tensor, "21"] = in_front(points_cam) & inside_image(camera, pixels[None])[0]
        observations.append(
            ViewObservation(
                camera=camera,
                world_from_rig=world_from_rig,
                keypoints_px=pixels,
                weights=seen.to(torch.float32),
                d_rel_mm=relative_distances(points_cam, torch.tensor(phi)),
            )
        )
    return HandObservation(side=side, views=tuple(observations))
