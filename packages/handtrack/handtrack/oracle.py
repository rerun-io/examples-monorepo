"""Ground-truth stand-ins for the two networks, to separate tracker and fit bugs from network quality.

- ``OracleDetector``: the ground-truth hand circle in the net frame (the smallest enclosing circle of the in-front
  keypoints, as DetNet is trained on), presence 1 where the hand is present in that image (>= 17 keypoints inside),
  0 otherwise.
- ``OracleKeypoints``: the projected ground-truth keypoints plus Gaussian noise (net px), the ground-truth d_rel plus
  Gaussian noise (mm), and as presence the fraction of the hand's in-front keypoints inside the crop (0 for an
  unlabelled hand), so a crop with 11 or more of the 21 keypoints keeps the track.
- ``KeyNetOnTruthBoxes``: the real KeyNet, but each requested view is cut from the ground-truth crop box (zero keypoint
  input) instead of the tracker's box: KeyNet's errors without the tracker's crop drift.
"""

from dataclasses import dataclass

import torch
from jaxtyping import Bool, Float32, Int64, UInt8
from torch import Tensor

from handtrack.data.segment_labels import SegmentLabels
from handtrack.labels.circles import square_boxes
from handtrack.labels.crops import apply_affine, count_inside_crop, crop_boxes, crop_from_net
from handtrack.labels.keypoint_input import relative_distances
from handtrack.labels.validity import HandLabel
from handtrack.models.detnet import Detections
from handtrack.tracker import MIN_BOX_RADIUS, CropRequest, KeypointEstimate, KeypointEstimator


@dataclass(frozen=True, slots=True)
class GroundTruthViews:
    """A segment's ground truth per frame f, camera c and hand slot, on the CPU."""

    net_xy: Float32[Tensor, "f c 2 21 2"]
    """Keypoints in the net frame (unclipped); NaN where the hand has no pose."""
    points_cam: Float32[Tensor, "f c 2 21 3"]
    in_front: Bool[Tensor, "f c 2 21"]
    """In front of the camera, and the hand labelled."""
    circles: Float32[Tensor, "f c 2 3"]
    """Smallest enclosing circle of the in-front keypoints in the net frame."""
    present: Bool[Tensor, "f c 2"]
    """``HandLabel.PRESENT`` in that image: labelled with >= 17 keypoints inside."""

    @staticmethod
    def from_labels(labels: SegmentLabels) -> "GroundTruthViews":
        return GroundTruthViews(
            net_xy=labels.projection.net_xy,
            points_cam=labels.projection.points_cam,
            in_front=labels.projection.in_front & labels.labelled[..., None],
            circles=labels.circles,
            present=labels.hand_label == int(HandLabel.PRESENT),
        )


@dataclass(frozen=True, slots=True)
class OracleDetector:
    """DetNet replaced by the ground truth."""

    truth: GroundTruthViews

    def __call__(self, images: UInt8[Tensor, "b 480 640"], frame: int, cameras: Int64[Tensor, "b"]) -> Detections:
        present: Bool[Tensor, "b 2"] = self.truth.present[frame, cameras]
        circle: Float32[Tensor, "b 2 3"] = torch.nan_to_num(self.truth.circles[frame, cameras], nan=0.0)
        probability: Float32[Tensor, "b 2"] = present.float()
        return Detections(circle=circle, probability=probability, present=present, box=square_boxes(circle))


class OracleKeypoints:
    """KeyNet replaced by the projected ground truth plus noise."""

    def __init__(self, truth: GroundTruthViews, phi: float, noise_px: float, noise_d_mm: float, seed: int = 0) -> None:
        self.truth: GroundTruthViews = truth
        self.phi: float = phi
        """ϕ that the ground-truth d_rel is divided by (the tracker's model)."""
        self.noise_px: float = noise_px
        self.noise_d_mm: float = noise_d_mm
        self.generator: torch.Generator = torch.Generator().manual_seed(seed)

    def __call__(self, images: UInt8[Tensor, "c 480 640"], frame: int, request: CropRequest) -> KeypointEstimate:
        n: int = request.camera.shape[0]
        points: Float32[Tensor, "n 21 2"] = torch.nan_to_num(self.truth.net_xy[frame, request.camera, request.side], nan=0.0)
        points_cam: Float32[Tensor, "n 21 3"] = torch.nan_to_num(self.truth.points_cam[frame, request.camera, request.side], nan=1.0)
        front: Bool[Tensor, "n 21"] = self.truth.in_front[frame, request.camera, request.side]
        inside: Int64[Tensor, "n"] = count_inside_crop(apply_affine(request.crop_from_net, points), front)
        d_rel: Float32[Tensor, "n 21"] = relative_distances(points_cam, torch.full((n,), self.phi))
        return KeypointEstimate(
            points_net=points + self.noise_px * torch.randn((n, 21, 2), generator=self.generator),
            d_rel_mm=d_rel + self.noise_d_mm * torch.randn((n, 21), generator=self.generator),
            presence=inside.float() / 21.0,
            confidence=torch.ones((n, 21)),
        )


@dataclass(frozen=True, slots=True)
class KeyNetOnTruthBoxes:
    """A keypoint estimator that sends every requested view to KeyNet on its ground-truth crop box; falls back to the tracker's crop without one."""

    truth: GroundTruthViews
    keynet: KeypointEstimator

    def __call__(self, images: UInt8[Tensor, "c 480 640"], frame: int, request: CropRequest) -> KeypointEstimate:
        circle: Float32[Tensor, "n 3"] = self.truth.circles[frame, request.camera, request.side]
        usable: Bool[Tensor, "n"] = torch.isfinite(circle).all(dim=-1)
        safe: Float32[Tensor, "n 3"] = torch.nan_to_num(circle, nan=100.0)
        floored: Float32[Tensor, "n 3"] = torch.cat([safe[:, :2], safe[:, 2:].clamp_min(MIN_BOX_RADIUS)], dim=-1)
        truth_map: Float32[Tensor, "n 3 3"] = crop_from_net(crop_boxes(floored), request.side == 1)
        crop_map: Float32[Tensor, "n 3 3"] = torch.where(usable[:, None, None], truth_map, request.crop_from_net)
        return self.keynet(images, frame, CropRequest(request.camera, request.side, crop_map, torch.zeros_like(request.keypoint_input)))
