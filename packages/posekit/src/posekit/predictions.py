"""GPU-resident prediction containers shared by all posekit models.

Predictions stay as torch tensors on the inference device so multi-stage
pipelines (detector -> pose -> triangulation) never round-trip through numpy.
Use the ``*_numpy`` helpers only at visualization/serialization boundaries.
"""

from dataclasses import dataclass

import numpy as np
import torch
from jaxtyping import Bool, Float32, Int64, UInt8
from numpy import ndarray
from torch import Tensor

from posekit.skeletons import KeypointSkeleton

TORCH_UINT8 = torch.__dict__["uint8"]


def validate_frames_rgb(frames_rgb: UInt8[Tensor, "b h w 3"]) -> None:
    """Validate the canonical posekit image-batch layout.

    Every posekit model consumes uint8 RGB NHWC frame batches (the layout
    TorchCodec's CUDA decoder produces), so the check lives here once.

    Args:
        frames_rgb: Frame batch to validate.

    Raises:
        ValueError: If ``frames_rgb`` is not a uint8 NHWC RGB tensor.
    """
    if frames_rgb.dtype != TORCH_UINT8 or frames_rgb.ndim != 4 or int(frames_rgb.shape[-1]) != 3:
        raise ValueError(f"Expected uint8 RGB frames with shape (batch, height, width, 3), got {frames_rgb.dtype} {tuple(frames_rgb.shape)}.")


@dataclass(frozen=True, slots=True)
class BoxDetections:
    """Flattened per-batch box detections from a detector, tracker, or segmenter.

    The optional ``masks``/``track_ids`` fields make detector-only, tracker,
    and segmenter outputs one type: a plain detector leaves both ``None``, a
    video segmenter fills both, and mask-consuming pose models (MammaNet-style
    ``TopDownDenseLandmarks2d``) require ``masks``.
    """

    xyxy: Float32[Tensor, "n 4"]
    """Image-space boxes in ``xyxy`` order."""
    scores: Float32[Tensor, "n"]
    """Detection confidence per box."""
    frame_indices: Int64[Tensor, "n"]
    """Index into the source frame batch for each box."""
    masks: Bool[Tensor, "n h w"] | None = None
    """Optional full-frame binary instance masks aligned with ``xyxy`` rows."""
    track_ids: Int64[Tensor, "n"] | None = None
    """Optional stable instance identities across frames (tracker/segmenter output)."""

    @property
    def num_detections(self) -> int:
        """Number of detections across the whole frame batch."""
        return int(self.xyxy.shape[0])

    @classmethod
    def empty(cls, device: torch.device | str, *, mask_hw: tuple[int, int] | None = None, with_track_ids: bool = False) -> "BoxDetections":
        """Build a zero-detection result on the given device.

        Args:
            device: Device the empty tensors live on.
            mask_hw: When given, populate an empty ``masks`` field at this
                ``(height, width)`` (segmenter outputs keep the field present).
            with_track_ids: Whether to populate an empty ``track_ids`` field.

        Returns:
            Empty detections with the requested optional fields present.
        """
        return cls(
            xyxy=torch.empty((0, 4), dtype=torch.float32, device=device),
            scores=torch.empty((0,), dtype=torch.float32, device=device),
            frame_indices=torch.empty((0,), dtype=torch.long, device=device),
            masks=torch.empty((0, mask_hw[0], mask_hw[1]), dtype=torch.bool, device=device) if mask_hw is not None else None,
            track_ids=torch.empty((0,), dtype=torch.long, device=device) if with_track_ids else None,
        )


@dataclass(frozen=True, slots=True)
class Keypoints2d:
    """Flattened per-instance 2D keypoints in source-image pixel coordinates."""

    xy: Float32[Tensor, "n k 2"]
    """Image-space keypoint locations per instance."""
    scores: Float32[Tensor, "n k"]
    """Per-keypoint confidence."""
    frame_indices: Int64[Tensor, "n"]
    """Index into the source frame batch for each instance."""
    skeleton: KeypointSkeleton
    """Skeleton format describing what each keypoint index means."""
    uncertainty: Float32[Tensor, "n k"] | None = None
    """Optional per-keypoint log-variance in pixel units (MammaNet-style heads).

    Consumers (BA weighting, weighted triangulation) may use it as a soft
    weight; scalar ``scores`` stays the mandatory confidence signal.
    """

    @property
    def num_instances(self) -> int:
        """Number of pose instances across the whole frame batch."""
        return int(self.xy.shape[0])

    @classmethod
    def empty(cls, skeleton: KeypointSkeleton, device: torch.device | str) -> "Keypoints2d":
        """Build a zero-instance result for a skeleton on the given device.

        Args:
            skeleton: Skeleton format the (absent) keypoints would follow.
            device: Device the empty tensors live on.

        Returns:
            Empty keypoints carrying the skeleton.
        """
        num_keypoints: int = skeleton.num_keypoints
        return cls(
            xy=torch.empty((0, num_keypoints, 2), dtype=torch.float32, device=device),
            scores=torch.empty((0, num_keypoints), dtype=torch.float32, device=device),
            frame_indices=torch.empty((0,), dtype=torch.long, device=device),
            skeleton=skeleton,
        )

    def xy_numpy(self) -> Float32[ndarray, "n k 2"]:
        """Copy keypoints to CPU for logging/serialization."""
        return self.xy.detach().cpu().numpy().astype(np.float32, copy=False)

    def scores_numpy(self) -> Float32[ndarray, "n k"]:
        """Copy scores to CPU for logging/serialization."""
        return self.scores.detach().cpu().numpy().astype(np.float32, copy=False)


@dataclass(frozen=True, slots=True)
class DenseLandmarks2d:
    """Flattened per-instance dense surface landmarks (MammaNet-style heads).

    Dense-landmark nets predict anonymous surface points (no fixed sparse
    skeleton) with richer per-point heads than ``Keypoints2d``; visibility
    doubles as the confidence signal.
    """

    xy: Float32[Tensor, "n p 2"]
    """Image-space landmark locations per instance."""
    log_variance: Float32[Tensor, "n p"]
    """Per-landmark predicted log-variance (aleatoric uncertainty)."""
    visibility: Float32[Tensor, "n p"]
    """Per-landmark visibility probability in ``[0, 1]`` (post-sigmoid)."""
    contact: Float32[Tensor, "n p"]
    """Per-landmark self/person-contact probability in ``[0, 1]``."""
    floor_contact: Float32[Tensor, "n p"]
    """Per-landmark floor-contact probability in ``[0, 1]``."""
    frame_indices: Int64[Tensor, "n"]
    """Index into the source frame batch for each instance."""

    @property
    def num_instances(self) -> int:
        """Number of landmark instances across the whole frame batch."""
        return int(self.xy.shape[0])


@dataclass(frozen=True, slots=True)
class Keypoints3d:
    """Flattened per-instance sparse 3D keypoints (RTMW3D-class models).

    Single-view 3D pose nets predict image-space ``xy`` plus a root-relative
    depth per keypoint; camera-space coordinates are only available when the
    model (or a consumer) resolves the root depth, so they stay optional.
    """

    xy: Float32[Tensor, "n k 2"]
    """Image-space keypoint locations per instance."""
    z_root_relative: Float32[Tensor, "n k"]
    """Depth per keypoint relative to the skeleton root, in model units."""
    scores: Float32[Tensor, "n k"]
    """Per-keypoint confidence."""
    frame_indices: Int64[Tensor, "n"]
    """Index into the source frame batch for each instance."""
    skeleton: KeypointSkeleton
    """Skeleton format describing what each keypoint index means."""
    root_indices: tuple[int, ...] = ()
    """Keypoint indices whose mean defines the root (e.g. hips ``(11, 12)``)."""
    xyz_camera: Float32[Tensor, "n k 3"] | None = None
    """Optional metric camera-space keypoints (OpenCV/RDF convention)."""

    @property
    def num_instances(self) -> int:
        """Number of pose instances across the whole frame batch."""
        return int(self.xy.shape[0])
