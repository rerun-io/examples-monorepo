"""Task-level model abstractions (the monoprior pattern, GPU-batched).

The model roles are the swap points of the design (docs/design.md §3):

- :class:`PersonDetector`: full frames -> boxes.
- :class:`TopDownPose2d`: frames + boxes -> keypoints in image space.
- :class:`InstancePose2d`: full frames -> boxes AND keypoints in one pass
  (one-stage/query-based nets that cannot consume external boxes).
- :class:`PromptableSegmenter`: frames + prompts -> instance masks.
- :class:`VideoSegmenter`: stateful per-frame mask tracking with stable ids.
- :class:`IdentityEncoder`: frames + boxes -> appearance embeddings (re-ID).
- :class:`TopDownDenseLandmarks2d`: frames + boxes WITH masks -> dense
  surface landmarks with uncertainty/visibility/contact heads.

All consume the canonical posekit image batch (uint8 RGB NHWC CUDA tensors)
and return GPU-resident predictions, so stages compose without host copies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from jaxtyping import Float32, Int64, UInt8
from torch import Tensor

from posekit.predictions import BoxDetections, DenseLandmarks2d, Keypoints2d
from posekit.skeletons import KeypointSkeleton


class PersonDetector(ABC):
    """Detects person/hand instances in full frames."""

    @abstractmethod
    def __call__(self, frames_rgb: UInt8[Tensor, "b h w 3"]) -> BoxDetections:
        """Detect instances across a frame batch.

        Args:
            frames_rgb: uint8 RGB NHWC frame batch on the inference device.

        Returns:
            Flattened detections across the batch.
        """
        raise NotImplementedError


class TopDownPose2d(ABC):
    """Estimates 2D keypoints for cropped instances (top-down paradigm)."""

    skeleton: KeypointSkeleton
    """Skeleton format of the predictions."""

    @abstractmethod
    def __call__(self, frames_rgb: UInt8[Tensor, "b h w 3"], detections: BoxDetections) -> Keypoints2d:
        """Estimate keypoints for every detection.

        Args:
            frames_rgb: uint8 RGB NHWC frame batch on the inference device.
            detections: Instance boxes referencing ``frames_rgb`` by index.

        Returns:
            Image-space keypoints, one instance per detection row.
        """
        raise NotImplementedError


class InstancePose2d(ABC):
    """Predicts boxes and keypoints jointly in one full-frame pass.

    One-stage (RTMO) and query-based (RF-DETR) nets condition keypoints on
    internal anchors/queries, so they structurally cannot consume external
    boxes — this role IS the "frames -> posed instances" slot, not a
    ``PersonDetector``/``TopDownPose2d`` pair. Dropping the keypoints adapts
    it into a ``PersonDetector``; the reverse is impossible.
    """

    skeleton: KeypointSkeleton
    """Skeleton format of the predictions."""

    @abstractmethod
    def __call__(self, frames_rgb: UInt8[Tensor, "b h w 3"]) -> tuple[BoxDetections, Keypoints2d]:
        """Predict posed instances across a frame batch.

        Args:
            frames_rgb: uint8 RGB NHWC frame batch on the inference device.

        Returns:
            Detections and their keypoints, row-aligned (instance ``i`` of the
            detections is instance ``i`` of the keypoints).
        """
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class SegmentationPrompts:
    """Instance prompts for promptable/video segmenters.

    Every prompt row targets one frame of the batch via ``frame_indices`` and
    proposes one instance; ``track_ids`` assigns stable identities when
    prompting a :class:`VideoSegmenter`.
    """

    frame_indices: Int64[Tensor, "m"]
    """Index into the frame batch each prompt targets."""
    boxes_xyxy: Float32[Tensor, "m 4"] | None = None
    """Optional image-space box prompt per instance."""
    points_xy: Float32[Tensor, "m 2"] | None = None
    """Optional image-space positive point prompt per instance."""
    track_ids: Int64[Tensor, "m"] | None = None
    """Optional identity to assign to each prompted instance (video mode)."""
    text: str | None = None
    """Optional open-vocabulary concept prompt shared by all frames (SAM3)."""


class PromptableSegmenter(ABC):
    """Segments instances in single images from box/point/text prompts."""

    @abstractmethod
    def __call__(self, frames_rgb: UInt8[Tensor, "b h w 3"], prompts: SegmentationPrompts) -> BoxDetections:
        """Segment prompted instances across a frame batch.

        Args:
            frames_rgb: uint8 RGB NHWC frame batch on the inference device.
            prompts: Box/point/text prompts referencing ``frames_rgb`` rows.

        Returns:
            Detections with ``masks`` populated (and ``track_ids`` when the
            prompts carried identities).
        """
        raise NotImplementedError


class VideoSegmenter(ABC):
    """Stateful streaming mask tracker (causal memory, forward-only).

    Call :meth:`step` once per timestep with all synchronized views stacked in
    the batch dimension; the segmenter keeps one memory state per batch slot,
    so the batch layout must stay identical across steps.
    """

    @abstractmethod
    def step(self, frames_rgb: UInt8[Tensor, "b h w 3"], prompts: SegmentationPrompts | None = None) -> BoxDetections:
        """Advance the tracker by one timestep.

        Args:
            frames_rgb: One frame per stream/view at the current timestep.
            prompts: New-instance prompts (bootstrap or re-detect); ``None``
                propagates existing tracks only.

        Returns:
            Detections with ``masks`` and ``track_ids`` populated; boxes are
            derived from the masks.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Drop all tracked instances and memory state."""
        raise NotImplementedError


class IdentityEncoder(ABC):
    """Embeds instance crops into an appearance space for re-identification."""

    embed_dim: int
    """Dimensionality of the returned embeddings."""

    @abstractmethod
    def __call__(self, frames_rgb: UInt8[Tensor, "b h w 3"], detections: BoxDetections) -> Float32[Tensor, "n embed_dim"]:
        """Embed every detection crop.

        Args:
            frames_rgb: uint8 RGB NHWC frame batch on the inference device.
            detections: Instance boxes referencing ``frames_rgb`` by index.

        Returns:
            One unnormalized embedding row per detection (cosine-compare after
            L2 normalization).
        """
        raise NotImplementedError


class TopDownDenseLandmarks2d(ABC):
    """Estimates dense surface landmarks for cropped instances (mask-aware).

    Same top-down contract as :class:`TopDownPose2d` but the detections must
    carry ``masks`` (the crop gets an extra mask channel) and the output has
    per-point uncertainty/visibility/contact heads instead of a fixed sparse
    skeleton.
    """

    num_landmarks: int
    """Number of dense landmarks per instance."""

    @abstractmethod
    def __call__(self, frames_rgb: UInt8[Tensor, "b h w 3"], detections: BoxDetections) -> DenseLandmarks2d:
        """Estimate dense landmarks for every detection.

        Args:
            frames_rgb: uint8 RGB NHWC frame batch on the inference device.
            detections: Instance boxes referencing ``frames_rgb`` by index;
                ``masks`` must be populated.

        Returns:
            Dense landmarks, one instance per detection row.

        Raises:
            ValueError: If ``detections.masks`` is ``None``.
        """
        raise NotImplementedError


class Pose2dPipeline:
    """Full-frame 2D pose: detector plus top-down estimator, GPU end to end."""

    def __init__(self, detector: PersonDetector, pose: TopDownPose2d) -> None:
        """Compose a detector and a top-down pose estimator.

        Args:
            detector: Person/hand detector stage.
            pose: Top-down keypoint estimator stage.
        """
        self.detector: PersonDetector = detector
        self.pose: TopDownPose2d = pose

    def __call__(self, frames_rgb: UInt8[Tensor, "b h w 3"]) -> tuple[BoxDetections, Keypoints2d]:
        """Run detection and pose estimation on a frame batch.

        Args:
            frames_rgb: uint8 RGB NHWC frame batch on the inference device.

        Returns:
            The detections and their estimated keypoints.
        """
        detections: BoxDetections = self.detector(frames_rgb)
        return detections, self.pose(frames_rgb, detections)
