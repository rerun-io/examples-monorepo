"""Structural stream contract and optional exact-validation metadata.

Training uses only BatchSource. For paper metrics a validation source also
implements the relevant metadata protocol. Metadata describes the most recent
batch returned by next_*; it must not advance the stream. Validation crops must
come from GT boxes without random geometry or intensity augmentation.
"""
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from jaxtyping import Bool, Float32, Int64
from torch import Tensor

from handtrack.data.batches import BatchSource

__all__ = ["BatchSource", "DetNetValidation", "DetNetValidationSource", "KeyNetValidation", "KeyNetValidationSource"]


@dataclass(frozen=True, slots=True)
class DetNetValidation:
    """Unaugmented GT in the 640x480 frame, including partially visible hands."""
    points: Float32[Tensor, 'b 2 21 2']
    """Projected GT landmarks per hand."""
    in_front: Bool[Tensor, 'b 2 21']
    """Points with positive camera depth; absent hands have all False."""
    camera: Int64[Tensor, 'b']
    """Globally unique camera IDs (namespace across datasets)."""


@dataclass(frozen=True, slots=True)
class KeyNetValidation:
    """Exact GT, avoiding heatmap quantization and distance clamping in metrics."""
    crop_from_net: Float32[Tensor, 'b 3 3']
    """Affine used to cut each GT crop, including mirror."""
    points_crop: Float32[Tensor, 'b 21 2']
    """Unquantized target keypoints in crop pixels."""
    distance_mm: Float32[Tensor, 'b 21']
    """Unclamped GT d_rel in mm."""


@runtime_checkable
class DetNetValidationSource(Protocol):
    """Optional metadata extension used only during DetNet validation."""
    def detnet_validation(self) -> DetNetValidation: ...


@runtime_checkable
class KeyNetValidationSource(Protocol):
    """Optional metadata extension used only during KeyNet validation."""
    def keynet_validation(self) -> KeyNetValidation: ...
