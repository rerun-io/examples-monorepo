"""Configuration for DPVO (Deep Patch Visual Odometry).

Defines a frozen dataclass :class:`DPVOConfig` with all tuneable
hyper-parameters.  Preset factory class methods produce ``accurate``
and ``fast`` configurations matching the former YAML presets.

Typical usage::

    from dpvo.config import DPVOConfig
    cfg = DPVOConfig.fast()
    # or
    cfg = DPVOConfig.accurate()

See Teed et al. (2022), "Deep Patch Visual Odometry" for details on how
each parameter affects the system.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class DPVOConfig:
    """All tuneable hyper-parameters for the DPVO visual odometry system.

    See Teed et al. (2022), "Deep Patch Visual Odometry" for details on
    how each parameter affects the system.
    """

    buffer_size: int = 2048
    """Maximum number of keyframes held in the sliding-window buffer.
    The main DPVO class allocates fixed-size GPU tensors of this length."""

    gradient_bias: bool = True
    """Use gradient-biased patch sampling: sample 3x candidates and keep
    the top-M by image gradient magnitude."""

    patches_per_frame: int = 80
    """Number of sparse 3×3 patches tracked per frame (M in the paper)."""

    removal_window: int = 20
    """Edges with source older than ``(current - removal_window)`` are pruned."""

    optimization_window: int = 12
    """Only poses within the last ``optimization_window`` frames are optimized
    during bundle adjustment (after initialization)."""

    patch_lifetime: int = 12
    """Maximum temporal distance (in frames) for creating measurement edges."""

    keyframe_index: int = 4
    """Index offset from the newest frame for the candidate keyframe removal check."""

    keyframe_thresh: float = 12.5
    """Pixel flow threshold below which a keyframe is removed as redundant."""

    motion_model: Literal["DAMPED_LINEAR"] = "DAMPED_LINEAR"
    """Motion model for pose extrapolation. Currently only ``DAMPED_LINEAR``
    is implemented: ``Pₙ = exp(d · log(Pₙ₋₁ · Pₙ₋₂⁻¹)) · Pₙ₋₁``."""

    motion_damping: float = 0.5
    """Damping factor for the damped linear motion model.
    0.0 = constant position, 1.0 = constant velocity."""

    mixed_precision: bool = True
    """Use float16 for correlation and GRU computations."""

    @classmethod
    def accurate(cls) -> "DPVOConfig":
        """Preset matching the former ``config/default.yaml`` (higher quality)."""
        return cls(
            patches_per_frame=96,
            removal_window=22,
            optimization_window=10,
            patch_lifetime=13,
            keyframe_thresh=15.0,
            gradient_bias=False,
            mixed_precision=True,
        )

    @classmethod
    def fast(cls) -> "DPVOConfig":
        """Preset matching the former ``config/fast.yaml`` (lower latency)."""
        return cls(
            patches_per_frame=48,
            removal_window=16,
            optimization_window=7,
            patch_lifetime=11,
            keyframe_thresh=15.0,
            gradient_bias=False,
            mixed_precision=True,
        )
