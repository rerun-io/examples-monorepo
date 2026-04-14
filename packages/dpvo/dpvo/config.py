"""Configuration for DPV-SLAM (Deep Patch Visual SLAM).

Defines a frozen dataclass :class:`DPVOConfig` with all tuneable
hyper-parameters for both visual odometry and SLAM modes.

Typical usage::

    from dpvo.config import DPVOConfig
    cfg = DPVOConfig.fast()           # VO only
    cfg = DPVOConfig.slam()           # SLAM with proximity loop closure
    cfg = DPVOConfig.slam_classic()   # SLAM with DBoW2 loop closure

See Teed et al. (2022), "Deep Patch Visual Odometry" and
Lipson et al. (2024), "Deep Patch Visual SLAM" for details.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class DPVOConfig:
    """All tuneable hyper-parameters for the DPV-SLAM system.

    SLAM features (``loop_closure``, ``classic_loop_closure``) are off by
    default for backward compatibility with pure VO usage.
    """

    buffer_size: int = 2048
    """Maximum number of keyframes held in the sliding-window buffer."""

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

    # ── Proximity loop closure (GPU-based, no external deps) ────────────

    loop_closure: bool = False
    """Enable proximity-based loop closure.  Detects loops via camera
    proximity and inserts long-range edges for global BA."""

    backend_thresh: float = 64.0
    """Maximum flow magnitude for a loop closure edge to be accepted."""

    max_edge_age: int = 1000
    """Maximum age (in frames) of patches used for loop closure edges.
    Also controls the patch memory buffer size when loop closure is on."""

    global_opt_freq: int = 15
    """How often (in frames) to attempt adding loop closure edges."""

    # ── Classical loop closure (DBoW2, requires dpretrieval) ────────────

    classic_loop_closure: bool = False
    """Enable classical (DBoW2-based) loop closure.  Requires ``dpretrieval``
    and ``kornia`` packages."""

    loop_close_window_size: int = 3
    """Number of consecutive retrieval hits required to confirm a loop."""

    loop_retr_thresh: float = 0.04
    """DBoW2 retrieval score threshold for loop detection."""

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

    @classmethod
    def slam(cls) -> "DPVOConfig":
        """SLAM preset: accurate() base + proximity loop closure.

        Uses the same VO parameters as ``accurate()`` (matching upstream
        ``config/default.yaml``) with proximity loop closure enabled.
        """
        return cls(
            buffer_size=4096,
            patches_per_frame=96,
            removal_window=22,
            optimization_window=10,
            patch_lifetime=13,
            keyframe_thresh=15.0,
            gradient_bias=False,
            mixed_precision=True,
            loop_closure=True,
            backend_thresh=64.0,
            max_edge_age=1000,
            global_opt_freq=15,
        )

    @classmethod
    def slam_classic(cls) -> "DPVOConfig":
        """SLAM preset: accurate() base + proximity + classical loop closure."""
        return cls(
            buffer_size=4096,
            patches_per_frame=96,
            removal_window=22,
            optimization_window=10,
            patch_lifetime=13,
            keyframe_thresh=15.0,
            gradient_bias=False,
            mixed_precision=True,
            loop_closure=True,
            backend_thresh=64.0,
            max_edge_age=1000,
            global_opt_freq=15,
            classic_loop_closure=True,
            loop_close_window_size=3,
            loop_retr_thresh=0.04,
        )
