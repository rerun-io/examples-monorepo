"""Depth alignment node.

Aligns a target depth map to a reference depth map's coordinate frame using
least-squares scale and shift estimation. This is a general-purpose utility
with no knowledge of any specific network — the caller decides which depth
is "reference" and which is "target".

Also provides a CLI entry point (``main``) for standalone usage with tyro.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from jaxtyping import Bool, Float32
from numpy import ndarray
from simplecv.rerun_log_utils import RerunTyroConfig

from monopriors.depth_utils import depth_edges_mask
from monopriors.scale_utils import compute_scale_and_shift


@dataclass
class DepthAlignmentConfig:
    """Configuration for aligning a target depth map to a reference depth's coordinate frame."""

    edge_threshold: float = 0.01
    """Threshold for depth edge masking on the aligned output."""
    scale_only: bool = False
    """Use scale-only alignment (no shift) when True."""


@dataclass
class DepthAlignmentResult:
    """Output of single-view depth alignment."""

    aligned_depth: Float32[ndarray, "H W"]
    """Target depth aligned to the reference depth's coordinate frame."""
    scale: float
    """Computed scale factor."""
    shift: float
    """Computed shift (0 if scale_only=True)."""


def run_depth_alignment(
    *,
    reference_depth: Float32[ndarray, "H W"],
    target_depth: Float32[ndarray, "H W"],
    confidence_mask: Bool[ndarray, "H W"] | None = None,
    exclusion_mask: Bool[ndarray, "H W"] | None = None,
    config: DepthAlignmentConfig | None = None,
) -> DepthAlignmentResult:
    """Align target_depth to reference_depth's scale/shift using valid pixels.

    Computes optimal scale and shift to transform ``target_depth`` into the
    coordinate frame of ``reference_depth``, then applies depth edge masking
    and optional exclusion masking.

    Args:
        reference_depth: Depth map defining the target coordinate frame.
        target_depth: Depth map to be aligned.
        confidence_mask: Binary mask of trusted pixels in reference_depth.
            When provided, only True pixels contribute to the alignment.
        exclusion_mask: Binary mask of pixels to exclude from the aligned
            output (e.g., people). These pixels are zeroed in the result.
        config: Alignment configuration.

    Returns:
        DepthAlignmentResult with aligned depth, scale, and shift.
    """
    if config is None:
        config = DepthAlignmentConfig()

    # Build mask of valid pixels for alignment
    if confidence_mask is not None:
        valid_mask: Bool[ndarray, "H W"] = confidence_mask
    else:
        valid_mask = reference_depth > 0

    # Compute scale and shift
    scale: float
    shift: float
    scale, shift = compute_scale_and_shift(
        target_depth, reference_depth, mask=valid_mask, scale_only=config.scale_only
    )

    # Apply alignment
    aligned: Float32[ndarray, "H W"] = target_depth.copy() * scale + shift

    # Apply depth edge masking
    edges: Bool[ndarray, "H W"] = depth_edges_mask(aligned, threshold=config.edge_threshold)
    aligned = aligned * ~edges

    # Apply exclusion mask (e.g., people)
    if exclusion_mask is not None:
        aligned = aligned * ~exclusion_mask

    return DepthAlignmentResult(
        aligned_depth=aligned.astype(np.float32),
        scale=float(scale),
        shift=float(shift),
    )


@dataclass
class DepthAlignmentCLIConfig:
    """CLI configuration for depth alignment."""

    rr_config: RerunTyroConfig
    """Rerun logging configuration."""
    reference_path: Path = Path("data/examples/depth_alignment/reference_depth.npy")
    """Path to reference depth .npy file."""
    target_path: Path = Path("data/examples/depth_alignment/target_depth.npy")
    """Path to target depth .npy file."""
    confidence_path: Path | None = Path("data/examples/depth_alignment/confidence.npy")
    """Optional path to confidence .npy file."""
    alignment_config: DepthAlignmentConfig = field(default_factory=DepthAlignmentConfig)
    """Depth alignment configuration."""


def main(config: DepthAlignmentCLIConfig) -> None:
    """CLI entry point for depth alignment with Rerun visualization."""
    import rerun as rr
    import rerun.blueprint as rrb

    parent_log_path: Path = Path("world")

    # Load depths
    ref_depth: Float32[ndarray, "H W"] = np.load(str(config.reference_path)).astype(np.float32)
    tgt_depth: Float32[ndarray, "H W"] = np.load(str(config.target_path)).astype(np.float32)

    # Load confidence if provided
    confidence_mask: Bool[ndarray, "H W"] | None = None
    if config.confidence_path is not None:
        raw_conf: ndarray = np.load(str(config.confidence_path))
        confidence_mask = (raw_conf > 0).astype(bool)

    # Run alignment
    result: DepthAlignmentResult = run_depth_alignment(
        reference_depth=ref_depth,
        target_depth=tgt_depth,
        confidence_mask=confidence_mask,
        config=config.alignment_config,
    )

    # Setup Rerun
    blueprint: rrb.Blueprint = rrb.Blueprint(
        rrb.Grid(
            rrb.Spatial2DView(origin=f"{parent_log_path}/reference_depth", name="Reference"),
            rrb.Spatial2DView(origin=f"{parent_log_path}/target_depth", name="Target"),
            rrb.Spatial2DView(origin=f"{parent_log_path}/aligned_depth", name="Aligned"),
            rrb.Spatial2DView(origin=f"{parent_log_path}/confidence", name="Confidence"),
        ),
        collapse_panels=True,
    )
    rr.send_blueprint(blueprint)
    rr.log(f"{parent_log_path}", rr.ViewCoordinates.RDF, static=True)

    rr.log(f"{parent_log_path}/reference_depth", rr.DepthImage(ref_depth, meter=1), static=True)
    rr.log(f"{parent_log_path}/target_depth", rr.DepthImage(tgt_depth, meter=1), static=True)
    rr.log(f"{parent_log_path}/aligned_depth", rr.DepthImage(result.aligned_depth, meter=1), static=True)

    if confidence_mask is not None:
        conf_vis: ndarray = confidence_mask.astype(np.uint8) * 255
    else:
        conf_vis = (ref_depth > 0).astype(np.uint8) * 255
    rr.log(f"{parent_log_path}/confidence", rr.Image(conf_vis, color_model=rr.ColorModel.L), static=True)

    print(f"Alignment complete (scale={result.scale:.4f}, shift={result.shift:.4f})")
