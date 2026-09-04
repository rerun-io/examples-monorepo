"""Ultrawide-camera policy, prompt placement, and mask helpers for the catalog lane.

The rectified ultrawide camera sees about ``ULTRAWIDE_FOV_RATIO`` times the wide
camera's field of view on both axes, so the wide LiDAR prompt only covers the
centre of an ultrawide frame. ZipDepth-PromptDA takes a fixed
``PROMPT_CANVAS_HW`` prompt that it bilinearly resamples to its decoder feature
sizes, so the prompt is placed as a scaled block inside a zero canvas rather
than reprojected: zero reads as "no prompt" through the model's own metric
range gate. ``docs/ultrawide.md`` records the measurement and the alternatives.
"""

from collections.abc import Generator, Iterable, Iterator, Sequence
from dataclasses import dataclass
from random import Random
from typing import Literal, TypeAlias, TypeVar

import torch
from jaxtyping import Bool, Float32, Shaped
from monopriors.models.depth_completion.base_completion_depth import MAX_METRIC_DEPTH_M, MIN_METRIC_DEPTH_M
from torch import Tensor
from torch.nn import functional as F

ItemT = TypeVar("ItemT")

Camera: TypeAlias = Literal["wide", "ultrawide"]
"""One physical ARKitScenes camera contributing training samples."""
CameraSelection: TypeAlias = Literal["wide", "ultrawide", "both"]
"""Cameras a catalog dataset streams; ``both`` interleaves them per segment."""

ULTRAWIDE_LAYER: str = "ultrawide_depth"
"""Catalog layer carrying the rectified ultrawide RGB and raycast depth."""

PROMPT_CANVAS_HW: tuple[int, int] = (192, 256)
"""Prompt grid the prompted ZipDepth decoder expects, in ``(height, width)``."""
ULTRAWIDE_FOV_RATIO: float = 2.1
"""Ultrawide-to-wide field-of-view ratio per axis, measured from the catalog calibration.

Over 24 segments spanning both stored orientations the per-axis ratio
``(resolution / focal_length)_ultrawide / (resolution / focal_length)_wide`` is
2.109 +/- 0.015 (2.069 to 2.138), identical on x and y because both cameras are
square-pixel and share a 4:3 aspect. The rectified ultrawide principal point sits
within 0.7% of the image centre, so a centred prompt block is the right model.
"""
DEFAULT_ULTRAWIDE_PROMPT_SCALE: float = 1.0 / ULTRAWIDE_FOV_RATIO
"""Fraction of an ultrawide frame the wide LiDAR prompt covers on each axis."""


@dataclass(frozen=True, slots=True)
class PromptPlacement:
    """Where the scaled wide prompt sits inside the fixed prompt canvas."""

    top: int
    """First canvas row occupied by the prompt block."""
    left: int
    """First canvas column occupied by the prompt block."""
    height: int
    """Prompt block height in canvas rows."""
    width: int
    """Prompt block width in canvas columns."""


@dataclass(frozen=True, slots=True)
class UltrawidePolicy:
    """Mask and prompt-placement policy applied to ultrawide samples only."""

    min_valid_fraction: float = 0.7
    """Reject an ultrawide frame whose in-range target fraction, measured before
    erosion, falls below this. The raycast depth has holes at glazing and
    outdoors; sparse frames teach the loss almost nothing per decoded frame."""
    valid_erosion_px: int = 1
    """Binary-erode the ultrawide target mask by this many pixels (a 3x3
    structuring element per pixel). Raycast silhouettes bleed one output pixel
    past the mesh, so the boundary ring is the least trustworthy supervision."""
    prompt_scale: float = DEFAULT_ULTRAWIDE_PROMPT_SCALE
    """Prompt block size as a fraction of the canvas, per axis."""

    def __post_init__(self) -> None:
        """Reject a fraction outside the unit interval or a negative erosion."""
        if not 0.0 <= self.min_valid_fraction <= 1.0:
            raise ValueError("min_valid_fraction must be in [0, 1]")
        if self.valid_erosion_px < 0:
            raise ValueError("valid_erosion_px must be non-negative")
        if not 0.0 < self.prompt_scale <= 1.0:
            raise ValueError("prompt_scale must be in (0, 1]")


def prompt_placement(scale: float, canvas_hw: tuple[int, int] = PROMPT_CANVAS_HW) -> PromptPlacement:
    """Centre a ``scale``-sized prompt block inside the prompt canvas.

    Args:
        scale: Prompt block size as a fraction of the canvas, per axis.
        canvas_hw: Prompt canvas ``(height, width)``.

    Returns:
        The block's top-left corner and size in canvas pixels.

    Raises:
        ValueError: If the scale is outside ``(0, 1]`` or rounds to an empty block.
    """
    if not 0.0 < scale <= 1.0:
        raise ValueError("prompt scale must be in (0, 1]")
    height: int = int(round(canvas_hw[0] * scale))
    width: int = int(round(canvas_hw[1] * scale))
    if height <= 0 or width <= 0:
        raise ValueError(f"prompt scale {scale} rounds to an empty block on a {canvas_hw} canvas")
    return PromptPlacement(top=(canvas_hw[0] - height) // 2, left=(canvas_hw[1] - width) // 2, height=height, width=width)


def resize_prompt_block(values_hw: Shaped[Tensor, "h w"], placement: PromptPlacement) -> Float32[Tensor, "block_h block_w"]:
    """Nearest-exact resample one prompt plane down to the placement block.

    Nearest resampling keeps every output value a real LiDAR reading, which
    matters because the block feeds the same metric range gate as the wide path.

    Args:
        values_hw: Prompt millimetres or confidence with shape ``(H, W)``.
        placement: Block size the values are resampled to.

    Returns:
        Float32 values with shape ``(placement.height, placement.width)`` on the
        input's device.
    """
    resampled: Float32[Tensor, "1 1 block_h block_w"] = F.interpolate(
        values_hw.to(dtype=torch.float32).unsqueeze(0).unsqueeze(0),
        size=(placement.height, placement.width),
        mode="nearest-exact",
    )
    return resampled.squeeze(0).squeeze(0)


def pad_prompt_block(
    block_hw: Float32[Tensor, "block_h block_w"],
    placement: PromptPlacement,
    canvas_hw: tuple[int, int] = PROMPT_CANVAS_HW,
) -> Float32[Tensor, "canvas_h canvas_w"]:
    """Write one prompt block into a zeroed canvas at its placement.

    Args:
        block_hw: Prompt values with shape ``(placement.height, placement.width)``.
        placement: Where the block sits in the canvas.
        canvas_hw: Prompt canvas ``(height, width)``.

    Returns:
        The canvas with the block written in and zeros elsewhere.

    Raises:
        ValueError: If the block does not match the placement size.
    """
    if tuple(block_hw.shape) != (placement.height, placement.width):
        raise ValueError(f"prompt block {tuple(block_hw.shape)} does not match placement {(placement.height, placement.width)}")
    canvas_hw_tensor: Float32[Tensor, "canvas_h canvas_w"] = torch.zeros(canvas_hw, dtype=block_hw.dtype, device=block_hw.device)
    canvas_hw_tensor[placement.top : placement.top + placement.height, placement.left : placement.left + placement.width] = block_hw
    return canvas_hw_tensor


def place_wide_prompt(
    prompt_depth_mm_hw: Shaped[Tensor, "canvas_h canvas_w"],
    prompt_confidence_hw: Shaped[Tensor, "canvas_h canvas_w"],
    placement: PromptPlacement,
    *,
    flipped: bool,
    canvas_hw: tuple[int, int] = PROMPT_CANVAS_HW,
) -> tuple[Float32[Tensor, "canvas_h canvas_w"], Bool[Tensor, "canvas_h canvas_w"]]:
    """Scale the oriented wide prompt into its ultrawide footprint on a zero canvas.

    Validity is derived after resampling, from the resampled millimetres and
    confidence, so depth and validity always describe the same LiDAR reading.
    Mirroring is applied to the block rather than the canvas, which keeps the
    flip exact even when the canvas margin is odd.

    Args:
        prompt_depth_mm_hw: Oriented wide LiDAR depth in millimetres, on the
            prompt canvas grid.
        prompt_confidence_hw: Oriented ARKit confidence on the same grid.
        placement: Where the scaled block sits inside the canvas.
        flipped: Whether the aligned RGB and target were mirrored horizontally.
        canvas_hw: Prompt canvas ``(height, width)``.

    Returns:
        Prompt depth in metres and its validity, both on the canvas grid, with
        zeros and ``False`` outside the footprint.

    Raises:
        ValueError: If either input does not match the canvas grid.
    """
    if tuple(prompt_depth_mm_hw.shape) != canvas_hw or tuple(prompt_confidence_hw.shape) != canvas_hw:
        raise ValueError(
            f"oriented prompt depth and confidence must both be {canvas_hw}, "
            f"got {tuple(prompt_depth_mm_hw.shape)} and {tuple(prompt_confidence_hw.shape)}"
        )
    block_mm_hw: Float32[Tensor, "block_h block_w"] = resize_prompt_block(prompt_depth_mm_hw, placement)
    block_confidence_hw: Float32[Tensor, "block_h block_w"] = resize_prompt_block(prompt_confidence_hw, placement)
    block_depth_m_hw: Float32[Tensor, "block_h block_w"] = block_mm_hw / 1000.0
    block_valid_hw: Bool[Tensor, "block_h block_w"] = (
        (block_confidence_hw >= 1.0) & (block_mm_hw >= MIN_METRIC_DEPTH_M * 1000.0) & (block_mm_hw <= MAX_METRIC_DEPTH_M * 1000.0)
    )
    if flipped:
        block_depth_m_hw = torch.flip(block_depth_m_hw, dims=(-1,))
        block_valid_hw = torch.flip(block_valid_hw, dims=(-1,))
    canvas_depth_m_hw: Float32[Tensor, "canvas_h canvas_w"] = pad_prompt_block(block_depth_m_hw, placement, canvas_hw)
    canvas_valid_hw: Bool[Tensor, "canvas_h canvas_w"] = (
        pad_prompt_block(block_valid_hw.to(dtype=torch.float32), placement, canvas_hw) > 0.5
    )
    return canvas_depth_m_hw, canvas_valid_hw


def footprint_box(placement: PromptPlacement, out_hw: tuple[int, int], canvas_hw: tuple[int, int] = PROMPT_CANVAS_HW) -> tuple[int, int, int, int]:
    """Map the prompt block onto an output image as ``(top, left, bottom, right)``.

    Args:
        placement: Prompt block position inside the canvas.
        out_hw: Output image ``(height, width)``.
        canvas_hw: Prompt canvas ``(height, width)``.

    Returns:
        Half-open output pixel bounds covered by the prompt.
    """
    scale_y: float = out_hw[0] / canvas_hw[0]
    scale_x: float = out_hw[1] / canvas_hw[1]
    return (
        int(round(placement.top * scale_y)),
        int(round(placement.left * scale_x)),
        int(round((placement.top + placement.height) * scale_y)),
        int(round((placement.left + placement.width) * scale_x)),
    )


def erode_valid(valid_chw: Bool[Tensor, "1 h w"], erosion_px: int) -> Bool[Tensor, "1 h w"]:
    """Binary-erode a validity mask with a ``2 * erosion_px + 1`` square element.

    Implemented as a max pool over the inverted mask, which runs on CPU and CUDA
    through one kernel so both sample builders agree exactly. Zero padding on the
    inverted mask leaves the image border valid: the rectified frame's genuine
    border holes are already invalid and erode inward on their own.

    Args:
        valid_chw: Boolean validity with shape ``(1, H, W)``.
        erosion_px: Erosion radius in pixels; zero returns the input unchanged.

    Returns:
        The eroded validity mask with the input's shape, dtype, and device.

    Raises:
        ValueError: If the radius is negative or the mask is not ``(1, H, W)``.
    """
    if erosion_px < 0:
        raise ValueError("erosion_px must be non-negative")
    if valid_chw.ndim != 3 or valid_chw.shape[0] != 1:
        raise ValueError(f"validity mask must have shape (1, H, W), got {tuple(valid_chw.shape)}")
    if erosion_px == 0:
        return valid_chw
    invalid_bchw: Float32[Tensor, "1 1 h w"] = (~valid_chw).to(dtype=torch.float32).unsqueeze(0)
    dilated_bchw: Float32[Tensor, "1 1 h w"] = F.max_pool2d(
        invalid_bchw,
        kernel_size=2 * erosion_px + 1,
        stride=1,
        padding=erosion_px,
    )
    return dilated_bchw.squeeze(0) < 0.5


def valid_fraction(valid_chw: Bool[Tensor, "1 h w"]) -> float:
    """Return the fraction of the mask that is valid."""
    return float(valid_chw.to(dtype=torch.float32).mean().item())


def subsample_indices(available: int, wanted: int, seed: int) -> list[int]:
    """Pick a seeded, sorted subset of frame positions.

    Args:
        available: Number of candidate frames in the segment.
        wanted: Frames to keep; values at or above ``available`` keep all of them.
        seed: Seed mixing the run seed, epoch, rank, and segment.

    Returns:
        Sorted candidate positions, so a producer still reads the segment in
        index order and keeps its decode locality.

    Raises:
        ValueError: If either count is negative.
    """
    if available < 0 or wanted < 0:
        raise ValueError("available and wanted frame counts must be non-negative")
    if wanted >= available:
        return list(range(available))
    return sorted(Random(seed).sample(range(available), wanted))


def interleave(  # noqa: UP047 — beartype does not support PEP 695 type parameters
    iterables: Sequence[Iterable[ItemT]],
) -> Generator[ItemT, None, None]:
    """Yield round-robin from every iterable until all are exhausted.

    Args:
        iterables: Sources drained one item at a time in the given order.

    Yields:
        Items alternating between the sources that still have output.
    """
    iterators: list[Iterator[ItemT]] = [iter(iterable) for iterable in iterables]
    while iterators:
        remaining: list[Iterator[ItemT]] = []
        iterator: Iterator[ItemT]
        for iterator in iterators:
            item: ItemT
            try:
                item = next(iterator)
            except StopIteration:
                continue
            remaining.append(iterator)
            yield item
        iterators = remaining
