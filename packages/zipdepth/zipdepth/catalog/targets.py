"""Helpers for building ZipDepth training targets (numpy/torch/albumentations only)."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

import albumentations as A
import cv2
import numpy as np
import torch
from jaxtyping import Bool, Float32, Int64, Shaped, UInt8, UInt16
from numpy import ndarray
from torch import Tensor
from torch.nn import functional as F

from zipdepth.data.transforms import AlbumentationsWrapper

MIN_DEPTH_MM: int = 100
"""Nearest valid pseudo-depth, in millimetres."""
MAX_DEPTH_MM: int = 4000
"""Furthest valid pseudo-depth, in millimetres."""
INV_DEPTH_SCALE: float = 6000.0
"""Inverse-depth quantization scale; 10 1/m maps to 60000 without uint16 overflow.

The SSI loss is scale-invariant, so only the resulting floating-point exponent
range matters after the trainer divides the integer target by 256.
"""

_DepthTransform: TypeAlias = Callable[
    [UInt8[ndarray, "h w 3"], Float32[ndarray, "h w"]],
    tuple[UInt8[ndarray, "out_h out_w 3"], Float32[ndarray, "out_h out_w"] | None],
]
TargetMode: TypeAlias = Literal["ssi", "metric"]


@dataclass(frozen=True, slots=True)
class AugmentPolicy:
    """Probabilities shared by CPU and CUDA training augmentation."""

    flip_p: float = 0.5
    """Probability of an aligned horizontal flip."""
    channel_shuffle_p: float = 0.05
    """Probability of a random RGB channel permutation."""
    gray_p: float = 0.05
    """Probability of grayscale conversion replicated to three channels."""

    def __post_init__(self) -> None:
        """Reject probabilities outside the closed unit interval."""
        probability: float
        for probability in (self.flip_p, self.channel_shuffle_p, self.gray_p):
            if not 0.0 <= probability <= 1.0:
                raise ValueError("augmentation probabilities must be in [0, 1]")


def inverse_depth_from_mm(depth_mm_hw: UInt16[ndarray, "h w"]) -> Float32[ndarray, "h w"]:
    """Convert valid millimetre depth to inverse metres and set invalid pixels to zero.

    Args:
        depth_mm_hw: Unsigned 16-bit depth in millimetres with shape ``(H, W)``.

    Returns:
        Float32 inverse depth in inverse metres with shape ``(H, W)``. Values
        outside ``[MIN_DEPTH_MM, MAX_DEPTH_MM]`` are zero.
    """
    valid_hw: Bool[ndarray, "h w"] = (depth_mm_hw >= MIN_DEPTH_MM) & (depth_mm_hw <= MAX_DEPTH_MM)
    inverse_depth_hw: Float32[ndarray, "h w"] = np.zeros(depth_mm_hw.shape, dtype=np.float32)
    inverse_depth_hw[valid_hw] = 1000.0 / depth_mm_hw[valid_hw]
    return inverse_depth_hw


def quantize_inverse_depth(inverse_depth_hw: Float32[ndarray, "h w"]) -> UInt16[ndarray, "h w"]:
    """Round inverse depth into the uint16 target range.

    Args:
        inverse_depth_hw: Float32 inverse depth in inverse metres with shape ``(H, W)``.

    Returns:
        Quantized unsigned 16-bit inverse depth with shape ``(H, W)``.
    """
    scaled_hw: Float32[ndarray, "h w"] = np.rint(inverse_depth_hw * INV_DEPTH_SCALE)
    quantized_hw: UInt16[ndarray, "h w"] = np.clip(scaled_hw, 0.0, float(np.iinfo(np.uint16).max)).astype(np.uint16)
    return quantized_hw


def build_train_transform(height: int, width: int, policy: AugmentPolicy) -> AlbumentationsWrapper:
    """Deterministic-resize training transform: flip and mild color, no crop zoom.

    The vendored ``get_train_transforms`` randomly zooms half the samples via
    SmallestMaxSize + RandomCrop; at 4:3 -> 1024x768 that trains a ~1.33x-zoom
    domain the full-FOV probe and eval never see. This keeps upstream's flip and
    color augmentation but resizes deterministically to the training size.
    """
    return AlbumentationsWrapper(
        A.ReplayCompose(
            [
                A.Resize(height=height, width=width, interpolation=cv2.INTER_LINEAR),
                A.HorizontalFlip(p=policy.flip_p),
                A.ChannelShuffle(p=policy.channel_shuffle_p),
                A.ToGray(num_output_channels=3, p=policy.gray_p),
            ],
            additional_targets={"mask": "mask"},
        )
    )


def build_eval_transform(height: int, width: int) -> AlbumentationsWrapper:
    """Build deterministic aligned RGB/mask resizing for evaluation."""
    if height <= 0 or width <= 0:
        raise ValueError("evaluation height and width must be positive")
    transform: A.ReplayCompose = A.ReplayCompose(
        [A.Resize(height=height, width=width, interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST)],
        additional_targets={"mask": "mask"},
    )
    return AlbumentationsWrapper(transform)


def _replay_applied_horizontal_flip(replay: dict[str, Any]) -> bool:
    """Return whether an Albumentations replay applied a horizontal flip."""
    transforms: Any = replay.get("transforms", [])
    if not isinstance(transforms, list):
        return False
    transform: Any
    for transform in transforms:
        if not isinstance(transform, dict):
            continue
        class_name: str = str(transform.get("__class_fullname__", ""))
        if class_name.endswith("HorizontalFlip") and bool(transform.get("applied", False)):
            return True
        if _replay_applied_horizontal_flip(transform):
            return True
    return False


def _transform_with_flip_state(
    rgb_hw3: UInt8[ndarray, "h w 3"],
    target_hw: Float32[ndarray, "h w"],
    transform: _DepthTransform,
) -> tuple[UInt8[ndarray, "out_h out_w 3"], Float32[ndarray, "out_h out_w"], bool]:
    """Apply RGB/target augmentation and report the replayed horizontal flip."""
    if not isinstance(transform, AlbumentationsWrapper) or not isinstance(transform.transform, A.ReplayCompose):
        transformed: tuple[UInt8[ndarray, "out_h out_w 3"], Float32[ndarray, "out_h out_w"] | None] = transform(rgb_hw3, target_hw)
        if transformed[1] is None:
            raise RuntimeError("training transform dropped the depth target")
        return transformed[0], transformed[1], False
    result: dict[str, Any] = transform.transform(image=rgb_hw3, mask=target_hw)
    augmented_rgb_hw3: UInt8[ndarray, "out_h out_w 3"] = result["image"]
    augmented_target_hw: Float32[ndarray, "out_h out_w"] = result["mask"]
    replay: Any = result.get("replay")
    if not isinstance(replay, dict):
        raise RuntimeError("ReplayCompose did not return augmentation metadata")
    return augmented_rgb_hw3, augmented_target_hw, _replay_applied_horizontal_flip(replay)


def _prompt_tensors(
    prompt_depth_mm_hw: UInt16[ndarray, "192 256"],
    prompt_confidence_hw: UInt8[ndarray, "192 256"],
    *,
    flipped: bool,
) -> dict[str, Tensor]:
    """Convert one oriented native-grid LiDAR prompt to collatable tensors."""
    if prompt_depth_mm_hw.shape != (192, 256) or prompt_confidence_hw.shape != (192, 256):
        raise ValueError(
            f"oriented prompt depth and confidence must both be 192x256, got {prompt_depth_mm_hw.shape} and {prompt_confidence_hw.shape}"
        )
    prompt_depth_m_hw: Float32[ndarray, "192 256"] = prompt_depth_mm_hw.astype(np.float32) / 1000.0
    prompt_valid_hw: Bool[ndarray, "192 256"] = (
        (prompt_confidence_hw >= 1) & (prompt_depth_mm_hw >= MIN_DEPTH_MM) & (prompt_depth_mm_hw <= MAX_DEPTH_MM)
    )
    if flipped:
        prompt_depth_m_hw = np.flip(prompt_depth_m_hw, axis=-1)
        prompt_valid_hw = np.flip(prompt_valid_hw, axis=-1)
    prompt_depth_1hw: Float32[ndarray, "1 192 256"] = np.ascontiguousarray(prompt_depth_m_hw[None])
    prompt_valid_1hw: Bool[ndarray, "1 192 256"] = np.ascontiguousarray(prompt_valid_hw[None])
    return {
        "prompt_depth": torch.from_numpy(prompt_depth_1hw),
        "prompt_valid": torch.from_numpy(prompt_valid_1hw),
    }


def depth_span_ratio(depth_mm_hw: UInt16[ndarray, "h w"]) -> float:
    """Return the p95/p5 ratio of valid depth, or 0.0 when nothing is valid.

    Capture-start closeups (a floor or wall filling the frame) have a nearly
    constant depth; under the scale-and-shift-invariant loss their MAD-normalized
    targets amplify sensor noise into large, unlearnable gradients. The span
    ratio lets the dataset skip such degenerate frames. A fixed stride-16 spatial
    sample preserves this coarse scene-span decision while reducing percentile
    input by 256x (about 200x faster on 1920x1440 targets).
    """
    sampled_mm_hw: UInt16[ndarray, "sample_h sample_w"] = depth_mm_hw[::16, ::16]
    valid: Float32[ndarray, "n"] = sampled_mm_hw[(sampled_mm_hw >= MIN_DEPTH_MM) & (sampled_mm_hw <= MAX_DEPTH_MM)].astype(np.float32)
    if valid.size == 0:
        return 0.0
    p5: float = float(np.percentile(valid, 5))
    p95: float = float(np.percentile(valid, 95))
    return p95 / max(p5, 1.0)


def depth_span_ratio_cuda(depth_mm_hw: Shaped[Tensor, "h w"]) -> Float32[Tensor, ""]:
    """Return the CPU span metric from the same stride-16 sample on CUDA.

    Invalid values sort behind the valid sample. GPU scalar indices then apply
    NumPy's linear percentile interpolation without copying the sample to the
    host. An all-invalid image returns a CUDA scalar zero.

    Args:
        depth_mm_hw: CUDA int32 or floating metric depth with shape ``(H, W)``.

    Returns:
        CUDA float32 scalar containing valid-depth p95/p5, or zero when no
        sampled depth is valid.

    Raises:
        ValueError: If the input is not a two-dimensional CUDA tensor.
    """
    if depth_mm_hw.ndim != 2 or not depth_mm_hw.is_cuda:
        raise ValueError("depth_mm_hw must be a two-dimensional CUDA tensor")
    sampled_n: Float32[Tensor, "sample_n"] = depth_mm_hw[::16, ::16].flatten().to(dtype=torch.float32)
    valid_n: Bool[Tensor, "sample_n"] = (sampled_n >= MIN_DEPTH_MM) & (sampled_n <= MAX_DEPTH_MM)
    sortable_n: Float32[Tensor, "sample_n"] = torch.where(valid_n, sampled_n, torch.full_like(sampled_n, torch.inf))
    sorted_n: Float32[Tensor, "sample_n"] = torch.sort(sortable_n).values
    valid_count: Tensor = valid_n.sum()
    last_valid_index: Float32[Tensor, ""] = torch.clamp(valid_count - 1, min=0).to(dtype=torch.float32)

    positions_2: Float32[Tensor, "2"] = last_valid_index * torch.tensor([0.05, 0.95], device=depth_mm_hw.device)
    lower_float_2: Float32[Tensor, "2"] = torch.floor(positions_2)
    lower_2: Int64[Tensor, "2"] = lower_float_2.to(dtype=torch.int64)
    upper_2: Int64[Tensor, "2"] = torch.ceil(positions_2).to(dtype=torch.int64)
    percentiles_2: Float32[Tensor, "2"] = torch.lerp(sorted_n[lower_2], sorted_n[upper_2], positions_2 - lower_float_2)
    p5: Float32[Tensor, ""] = percentiles_2[0]
    p95: Float32[Tensor, ""] = percentiles_2[1]
    ratio: Float32[Tensor, ""] = p95 / torch.clamp(p5, min=1.0)
    return torch.where(valid_count > 0, ratio, torch.zeros_like(ratio))


def build_training_sample(
    rgb_hw3: UInt8[ndarray, "h w 3"],
    depth_mm_hw: UInt16[ndarray, "h w"],
    transform: _DepthTransform,
    *,
    prompt_depth_mm_hw: UInt16[ndarray, "192 256"] | None = None,
    prompt_confidence_hw: UInt8[ndarray, "192 256"] | None = None,
) -> dict[str, Tensor]:
    """Augment aligned RGB and inverse depth, then build one collatable sample.

    Args:
        rgb_hw3: Unsigned 8-bit RGB image with shape ``(H, W, 3)``.
        depth_mm_hw: Unsigned 16-bit metric depth with shape ``(H, W)``.
        transform: ZipDepth augmentation callable applied to RGB and float32
            inverse depth together. It treats depth as an Albumentations mask,
            so spatial resizing uses nearest-neighbor interpolation.

    Returns:
        A dictionary containing contiguous ``image`` uint8 ``(3, H, W)``,
        ``depth`` uint16 ``(1, H, W)``, and ``mask`` bool ``(1, H, W)`` tensors.
        This preserves the vendored loader's uint16 storage dtype; the trainer
        immediately converts it to float32 and divides by 256.

    Raises:
        RuntimeError: If the transform drops the depth target.
    """
    inverse_depth_hw: Float32[ndarray, "h w"] = inverse_depth_from_mm(depth_mm_hw)
    transformed: tuple[UInt8[ndarray, "out_h out_w 3"], Float32[ndarray, "out_h out_w"], bool] = _transform_with_flip_state(
        rgb_hw3, inverse_depth_hw, transform
    )
    augmented_rgb_hw3: UInt8[ndarray, "out_h out_w 3"] = transformed[0]
    augmented_inverse_depth_hw: Float32[ndarray, "out_h out_w"] = transformed[1]
    quantized_hw: UInt16[ndarray, "out_h out_w"] = quantize_inverse_depth(augmented_inverse_depth_hw)
    image_chw: UInt8[ndarray, "3 out_h out_w"] = np.ascontiguousarray(np.moveaxis(augmented_rgb_hw3, -1, 0))
    depth_1hw: UInt16[ndarray, "1 out_h out_w"] = np.ascontiguousarray(quantized_hw[None])
    mask_1hw: Bool[ndarray, "1 out_h out_w"] = np.ascontiguousarray(depth_1hw > 0)
    image_tensor: UInt8[Tensor, "3 out_h out_w"] = torch.from_numpy(image_chw)
    depth_tensor: UInt16[Tensor, "1 out_h out_w"] = torch.from_numpy(depth_1hw)
    mask_tensor: Bool[Tensor, "1 out_h out_w"] = torch.from_numpy(mask_1hw)
    sample: dict[str, Tensor] = {"image": image_tensor, "depth": depth_tensor, "mask": mask_tensor}
    if (prompt_depth_mm_hw is None) != (prompt_confidence_hw is None):
        raise ValueError("prompt depth and confidence must be provided together")
    if prompt_depth_mm_hw is not None and prompt_confidence_hw is not None:
        sample.update(_prompt_tensors(prompt_depth_mm_hw, prompt_confidence_hw, flipped=transformed[2]))
    return sample


def build_metric_training_sample(
    rgb_hw3: UInt8[ndarray, "h w 3"],
    target_depth_mm_hw: UInt16[ndarray, "h w"],
    prompt_depth_mm_hw: UInt16[ndarray, "192 256"],
    prompt_confidence_hw: UInt8[ndarray, "192 256"],
    transform: _DepthTransform,
) -> dict[str, Tensor]:
    """Build a metric prompted sample without inverse-depth quantization.

    Args:
        rgb_hw3: Oriented uint8 RGB with shape ``(H, W, 3)``.
        target_depth_mm_hw: Oriented dense PromptDA depth in millimetres with
            shape ``(H, W)``.
        prompt_depth_mm_hw: Oriented ARKit LiDAR depth in millimetres with
            native shape ``(192, 256)``.
        prompt_confidence_hw: Oriented ARKit confidence with native shape
            ``(192, 256)``.
        transform: Replayable aligned RGB/target augmentation.

    Returns:
        A dictionary with uint8 ``image``; float32 metre ``target_depth`` and
        ``prompt_depth``; and bool ``target_valid`` and ``prompt_valid``.
    """
    # Keep every real teacher value (0 mm is the invalid encoding): the loss windows L1 at
    # 4 m itself, and the gradient term may supervise edge pairs beyond it. The emitted
    # target_valid mask (recomputed post-augmentation below) still carries the [MIN, MAX]
    # window for L1, eval, and the flat-frame filter.
    target_depth_m_hw: Float32[ndarray, "h w"] = np.where(
        target_depth_mm_hw > 0,
        target_depth_mm_hw.astype(np.float32) / 1000.0,
        0.0,
    ).astype(np.float32)
    transformed: tuple[UInt8[ndarray, "out_h out_w 3"], Float32[ndarray, "out_h out_w"], bool] = _transform_with_flip_state(
        rgb_hw3, target_depth_m_hw, transform
    )
    augmented_rgb_hw3: UInt8[ndarray, "out_h out_w 3"] = transformed[0]
    augmented_target_depth_hw: Float32[ndarray, "out_h out_w"] = transformed[1]
    image_chw: UInt8[ndarray, "3 out_h out_w"] = np.ascontiguousarray(np.moveaxis(augmented_rgb_hw3, -1, 0))
    target_depth_1hw: Float32[ndarray, "1 out_h out_w"] = np.ascontiguousarray(augmented_target_depth_hw[None])
    target_valid_1hw: Bool[ndarray, "1 out_h out_w"] = np.ascontiguousarray(
        (target_depth_1hw >= MIN_DEPTH_MM / 1000.0) & (target_depth_1hw <= MAX_DEPTH_MM / 1000.0)
    )
    sample: dict[str, Tensor] = {
        "image": torch.from_numpy(image_chw),
        "target_depth": torch.from_numpy(target_depth_1hw),
        "target_valid": torch.from_numpy(target_valid_1hw),
    }
    sample.update(_prompt_tensors(prompt_depth_mm_hw, prompt_confidence_hw, flipped=transformed[2]))
    return sample


def build_training_sample_cuda(
    rgb_chw: UInt8[Tensor, "3 h w"],
    depth_mm_hw: Shaped[Tensor, "h w"],
    quarter_turns: int,
    out_hw: tuple[int, int],
    generator: torch.Generator,
    policy: AugmentPolicy,
    *,
    prompt_depth_mm_hw: Shaped[Tensor, "prompt_h prompt_w"] | None = None,
    prompt_confidence_hw: UInt8[Tensor, "prompt_h prompt_w"] | None = None,
    target_mode: TargetMode = "ssi",
) -> dict[str, Tensor]:
    """Build one augmented training sample without leaving CUDA.

    Rotation follows NumPy's counter-clockwise ``rot90`` convention. Spatial
    resizing matches the CPU path's bilinear RGB and nearest-neighbor inverse
    depth policy. The returned depth uses uint16 storage, but all arithmetic is
    performed in float32 or int32 because PyTorch has limited uint16 support.

    Args:
        rgb_chw: CUDA uint8 RGB with shape ``(3, H, W)``.
        depth_mm_hw: CUDA int32 or floating metric depth with shape ``(H, W)``.
        quarter_turns: Counter-clockwise rotations, normalized modulo four.
        out_hw: Output ``(height, width)``.
        generator: Random generator seeded for this sample. A CPU generator
            avoids a device synchronization when selecting augmentations.
        policy: Shared probabilities for flip, channel shuffle, and grayscale.

    Returns:
        CUDA tensors: uint8 ``image`` ``(3, h, w)``, uint16 ``depth``
        ``(1, h, w)``, and bool ``mask`` ``(1, h, w)``.

    Raises:
        ValueError: If inputs, output size, or probabilities violate the CUDA
            sample-building contract.
    """
    if rgb_chw.ndim != 3 or rgb_chw.shape[0] != 3 or rgb_chw.dtype != torch.uint8 or not rgb_chw.is_cuda:
        raise ValueError("rgb_chw must be a CUDA uint8 tensor with shape (3, H, W)")
    if depth_mm_hw.ndim != 2 or not depth_mm_hw.is_cuda or tuple(depth_mm_hw.shape) != tuple(rgb_chw.shape[-2:]):
        raise ValueError("depth_mm_hw must be a CUDA tensor aligned with rgb_chw")
    if depth_mm_hw.dtype not in (torch.int32, torch.float16, torch.float32, torch.float64, torch.bfloat16):
        raise ValueError("depth_mm_hw must use int32 or a floating dtype")
    out_height: int = out_hw[0]
    out_width: int = out_hw[1]
    if out_height <= 0 or out_width <= 0:
        raise ValueError("output height and width must be positive")
    turns: int = quarter_turns % 4
    rotated_rgb_chw: UInt8[Tensor, "3 rotated_h rotated_w"] = torch.rot90(rgb_chw, turns, dims=(-2, -1))
    rotated_depth_mm_hw: Shaped[Tensor, "rotated_h rotated_w"] = torch.rot90(depth_mm_hw, turns, dims=(-2, -1))
    depth_float_hw: Float32[Tensor, "rotated_h rotated_w"] = rotated_depth_mm_hw.to(dtype=torch.float32)
    valid_hw: Bool[Tensor, "rotated_h rotated_w"] = (depth_float_hw >= MIN_DEPTH_MM) & (depth_float_hw <= MAX_DEPTH_MM)
    if target_mode not in ("ssi", "metric"):
        raise ValueError(f"unknown target mode {target_mode!r}")
    if (prompt_depth_mm_hw is None) != (prompt_confidence_hw is None):
        raise ValueError("prompt depth and confidence must be provided together")
    if target_mode == "ssi":
        target_hw: Float32[Tensor, "rotated_h rotated_w"] = torch.where(
            valid_hw,
            1000.0 / depth_float_hw,
            torch.zeros_like(depth_float_hw),
        )
    else:
        # Keep every real teacher value (0 mm is the invalid encoding): the loss windows
        # L1 at 4 m itself, and the gradient term may supervise edge pairs beyond it.
        # The emitted validity mask still carries the [MIN, MAX] window.
        target_hw = torch.where(
            depth_float_hw > 0,
            depth_float_hw / 1000.0,
            torch.zeros_like(depth_float_hw),
        )
    resized_rgb_float_chw: Float32[Tensor, "3 out_h out_w"] = F.interpolate(
        rotated_rgb_chw.unsqueeze(0).to(dtype=torch.float32),
        size=(out_height, out_width),
        mode="bilinear",
        align_corners=False,
        antialias=False,
    ).squeeze(0)
    resized_rgb_chw: UInt8[Tensor, "3 out_h out_w"] = torch.round(resized_rgb_float_chw).clamp_(0.0, 255.0).to(dtype=torch.uint8)
    resized_target_hw: Float32[Tensor, "out_h out_w"] = F.interpolate(
        target_hw.unsqueeze(0).unsqueeze(0),
        size=(out_height, out_width),
        mode="nearest",
    ).squeeze(0).squeeze(0)
    prompt_depth_m_hw: Float32[Tensor, "192 256"] | None = None
    prompt_valid_hw: Bool[Tensor, "192 256"] | None = None
    if prompt_depth_mm_hw is not None and prompt_confidence_hw is not None:
        rotated_prompt_mm_hw: Shaped[Tensor, "oriented_prompt_h oriented_prompt_w"] = torch.rot90(
            prompt_depth_mm_hw, turns, dims=(-2, -1)
        )
        rotated_confidence_hw: UInt8[Tensor, "oriented_prompt_h oriented_prompt_w"] = torch.rot90(
            prompt_confidence_hw, turns, dims=(-2, -1)
        )
        if tuple(rotated_prompt_mm_hw.shape) != (192, 256) or tuple(rotated_confidence_hw.shape) != (192, 256):
            raise ValueError(
                f"oriented prompt depth and confidence must both be 192x256, got {tuple(rotated_prompt_mm_hw.shape)} and {tuple(rotated_confidence_hw.shape)}"
            )
        prompt_depth_m_hw = rotated_prompt_mm_hw.to(dtype=torch.float32) / 1000.0
        prompt_valid_hw = (
            (rotated_confidence_hw >= 1)
            & (rotated_prompt_mm_hw >= MIN_DEPTH_MM)
            & (rotated_prompt_mm_hw <= MAX_DEPTH_MM)
        )

    random_values_n: Float32[Tensor, "3"] = torch.rand(3, device=generator.device, generator=generator)
    random_values: list[float] = random_values_n.cpu().tolist()

    if random_values[0] < policy.flip_p:
        resized_rgb_chw = torch.flip(resized_rgb_chw, dims=(-1,))
        resized_target_hw = torch.flip(resized_target_hw, dims=(-1,))
        if prompt_depth_m_hw is not None and prompt_valid_hw is not None:
            prompt_depth_m_hw = torch.flip(prompt_depth_m_hw, dims=(-1,))
            prompt_valid_hw = torch.flip(prompt_valid_hw, dims=(-1,))

    if random_values[1] < policy.channel_shuffle_p:
        channel_order: Int64[Tensor, "3"] = torch.randperm(3, device=generator.device, generator=generator).to(device=rgb_chw.device)
        resized_rgb_chw = resized_rgb_chw[channel_order]

    if random_values[2] < policy.gray_p:
        rgb_float_chw: Float32[Tensor, "3 out_h out_w"] = resized_rgb_chw.to(dtype=torch.float32)
        gray_hw: Float32[Tensor, "out_h out_w"] = (
            0.299 * rgb_float_chw[0] + 0.587 * rgb_float_chw[1] + 0.114 * rgb_float_chw[2]
        )
        resized_rgb_chw = torch.round(gray_hw).clamp_(0.0, 255.0).to(dtype=torch.uint8).expand_as(resized_rgb_chw)

    image_chw: UInt8[Tensor, "3 out_h out_w"] = resized_rgb_chw.contiguous()
    if target_mode == "ssi":
        quantized_i32_hw: Shaped[Tensor, "out_h out_w"] = torch.round(resized_target_hw * INV_DEPTH_SCALE).clamp_(0.0, 65535.0).to(dtype=torch.int32)
        mask_1hw: Bool[Tensor, "1 out_h out_w"] = (quantized_i32_hw > 0).unsqueeze(0).contiguous()
        depth_1hw: UInt16[Tensor, "1 out_h out_w"] = quantized_i32_hw.to(dtype=torch.uint16).unsqueeze(0).contiguous()
        sample: dict[str, Tensor] = {"image": image_chw, "depth": depth_1hw, "mask": mask_1hw}
    else:
        target_depth_1hw: Float32[Tensor, "1 out_h out_w"] = resized_target_hw.unsqueeze(0).contiguous()
        target_valid_1hw: Bool[Tensor, "1 out_h out_w"] = (
            (target_depth_1hw >= MIN_DEPTH_MM / 1000.0) & (target_depth_1hw <= MAX_DEPTH_MM / 1000.0)
        )
        sample = {"image": image_chw, "target_depth": target_depth_1hw, "target_valid": target_valid_1hw}
    if prompt_depth_m_hw is not None and prompt_valid_hw is not None:
        sample["prompt_depth"] = prompt_depth_m_hw.unsqueeze(0).contiguous()
        sample["prompt_valid"] = prompt_valid_hw.unsqueeze(0).contiguous()
    return sample
