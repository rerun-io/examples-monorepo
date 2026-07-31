from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from numbers import Number
from typing import TypeAlias

import torch
from jaxtyping import Bool, Float
from torch import Tensor

from monopriors.third_party.moge.utils.geometry_torch import intrinsics_from_focal_center, recover_focal_shift

TensorValue: TypeAlias = Float[Tensor, "*shape"] | Bool[Tensor, "*shape"]
InferenceOutput: TypeAlias = dict[str, TensorValue]
FieldOfView: TypeAlias = Number | Float[Tensor, ""] | Float[Tensor, "b"] | None


@dataclass(frozen=True, slots=True)
class InferenceInput:
    """Normalized batched model input and its image geometry."""

    image_bchw: Float[Tensor, "b 3 h w"]
    """Input image with a batch dimension on the model device and dtype."""
    omit_batch_dim: bool
    """Whether the caller supplied an unbatched image."""
    height: int
    """Original input height in pixels."""
    width: int
    """Original input width in pixels."""
    aspect_ratio: float
    """Image width divided by height."""


@dataclass(frozen=True, slots=True)
class CameraRecovery:
    """Recovered camera-space translation and normalized intrinsics."""

    shift_b: Float[Tensor, "b"]
    """Per-image Z-axis translation."""
    intrinsics_b33: Float[Tensor, "b 3 3"]
    """Per-image normalized OpenCV intrinsics."""


@dataclass(frozen=True, slots=True)
class MaskedGeometry:
    """Point and depth tensors after applying a validity mask."""

    points_bhw3: Float[Tensor, "b h w 3"]
    """Masked camera-space point map."""
    depth_bhw: Float[Tensor, "b h w"]
    """Masked depth map."""


def prepare_inference_input(
    image: Float[Tensor, "3 h w"] | Float[Tensor, "b 3 h w"],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> InferenceInput:
    """Batch and move an inference image without changing its values.

    Args:
        image: Float tensor shaped ``3 h w`` or ``b 3 h w``.
        device: Destination model device.
        dtype: Destination model dtype.

    Returns:
        Batched image and original image geometry.
    """
    omit_batch_dim: bool = image.dim() == 3
    image_bchw: Float[Tensor, "b 3 h w"] = image.unsqueeze(0) if omit_batch_dim else image
    image_bchw = image_bchw.to(dtype=dtype, device=device)
    original_height: int = image_bchw.shape[-2]
    original_width: int = image_bchw.shape[-1]
    aspect_ratio: float = original_width / original_height
    return InferenceInput(
        image_bchw=image_bchw,
        omit_batch_dim=omit_batch_dim,
        height=original_height,
        width=original_width,
        aspect_ratio=aspect_ratio,
    )


def select_num_tokens(
    num_tokens_range: tuple[Number, Number],
    *,
    resolution_level: int,
    num_tokens: int | None,
) -> int:
    """Resolve an explicit or resolution-derived inference token count.

    Args:
        num_tokens_range: Inclusive training-time token-count range.
        resolution_level: Detail level from 0 through 9.
        num_tokens: Explicit token count, or ``None`` to derive one.

    Returns:
        Token count passed to the model forward method.
    """
    if num_tokens is not None:
        return num_tokens
    min_tokens: Number = num_tokens_range[0]
    max_tokens: Number = num_tokens_range[1]
    return int(min_tokens + (resolution_level / 9) * (max_tokens - min_tokens))


@contextmanager
def inference_autocast(*, device: torch.device, dtype: torch.dtype, use_fp16: bool) -> Iterator[None]:
    """Enter the shared mixed-precision inference context.

    Args:
        device: Model device controlling the autocast backend.
        dtype: Model parameter dtype.
        use_fp16: Whether callers requested float16 mixed precision.

    Yields:
        Control while the matching Torch autocast context is active.
    """
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_fp16 and dtype != torch.float16):
        yield


def recover_shift_and_intrinsics(
    points_bhw3: Float[Tensor, "b h w 3"],
    mask_bhw: Bool[Tensor, "b h w"] | None,
    *,
    fov_x: FieldOfView,
    aspect_ratio: float,
) -> CameraRecovery:
    """Recover the point-map Z shift and normalized camera intrinsics.

    Args:
        points_bhw3: Affine point map as a float tensor shaped ``b h w 3``.
        mask_bhw: Optional validity mask as a bool tensor shaped ``b h w``.
        fov_x: Optional horizontal field of view in degrees, scalar or batch-shaped.
        aspect_ratio: Image width divided by height.

    Returns:
        Recovered per-image shift and normalized intrinsics.
    """
    focal_b: Float[Tensor, "b"]
    shift_b: Float[Tensor, "b"]
    if fov_x is None:
        focal_b, shift_b = recover_focal_shift(points_bhw3, mask_bhw)
    else:
        focal_b = (
            aspect_ratio
            / (1 + aspect_ratio**2) ** 0.5
            / torch.tan(torch.deg2rad(torch.as_tensor(fov_x, device=points_bhw3.device, dtype=points_bhw3.dtype) / 2))
        )
        if focal_b.ndim == 0:
            focal_b = focal_b[None].expand(points_bhw3.shape[0])
        _, shift_b = recover_focal_shift(points_bhw3, mask_bhw, focal=focal_b)
    fx_b: Float[Tensor, "b"] = focal_b / 2 * (1 + aspect_ratio**2) ** 0.5 / aspect_ratio
    fy_b: Float[Tensor, "b"] = focal_b / 2 * (1 + aspect_ratio**2) ** 0.5
    center_x: Float[Tensor, ""] = torch.tensor(0.5, device=points_bhw3.device, dtype=points_bhw3.dtype)
    center_y: Float[Tensor, ""] = torch.tensor(0.5, device=points_bhw3.device, dtype=points_bhw3.dtype)
    intrinsics_b33: Float[Tensor, "b 3 3"] = intrinsics_from_focal_center(fx_b, fy_b, center_x, center_y)
    return CameraRecovery(shift_b=shift_b, intrinsics_b33=intrinsics_b33)


def mask_depth_and_points(
    points_bhw3: Float[Tensor, "b h w 3"],
    depth_bhw: Float[Tensor, "b h w"],
    mask_bhw: Bool[Tensor, "b h w"],
) -> MaskedGeometry:
    """Apply a shared validity mask to point and depth maps.

    Args:
        points_bhw3: Float point map shaped ``b h w 3``.
        depth_bhw: Float depth map shaped ``b h w``.
        mask_bhw: Bool validity mask shaped ``b h w``.

    Returns:
        Point and depth maps with invalid pixels filled by infinity.
    """
    masked_points_bhw3: Float[Tensor, "b h w 3"] = torch.where(mask_bhw[..., None], points_bhw3, torch.inf)
    masked_depth_bhw: Float[Tensor, "b h w"] = torch.where(mask_bhw, depth_bhw, torch.inf)
    return MaskedGeometry(points_bhw3=masked_points_bhw3, depth_bhw=masked_depth_bhw)


def finalize_inference_output(output: dict[str, TensorValue | None], *, omit_batch_dim: bool) -> InferenceOutput:
    """Drop unavailable outputs and restore an optional omitted batch dimension.

    Args:
        output: Ordered inference outputs, including optional unavailable values.
        omit_batch_dim: Whether to squeeze the leading batch dimension.

    Returns:
        Public inference dictionary containing only tensors.
    """
    available_output: InferenceOutput = {key: value for key, value in output.items() if value is not None}
    if omit_batch_dim:
        available_output = {key: value.squeeze(0) for key, value in available_output.items()}
    return available_output
