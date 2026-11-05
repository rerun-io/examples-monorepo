"""Sparse bilinear feature sampling with parallel work across image channels."""

from jaxtyping import Float32
from torch import Tensor

from posekit.ops.compile import compile_unless_dev


@compile_unless_dev
def sample_keypoint_features(
    features: Float32[Tensor, "b c h w"], points: Float32[Tensor, "b k 2"],
) -> Float32[Tensor, "b k c"]:
    """Match zero-padded, align_corners=False grid_sample for sparse query points.

    The dense image sampler assigns a thread to each output pixel and loops
    over channels. Seventy query points leave that kernel mostly idle. These
    gathers expose the channel dimension to Torch's fused kernel instead.
    """
    height, width = features.shape[-2:]
    x: Tensor = ((points[..., 0] + 1) * width - 1) / 2
    y: Tensor = ((points[..., 1] + 1) * height - 1) / 2
    left: Tensor = x.floor().long()
    top: Tensor = y.floor().long()
    right: Tensor = left + 1
    bottom: Tensor = top + 1
    flat: Tensor = features.flatten(2)

    def corner(col: Tensor, row: Tensor, weight: Tensor) -> Tensor:
        valid: Tensor = (col >= 0) & (col < width) & (row >= 0) & (row < height)
        index: Tensor = (row.clamp(0, height - 1) * width + col.clamp(0, width - 1))[:, None].expand(-1, features.shape[1], -1)
        return flat.gather(2, index) * (weight * valid)[:, None]

    result: Tensor = corner(left, top, (right - x) * (bottom - y))
    result = result + corner(right, top, (x - left) * (bottom - y))
    result = result + corner(left, bottom, (right - x) * (y - top))
    result = result + corner(right, bottom, (x - left) * (y - top))
    return result.transpose(1, 2).contiguous()
