"""Released SAM/MHR rotations with fixed-size GPU indexing."""

import math
from typing import Literal

import roma
import torch
from jaxtyping import Float
from sam_3d_body.models.modules.mhr_utils import batchXYZfrom6D
from torch import Tensor, nn

from posekit.ops.compile import compile_unless_dev


@compile_unless_dev
def rotmat_to_euler(convention: str, rotation: Float[Tensor, "... 3 3"]) -> Float[Tensor, "... 3"]:
    """RoMa's asymmetric Euler conversion, with ``where`` in its singular branches."""
    extrinsic: bool = convention.islower()
    order: str = convention.lower() if extrinsic else convention.lower()[::-1]
    i, j, k = ("xyz".index(axis) for axis in order)
    if len(set(order)) != 3:
        raise ValueError("SAM body requires three distinct Euler axes.")
    sign: int = (i - j) * (j - k) * (k - i) // 2
    q: Tensor = roma.rotmat_to_unitquat(rotation)
    a: Tensor = q[..., 3] - q[..., j]
    b: Tensor = q[..., i] + q[..., k] * sign
    c: Tensor = q[..., j] + q[..., 3]
    d: Tensor = q[..., k] * sign - q[..., i]
    middle: Tensor = 2 * torch.atan2(torch.hypot(c, d), torch.hypot(a, b))
    near_zero: Tensor = middle.abs() <= 1e-7
    near_pi: Tensor = (middle - math.pi).abs() <= 1e-7
    half_sum: Tensor = torch.atan2(b, a)
    half_diff: Tensor = torch.atan2(d, c)
    first: Tensor = half_sum - half_diff if extrinsic else half_sum + half_diff
    third: Tensor = half_sum + half_diff if extrinsic else half_sum - half_diff
    third = torch.where(near_zero | near_pi, 0.0, third)
    first = torch.where(near_zero, 2 * half_sum, first)
    first = torch.where(near_pi, 2 * (-1 if extrinsic else 1) * half_diff, first)
    result: Tensor = torch.stack((first if extrinsic else first * sign, middle - math.pi / 2, third * sign if extrinsic else third), -1)
    result = torch.where(result < -math.pi, result + 2 * math.pi, result)
    return torch.where(result > math.pi, result - 2 * math.pi, result)


class CompactPose(nn.Module):
    """Convert SAM's continuous body/hand parameters using cached gather indices."""

    def __init__(self, kind: Literal["body", "hand"], device: torch.device) -> None:
        super().__init__()
        self.kind = kind
        if kind == "body":
            triples: list[tuple[int, int, int]] = [
                (0, 2, 4), (6, 8, 10), (12, 13, 14), (15, 16, 17), (18, 19, 20), (21, 22, 23),
                (24, 25, 26), (27, 28, 29), (34, 35, 36), (37, 38, 39), (44, 45, 46), (53, 54, 55),
                (64, 65, 66), (85, 69, 73), (86, 70, 79), (87, 71, 82), (88, 72, 76), (91, 92, 93),
                (112, 96, 100), (113, 97, 106), (114, 98, 109), (115, 99, 103), (130, 131, 132),
            ]
            one: list[int] = [
                1, 3, 5, 7, 9, 11, 30, 31, 32, 33, 40, 41, 42, 43, 47, 48, 49, 50, 51, 52,
                56, 57, 58, 59, 60, 61, 62, 63, 67, 68, 74, 75, 77, 78, 80, 81, 83, 84, 89, 90,
                94, 95, 101, 102, 104, 105, 107, 108, 110, 111, 116, 117, 118, 119, 120, 121, 122, 123,
            ]
            three: list[int] = [index for triple in triples for index in triple]
            translation: list[int] = list(range(124, 130))
            self.three_count: int = len(three)
            self.one_count: int = len(one)
            self.restore: Tensor = torch.tensor(three + one + translation, device=device).argsort()
            self.input_three: Tensor = torch.arange(2 * len(three), device=device)
            self.input_one: Tensor = torch.arange(2 * len(three), 2 * (len(three) + len(one)), device=device)
        else:
            dofs: tuple[int, ...] = (3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 2, 3, 1, 1)
            three, one = [], []
            start: int = 0
            for width in dofs:
                (three if width == 3 else one).extend(range(start, start + width))
                start += width
            self.three_count = len(three)
            self.one_count = len(one)
            self.restore = torch.tensor(three + one, device=device).argsort()
            self.input_three = torch.tensor([2 * index + offset for index in three for offset in range(2)], device=device)
            self.input_one = torch.tensor([2 * index + offset for index in one for offset in range(2)], device=device)

    @compile_unless_dev
    def forward(self, continuous: Float[Tensor, "b c"]) -> Float[Tensor, "b p"]:
        """Use the released Gram-Schmidt and atan2 conversions, then restore parameter order."""
        three: Tensor = batchXYZfrom6D(continuous[:, self.input_three].unflatten(-1, (-1, 6))).flatten(1)
        pairs: Tensor = continuous[:, self.input_one].unflatten(-1, (-1, 2))
        one: Tensor = torch.atan2(pairs[..., 0], pairs[..., 1])
        pieces: list[Tensor] = [three, one]
        if self.kind == "body":
            pieces.append(continuous[:, 2 * (self.three_count + self.one_count) :])
        return torch.cat(pieces, -1)[:, self.restore]
