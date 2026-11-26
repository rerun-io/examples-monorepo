"""Strided pixel sampling (PennyGS 3.4) without TRiGS: the offsets, the shared per-step state and the intrinsics.

A strided step renders and supervises full-size pixels ``(oy + STRIDE * i, ox + STRIDE * j)``; the trainer
(``open4dgs.trigs_trainer``) and the MoGe-2 prior loss (``open4dgs.prior``) read the step's offset from one ``StrideState``.
"""

from dataclasses import dataclass

from jaxtyping import Float32
from torch import Tensor

STRIDE: int = 2
"""Strided pixel sampling: one pixel of every STRIDE x STRIDE block per step."""

STRIDE_OFFSETS: tuple[tuple[int, int], ...] = ((0, 0), (1, 1), (0, 1), (1, 0))
"""(row, column) offsets inside the block, one per step in turn: every pixel is supervised once every four steps."""

GSPLAT_EPS2D: float = 0.3
"""gsplat's default 2D dilation (pixels squared) added to every projected covariance."""


@dataclass
class StrideState:
    """The strided offset of the step being trained, None outside a strided step (evaluation renders full size)."""

    offset: tuple[int, int] | None = None


def strided_intrinsics(Ks: Float32[Tensor, "c 3 3"], offset: tuple[int, int], stride: int = STRIDE) -> Float32[Tensor, "c 3 3"]:
    """Intrinsics whose pixel q is full-size pixel ``stride * q + offset`` (see ``strided_camera_config``)."""
    oy, ox = offset
    out: Tensor = Ks.clone()
    out[:, 0, 0] /= stride
    out[:, 1, 1] /= stride
    out[:, 0, 2] = (out[:, 0, 2] - ox - 0.5) / stride + 0.5
    out[:, 1, 2] = (out[:, 1, 2] - oy - 0.5) / stride + 0.5
    return out


def strided_pixels(x: Tensor, offset: tuple[int, int]) -> Tensor:
    """The (h, w, ...) image or map ``x`` at the pixels a step with ``offset`` renders (a view)."""
    oy, ox = offset
    return x[oy::STRIDE, ox::STRIDE]
