"""Utility functions for DPVO: timing, coordinate grids, patch extraction, and more.

This module provides low-level helpers used throughout the DPVO pipeline:

- :class:`Timer` -- CUDA-event-based timing context manager for profiling.
- :func:`coords_grid`, :func:`coords_grid_with_index` -- pixel coordinate
  grid generation for building patch representations.
- :func:`patchify` -- extract dense overlapping patches from feature maps
  using :func:`torch.nn.functional.unfold`.
- :func:`pyramidify` -- build a multi-scale feature pyramid via average
  pooling.
- :func:`flatmeshgrid` -- flattened Cartesian product of index tensors,
  used to enumerate measurement edges in the factor graph.
- :func:`all_pairs_exclusive` -- all ordered pairs ``(i, j)`` with ``i != j``.
- :func:`set_depth` -- write inverse-depth values into a patch tensor.
"""

from collections.abc import Generator
from types import TracebackType

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor

all_times: list[float] = []
"""Module-level accumulator for elapsed times recorded by :class:`Timer`."""


class Timer:
    """CUDA event-based timing context manager.

    Uses :class:`torch.cuda.Event` for accurate GPU timing that accounts
    for asynchronous kernel execution.  Elapsed time (in ms) is printed
    on exit and appended to the module-level :data:`all_times` list.

    Attributes:
        name: Label printed alongside the elapsed time.
        enabled: When False, no events are recorded and no output is
            produced.  Allows timing to be toggled without restructuring
            calling code.
    """

    def __init__(self, name: str, enabled: bool = True) -> None:
        self.name: str = name
        self.enabled: bool = enabled

        if self.enabled:
            self.start: torch.cuda.Event = torch.cuda.Event(enable_timing=True)
            self.end: torch.cuda.Event = torch.cuda.Event(enable_timing=True)

    def __enter__(self) -> "Timer":
        if self.enabled:
            self.start.record()
        return self

    def __exit__(
        self,
        type: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        global all_times
        if self.enabled:
            self.end.record()
            torch.cuda.synchronize()

            elapsed: float = self.start.elapsed_time(self.end)
            all_times.append(elapsed)
            print(f"{self.name}: {elapsed:.2f}ms")


def coords_grid(b: int, n: int, h: int, w: int, **kwargs: object) -> Float[Tensor, "b n 2 h w"]:
    """Create a batch of 2-D pixel coordinate grids.

    Returns a tensor where ``result[b, n, 0, y, x] = x`` and
    ``result[b, n, 1, y, x] = y``, i.e. channel 0 is the horizontal
    (column) coordinate and channel 1 is the vertical (row) coordinate.

    Args:
        b: Batch size.
        n: Number of images / frames.
        h: Grid height (pixels).
        w: Grid width (pixels).
        **kwargs: Passed to :func:`torch.arange` (e.g. ``device``).

    Returns:
        Coordinate grid of shape ``(b, n, 2, h, w)``.
    """
    x: Float[Tensor, "w"] = torch.arange(0, w, dtype=torch.float, **kwargs)
    y: Float[Tensor, "h"] = torch.arange(0, h, dtype=torch.float, **kwargs)
    coords: Float[Tensor, "2 h w"] = torch.stack(torch.meshgrid(y, x, indexing="ij"))
    # Swap from (y, x) to (x, y) ordering to match DPVO's (u, v) convention
    result: Float[Tensor, "b n 2 h w"] = coords[[1, 0]].view(1, 1, 2, h, w).repeat(b, n, 1, 1, 1)
    return result


def coords_grid_with_index(
    d: Float[Tensor, "b n h w"], **kwargs: object
) -> tuple[Float[Tensor, "b n 3 h w"], Float[Tensor, "b n 1 h w"]]:
    """Create a 3-channel coordinate grid ``(x, y, inverse_depth)`` plus a frame index volume.

    This is used by the :class:`~dpvo.net.Patchifier` to build the
    initial patch representation.  Each pixel gets coordinates ``(x, y, d)``
    where ``d`` comes from the input disparity map, and a frame index
    channel that identifies which frame the pixel belongs to.

    Args:
        d: Inverse-depth (disparity) map of shape ``(b, n, h, w)``.
        **kwargs: Passed to :func:`torch.arange` (e.g. ``device``).

    Returns:
        A 2-tuple of:
        - ``coords``: shape ``(b, n, 3, h, w)`` with channels ``(x, y, d)``.
        - ``index``: shape ``(b, n, 1, h, w)`` with the frame index
          (0, 1, ..., n-1) broadcast over spatial dimensions.
    """
    b: int
    n: int
    h: int
    w: int
    b, n, h, w = d.shape
    _i: Float[Tensor, "b n h w"] = torch.ones_like(d)
    x: Float[Tensor, "w"] = torch.arange(0, w, dtype=torch.float, **kwargs)
    y: Float[Tensor, "h"] = torch.arange(0, h, dtype=torch.float, **kwargs)

    y, x = torch.stack(torch.meshgrid(y, x, indexing="ij"))
    y: Float[Tensor, "b n h w"] = y.view(1, 1, h, w).repeat(b, n, 1, 1)
    x: Float[Tensor, "b n h w"] = x.view(1, 1, h, w).repeat(b, n, 1, 1)

    # Stack (x, y, inverse_depth) to form the 3-channel patch coordinate
    coords: Float[Tensor, "b n 3 h w"] = torch.stack([x, y, d], dim=2)
    index: Float[Tensor, "b n 1 h w"] = torch.arange(0, n, dtype=torch.float, **kwargs).view(1, n, 1, 1, 1).repeat(b, 1, 1, h, w)

    return coords, index


def patchify(x: Float[Tensor, "b n c h w"], patch_size: int = 3) -> Float[Tensor, "b n_patches c ps ps"]:
    """Extract all overlapping patches from a feature map using ``unfold``.

    Each spatial location produces one ``(c, patch_size, patch_size)``
    patch.  This is used to build the dense patch tensor from coordinate
    grids during patchification.

    Args:
        x: Input feature map of shape ``(b, n, c, h, w)``.
        patch_size: Side length of each square patch (default 3, matching
            DPVO's 3x3 patches).

    Returns:
        Tensor of shape ``(b, n_patches, c, patch_size, patch_size)``
        where ``n_patches = n * (h - patch_size + 1) * (w - patch_size + 1)``.
    """
    b: int
    n: int
    c: int
    h: int
    w: int
    b, n, c, h, w = x.shape
    x_reshaped: Float[Tensor, "bn c h w"] = x.view(b * n, c, h, w)
    # unfold extracts all patch_size x patch_size patches as columns
    y: Float[Tensor, "bn patches_times_c patch_elements"] = F.unfold(x_reshaped, patch_size)
    y_t: Float[Tensor, "bn n_patches patch_elements"] = y.transpose(1, 2)
    result: Float[Tensor, "b n_patches c ps ps"] = y_t.reshape(b, -1, c, patch_size, patch_size)
    return result


def pyramidify(fmap: Float[Tensor, "b n c h w"], lvls: list[int] | None = None) -> list[Float[Tensor, "b n c h_l w_l"]]:
    """Build a multi-scale feature pyramid via average pooling.

    The correlation volume in DPVO is computed at two scales: stride-4
    (``lvl=1``, i.e. the original feature map) and stride-16 (``lvl=4``).
    This yields a ``(2R+1)^2 x 2`` correlation feature vector per edge.
    See Sec. 3.2 of Teed et al. (2022).

    Args:
        fmap: Feature map of shape ``(b, n, c, h, w)`` at the base
            resolution (stride-4 relative to the input image).
        lvls: List of pooling factors.  ``lvl=1`` keeps the original
            resolution; ``lvl=4`` pools 4x to stride-16.

    Returns:
        List of feature maps, one per level, each of shape
        ``(b, n, c, h // lvl, w // lvl)``.
    """
    b: int
    n: int
    c: int
    h: int
    w: int
    if lvls is None:
        lvls = [1]
    b, n, c, h, w = fmap.shape

    pyramid: list[Float[Tensor, "b n c h_l w_l"]] = []
    for lvl in lvls:
        gmap: Float[Tensor, "bn c h_l w_l"] = F.avg_pool2d(fmap.view(b * n, c, h, w), lvl, stride=lvl)
        pyramid += [gmap.view(b, n, c, h // lvl, w // lvl)]

    return pyramid


def all_pairs_exclusive(n: int, **kwargs: object) -> tuple[Int[Tensor, "n_pairs"], Int[Tensor, "n_pairs"]]:
    """Generate all ordered pairs ``(i, j)`` with ``i != j`` for ``i, j in 0..n-1``.

    Used during training to create a fully-connected measurement graph
    (excluding self-loops) between frames.

    Args:
        n: Number of elements.
        **kwargs: Passed to :func:`torch.arange` (e.g. ``device``).

    Returns:
        A 2-tuple ``(ii, jj)`` each of shape ``(n*(n-1),)`` containing
        the source and target indices.
    """
    ii: Int[Tensor, "n n"]
    jj: Int[Tensor, "n n"]
    ii, jj = torch.meshgrid(torch.arange(n, **kwargs), torch.arange(n, **kwargs))
    k: torch.Tensor = ii != jj
    return ii[k].reshape(-1), jj[k].reshape(-1)


def set_depth(
    patches: Float[Tensor, "*batch 3 ps ps"], depth: Float[Tensor, "*batch"]
) -> Float[Tensor, "*batch 3 ps ps"]:
    """Set the inverse-depth channel of a patch tensor to a uniform value.

    Each patch has three channels ``(x, y, inverse_depth)``.  This function
    broadcasts a per-patch scalar ``depth`` across the spatial dimensions
    of the inverse-depth channel (channel index 2).

    Args:
        patches: Patch tensor with shape ``(..., 3, ps, ps)``.
        depth: Scalar inverse-depth per patch, shape ``(...)``.

    Returns:
        The modified ``patches`` tensor (mutated in-place and returned).
    """
    patches[..., 2, :, :] = depth[..., None, None]
    return patches


def flatmeshgrid(*args: torch.Tensor, **kwargs: object) -> Generator[torch.Tensor, None, None]:
    """Flattened Cartesian product of 1-D index tensors.

    Equivalent to :func:`torch.meshgrid` followed by ``.reshape(-1)`` on
    each output.  Used extensively to enumerate measurement edges: for
    example, ``flatmeshgrid(patch_indices, frame_indices)`` yields all
    ``(patch, frame)`` pairs as flat index vectors.

    Args:
        *args: 1-D tensors whose Cartesian product is computed.
        **kwargs: Passed to :func:`torch.meshgrid` (e.g.
            ``indexing="ij"``).

    Yields:
        One flattened tensor per input, each of length equal to the
        product of input lengths.
    """
    grid: tuple[torch.Tensor, ...] = tuple(torch.meshgrid(*args, **kwargs))
    return (x.reshape(-1) for x in grid)
