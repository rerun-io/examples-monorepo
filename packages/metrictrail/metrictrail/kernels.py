"""Smoke checks of the CUDA kernels that `recipes/vipe-src` compiles at package build time.

Each check runs one tiny op of a compiled module on the GPU and returns its largest absolute error against a
plain-torch reference, so a wrong build (missing architecture, broken ABI, wrong module) fails loudly.

- ``vipe.metrictrail.dpvo_ext.cuda_corr``: ``patchify_forward`` (DPVO patch extraction).
- ``vipe.metrictrail.dpvo_ext.cuda_ba``: ``reproject`` (DPVO patch reprojection under a relative pose).
- ``vipe_ext.scatter_ext``: ``scatter_max`` (ViPE's own extension, used by its keyframe SLAM).

None of these import lietorch, so they run before a lietorch provider is installed.
"""

import importlib
from dataclasses import dataclass
from types import ModuleType

import torch
from jaxtyping import Float32, Int64
from torch import Tensor

VIPE_EXT_SUBMODULES: tuple[str, ...] = ("droid_net_ext", "grounding_dino_ext", "utils_ext", "slam_ext", "scatter_ext", "corr_ext")
"""The vipe_ext submodules the recipe builds."""
VIPE_EXT_DROPPED: tuple[str, ...] = ("lietorch_ext",)
"""The vipe_ext submodule the recipe deliberately does not build (patch 0001: lietorch comes from lietorch-mojo)."""


@dataclass(frozen=True, slots=True)
class KernelCheck:
    """Result of one kernel smoke check."""

    module: str
    """Fully qualified compiled module."""
    op: str
    """The op that ran."""
    max_abs_error: float
    """Largest absolute difference from the torch reference."""


def _import(name: str) -> ModuleType:
    return importlib.import_module(name)


def check_patchify(device: torch.device) -> KernelCheck:
    """``cuda_corr.patchify_forward`` against direct indexing (integer coordinates, radius 1)."""
    cuda_corr: ModuleType = _import("vipe.metrictrail.dpvo_ext.cuda_corr")
    generator: torch.Generator = torch.Generator().manual_seed(0)
    net: Float32[Tensor, "b c h w"] = torch.rand(1, 4, 12, 16, generator=generator).to(device)
    coords_xy: Float32[Tensor, "b m 2"] = torch.tensor([[[5.0, 4.0], [10.0, 7.0], [3.0, 8.0]]], device=device)
    radius: int = 1
    patches: Float32[Tensor, "b m c d d"] = cuda_corr.patchify_forward(net, coords_xy, radius)[0]
    diameter: int = 2 * radius + 2
    expected: Float32[Tensor, "b m c d d"] = torch.stack(
        [net[0, :, int(y) - radius : int(y) - radius + diameter, int(x) - radius : int(x) - radius + diameter] for x, y in coords_xy[0].tolist()]
    )[None]
    return KernelCheck("vipe.metrictrail.dpvo_ext.cuda_corr", "patchify_forward", float((patches - expected).abs().max()))


def check_reproject(device: torch.device) -> KernelCheck:
    """``cuda_ba.reproject`` of 3x3 patches from frame 0 (identity) into frame 1 (pure translation)."""
    cuda_ba: ModuleType = _import("vipe.metrictrail.dpvo_ext.cuda_ba")
    fx, fy, cx, cy = 200.0, 210.0, 64.0, 48.0
    translation: Float32[Tensor, "3"] = torch.tensor([0.1, -0.05, 0.2])
    # DPVO layout: [tx, ty, tz, qx, qy, qz, qw]; world-to-camera, so the relative pose of frame 1 to 0 is pose 1.
    poses: Float32[Tensor, "1 n 7"] = torch.tensor([[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [*translation.tolist(), 0.0, 0.0, 0.0, 1.0]]])
    generator: torch.Generator = torch.Generator().manual_seed(0)
    patch_size: int = 3
    num_patches: int = 4
    pixel_x: Float32[Tensor, "m p p"] = 20.0 + 80.0 * torch.rand(num_patches, patch_size, patch_size, generator=generator)
    pixel_y: Float32[Tensor, "m p p"] = 10.0 + 70.0 * torch.rand(num_patches, patch_size, patch_size, generator=generator)
    inverse_depth: Float32[Tensor, "m p p"] = 0.2 + 0.8 * torch.rand(num_patches, patch_size, patch_size, generator=generator)
    patches: Float32[Tensor, "1 m 3 p p"] = torch.stack([pixel_x, pixel_y, inverse_depth], dim=1)[None]
    intrinsics: Float32[Tensor, "1 n 4"] = torch.tensor([[[fx, fy, cx, cy], [fx, fy, cx, cy]]])
    ii: Int64[Tensor, "e"] = torch.zeros(num_patches, dtype=torch.int64)
    jj: Int64[Tensor, "e"] = torch.ones(num_patches, dtype=torch.int64)
    kk: Int64[Tensor, "e"] = torch.arange(num_patches, dtype=torch.int64)
    coords: Float32[Tensor, "1 e 2 p p"] = cuda_ba.reproject(
        poses.to(device), patches.to(device), intrinsics.to(device), ii.to(device), jj.to(device), kk.to(device)
    )
    # X_j = X_i + t * w for the homogeneous point X_i = [(u - cx) / fx, (v - cy) / fy, 1, w].
    depth_j: Float32[Tensor, "m p p"] = 1.0 + translation[2] * inverse_depth
    expected_x: Float32[Tensor, "m p p"] = fx * ((pixel_x - cx) / fx + translation[0] * inverse_depth) / depth_j + cx
    expected_y: Float32[Tensor, "m p p"] = fy * ((pixel_y - cy) / fy + translation[1] * inverse_depth) / depth_j + cy
    expected: Float32[Tensor, "1 e 2 p p"] = torch.stack([expected_x, expected_y], dim=1)[None]
    return KernelCheck("vipe.metrictrail.dpvo_ext.cuda_ba", "reproject", float((coords.cpu() - expected).abs().max()))


def check_scatter_max(device: torch.device) -> KernelCheck:
    """``vipe_ext.scatter_ext.scatter_max`` against ``torch.Tensor.scatter_reduce(amax)``."""
    vipe_ext: ModuleType = _import("vipe_ext")
    generator: torch.Generator = torch.Generator().manual_seed(0)
    src: Float32[Tensor, "n"] = torch.rand(64, generator=generator).to(device)
    index: Int64[Tensor, "n"] = torch.randint(0, 8, (64,), generator=generator).to(device)
    result: tuple[Float32[Tensor, "k"], Int64[Tensor, "k"]] = vipe_ext.scatter_ext.scatter_max(src, index, 0, None, 8)
    expected: Float32[Tensor, "k"] = torch.zeros(8, device=device).scatter_reduce(0, index, src, reduce="amax", include_self=False)
    return KernelCheck("vipe_ext.scatter_ext", "scatter_max", float((result[0] - expected).abs().max()))


def missing_vipe_ext_submodules() -> list[str]:
    """Names in :data:`VIPE_EXT_SUBMODULES` that the installed ``vipe_ext`` lacks."""
    vipe_ext: ModuleType = _import("vipe_ext")
    return [name for name in VIPE_EXT_SUBMODULES if not hasattr(vipe_ext, name)]


def run_all(device: torch.device) -> list[KernelCheck]:
    """Every kernel check, in build order."""
    return [check_scatter_max(device), check_patchify(device), check_reproject(device)]
