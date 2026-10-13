"""Pixel-aligned 3D Gaussians from RGB-D views: initialise, rasterise (gsplat) and optimise against the views' colours."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeAlias

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float32, Float64
from numpy import ndarray
from torch import Tensor
from torchmetrics.image import StructuralSimilarityIndexMeasure

from vistadream.ops.trajectory import backproject
from vistadream.types import CameraView, Splats

Render: TypeAlias = tuple[Float32[Tensor, "h w 3"], Float32[Tensor, "h w"], Float32[Tensor, "h w"]]
"""Colour, expected depth and alpha of one view."""
Rasterize: TypeAlias = Callable[[Splats, Float64[ndarray, "3 3"], Float64[ndarray, "4 4"], int, int], Render]
"""``(splats, K, cam_T_world, width, height) -> Render``; nodes take one so unit tests run without a GPU."""
LEARNING_RATES: dict[str, float] = {"means": 1e-4, "rgb_logits": 5e-4, "log_scales": 5e-3, "opacity_logits": 5e-2, "quats_wxyz": 1e-3}
"""Adam step size per parameter (VistaDream's, after pointrix-project/msplat)."""


@dataclass(slots=True)
class Trainable:
    """Raw (pre-activation) parameters of one view's Gaussians, the tensors Adam updates."""

    means: Float32[Tensor, "n 3"]
    log_scales: Float32[Tensor, "n 3"]
    quats_wxyz: Float32[Tensor, "n 4"]
    """Not normalised; rendering normalises."""
    rgb_logits: Float32[Tensor, "n 3"]
    opacity_logits: Float32[Tensor, " n"]

    def activated(self) -> Splats:
        return Splats(self.means, self.log_scales.exp(), F.normalize(self.quats_wxyz, dim=1), self.rgb_logits.sigmoid(), self.opacity_logits.sigmoid())


def gaussians(view: CameraView, spawn: Bool[ndarray, "h w"], device: str) -> Trainable:
    """One Gaussian per ``spawn`` pixel at its back-projected depth (the first row when fewer than 3): isotropic, sized to a
    pixel's footprint (``depth / fx / sqrt 2``), identity rotation, opacity 0.8, the pixel's colour."""
    assert view.depth is not None, "a training view carries depth"
    if spawn.sum() < 3:
        spawn = np.zeros_like(spawn)
        spawn[0] = True
    world_T_cam: Float64[ndarray, "4 4"] = np.linalg.inv(view.cam_T_world)
    points: Float64[ndarray, "n 3"] = backproject(view.depth, view.K)[spawn] @ world_T_cam[:3, :3].T + world_T_cam[:3, 3]
    scales: Float64[ndarray, "n"] = view.depth[spawn] / view.K[0, 0] / np.sqrt(2)
    n: int = len(points)

    def tensor(array: ndarray, deactivate: Callable[[Tensor], Tensor] = torch.clone) -> Float32[Tensor, "..."]:
        return deactivate(torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32)).to(device)).requires_grad_(True)

    return Trainable(
        tensor(points),
        tensor(np.repeat(scales[:, None], 3, axis=1), torch.log),
        tensor(np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))),
        tensor(view.rgb[spawn].astype(np.float32) / 255.0, torch.logit),  # 0 and 255 become -inf / inf; sigmoid maps them back
        tensor(np.full(n, 0.8), torch.logit),
    )


def merge(parts: Sequence[Splats]) -> Splats:
    return Splats(*(torch.cat([getattr(p, f) for p in parts]) for f in ("centers", "scales", "quats_wxyz", "rgb", "opacity")))


def render(splats: Splats, K: Float64[ndarray, "3 3"], cam_T_world: Float64[ndarray, "4 4"], width: int, height: int) -> Render:
    """gsplat's rasteriser (RGB + expected depth, black background, near plane 1 cm)."""
    import gsplat

    device: torch.device = splats.centers.device
    colors, alphas, _ = gsplat.rasterization(
        means=splats.centers,
        quats=splats.quats_wxyz,
        scales=splats.scales,
        opacities=splats.opacity,
        colors=splats.rgb,
        viewmats=torch.from_numpy(cam_T_world.astype(np.float32))[None].to(device),
        Ks=torch.from_numpy(K.astype(np.float32))[None].to(device),
        width=width,
        height=height,
        packed=False,
        near_plane=0.01,
        render_mode="RGB+ED",
    )
    return colors[0, ..., :3], colors[0, ..., 3], alphas[0, ..., 0]


def train(
    frozen: Splats | None, trainable: Sequence[Trainable], views: Sequence[CameraView], iters: int, rng: np.random.RandomState, rasterize: Rasterize = render
) -> Splats:
    """Adam on ``trainable`` (``frozen`` Gaussians render but stay fixed): each step renders one view drawn by ``rng`` and
    compares its supervised pixels, smooth-L1 + 0.2 x (1 - SSIM). Returns the activated, detached scene."""
    device: torch.device = trainable[0].means.device
    groups: list[dict] = [{"params": [getattr(t, name) for t in trainable], "lr": lr} for name, lr in LEARNING_RATES.items()]
    optimizer = torch.optim.Adam(groups)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    targets: list[Float32[Tensor, "h w 3"]] = [torch.from_numpy(v.rgb.astype(np.float32) / 255.0).to(device) for v in views]

    def scene() -> Splats:
        return merge(([frozen] if frozen is not None else []) + [t.activated() for t in trainable])

    for _ in range(iters):
        index: int = rng.randint(0, len(views))
        view: CameraView = views[index]
        height, width = view.rgb.shape[:2]
        rgb: Float32[Tensor, "h w 3"] = torch.nan_to_num(rasterize(scene(), view.K, view.cam_T_world, width, height)[0])
        target: Float32[Tensor, "h w 3"] = torch.nan_to_num(targets[index])
        mask: Bool[ndarray, "h w"] = np.ones((height, width), dtype=bool) if view.supervise is None else view.supervise
        loss: Tensor = F.smooth_l1_loss(rgb[mask], target[mask]) + 0.2 * (1.0 - ssim(rgb[None].permute(0, 3, 1, 2), target[None].permute(0, 3, 1, 2)))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    with torch.no_grad():
        return scene()


def hole_mask(alpha: Float32[Tensor, "h w"]) -> Bool[ndarray, "h w"]:
    """Pixels the scene leaves uncovered (alpha < 0.5), opened by one 5x5 erosion and grown by three dilations."""
    holes: Float32[ndarray, "h w"] = 1.0 - np.around(alpha.detach().cpu().numpy())
    kernel: ndarray = np.ones((5, 5), np.uint8)
    return cv2.dilate(cv2.erode(holes, kernel, iterations=1), kernel, iterations=3) > 0.5
