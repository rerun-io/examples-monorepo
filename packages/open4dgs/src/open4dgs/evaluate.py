"""Held-out (and training) camera metrics, their floors, and the 360-degree orbit.

Every setting is scored against the catalog frame and the ground-truth person mask (threshold 0.5 for boxes and pixel
sets; the soft mask for compositing). The ground truth is always ``gt * mask`` (the person matted onto black). The
render is the TRiGS render, premultiplied over black (gsplat with no background), clamped to [0, 1].

- ``crop_raw`` (HEADLINE): the raw render against ``gt * mask`` inside the mask's bounding box padded by 5 % of its
  size per side. Silhouette errors (splats outside the person, holes inside it) count. PSNR, SSIM, LPIPS (alex).
- ``crop_masked``: as ``crop_raw`` but the render is also multiplied by the ground-truth mask, which hides every
  splat outside the true silhouette (the pre-2026-11-21 ``crop``; kept for comparison, +1.0 dB on 4DAnyone run C).
- ``fg``: foreground-only PSNR, the squared error of the raw render against the raw frame over the mask pixels only.
- ``full``: the whole frame, raw render against ``gt * mask``. Mostly black background, so a black image already
  scores high; for comparison with papers only.
- ``alpha_iou``: IoU of the rendered alpha (> 0.5) with the ground-truth mask (> 0.5).

Floors (``baseline_scores``) are the same metrics for two trivial renders: ``black`` (all zero, alpha 0) and ``mean``
(the mean ground-truth colour inside the mask, painted with the soft mask; alpha = mask). Every number in the run has its
floor beside it.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float32
from torch import Tensor
from torchmetrics.functional.image import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from trigs.core.camera import CameraConfig

from open4dgs.catalog_clip import MultiViewClip
from open4dgs.trigs_adapter import TrainedClip, camera_config

CROP_PAD: float = 0.05
"""Padding of the person crop, as a fraction of the box size per side."""
MASK_THRESHOLD: float = 0.5
"""Mask value above which a pixel is the person (boxes, fg pixel set)."""


def psnr(pred: Float32[Tensor, "h w 3"], target: Float32[Tensor, "h w 3"]) -> float:
    """PSNR in dB for images in [0, 1] (inf when equal)."""
    mse: float = float(((pred - target) ** 2).mean())
    return math.inf if mse == 0.0 else -10.0 * math.log10(mse)


def masked_psnr(pred: Float32[Tensor, "h w 3"], target: Float32[Tensor, "h w 3"], mask: Bool[Tensor, "h w"]) -> float:
    """PSNR over the ``mask`` pixels only (nan when the mask is empty)."""
    if not bool(mask.any()):
        return math.nan
    mse: float = float(((pred - target) ** 2)[mask].mean())
    return math.inf if mse == 0.0 else -10.0 * math.log10(mse)


def ssim(pred: Float32[Tensor, "h w 3"], target: Float32[Tensor, "h w 3"]) -> float:
    """SSIM (11x11 Gaussian window, torchmetrics) for images in [0, 1]."""
    value = structural_similarity_index_measure(pred.permute(2, 0, 1)[None], target.permute(2, 0, 1)[None], data_range=1.0)
    return float(value if isinstance(value, Tensor) else value[0])


def mask_box(mask: Bool[Tensor, "h w"], pad: float = CROP_PAD) -> tuple[int, int, int, int] | None:
    """``(x0, y0, x1, y1)`` (exclusive end) of the mask, padded by ``pad`` of the box size per side; None when empty."""
    ys, xs = torch.nonzero(mask, as_tuple=True)
    if ys.numel() == 0:
        return None
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    px, py = round(pad * (x1 - x0)), round(pad * (y1 - y0))
    height, width = mask.shape
    return max(0, x0 - px), max(0, y0 - py), min(width, x1 + px), min(height, y1 + py)


@dataclass(frozen=True, slots=True)
class FrameMetrics:
    """One camera at one frame; nan where the mask is empty."""

    camera: str
    frame_index: int
    crop_raw_psnr: float
    crop_raw_ssim: float
    crop_raw_lpips: float
    crop_masked_psnr: float
    crop_masked_ssim: float
    crop_masked_lpips: float
    fg_psnr: float
    full_psnr: float
    full_ssim: float
    full_lpips: float
    alpha_iou: float


METRIC_NAMES: tuple[str, ...] = (
    "crop_raw_psnr",
    "crop_raw_ssim",
    "crop_raw_lpips",
    "crop_masked_psnr",
    "crop_masked_ssim",
    "crop_masked_lpips",
    "fg_psnr",
    "full_psnr",
    "full_ssim",
    "full_lpips",
    "alpha_iou",
)
HEADLINE: tuple[str, str, str] = ("crop_raw_psnr", "crop_raw_ssim", "crop_raw_lpips")
"""The headline metrics (PSNR / SSIM / LPIPS of ``crop_raw``)."""
BASELINES: tuple[str, ...] = ("black", "mean")
"""Floor renders scored beside every run (see the module docstring)."""

LpipsFn = Callable[[Tensor, Tensor], float]


def score_frame(
    camera: str,
    frame_index: int,
    render: Float32[Tensor, "h w 3"],
    alpha: Float32[Tensor, "h w"],
    gt: Float32[Tensor, "h w 3"],
    mask: Float32[Tensor, "h w"],
    lpips: LpipsFn,
) -> FrameMetrics:
    """Every setting of the module docstring on one image pair."""
    person: Bool[Tensor, "h w"] = mask > MASK_THRESHOLD
    box = mask_box(person)
    if box is None:
        nan: float = math.nan
        return FrameMetrics(camera, frame_index, *([nan] * (len(METRIC_NAMES))))
    gt_c: Float32[Tensor, "h w 3"] = gt * mask[..., None]
    render_m: Float32[Tensor, "h w 3"] = render * mask[..., None]
    x0, y0, x1, y1 = box
    raw, masked, target = render[y0:y1, x0:x1], render_m[y0:y1, x0:x1], gt_c[y0:y1, x0:x1]
    drawn: Bool[Tensor, "h w"] = alpha > MASK_THRESHOLD
    union: int = int((drawn | person).sum())
    return FrameMetrics(
        camera=camera,
        frame_index=frame_index,
        crop_raw_psnr=psnr(raw, target),
        crop_raw_ssim=ssim(raw, target),
        crop_raw_lpips=lpips(raw, target),
        crop_masked_psnr=psnr(masked, target),
        crop_masked_ssim=ssim(masked, target),
        crop_masked_lpips=lpips(masked, target),
        fg_psnr=masked_psnr(render, gt, person),
        full_psnr=psnr(render, gt_c),
        full_ssim=ssim(render, gt_c),
        full_lpips=lpips(render, gt_c),
        alpha_iou=float((drawn & person).sum()) / union,
    )


def baseline_renders(
    gt: Float32[Tensor, "h w 3"], mask: Float32[Tensor, "h w"]
) -> dict[str, tuple[Float32[Tensor, "h w 3"], Float32[Tensor, "h w"]]]:
    """The floor renders (render, alpha) by name: ``black`` and ``mean`` (mean gt colour inside the mask)."""
    person: Bool[Tensor, "h w"] = mask > MASK_THRESHOLD
    colour: Tensor = gt[person].mean(dim=0) if bool(person.any()) else torch.zeros(3, device=gt.device)
    return {"black": (torch.zeros_like(gt), torch.zeros_like(mask)), "mean": (mask[..., None] * colour, mask.clone())}


class Metrics:
    """Scores image pairs; holds the LPIPS network."""

    def __init__(self, device: torch.device) -> None:
        self.device: torch.device = device
        self.lpips_net: LearnedPerceptualImagePatchSimilarity = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device)

    def lpips(self, pred: Float32[Tensor, "h w 3"], target: Float32[Tensor, "h w 3"]) -> float:
        with torch.no_grad():
            return float(self.lpips_net(pred.permute(2, 0, 1)[None], target.permute(2, 0, 1)[None]))

    def frame(
        self,
        camera: str,
        frame_index: int,
        render: Float32[Tensor, "h w 3"],
        alpha: Float32[Tensor, "h w"],
        gt: Float32[Tensor, "h w 3"],
        mask: Float32[Tensor, "h w"],
    ) -> FrameMetrics:
        return score_frame(camera, frame_index, render, alpha, gt, mask, self.lpips)


def mean_metrics(rows: list[FrameMetrics]) -> dict[str, float]:
    """Mean of each metric over finite values."""
    out: dict[str, float] = {}
    for name in METRIC_NAMES:
        values: list[float] = [getattr(row, name) for row in rows if math.isfinite(getattr(row, name))]
        out[name] = float(np.mean(values)) if values else math.nan
    return out


@dataclass(frozen=True)
class CameraRenders:
    """Renders of one clip camera over the clip, for the run recording."""

    camera: str
    render: torch.Tensor
    """UInt8 (frames, h, w, 3) on the CPU."""
    metrics: list[FrameMetrics]
    baselines: dict[str, list[FrameMetrics]] = field(default_factory=dict)
    """Floor metrics by baseline name (``black``, ``mean``); empty unless requested."""


def evaluate_cameras(
    model: TrainedClip,
    clip: MultiViewClip,
    masks: tuple[Float32[Tensor, "frames h w"], ...],
    camera_names: tuple[str, ...],
    metrics: Metrics,
    keep_renders: bool = True,
    with_baselines: bool = False,
) -> list[CameraRenders]:
    """Render ``camera_names`` at every clip frame and score them (and, optionally, the floors) against the clip frames and masks."""
    out: list[CameraRenders] = []
    for name in camera_names:
        cam: int = clip.camera_names.index(name)
        rows: list[FrameMetrics] = []
        floors: dict[str, list[FrameMetrics]] = {kind: [] for kind in BASELINES} if with_baselines else {}
        renders: list[Tensor] = []
        for f in range(model.num_frames):
            rgb, alpha = model.render(model.cameras, cam, f)
            gt: Float32[Tensor, "h w 3"] = clip.frames[cam][f].to(metrics.device).float() / 255.0
            mask: Float32[Tensor, "h w"] = masks[cam][f].to(metrics.device).float()
            frame_index: int = int(clip.frame_indices[f])
            rows.append(metrics.frame(name, frame_index, rgb, alpha[..., 0], gt, mask))
            if with_baselines:
                for kind, (floor_rgb, floor_alpha) in baseline_renders(gt, mask).items():
                    floors[kind].append(metrics.frame(name, frame_index, floor_rgb, floor_alpha, gt, mask))
            if keep_renders:
                renders.append((rgb * 255.0).round().to(torch.uint8).cpu())
        out.append(CameraRenders(name, torch.stack(renders) if renders else torch.empty(0, dtype=torch.uint8), rows, floors))
    return out


def look_at(eye: Float32[Tensor, "3"], target: Float32[Tensor, "3"], up: Float32[Tensor, "3"]) -> Float32[Tensor, "4 4"]:
    """Camera-to-world (OpenCV RDF: x right, y down, z forward) at ``eye`` looking at ``target``."""
    forward: Tensor = F.normalize(target - eye, dim=0)
    right: Tensor = F.normalize(torch.linalg.cross(forward, up), dim=0)
    down: Tensor = torch.linalg.cross(forward, right)
    pose: Tensor = torch.eye(4, dtype=eye.dtype, device=eye.device)
    pose[:3, 0], pose[:3, 1], pose[:3, 2], pose[:3, 3] = right, down, forward, eye
    return pose


def orbit_poses(world_T_cam: Float32[Tensor, "c 4 4"], centre: Float32[Tensor, "3"], num_views: int) -> Float32[Tensor, "n 4 4"]:
    """One turn of cameras around ``centre`` at the cameras' mean distance (in the ground plane) and mean height.

    Up is the direction orthogonal to the camera x axes (level cameras), signed like the mean camera up.
    """
    # Camera x axes are horizontal when the cameras have no roll: up is the direction most orthogonal to all of them,
    # signed to agree with the mean camera up (-y). Exact for level cameras, robust to an asymmetric rig.
    rights: Tensor = world_T_cam[:, :3, 0].double()
    up: Tensor = torch.linalg.svd(rights.T @ rights).U[:, -1].float()
    if float(up @ -world_T_cam[:, :3, 1].mean(dim=0)) < 0.0:
        up = -up
    offsets: Tensor = world_T_cam[:, :3, 3] - centre
    height: float = float((offsets @ up).mean())
    planar: Tensor = offsets - (offsets @ up)[:, None] * up
    radius: float = float(planar.norm(dim=-1).mean())
    axis_a: Tensor = F.normalize(planar[0], dim=0)
    axis_b: Tensor = torch.linalg.cross(up, axis_a)
    poses: list[Tensor] = []
    for k in range(num_views):
        angle: float = 2.0 * math.pi * k / num_views
        eye: Tensor = centre + height * up + radius * (math.cos(angle) * axis_a + math.sin(angle) * axis_b)
        poses.append(look_at(eye, centre, up))
    return torch.stack(poses)


def render_orbit(
    model: TrainedClip, clip: MultiViewClip, train_names: tuple[str, ...], device: torch.device
) -> tuple[torch.Tensor, Float32[Tensor, "n 4 4"]]:
    """A 360-degree orbit over the clip time (one view per frame), around the people's centre.

    Intrinsics and size come from the first training camera. Returns UInt8 (frames, h, w, 3) renders on the CPU and
    the orbit poses (camera-to-world).
    """
    cams: list[int] = [clip.camera_names.index(name) for name in train_names]
    if clip.people is not None and bool(torch.isfinite(clip.people.vertices).any()):
        verts: Tensor = clip.people.vertices.reshape(-1, 3)
        centre: Tensor = verts[torch.isfinite(verts).all(dim=-1)].mean(dim=0)
    else:
        centre = clip.world_T_cam[cams, :3, 3].mean(dim=0)
    poses: Tensor = orbit_poses(clip.world_T_cam[cams].float(), centre.float(), model.num_frames)
    size: tuple[int, int] = clip.sizes[cams[0]]
    orbit_cameras: CameraConfig = camera_config(poses, clip.K[cams[0]].expand(len(poses), 3, 3), [size] * len(poses), device)
    frames: list[Tensor] = []
    for f in range(model.num_frames):
        rgb, _ = model.render(orbit_cameras, f, f)
        frames.append((rgb * 255.0).round().to(torch.uint8).cpu())
    return torch.stack(frames), poses
