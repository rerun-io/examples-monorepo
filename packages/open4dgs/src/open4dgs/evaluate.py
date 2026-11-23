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

Every score is computed in full float32 (``ieee_fp32``): PyTorch lets cuDNN run float32 convolutions in TF32 by default
(10-bit mantissa), and the rounding then depends on the convolution algorithm cuDNN picks for the batch shape. LPIPS
(AlexNet convolutions) moved by up to 2e-4 per frame between the batched ``score_camera`` and the per-frame
``score_frame``, most on the all-black floor. The renders themselves and training are not affected: the flags are set
only around the scoring and restored afterwards.
"""

import math
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float32, UInt8
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


@contextmanager
def ieee_fp32() -> Iterator[None]:
    """Run the block with TF32 off for cuDNN convolutions and CUDA matmuls (full float32); restores the previous flags.

    Uses PyTorch's ``fp32_precision`` flags and puts back the exact previous values, also when the block raises. Inside
    the block PyTorch 2.12's legacy getter ``torch.backends.cudnn.allow_tf32`` raises (it does not handle ``ieee``);
    nothing in the scoring reads it.
    """
    # The stubs lack cuDNN's conv / rnn flag modules (set at runtime) and type matmul's flag as bool | str.
    cudnn: Any = torch.backends.cudnn
    conv, rnn, matmul = cudnn.conv, cudnn.rnn, cast(Any, torch.backends.cuda.matmul)
    saved: tuple[str, str, str] = (conv.fp32_precision, rnn.fp32_precision, matmul.fp32_precision)
    conv.fp32_precision = rnn.fp32_precision = matmul.fp32_precision = "ieee"
    try:
        yield
    finally:
        conv.fp32_precision, rnn.fp32_precision, matmul.fp32_precision = saved


@ieee_fp32()
def score_frame(
    camera: str,
    frame_index: int,
    render: Float32[Tensor, "h w 3"],
    alpha: Float32[Tensor, "h w"],
    gt: Float32[Tensor, "h w 3"],
    mask: Float32[Tensor, "h w"],
    lpips: LpipsFn,
) -> FrameMetrics:
    """Every setting of the module docstring on one image pair (full float32, ``ieee_fp32``)."""
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
        with torch.no_grad(), ieee_fp32():
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


EVAL_BATCH_FRAMES: int = 8
"""Frames rendered and scored together by ``score_camera`` (bounds the GPU memory of the batched full-frame metrics)."""


def _psnr_from_mse(mse: Tensor) -> Tensor:
    """``psnr`` on a batch of float32 MSEs: the log in float64 like ``math.log10``, inf where the MSE is 0."""
    db: Tensor = -10.0 * torch.log10(mse.double())
    return torch.where(mse == 0.0, torch.full_like(db, math.inf), db)


def _batch_ssim(pred: Float32[Tensor, "n h w 3"], target: Float32[Tensor, "n h w 3"]) -> Tensor:
    """``ssim`` of each image pair of the batch."""
    value = structural_similarity_index_measure(pred.permute(0, 3, 1, 2), target.permute(0, 3, 1, 2), data_range=1.0, reduction="none")
    return value if isinstance(value, Tensor) else value[0]


def _batch_lpips(metrics: Metrics, pred: Float32[Tensor, "n h w 3"], target: Float32[Tensor, "n h w 3"]) -> Tensor:
    """``Metrics.lpips`` of each image pair, through the metric's network (no torchmetrics state update, no range check)."""
    return metrics.lpips_net.net(pred.permute(0, 3, 1, 2), target.permute(0, 3, 1, 2), normalize=True).reshape(-1)


def _mask_boxes(person: Bool[Tensor, "n h w"], pad: float = CROP_PAD) -> list[tuple[int, int, int, int] | None]:
    """``mask_box`` of every frame of ``person``, with one device-to-host read."""
    height, width = person.shape[1:]
    rows: Tensor = person.any(dim=2)
    cols: Tensor = person.any(dim=1)
    extents: Tensor = torch.stack(
        [
            rows.any(dim=1).long(),
            cols.long().argmax(dim=1),
            width - cols.flip(1).long().argmax(dim=1),
            rows.long().argmax(dim=1),
            height - rows.flip(1).long().argmax(dim=1),
        ],
        dim=1,
    )
    boxes: list[tuple[int, int, int, int] | None] = []
    for present, x0, x1, y0, y1 in extents.tolist():
        if not present:
            boxes.append(None)
            continue
        px, py = round(pad * (x1 - x0)), round(pad * (y1 - y0))
        boxes.append((max(0, x0 - px), max(0, y0 - py), min(width, x1 + px), min(height, y1 + py)))
    return boxes


def _score_batch(
    renders: dict[str, tuple[Float32[Tensor, "n h w 3"], Float32[Tensor, "n h w"]]],
    gt: Float32[Tensor, "n h w 3"],
    mask: Float32[Tensor, "n h w"],
    boxes: list[tuple[int, int, int, int] | None],
    metrics: Metrics,
) -> dict[str, dict[str, Tensor]]:
    """``score_frame`` of every render set (model, floors) on a batch of frames: per set, per metric, an (n,) tensor on the device.

    Values stay on the device (no host read). Crops of the same size are stacked; frames with an empty mask are nan.
    """
    n: int = gt.shape[0]
    device: torch.device = gt.device
    person: Tensor = mask > MASK_THRESHOLD
    present: Tensor = torch.tensor([box is not None for box in boxes], device=device)
    gt_c: Tensor = gt * mask[..., None]
    count: Tensor = person.sum(dim=(1, 2))
    out: dict[str, dict[str, Tensor]] = {}
    for kind, (render, alpha) in renders.items():
        drawn: Tensor = alpha > MASK_THRESHOLD
        fg_mse: Tensor = (((render - gt) ** 2) * person[..., None]).sum(dim=(1, 2, 3)) / (count * 3)
        values: dict[str, Tensor] = {
            "fg_psnr": _psnr_from_mse(fg_mse),
            "full_psnr": _psnr_from_mse(((render - gt_c) ** 2).mean(dim=(1, 2, 3))),
            "full_ssim": _batch_ssim(render, gt_c).double(),
            "full_lpips": _batch_lpips(metrics, render, gt_c).double(),
            "alpha_iou": (drawn & person).sum(dim=(1, 2)).double() / (drawn | person).sum(dim=(1, 2)).double(),
        }
        out[kind] = {name: torch.where(present, value, torch.full_like(value, math.nan)) for name, value in values.items()}
        for name in ("crop_raw_psnr", "crop_raw_ssim", "crop_raw_lpips", "crop_masked_psnr", "crop_masked_ssim", "crop_masked_lpips"):
            out[kind][name] = torch.full((n,), math.nan, dtype=torch.float64, device=device)
    by_size: dict[tuple[int, int], list[int]] = {}
    for i, box in enumerate(boxes):
        if box is not None:
            by_size.setdefault((box[3] - box[1], box[2] - box[0]), []).append(i)
    kinds: list[str] = list(renders)
    for frames in by_size.values():
        slices = [(i, slice(box[1], box[3]), slice(box[0], box[2])) for i in frames if (box := boxes[i]) is not None]
        target: Tensor = torch.stack([gt_c[i, ys, xs] for i, ys, xs in slices])
        crop_mask: Tensor = torch.stack([mask[i, ys, xs] for i, ys, xs in slices])[..., None]
        preds: list[Tensor] = []
        for kind in kinds:
            raw: Tensor = torch.stack([renders[kind][0][i, ys, xs] for i, ys, xs in slices])
            preds += [raw, raw * crop_mask]
        pred_all: Tensor = torch.cat(preds)
        target_all: Tensor = target.repeat(len(preds), 1, 1, 1)
        scores: dict[str, Tensor] = {
            "psnr": _psnr_from_mse(((pred_all - target_all) ** 2).mean(dim=(1, 2, 3))),
            "ssim": _batch_ssim(pred_all, target_all).double(),
            "lpips": _batch_lpips(metrics, pred_all, target_all).double(),
        }
        index: Tensor = torch.tensor(frames, device=device)
        k: int = len(frames)
        for s, kind in enumerate(kinds):
            for v, variant in enumerate(("crop_raw", "crop_masked")):
                start: int = (2 * s + v) * k
                for metric, value in scores.items():
                    out[kind][f"{variant}_{metric}"][index] = value[start : start + k]
    return out


def score_camera(
    camera: str,
    frame_indices: list[int],
    render: Callable[[int], tuple[Float32[Tensor, "h w 3"], Float32[Tensor, "h w 1"]]],
    frames: UInt8[Tensor, "frames h w 3"],
    masks: Float32[Tensor, "frames h w"],
    metrics: Metrics,
    keep_renders: bool = True,
    with_baselines: bool = False,
) -> CameraRenders:
    """``score_frame`` (and the floors of ``baseline_renders``) of one camera at every clip frame, batched.

    ``render(f)`` gives the render and alpha at clip frame ``f``; ``frames`` are the camera's UInt8 (frames, h, w, 3)
    ground truth. Frames are rendered and scored ``EVAL_BATCH_FRAMES`` at a time; each metric is read to the host once.
    Same definitions as the per-frame ``score_frame`` (equal to float32 rounding; ``test_score_camera_matches_score_frame``).
    """
    device: torch.device = metrics.device
    num_frames: int = len(frame_indices)
    boxes: list[tuple[int, int, int, int] | None] = _mask_boxes(masks.to(device) > MASK_THRESHOLD)
    kinds: tuple[str, ...] = ("model", *BASELINES) if with_baselines else ("model",)
    parts: dict[str, dict[str, list[Tensor]]] = {kind: {name: [] for name in METRIC_NAMES} for kind in kinds}
    renders_cpu: list[Tensor] = []
    with torch.no_grad():
        for start in range(0, num_frames, EVAL_BATCH_FRAMES):
            stop: int = min(num_frames, start + EVAL_BATCH_FRAMES)
            outputs = [render(f) for f in range(start, stop)]
            rgb: Tensor = torch.stack([o[0] for o in outputs])
            alpha: Tensor = torch.stack([o[1][..., 0] for o in outputs])
            gt: Tensor = frames[start:stop].to(device).float() / 255.0
            mask: Tensor = masks[start:stop].to(device).float()
            sets: dict[str, tuple[Tensor, Tensor]] = {"model": (rgb, alpha)}
            if with_baselines:
                person: Tensor = mask > MASK_THRESHOLD
                colour: Tensor = (gt * person[..., None]).sum(dim=(1, 2)) / person.sum(dim=(1, 2)).clamp(min=1)[:, None]
                sets["black"] = (torch.zeros_like(gt), torch.zeros_like(mask))
                sets["mean"] = (mask[..., None] * colour[:, None, None, :], mask.clone())
            with ieee_fp32():
                scores = _score_batch(sets, gt, mask, boxes[start:stop], metrics)
            for kind, values in scores.items():
                for name, value in values.items():
                    parts[kind][name].append(value)
            if keep_renders:
                renders_cpu.append((rgb * 255.0).round().to(torch.uint8).cpu())
    host: dict[str, dict[str, list[float]]] = {
        kind: {name: torch.cat(parts[kind][name]).tolist() for name in METRIC_NAMES} for kind in kinds
    }
    rows: dict[str, list[FrameMetrics]] = {
        kind: [FrameMetrics(camera, frame_indices[f], *(host[kind][name][f] for name in METRIC_NAMES)) for f in range(num_frames)]
        for kind in kinds
    }
    return CameraRenders(
        camera,
        torch.cat(renders_cpu) if renders_cpu else torch.empty(0, dtype=torch.uint8),
        rows["model"],
        {kind: rows[kind] for kind in BASELINES} if with_baselines else {},
    )


def evaluate_cameras(
    model: TrainedClip,
    clip: MultiViewClip,
    masks: tuple[Float32[Tensor, "frames h w"], ...],
    camera_names: tuple[str, ...],
    metrics: Metrics,
    keep_renders: bool = True,
    with_baselines: bool = False,
    appearance: bool = False,
) -> list[CameraRenders]:
    """Render ``camera_names`` at every clip frame and score them (and, optionally, the floors) against the clip frames and masks.

    ``appearance``: apply each camera's learned appearance affine (``TrainedClip.render``); raw renders otherwise.
    """
    frame_indices: list[int] = clip.frame_indices[: model.num_frames].tolist()
    out: list[CameraRenders] = []
    for name in camera_names:
        cam: int = clip.camera_names.index(name)
        out.append(
            score_camera(
                name,
                frame_indices,
                lambda f, cam=cam: model.render(model.cameras, cam, f, appearance=appearance),
                clip.frames[cam][: model.num_frames],
                masks[cam][: model.num_frames],
                metrics,
                keep_renders=keep_renders,
                with_baselines=with_baselines,
            )
        )
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
    model: TrainedClip, clip: MultiViewClip, train_names: tuple[str, ...], device: torch.device, centre: Tensor | None = None
) -> tuple[torch.Tensor, Float32[Tensor, "n 4 4"]]:
    """A 360-degree orbit over the clip time (one view per frame), around the people's centre.

    Intrinsics and size come from the first training camera. Returns UInt8 (frames, h, w, 3) renders on the CPU and
    the orbit poses (camera-to-world).
    """
    cams: list[int] = [clip.camera_names.index(name) for name in train_names]
    if centre is None:
        if clip.people is not None and bool(torch.isfinite(clip.people.vertices).any()):
            verts: Tensor = clip.people.vertices.reshape(-1, 3)
            centre = verts[torch.isfinite(verts).all(dim=-1)].mean(dim=0)
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
