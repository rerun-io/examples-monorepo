"""Held-out scoring of the per-camera compensations (appearance affine, time offset) and the brightness split.

A held-out camera has no learned compensation. Two protocols:

- identity (strict): the held-out camera renders with gain 1, bias 0 and the gauge offset the trainer leaves on it
  (TRiGS' own eval, ``_eval_render_time``: nothing is fitted on its pixels);
- test-time fit: only that camera's compensation parameters (the ones the run learned: affine and/or time offset) are
  fitted on its EVEN clip frames with the training loss (``_photometric_losses``: L1 + D-SSIM over black, the alpha
  terms, the same regularisers and clamps), every Gaussian parameter frozen (checked bit for bit), and both protocols
  are scored on its ODD clip frames.
"""

from dataclasses import dataclass

import torch
from jaxtyping import Float32
from torch import Tensor

from open4dgs.catalog_clip import MultiViewClip
from open4dgs.evaluate import MASK_THRESHOLD, CameraRenders, Metrics, score_camera
from open4dgs.trigs_adapter import CameraCompensation, TrainedClip

FIT_LR_APPEARANCE: float = 5e-3
"""Adam step of the test-time affine (5x the trainer's 1e-3: a few hundred steps instead of thousands)."""
FIT_LR_OFFSET: float = 1e-4
"""Adam step of the test-time offset, seconds (the trainer's ``lr_camera_time_offsets``)."""
LUMA: tuple[float, float, float] = (0.299, 0.587, 0.114)


def fit_compensation(
    model: TrainedClip,
    clip: MultiViewClip,
    masks: tuple[Float32[Tensor, "frames h w"], ...],
    camera: str,
    frames: list[int],
    steps: int,
    fit_appearance: bool,
    fit_offset: bool,
) -> CameraCompensation:
    """Fit ``camera``'s affine and/or time offset on clip frames ``frames`` (one frame per step, cycled); Gaussians frozen.

    Starts from the identity (the held-out camera's learned values). Raises if any model parameter changed.
    """
    cam: int = clip.camera_names.index(camera)
    start: CameraCompensation = model.learned()[cam]
    device: torch.device = model.params["means"].device
    config = model.trigs_config
    gain: Tensor = torch.tensor(start.gain, device=device).requires_grad_(fit_appearance)
    bias: Tensor = torch.tensor(start.bias, device=device).requires_grad_(fit_appearance)
    offset: Tensor = torch.tensor(start.offset_s, device=device, dtype=torch.float32).requires_grad_(fit_offset)
    groups: list[dict[str, object]] = []
    if fit_appearance:
        groups.append({"params": [gain, bias], "lr": FIT_LR_APPEARANCE})
    if fit_offset:
        groups.append({"params": [offset], "lr": FIT_LR_OFFSET})
    if not groups:
        return start
    grad_flags: dict[str, bool] = {name: p.requires_grad for name, p in model.params.items()}
    before: list[Tensor] = [p.detach().clone() for p in model.params.values()]
    for p in model.params.values():
        p.requires_grad_(False)
    clamp_s: float = float(config.camera_time_offset_clamp_s)
    optimizer = torch.optim.Adam(groups)  # pyrefly: ignore[bad-argument-type]
    cached: dict[int, tuple[Tensor, Tensor]] = {}
    try:
        for step in range(steps):
            f: int = frames[step % len(frames)]
            if fit_offset:
                t: Tensor = offset + model.capture_time(cam, f)
                colors, alphas, _, _ = model.rasterize(model.cameras, cam, t)
            else:
                if f not in cached:
                    with torch.no_grad():
                        rgb, a, _, _ = model.rasterize(model.cameras, cam, model.capture_time(cam, f) + start.offset_s)
                    cached[f] = (rgb, a)
                colors, alphas = cached[f]
            gt: Tensor = clip.frames[cam][f].to(device).float() / 255.0
            affine: dict[str, Tensor] = {"cam_gain": gain[None], "cam_bias": bias[None]} if fit_appearance else {}
            rgb_loss, alpha_out, alpha_mask = model.photometric_losses(affine, colors, alphas, gt, {f: [masks[cam][f].to(device)]}, f, 0)
            loss: Tensor = config.lambda_render * rgb_loss + config.lambda_alpha_outside_mask * alpha_out + config.lambda_alpha_mask * alpha_mask
            if fit_appearance:
                loss = loss + config.lambda_camera_appearance_reg * ((gain - 1.0).pow(2).mean() + bias.pow(2).mean())
            if fit_offset:
                loss = loss + config.lambda_camera_time_offset_reg * offset.pow(2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                gain.clamp_(0.5, 2.0)
                bias.clamp_(-0.25, 0.25)
                offset.clamp_(-clamp_s, clamp_s)
    finally:
        for name, p in model.params.items():
            p.requires_grad_(grad_flags[name])
    if not all(torch.equal(a, p.detach()) for a, p in zip(before, model.params.values(), strict=True)):
        raise RuntimeError("the test-time fit changed a model parameter")
    return CameraCompensation(
        gain=tuple(gain.detach().cpu().tolist()),  # type: ignore[arg-type]
        bias=tuple(bias.detach().cpu().tolist()),  # type: ignore[arg-type]
        offset_s=float(offset.detach().cpu()),
    )


@dataclass(frozen=True)
class HoldoutFit:
    """One held-out camera under both protocols, scored on its odd clip frames."""

    camera: str
    compensation: CameraCompensation
    identity: CameraRenders
    fitted: CameraRenders


def evaluate_holdout_fit(
    model: TrainedClip,
    clip: MultiViewClip,
    masks: tuple[Float32[Tensor, "frames h w"], ...],
    holdout_names: tuple[str, ...],
    metrics: Metrics,
    steps: int,
    fit_appearance: bool,
    fit_offset: bool,
) -> list[HoldoutFit]:
    """Fit each held-out camera's compensation on its even clip frames; score identity and fit on its odd frames."""
    num_frames: int = model.num_frames
    even: list[int] = list(range(0, num_frames, 2))
    odd: list[int] = list(range(1, num_frames, 2))
    frame_indices: list[int] = [int(clip.frame_indices[f]) for f in odd]
    out: list[HoldoutFit] = []
    for name in holdout_names:
        cam: int = clip.camera_names.index(name)
        comp: CameraCompensation = fit_compensation(model, clip, masks, name, even, steps, fit_appearance, fit_offset)
        frames: Tensor = clip.frames[cam][odd]
        cam_masks: Tensor = masks[cam][odd]
        identity: CameraRenders = score_camera(
            name, frame_indices, lambda j, cam=cam: model.render(model.cameras, cam, odd[j]), frames, cam_masks, metrics, keep_renders=True
        )
        fitted: CameraRenders = score_camera(
            name,
            frame_indices,
            lambda j, cam=cam, comp=comp: model.render(model.cameras, cam, odd[j], compensation=comp),
            frames,
            cam_masks,
            metrics,
            keep_renders=True,
        )
        out.append(HoldoutFit(name, comp, identity, fitted))
    return out


@dataclass(frozen=True, slots=True)
class BrightnessSplit:
    """Luminance of the render against the truth inside the ground-truth mask (> 0.5), summed over frames.

    ``ratio = alpha_part * colour_part`` exactly: ``alpha_part`` is the darkening that the rendered opacity alone explains
    (sum alpha * Y_gt / sum Y_gt), ``colour_part`` the rest (sum Y_render / sum alpha * Y_gt; 1 = the colour of the
    splats is right where they are opaque). ``alpha_mean`` is the mean rendered alpha inside the mask.
    """

    ratio: float
    alpha_part: float
    colour_part: float
    alpha_mean: float


def brightness_split(
    render: Float32[Tensor, "n h w 3"], alpha: Float32[Tensor, "n h w"], gt: Float32[Tensor, "n h w 3"], mask: Float32[Tensor, "n h w"]
) -> BrightnessSplit:
    """``BrightnessSplit`` of a batch of renders (premultiplied, [0, 1]) against ``gt`` ([0, 1])."""
    person: Tensor = (mask > MASK_THRESHOLD).float()
    luma: Tensor = render.new_tensor(LUMA)
    y_render: Tensor = (render * luma).sum(-1) * person
    y_gt: Tensor = (gt * luma).sum(-1) * person
    gt_sum: float = float(y_gt.double().sum())
    weighted: float = float((alpha * y_gt).double().sum())
    render_sum: float = float(y_render.double().sum())
    return BrightnessSplit(
        ratio=render_sum / max(gt_sum, 1e-12),
        alpha_part=weighted / max(gt_sum, 1e-12),
        colour_part=render_sum / max(weighted, 1e-12),
        alpha_mean=float((alpha * person).double().sum()) / max(float(person.double().sum()), 1.0),
    )


def camera_brightness(
    model: TrainedClip,
    clip: MultiViewClip,
    masks: tuple[Float32[Tensor, "frames h w"], ...],
    camera: str,
    frames: list[int],
    compensation: CameraCompensation | None = None,
) -> BrightnessSplit:
    """``brightness_split`` of one clip camera over clip frames ``frames`` (identity, or with ``compensation``)."""
    cam: int = clip.camera_names.index(camera)
    device: torch.device = model.params["means"].device
    renders: list[tuple[Tensor, Tensor]] = [model.render(model.cameras, cam, f, compensation=compensation) for f in frames]
    return brightness_split(
        torch.stack([r for r, _ in renders]),
        torch.stack([a[..., 0] for _, a in renders]),
        clip.frames[cam][frames].to(device).float() / 255.0,
        masks[cam][frames].to(device).float(),
    )
