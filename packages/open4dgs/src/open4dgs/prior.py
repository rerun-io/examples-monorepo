"""MoGe-2 monocular depth + normal prior for the TRiGS trainer: the loss.

Loss (per training camera ``c`` and clip frame ``f``; ``notes/prior-design.md`` sections 2-3). ``D`` = rendered expected
depth (gsplat ``RGB+ED``, recipe patch 0003), ``A`` = rendered alpha, ``M`` = person mask, ``Dm``/``Nm`` = MoGe-2 depth
and camera-frame normals. Valid pixels ``P = erode(M > 0.5, erode_px) & (Dm > 0) & (A.detach() > alpha_min) &
no MoGe depth edge (|grad log Dm| <= depth_edge_max per px)``.

- depth: ``L_d = 1 - pearson_P(D, Dm)``; scale and shift invariant, so MoGe's per-image metric bias does not matter;
- normals: ``L_n = mean_P (1 - <Nd, Nm>)`` with ``Nd = depth_to_normal(D)`` (gsplat, camera frame); from
  ``contradiction_start_step`` pixels with ``angle(Nd, Nm) > contradiction_deg`` leave P for this term (NeuRIS gate);
- total: ``ramp(step) * (lambda_depth L_d + lambda_normal L_n)``, ramp linear from ``start_step`` to ``ramp_end_step``.
  A camera/frame with fewer than ``min_valid_fraction`` of its pixels in P adds a differentiable zero (default 0.005:
  the person covers 2-4 % of a 514 px MAMMA frame, so 0.02 skipped many camera/frames).

Sign conventions (pinned by tests/test_prior.py): camera frame is OpenCV RDF; ``depth_to_normal`` crosses the row
difference (down) with the column difference (right), so a surface facing the camera has normal ``(0, 0, -1)``, the
MoGe convention (normals point back at the camera). Nothing is rotated to the world frame.

Prior maps and their catalog layer ``moge2_prior``: ``open4dgs.prior_layer``.
"""

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn.functional as F
from gsplat.utils import depth_to_normal
from jaxtyping import Bool, Float16, Float32, Int64, UInt8
from torch import Tensor


@dataclass(frozen=True, slots=True)
class PriorConfig:
    """Weights, schedule and gates of the MoGe-2 prior (defaults: prior-design.md section 3, schedule from the p2 experiment).

    With ``schedule="fraction"`` (default) ``at_steps(total)`` sets the three step fields from the ``*_fraction``
    fields (15 % / 40 % / 75 % of the run); ``schedule="steps"`` keeps the step fields as given.
    """

    lambda_depth: float = 0.05
    """Weight of the Pearson depth term."""
    lambda_normal: float = 0.05
    """Weight of the depth-normal term."""
    start_step: int = 500
    """First step with a non-zero weight."""
    ramp_end_step: int = 1500
    """Step at which the weights reach their full value (linear ramp from ``start_step``)."""
    erode_px: int = 3
    """Erosion of the person mask, pixels."""
    alpha_min: float = 0.5
    """Rendered alpha (detached) a pixel needs."""
    depth_edge_max: float = 0.05
    """Largest |grad log Dm| per pixel (MoGe depth steps, e.g. an arm over the torso, leave P)."""
    min_valid_fraction: float = 0.005
    """Below this share of the image in P the camera adds nothing."""
    normal_form: Literal["cos", "spirula"] = "cos"
    """``cos``: 1 - cos; ``spirula``: (1 - cos)/2 + sqrt((1 - cos)/2) (steeper near agreement)."""
    contradiction_deg: float = 30.0
    """NeuRIS gate: from ``contradiction_start_step`` the normal term ignores pixels where render and prior disagree more."""
    contradiction_start_step: int = 2000
    log_every: int = 25
    """Interval of the ``prior/*`` scalars."""
    schedule: Literal["fraction", "steps"] = "fraction"
    """``fraction``: the step fields follow the run length (``at_steps``); ``steps``: they are absolute."""
    start_fraction: float = 0.15
    """``start_step`` as a fraction of the run (``schedule="fraction"``)."""
    ramp_end_fraction: float = 0.40
    """``ramp_end_step`` as a fraction of the run."""
    contradiction_start_fraction: float = 0.75
    """``contradiction_start_step`` as a fraction of the run."""

    def at_steps(self, total_steps: int) -> "PriorConfig":
        """The config for a run of ``total_steps`` steps: the step fields from the fractions (unchanged for ``schedule="steps"``)."""
        if self.schedule == "steps":
            return self
        return dataclasses.replace(
            self,
            start_step=round(self.start_fraction * total_steps),
            ramp_end_step=round(self.ramp_end_fraction * total_steps),
            contradiction_start_step=round(self.contradiction_start_fraction * total_steps),
        )

    def ramp(self, step: int) -> float:
        """Weight multiplier in [0, 1] at ``step``."""
        if step < self.start_step:
            return 0.0
        if self.ramp_end_step <= self.start_step:
            return 1.0
        return min(1.0, (step - self.start_step) / (self.ramp_end_step - self.start_step))


@dataclass(frozen=True, slots=True)
class ClipPriors:
    """MoGe-2 maps for some clip cameras, on each camera's clip pixel grid; held on the CPU."""

    camera_names: tuple[str, ...]
    frame_indices: Int64[Tensor, " frames"]
    """Source ``frame_index`` values (equal to the clip's)."""
    depth: tuple[Float16[Tensor, "frames h w"], ...]
    """Metric MoGe-2 z-depth, metres; 0 where MoGe-2 is invalid."""
    normals: tuple[UInt8[Tensor, "frames h w 3"], ...]
    """Camera-frame (RDF) normals as ``127.5 + 127.5 n``; (0, 0, 0) where invalid."""


def encode_normals(normals: Float32[Tensor, "... 3"], valid: Bool[Tensor, "..."]) -> UInt8[Tensor, "... 3"]:
    """Unit normals to ``127.5 + 127.5 n`` uint8 (rounded, clamped to 1..255 so a valid pixel is never black); black where invalid."""
    coded: Tensor = (127.5 + 127.5 * normals).round().clamp(1, 255).to(torch.uint8)
    return torch.where(valid[..., None], coded, torch.zeros_like(coded))


def decode_normals(coded: UInt8[Tensor, "... 3"]) -> tuple[Float32[Tensor, "... 3"], Bool[Tensor, "..."]]:
    """Inverse of ``encode_normals``: renormalised unit normals and the validity (not black)."""
    valid: Tensor = coded.amax(dim=-1) > 0
    normals: Tensor = F.normalize((coded.float() - 127.5) / 127.5, dim=-1)
    return torch.where(valid[..., None], normals, torch.zeros_like(normals)), valid


def camera_normals(depth: Float32[Tensor, "h w"], K: Float32[Tensor, "3 3"]) -> Float32[Tensor, "h w 3"]:
    """Camera-frame normals of a z-depth map (gsplat ``depth_to_normal`` with an identity pose; 1-px zero border)."""
    eye: Tensor = torch.eye(4, dtype=depth.dtype, device=depth.device)
    return depth_to_normal(depth[None, :, :, None], eye[None], K.to(depth)[None])[0]


def pearson_loss(depth: Float32[Tensor, " n"], prior: Float32[Tensor, " n"]) -> Float32[Tensor, ""]:
    """``1 - pearson(depth, prior)`` over the given pixels (0 = perfectly correlated, 2 = anti-correlated)."""
    d: Tensor = depth - depth.mean()
    p: Tensor = prior - prior.mean()
    corr: Tensor = (d * p).mean() / (d.pow(2).mean() * p.pow(2).mean()).clamp(min=1e-12).sqrt()
    return 1.0 - corr


def depth_edges(prior_depth: Float32[Tensor, "h w"], max_step: float) -> Bool[Tensor, "h w"]:
    """Pixels next to a MoGe depth step: |grad log Dm| > ``max_step`` in x or y, dilated by one pixel; invalid depth counts as a step."""
    log_d: Tensor = torch.log(prior_depth.clamp(min=1e-6))
    invalid: Tensor = prior_depth <= 0
    gx: Tensor = (log_d[:, 1:] - log_d[:, :-1]).abs() > max_step
    gy: Tensor = (log_d[1:, :] - log_d[:-1, :]).abs() > max_step
    gx = gx | invalid[:, 1:] | invalid[:, :-1]
    gy = gy | invalid[1:, :] | invalid[:-1, :]
    edge: Tensor = torch.zeros_like(invalid)
    edge[:, 1:] |= gx
    edge[:, :-1] |= gx
    edge[1:, :] |= gy
    edge[:-1, :] |= gy
    return F.max_pool2d(edge.float()[None, None], 3, stride=1, padding=1)[0, 0] > 0


def erode(mask: Bool[Tensor, "h w"], px: int) -> Bool[Tensor, "h w"]:
    """Square erosion by ``px`` pixels (outside the image counts as background)."""
    if px <= 0:
        return mask
    outside: Tensor = F.max_pool2d((~mask).float()[None, None], 2 * px + 1, stride=1, padding=px)[0, 0] > 0
    border: Tensor = torch.zeros_like(mask)
    border[:px, :] = border[-px:, :] = border[:, :px] = border[:, -px:] = True
    return mask & ~outside & ~border


def static_valid(mask: Float32[Tensor, "h w"], prior_depth: Float32[Tensor, "h w"], config: PriorConfig) -> Bool[Tensor, "h w"]:
    """The render-independent part of P: eroded person mask, valid MoGe depth, no MoGe depth edge, no 1-px border."""
    valid: Tensor = erode(mask > 0.5, config.erode_px) & (prior_depth > 0)
    valid = valid & ~depth_edges(prior_depth, config.depth_edge_max)
    valid[0, :] = valid[-1, :] = valid[:, 0] = valid[:, -1] = False  # depth_to_normal border
    return valid


def _masked_mean(values: Tensor, weight: Tensor, count: Tensor) -> Tensor:
    return (values * weight).sum() / count.clamp(min=1.0)


@dataclass(frozen=True, slots=True)
class PriorTerms:
    """Unweighted terms of one camera at one frame; 0-d tensors, so nothing here waits for the GPU."""

    depth: Float32[Tensor, ""]
    normal: Float32[Tensor, ""]
    valid_fraction: Float32[Tensor, ""]
    """Share of the image in P (0 when the camera was skipped)."""


def prior_terms(
    depth: Float32[Tensor, "h w"],
    alpha: Float32[Tensor, "h w"],
    static: Bool[Tensor, "h w"],
    prior_depth: Float32[Tensor, "h w"],
    prior_normals: Float32[Tensor, "h w 3"],
    K: Float32[Tensor, "3 3"],
    step: int,
    config: PriorConfig,
) -> PriorTerms:
    """Both prior terms of one rendered camera (differentiable in ``depth``), ``static`` from ``static_valid``.

    Masked means over P instead of boolean indexing and a tensor skip flag instead of ``if``: no GPU->CPU sync. A
    skipped camera (P below ``min_valid_fraction``) or an empty normal set gives a differentiable zero.
    """
    weight: Tensor = (static & (alpha.detach() > config.alpha_min)).float()
    count: Tensor = weight.sum()
    fraction: Tensor = count / weight.numel()
    keep: Tensor = (fraction >= config.min_valid_fraction).float()
    d: Tensor = depth - _masked_mean(depth, weight, count)
    p: Tensor = prior_depth - _masked_mean(prior_depth, weight, count)
    cov: Tensor = _masked_mean(d * p, weight, count)
    var: Tensor = (_masked_mean(d.pow(2), weight, count) * _masked_mean(p.pow(2), weight, count)).clamp(min=1e-12).sqrt()
    l_depth: Tensor = 1.0 - cov / var
    rendered_normals: Tensor = camera_normals(depth, K)
    normal_weight: Tensor = weight
    if step >= config.contradiction_start_step:
        cos: Tensor = (rendered_normals.detach() * prior_normals).sum(dim=-1)
        normal_weight = weight * (cos >= math.cos(math.radians(config.contradiction_deg))).float()
    one_minus_cos: Tensor = (1.0 - (rendered_normals * prior_normals).sum(dim=-1)).clamp(min=0.0)
    if config.normal_form == "spirula":
        half: Tensor = one_minus_cos / 2.0
        one_minus_cos = half + (half + 1e-8).sqrt()
    l_normal: Tensor = _masked_mean(one_minus_cos, normal_weight, normal_weight.sum())
    return PriorTerms(depth=l_depth * keep, normal=l_normal * keep, valid_fraction=fraction * keep)


@dataclass
class MonoPriorLoss:
    """TRiGS ``camera_loss_hook``: the MoGe-2 prior terms of the training cameras that have maps.

    ``priors`` must hold TRAINING cameras only (train_clip checks it); ``positions[i]`` is the clip camera position
    of ``priors.camera_names[i]``. ``rows`` collects ``prior/{depth,normal,valid_fraction,weight}`` (the unweighted
    terms averaged over the non-skipped cameras of a step, every ``config.log_every`` steps).

    On the first call the maps move to the render's device and stay there for the run: prior depth (fp16), coded
    normals (uint8) and the render-independent gate (``static_valid``, bool), 6 bytes per pixel per camera and frame;
    the masks are not kept. Scalars accumulate as GPU tensors and are read once per logged step.
    """

    config: PriorConfig
    priors: ClipPriors
    masks: tuple[Float32[Tensor, "frames h w"], ...]
    """Person masks of the cameras in ``priors``, same order."""
    positions: tuple[int, ...]
    K: Float32[Tensor, "cams 3 3"]
    """Intrinsics of every clip camera (indexed by position)."""
    rows: dict[str, list[tuple[int, float]]] = field(default_factory=dict)
    _step: int = -1
    _sums: Tensor | None = None
    _device: torch.device | None = None
    _depth: tuple[Tensor, ...] = ()
    _normals: tuple[Tensor, ...] = ()
    _static: tuple[Tensor, ...] = ()
    _K: Tensor | None = None

    def _to_device(self, device: torch.device) -> None:
        self._device = device
        self._depth = tuple(d.to(device) for d in self.priors.depth)
        self._normals = tuple(n.to(device) for n in self.priors.normals)
        self._static = tuple(
            torch.stack([static_valid(m[f].to(device).float(), d[f].float(), self.config) for f in range(d.shape[0])])
            for m, d in zip(self.masks, self._depth, strict=True)
        )
        self._K = self.K.to(device)

    def _flush(self) -> None:
        if self._step >= 0 and self._sums is not None:
            depth, normal, fraction, n = self._sums.tolist()
            if n > 0:
                for name, value in (("depth", depth), ("normal", normal), ("valid_fraction", fraction)):
                    self.rows.setdefault(f"prior/{name}", []).append((self._step, value / n))
                self.rows.setdefault("prior/weight", []).append((self._step, self.config.ramp(self._step)))
        self._sums = None

    def finish(self) -> None:
        """Flush the last step's scalars."""
        self._flush()
        self._step = -1

    def __call__(self, *, step: int, frame_idx: int, cam_idx: int, colors: Tensor, alphas: Tensor, info: dict, cameras: object) -> Tensor:
        del colors, cameras
        if step != self._step:
            self._flush()
            self._step = step
        depth_rendered: Tensor = info["render_depth"][0, ..., 0]
        weight: float = self.config.ramp(step)
        logged: bool = step % self.config.log_every == 0
        if cam_idx not in self.positions or (weight == 0.0 and not logged):
            return depth_rendered.sum() * 0.0
        if self._device != depth_rendered.device:
            self._to_device(depth_rendered.device)
        assert self._K is not None
        slot: int = self.positions.index(cam_idx)
        prior_normals, _ = decode_normals(self._normals[slot][frame_idx])
        terms: PriorTerms = prior_terms(
            depth_rendered,
            alphas[0, ..., 0],
            self._static[slot][frame_idx],
            self._depth[slot][frame_idx].float(),
            prior_normals,
            self._K[cam_idx],
            step,
            self.config,
        )
        if logged:
            kept: Tensor = (terms.valid_fraction > 0).float()
            row: Tensor = torch.stack([terms.depth.detach(), terms.normal.detach(), terms.valid_fraction, kept])
            self._sums = row if self._sums is None else self._sums + row
        return weight * (self.config.lambda_depth * terms.depth + self.config.lambda_normal * terms.normal)
