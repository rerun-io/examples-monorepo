"""Metric-depth prior of MetricDPVO: patch-depth initialisation, the BA unary and its warm-up.

Ported from ViPE ``vipe/metrictrail/dpvo/metric_depth.py`` and the metric
branches of ``DPVO.__call__``/``update``/``__run_global_BA`` (dpvo.py:1046-1192,
985-1003, 796-841):

1. Before patchify, the metric z-depth becomes inverse depth on DPVO's ``1::4``
   grid; invalid pixels take the median valid inverse depth of the full-resolution
   frame. A frame with no valid grid cell is rejected.
2. Right after patchify every patch's inverse depth is its centre's grid value
   (no random draw); the prior weight is 1 for a valid centre and 0 otherwise.
   A frame whose patches all have weight 0 is rejected.
3. The local and the global BA get the prior with ``lam = prior_weight`` once
   ``prior_warmup`` keyframes were accepted (the switch flips at the end of the
   frame that reaches the count and never flips back); with ``scale_latent`` the
   global BA sees the prior rescaled per keyframe (:mod:`dpvo.metric.scale_latent`).
"""

from dataclasses import dataclass

import torch
from jaxtyping import Bool, Float32, Int64
from torch import Tensor

from dpvo.dpvo import DPVO
from dpvo.fastba import BADepthPrior
from dpvo.hooks import BAKind, DPVOObserver, FrameDepth, FrameEvent
from dpvo.metric.config import MetricDPVOConfig
from dpvo.metric.scale_latent import keyframe_scale_latent

METRIC_DISPS: str = "metric_disps"
"""Frame attribute (``pg.frame_attrs``): prior inverse depth per keyframe slot and patch, ``(N, M)`` float32."""
METRIC_WEIGHTS: str = "metric_weights"
"""Frame attribute: prior weight (0 or 1) per keyframe slot and patch, ``(N, M)`` float32."""


@dataclass(frozen=True, slots=True)
class MetricDisparityGrid:
    """Metric inverse depth sampled on the patchifier's quarter-resolution grid."""

    disps: Float32[Tensor, "1 1 h4 w4"]
    """Inverse depth (1/m) at pixels ``[1::4, 1::4]``; invalid cells hold the frame's median valid inverse depth."""
    valid: Bool[Tensor, "1 1 h4 w4"]
    """Cells whose depth was valid."""


@dataclass(frozen=True, slots=True)
class PatchPriors:
    """Prior of each patch of one frame."""

    disps: Float32[Tensor, "M"]
    """Inverse depth at the patch centre."""
    weights: Float32[Tensor, "M"]
    """1 where the centre is valid, in bounds, finite and positive; else 0."""


def prepare_metric_disparity(
    depth: Float32[Tensor, "ht wd"],
    mask: Bool[Tensor, "ht wd"] | None,
    *,
    image_hw: tuple[int, int],
    min_depth: float,
    max_depth: float,
) -> MetricDisparityGrid:
    """Turn a full-resolution metric z-depth map into inverse depth on DPVO's ``1::4`` grid.

    DPVO training subsamples inverse depth with ``[..., 1::4, 1::4]``; the same
    convention keeps the grid aligned with the patchifier's coordinates. The
    caller's ``mask`` is not modified.

    Args:
        depth: z-depth in metres at image resolution.
        mask: Pixels that may be used, or ``None`` for all.
        image_hw: Image height and width; both divisible by 4.
        min_depth: Smallest valid depth (m).
        max_depth: Largest valid depth (m).

    Raises:
        ValueError: The image size is not divisible by 4, the range or the
            shapes are wrong, or no grid cell is valid (the frame is rejected).
    """
    height, width = image_hw
    if height % 4 or width % 4:
        raise ValueError(f"DPVO metric-depth input must be divisible by 4: {image_hw}")
    if not 0.0 < min_depth < max_depth:
        raise ValueError(f"Invalid metric depth range: {min_depth}, {max_depth}")
    if depth.shape != (height, width) or (mask is not None and mask.shape != depth.shape):
        raise ValueError(f"Metric depth {tuple(depth.shape)} / mask does not match image {image_hw}")

    valid: Bool[Tensor, "ht wd"] = torch.ones_like(depth, dtype=torch.bool) if mask is None else mask
    valid = valid & torch.isfinite(depth) & (depth >= min_depth) & (depth <= max_depth)
    valid_grid: Bool[Tensor, "1 1 h4 w4"] = valid[1::4, 1::4].contiguous()[None, None]
    if not bool(valid_grid.any()):
        raise ValueError("metric depth has no valid pixels; rejecting the frame instead of injecting a 1 m initialization")

    disparity: Float32[Tensor, "ht wd"] = torch.zeros_like(depth)
    disparity[valid] = depth[valid].reciprocal()
    fallback: Float32[Tensor, ""] = disparity[valid].median()
    disparity = torch.where(valid, disparity, fallback)
    return MetricDisparityGrid(disps=disparity[1::4, 1::4].contiguous()[None, None], valid=valid_grid)


def patch_metric_priors(patches: Float32[Tensor, "1 M 3 ps ps"], valid_grid: Bool[Tensor, "1 1 h4 w4"]) -> PatchPriors:
    """Read each patch's centre inverse depth (sampled from the prior grid) and its validity.

    Args:
        patches: Patches of one frame, sampled with ``disps`` = the prior grid.
        valid_grid: Validity of the grid cells.
    """
    centre: int = patches.shape[-1] // 2
    x: Int64[Tensor, "M"] = patches[0, :, 0, centre, centre].round().long()
    y: Int64[Tensor, "M"] = patches[0, :, 1, centre, centre].round().long()
    height, width = valid_grid.shape[-2:]
    in_bounds: Bool[Tensor, "M"] = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    disps: Float32[Tensor, "M"] = patches[0, :, 2, centre, centre].float()
    weights: Float32[Tensor, "M"] = valid_grid[0, 0, y.clamp(0, height - 1), x.clamp(0, width - 1)].float() * in_bounds.float()
    weights *= torch.isfinite(disps).float() * (disps > 0).float()
    return PatchPriors(disps=disps.contiguous(), weights=weights.contiguous())


class MetricDepthPrior(DPVOObserver):
    """:class:`~dpvo.hooks.DepthPrior` from per-frame metric depth, plus the observer that runs its warm-up.

    Reads and writes the frame attributes :data:`METRIC_DISPS` and
    :data:`METRIC_WEIGHTS`, which :class:`~dpvo.metric.dpvo.MetricDPVO` registers.

    Args:
        cfg: Tracker config (depth range, prior weight and Huber threshold, warm-up, scale latent).
    """

    def __init__(self, cfg: MetricDPVOConfig) -> None:
        self.cfg: MetricDPVOConfig = cfg
        self.keyframes_seen: int = 0
        """Frames accepted as keyframes so far (culled ones included)."""
        self.regularization_enabled: bool = False
        """The BA unary is on (latched after ``prior_warmup`` keyframes)."""
        self.ba_calls_with_prior: int = 0
        """Local and global BA solves that got the prior with ``lam > 0``."""
        self.global_ba_calls: int = 0
        """Global BA solves."""
        self.global_ba_calls_with_prior: int = 0
        """Global BA solves that got the prior with ``lam > 0``."""
        self._valid_grid: Bool[Tensor, "1 1 h4 w4"] | None = None

    def prepare(self, dpvo: DPVO, image_hw: tuple[int, int], frame: FrameDepth) -> Float32[Tensor, "1 1 h4 w4"] | None:
        """The prior grid for patchify; rejects a frame without metric depth or without a valid cell."""
        if frame.depth is None:
            raise ValueError("MetricDPVO requires metric_depth for every frame")
        grid: MetricDisparityGrid = prepare_metric_disparity(
            frame.depth, frame.mask, image_hw=image_hw, min_depth=self.cfg.depth_min, max_depth=self.cfg.depth_max
        )
        self._valid_grid = grid.valid.to(device=dpvo.device)
        return grid.disps.to(device=dpvo.device)

    def init_patch_depth(self, dpvo: DPVO, patches: Float32[Tensor, "1 M 3 ps ps"]) -> None:
        """Set every patch texel's inverse depth to its centre prior and store the prior at slot ``n``."""
        assert self._valid_grid is not None, "prepare() runs before init_patch_depth()"
        priors: PatchPriors = patch_metric_priors(patches, self._valid_grid)
        if not bool(priors.weights.any()):
            raise ValueError("metric depth has no valid sampled DPVO patches; rejecting the frame instead of injecting an unobserved depth")
        patches[:, :, 2] = priors.disps[None, :, None, None]
        dpvo.pg.frame_attrs[METRIC_DISPS][dpvo.n] = priors.disps
        dpvo.pg.frame_attrs[METRIC_WEIGHTS][dpvo.n] = priors.weights

    def ba_prior(self, dpvo: DPVO, kind: BAKind) -> BADepthPrior | None:
        """The prior for the next solve; ``lam = 0`` before the warm-up ends. Counts the solves."""
        lam: float = self.cfg.prior_weight if self.regularization_enabled else 0.0
        disps: Float32[Tensor, "N M"] = dpvo.pg.frame_attrs[METRIC_DISPS]
        weights: Float32[Tensor, "N M"] = dpvo.pg.frame_attrs[METRIC_WEIGHTS]
        flat_disps: Float32[Tensor, "NM"] = disps.reshape(-1)
        if kind == "global":
            self.global_ba_calls += 1
            if lam > 0.0:
                self.global_ba_calls_with_prior += 1
            if lam > 0.0 and self.cfg.scale_latent:
                # Rows >= n stay at exp(0) = 1; the BA reads only patches of keyframes < n
                log_scale: Float32[Tensor, "N"] = torch.zeros(disps.shape[0], dtype=torch.float32, device=disps.device)
                log_scale[: dpvo.n] = keyframe_scale_latent(
                    dpvo,
                    disps,
                    weights,
                    clamp=self.cfg.scale_clamp,
                    min_patches=self.cfg.scale_min_patches,
                    min_parallax_px=self.cfg.scale_min_parallax_px,
                    lambda_smooth=self.cfg.scale_smooth,
                )
                flat_disps = (disps * torch.exp(log_scale)[:, None]).reshape(-1)
        if lam > 0.0:
            self.ba_calls_with_prior += 1
        return BADepthPrior(disps=flat_disps, weights=weights.reshape(-1), lam=lam, huber=self.cfg.prior_huber)

    def on_frame(self, dpvo: DPVO, event: FrameEvent) -> None:
        """Count accepted keyframes; latch the unary on once the warm-up count is reached."""
        if event.keyframe_added:
            self.keyframes_seen += 1
            if self.keyframes_seen >= self.cfg.prior_warmup:
                self.regularization_enabled = True
