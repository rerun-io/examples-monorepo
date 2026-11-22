"""MoGe-2 monocular depth + normal prior for the TRiGS trainer: the loss, the prior maps and their catalog layer.

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

Prior maps: MoGe-2 (``moge-2-vitl-normal``) on the EXACT frames the trainer sees (``read_catalog_clip``: undistorted,
resized to ``max_side``), with the calibrated horizontal field of view given to MoGe's shift recovery instead of
letting it guess the focal. Layer ``moge2_prior``: per camera and frame ``/world/rig_NN/cam_00/pinhole/moge2/depth``
(``EncodedDepthImage``, 16-bit PNG in mm, 0 = invalid, ``meter=1000``, ``depth_range`` set) and ``.../moge2/normals``
(``EncodedImage`` RGB PNG ``127.5 + 127.5 n``, black = invalid) on ``frame_index``, at the CLIP's pixel grid (not the
stored video's), with recording property ``moge2_prior`` (model, window, max_side, image sizes).
"""

import dataclasses
import hashlib
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import pyarrow as pa
import rerun as rr
import torch
import torch.nn.functional as F
from datafusion import col
from gsplat.utils import depth_to_normal
from jaxtyping import Bool, Float16, Float32, Int64, UInt8
from numpy import ndarray
from rerun.catalog import CatalogClient, DatasetEntry
from simplecv.catalog_layers import atomic_recording, ensure_dataset, register_layer
from simplecv.catalog_rig_layout import catalog_components
from torch import Tensor

from open4dgs.catalog_clip import FRAME_TIMELINE, MultiViewClip

PRIOR_LAYER: str = "moge2_prior"
"""Catalog layer name of the MoGe-2 prior maps."""
MODEL_NAME: str = "moge-2-vitl-normal"
"""Recorded in the layer's properties."""


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


# ---------------------------------------------------------------------------------------------------------------------
# Computing the maps (MoGe-2) and the catalog layer
# ---------------------------------------------------------------------------------------------------------------------


def compute_clip_priors(
    clip: MultiViewClip, camera_names: Sequence[str], predictor: object, batch_size: int = 8, device: torch.device | None = None
) -> ClipPriors:
    """MoGe-2 maps for ``camera_names`` at every clip frame, from the clip's own (undistorted, resized) frames."""
    dev: torch.device = device if device is not None else torch.device("cuda")
    depths: list[Tensor] = []
    normals: list[Tensor] = []
    for name in camera_names:
        cam: int = clip.camera_names.index(name)
        # The calibrated focal fixes MoGe's shift recovery (metrictrail ``recover_metric_depth`` pattern); MoGe assumes a
        # centred principal point.
        fov_x: float = math.degrees(2.0 * math.atan(clip.frames[cam].shape[2] / (2.0 * float(clip.K[cam, 0, 0]))))
        d_parts: list[Tensor] = []
        n_parts: list[Tensor] = []
        for start in range(0, clip.frames[cam].shape[0], batch_size):
            batch: Tensor = clip.frames[cam][start : start + batch_size].to(dev)
            out = predictor.predict_geometry(batch, fov_x=fov_x)  # type: ignore[attr-defined]
            valid: Tensor = (out.mask_bhw > 0.5) & (out.depth_bhw > 0)
            d_parts.append(torch.where(valid, out.depth_bhw, 0.0).half().cpu())
            n_parts.append(encode_normals(out.normals_bhw3, valid).cpu())
        depths.append(torch.cat(d_parts))
        normals.append(torch.cat(n_parts))
    return ClipPriors(camera_names=tuple(camera_names), frame_indices=clip.frame_indices.clone(), depth=tuple(depths), normals=tuple(normals))


def as_stored(priors: ClipPriors) -> ClipPriors:
    """``priors`` as ``read_clip_priors`` hands them back from the layer: depth rounded to whole millimetres (the 16-bit
    PNG), then float16; normals are stored losslessly. A run that computes its maps trains on the same values as the
    next run that reads them."""
    depth: tuple[Tensor, ...] = tuple(((d.float() * 1000.0).round().clamp(0, 65535) / 1000.0).half() for d in priors.depth)
    return dataclasses.replace(priors, depth=depth)


def layer_properties(clip: MultiViewClip, source: str, max_side: int | None, resolution_level: int) -> dict[str, str | int | float]:
    """The ``moge2_prior`` recording property: model, source, window, sizes and the camera grid (``roi_properties``)."""
    return {
        "model": MODEL_NAME,
        "predictor": "MoGeV2TorchPredictor fp16 autocast",
        "resolution_level": resolution_level,
        "fov_source": "calibrated fx (fov_x = 2 atan(w / 2 fx))",
        "source": source,
        "frame_start": int(clip.frame_indices[0]),
        "frame_end": int(clip.frame_indices[-1]),
        "num_frames": len(clip.frame_indices),
        "max_side": max_side or 0,
        "cameras": ",".join(clip.camera_names),
        "image_sizes": ",".join(f"{w}x{h}" for w, h in clip.sizes),
        "loader": "open4dgs.catalog_clip.read_catalog_clip (undistorted)",
        **roi_properties(clip),
    }


def compute_prior_layer(
    clip: MultiViewClip,
    segment_id: str,
    *,
    source: str,
    catalog_url: str,
    dataset: str,
    output_dir: Path,
    max_side: int | None,
    register: bool = True,
    batch_size: int = 8,
    resolution_level: int = 9,
) -> tuple[ClipPriors, Path, dict[str, float]]:
    """MoGe-2 (monopriors torch predictor, vitl) on every clip camera and frame, written as the ``moge2_prior`` layer
    recording and registered on ``catalog_url``/``dataset`` (get-or-create; replaces the segment's previous layer).

    The file is ``<output_dir>/<dataset>/<segment_id>.moge2_prior-<grid>.rrd``, ``<grid>`` a hash of the window, cameras,
    sizes and ROI grid, so a layer of another grid is never overwritten on disk. Returns the maps as stored
    (``as_stored``), the path and the timings (``moge_s``, ``images``, ``write_s``, ``register_s``). The predictor is
    freed before returning.
    """
    from monopriors.models.moge_v2.predictor import MoGeV2TorchPredictor

    predictor = MoGeV2TorchPredictor(encoder="vitl", heads="geometry", resolution_level=resolution_level)
    torch.cuda.synchronize()
    t0: float = time.perf_counter()
    priors: ClipPriors = compute_clip_priors(clip, clip.camera_names, predictor, batch_size=batch_size)
    torch.cuda.synchronize()
    timings: dict[str, float] = {"moge_s": time.perf_counter() - t0, "images": float(len(clip.camera_names) * len(clip.frame_indices))}
    del predictor
    torch.cuda.empty_cache()
    properties = layer_properties(clip, source, max_side, resolution_level)
    grid: str = hashlib.sha1(
        "|".join(str(properties[k]) for k in ("frame_start", "frame_end", "cameras", "image_sizes", "max_side", "roi", "roi_cameras")).encode()
    ).hexdigest()[:10]
    path: Path = output_dir / dataset / f"{segment_id}.{PRIOR_LAYER}-{grid}.rrd"
    t0 = time.perf_counter()
    write_prior_layer(path, segment_id, priors, properties)
    timings["write_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    if register:
        register_prior_layer(catalog_url, dataset, path)
    timings["register_s"] = time.perf_counter() - t0
    return as_stored(priors), path, timings


def select_cameras(priors: ClipPriors, camera_names: Sequence[str]) -> ClipPriors:
    """The maps of ``camera_names`` (each must be in ``priors``)."""
    slots: list[int] = [priors.camera_names.index(name) for name in camera_names]
    return ClipPriors(tuple(camera_names), priors.frame_indices, tuple(priors.depth[i] for i in slots), tuple(priors.normals[i] for i in slots))


def roi_signature(clip: MultiViewClip, camera_names: Sequence[str]) -> dict[str, str]:
    """Per camera, the grid a prior map belongs to: ``none`` for a real camera, else the ROI pad, cap, size, ``K`` and rotation.

    Written into the layer's properties (``roi``, ``roi_cameras``); ``read_clip_priors`` compares it with the clip it reads into.
    """
    if clip.roi is None:
        return dict.fromkeys(camera_names, "none")
    signatures: dict[str, str] = {}
    for name in camera_names:
        c: int = clip.camera_names.index(name)
        w, h = clip.sizes[c]
        k: Float32[ndarray, "3 3"] = clip.K[c].numpy()
        r: str = ",".join(f"{v:.4f}" for v in clip.roi.cameras[c].cam_R_virtual.ravel())
        signatures[name] = (
            f"pad={clip.roi.pad:g};max_side={clip.roi.max_side};{w}x{h};K={k[0, 0]:.2f},{k[1, 1]:.2f},{k[0, 2]:.2f},{k[1, 2]:.2f};R={r}"
        )
    return signatures


def roi_properties(clip: MultiViewClip) -> dict[str, str]:
    """Layer properties that pin the prior to the clip's camera grid (see ``roi_signature``)."""
    signatures: dict[str, str] = roi_signature(clip, clip.camera_names)
    roi: str = "none" if clip.roi is None else f"people pad={clip.roi.pad:g} max_side={clip.roi.max_side}"
    return {"roi": roi, "roi_cameras": "|".join(f"{name}@{sig}" for name, sig in signatures.items())}


def check_roi(clip: MultiViewClip, camera_names: Sequence[str], stored: str | None, source: str) -> None:
    """Raise when the layer's ``roi_cameras`` property (None: a layer written before ROI cameras, real cameras) does not
    describe ``clip``'s cameras ``camera_names``.

    Raises:
        ValueError: On a camera whose grid differs (an ROI clip read against a full-frame layer, another pad, cap or window).
    """
    stored_of: dict[str, str] = dict(item.split("@", 1) for item in stored.split("|")) if stored else {}
    for name, expected in roi_signature(clip, camera_names).items():
        got: str = stored_of.get(name, "none")
        if got != expected:
            raise ValueError(f"{source}: prior layer camera {name} was computed on grid {got!r}, the clip camera is {expected!r}")


def prior_entity(camera_name: str, kind: Literal["depth", "normals"]) -> str:
    """``/world/rig_NN/cam_00/pinhole/moge2/depth`` or ``.../normals``."""
    return f"/world/{camera_name}/pinhole/moge2/{kind}"


def write_prior_layer(path: Path, segment_id: str, priors: ClipPriors, properties: dict[str, str | int | float]) -> None:
    """Write the ``moge2_prior`` layer recording (recording id = ``segment_id``)."""
    with atomic_recording(path, application_id="open4dgs-moge2-prior", recording_id=segment_id, send_properties=False) as rec:
        rec.send_property(PRIOR_LAYER, rr.AnyValues(**properties))  # type: ignore[bad-argument-type]
        frames: Int64[ndarray, " t"] = priors.frame_indices.numpy()
        for camera, depth, normals in zip(priors.camera_names, priors.depth, priors.normals, strict=True):
            depth_mm: np.ndarray = (depth.float() * 1000.0).round().clamp(0, 65535).numpy().astype(np.uint16)
            valid_mm: np.ndarray = depth_mm[depth_mm > 0]
            lo, hi = (float(np.percentile(valid_mm, 1)), float(np.percentile(valid_mm, 99))) if valid_mm.size else (0.0, 1.0)
            depth_blobs: list[bytes] = []
            normal_blobs: list[bytes] = []
            for f in range(len(frames)):
                ok_d, png_d = cv2.imencode(".png", depth_mm[f])
                ok_n, png_n = cv2.imencode(".png", np.ascontiguousarray(normals[f].numpy()[..., ::-1]))  # cv2 writes BGR
                if not (ok_d and ok_n):
                    raise RuntimeError(f"PNG encode failed for {camera} frame {int(frames[f])}")
                depth_blobs.append(png_d.tobytes())
                normal_blobs.append(png_n.tobytes())
            index = [rr.TimeColumn(FRAME_TIMELINE, sequence=frames)]
            rec.send_columns(
                prior_entity(camera, "depth"),
                indexes=index,
                columns=rr.EncodedDepthImage.columns(
                    blob=depth_blobs, media_type=["image/png"] * len(frames), meter=[1000.0] * len(frames), depth_range=[[lo, hi]] * len(frames)
                ),
            )
            rec.send_columns(
                prior_entity(camera, "normals"), indexes=index, columns=rr.EncodedImage.columns(blob=normal_blobs, media_type=["image/png"] * len(frames))
            )


def register_prior_layer(catalog_url: str, dataset: str, path: Path) -> None:
    """Register ``path`` as layer ``moge2_prior`` of its segment on ``dataset`` (``ensure_dataset``: get-or-create with one retry; replaces)."""
    register_layer(ensure_dataset(CatalogClient(catalog_url), dataset), [path], PRIOR_LAYER)


def _cell_bytes(cell: object | None) -> bytes | None:
    if isinstance(cell, list):
        cell = cell[0] if cell else None
    return None if cell is None else bytes(cell)  # type: ignore[arg-type]


def read_clip_priors(clip: MultiViewClip, catalog_url: str, dataset: str, segment_id: str, camera_names: Sequence[str] | None = None) -> ClipPriors:
    """The ``moge2_prior`` layer for ``clip`` (default: every clip camera), checked against the clip's frames and sizes.

    Raises:
        ValueError: If a camera or a clip frame is missing from the layer, or a map's size is not the clip camera's size
            (the layer was computed for another ``max_side``).
    """
    names: list[str] = list(camera_names) if camera_names is not None else list(clip.camera_names)
    client = CatalogClient(catalog_url)  # held for the whole read: a dropped client ends the queries
    entry: DatasetEntry = client.get_dataset(dataset)
    entities: list[str] = [prior_entity(name, kind) for name in names for kind in ("depth", "normals")]
    frames: list[int] = clip.frame_indices.tolist()
    columns: list[str] = [f"{prior_entity(name, 'depth')}:EncodedDepthImage:blob" for name in names] + [
        f"{prior_entity(name, 'normals')}:EncodedImage:blob" for name in names
    ]
    view = entry.filter_segments(segment_id).filter_contents(entities)
    properties: pa.Table = entry.segment_table().to_arrow_table()
    roi_column: str = f"property:{PRIOR_LAYER}:roi_cameras"
    stored: str | None = None
    if roi_column in properties.column_names:
        segment_column: str = "rerun_segment_id"
        rows: list[dict[str, object]] = [r for r in properties.select([segment_column, roi_column]).to_pylist() if r[segment_column] == segment_id]
        cell: object = rows[0][roi_column] if rows else None
        while isinstance(cell, list):
            cell = cell[0] if cell else None
        stored = None if cell is None else str(cell)
    check_roi(clip, names, stored, f"{dataset}/{segment_id}")
    available: set[str] = {f"{c.entity_path}:{c.component}" for c in catalog_components(view.schema())}
    missing: list[str] = [c for c in columns if c not in available]
    if missing:
        raise ValueError(f"{dataset}/{segment_id}: layer {PRIOR_LAYER!r} lacks {missing[:3]} (cameras not computed?)")
    table: pa.Table = (
        view.reader(index=FRAME_TIMELINE)
        .filter((col(FRAME_TIMELINE) >= min(frames)) & (col(FRAME_TIMELINE) <= max(frames)))
        .select(FRAME_TIMELINE, *columns)
        .to_arrow_table()
    )
    row_of: dict[int, int] = {int(frame): row for row, frame in enumerate(table.column(FRAME_TIMELINE).to_numpy())}
    depths: list[Tensor] = []
    normals: list[Tensor] = []
    for name in names:
        width, height = clip.sizes[clip.camera_names.index(name)]
        d_cells: list[object | None] = table.column(f"{prior_entity(name, 'depth')}:EncodedDepthImage:blob").to_pylist()
        n_cells: list[object | None] = table.column(f"{prior_entity(name, 'normals')}:EncodedImage:blob").to_pylist()
        d_stack: np.ndarray = np.empty((len(frames), height, width), dtype=np.float16)
        n_stack: np.ndarray = np.empty((len(frames), height, width, 3), dtype=np.uint8)
        for slot, frame in enumerate(frames):
            row: int | None = row_of.get(frame)
            d_blob: bytes | None = None if row is None else _cell_bytes(d_cells[row])
            n_blob: bytes | None = None if row is None else _cell_bytes(n_cells[row])
            if d_blob is None or n_blob is None:
                raise ValueError(f"{dataset}/{segment_id}: {name} has no prior at frame {frame} (window {frames[0]}..{frames[-1]} not covered)")
            depth_mm: np.ndarray | None = cv2.imdecode(np.frombuffer(d_blob, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            normal_bgr: np.ndarray | None = cv2.imdecode(np.frombuffer(n_blob, dtype=np.uint8), cv2.IMREAD_COLOR)
            if depth_mm is None or normal_bgr is None or depth_mm.shape != (height, width) or normal_bgr.shape[:2] != (height, width):
                got = None if depth_mm is None else depth_mm.shape[::-1]
                raise ValueError(f"{name} frame {frame}: prior is {got}, the clip camera is {(width, height)} (layer computed for another max_side?)")
            d_stack[slot] = depth_mm.astype(np.float32) / 1000.0
            n_stack[slot] = normal_bgr[..., ::-1]
        depths.append(torch.from_numpy(d_stack))
        normals.append(torch.from_numpy(n_stack))
    return ClipPriors(camera_names=tuple(names), frame_indices=clip.frame_indices.clone(), depth=tuple(depths), normals=tuple(normals))
