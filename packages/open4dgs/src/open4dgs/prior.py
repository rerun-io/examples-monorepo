"""MoGe-2 monocular depth + normal prior for the TRiGS trainer: the loss, the prior maps and their catalog layer.

Loss (per training camera ``c`` and clip frame ``f``; ``notes/prior-design.md`` sections 2-3). ``D`` = rendered expected
depth (gsplat ``RGB+ED``, recipe patch 0003), ``A`` = rendered alpha, ``M`` = person mask, ``Dm``/``Nm`` = MoGe-2 depth
and camera-frame normals. Valid pixels ``P = erode(M > 0.5, erode_px) & (Dm > 0) & (A.detach() > alpha_min) &
no MoGe depth edge (|grad log Dm| <= depth_edge_max per px)``.

- depth: ``L_d = 1 - pearson_P(D, Dm)``; scale and shift invariant, so MoGe's per-image metric bias does not matter;
- normals: ``L_n = mean_P (1 - <Nd, Nm>)`` with ``Nd = depth_to_normal(D)`` (gsplat, camera frame); from
  ``contradiction_start_step`` pixels with ``angle(Nd, Nm) > contradiction_deg`` leave P for this term (NeuRIS gate);
- total: ``ramp(step) * (lambda_depth L_d + lambda_normal L_n)``, ramp linear from ``start_step`` to ``ramp_end_step``.
  A camera/frame with fewer than ``min_valid_fraction`` of its pixels in P adds a differentiable zero.

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

import math
import sys
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
from rerun.catalog import CatalogClient, DatasetEntry, OnDuplicateSegmentLayer
from simplecv.catalog_rig_layout import catalog_components
from torch import Tensor

from open4dgs.catalog_clip import FRAME_TIMELINE, MultiViewClip

BeartypeException: tuple[type[Exception], ...] = (sys.modules["beartype.roar"].BeartypeException,) if "beartype.roar" in sys.modules else ()

PRIOR_LAYER: str = "moge2_prior"
"""Catalog layer name of the MoGe-2 prior maps."""
MODEL_NAME: str = "moge-2-vitl-normal"
"""Recorded in the layer's properties."""


@dataclass(frozen=True, slots=True)
class PriorConfig:
    """Weights, schedule and gates of the MoGe-2 prior (defaults: prior-design.md section 3)."""

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
    min_valid_fraction: float = 0.02
    """Below this share of the image in P the camera adds nothing."""
    normal_form: Literal["cos", "spirula"] = "cos"
    """``cos``: 1 - cos; ``spirula``: (1 - cos)/2 + sqrt((1 - cos)/2) (steeper near agreement)."""
    contradiction_deg: float = 30.0
    """NeuRIS gate: from ``contradiction_start_step`` the normal term ignores pixels where render and prior disagree more."""
    contradiction_start_step: int = 2000
    log_every: int = 25
    """Interval of the ``prior/*`` scalars."""

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

    def subset(self, names: Sequence[str]) -> "ClipPriors":
        """The maps of ``names`` only (e.g. the training cameras)."""
        pick: list[int] = [self.camera_names.index(name) for name in names]
        return ClipPriors(
            camera_names=tuple(names),
            frame_indices=self.frame_indices,
            depth=tuple(self.depth[i] for i in pick),
            normals=tuple(self.normals[i] for i in pick),
        )


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


def normal_loss(rendered: Float32[Tensor, "n 3"], prior: Float32[Tensor, "n 3"], form: Literal["cos", "spirula"]) -> Float32[Tensor, ""]:
    """Mean angular disagreement of unit normals."""
    one_minus_cos: Tensor = (1.0 - (rendered * prior).sum(dim=-1)).clamp(min=0.0)
    if form == "cos":
        return one_minus_cos.mean()
    half: Tensor = one_minus_cos / 2.0
    return (half + (half + 1e-8).sqrt()).mean()


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


@dataclass(frozen=True, slots=True)
class PriorTerms:
    """Unweighted terms of one camera at one frame."""

    depth: Float32[Tensor, ""]
    normal: Float32[Tensor, ""]
    valid_fraction: float
    """Share of the image in P (0 when the camera was skipped)."""


def prior_terms(
    depth: Float32[Tensor, "h w"],
    alpha: Float32[Tensor, "h w"],
    mask: Float32[Tensor, "h w"],
    prior_depth: Float32[Tensor, "h w"],
    prior_normals: Float32[Tensor, "h w 3"],
    K: Float32[Tensor, "3 3"],
    step: int,
    config: PriorConfig,
) -> PriorTerms:
    """Both prior terms of one rendered camera (differentiable in ``depth``)."""
    valid: Tensor = erode(mask > 0.5, config.erode_px) & (prior_depth > 0) & (alpha.detach() > config.alpha_min)
    valid = valid & ~depth_edges(prior_depth, config.depth_edge_max)
    valid[0, :] = valid[-1, :] = valid[:, 0] = valid[:, -1] = False  # depth_to_normal border
    fraction: float = float(valid.float().mean())
    if fraction < config.min_valid_fraction:
        zero: Tensor = depth.sum() * 0.0
        return PriorTerms(depth=zero, normal=zero, valid_fraction=0.0)
    l_depth: Tensor = pearson_loss(depth[valid], prior_depth[valid])
    rendered_normals: Tensor = camera_normals(depth, K)
    normal_valid: Tensor = valid
    if step >= config.contradiction_start_step:
        cos: Tensor = (rendered_normals.detach() * prior_normals).sum(dim=-1)
        normal_valid = valid & (cos >= math.cos(math.radians(config.contradiction_deg)))
    l_normal: Tensor = (
        normal_loss(rendered_normals[normal_valid], prior_normals[normal_valid], config.normal_form)
        if bool(normal_valid.any())
        else depth.sum() * 0.0
    )
    return PriorTerms(depth=l_depth, normal=l_normal, valid_fraction=fraction)


@dataclass
class MonoPriorLoss:
    """TRiGS ``camera_loss_hook``: the MoGe-2 prior terms of the training cameras that have maps.

    ``priors`` must hold TRAINING cameras only (train_clip checks it); ``positions[i]`` is the clip camera position
    of ``priors.camera_names[i]``. ``rows`` collects ``prior/{depth,normal,valid_fraction,weight}`` (the unweighted
    terms averaged over the cameras of a step, every ``config.log_every`` steps).
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
    _sums: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])

    def _flush(self) -> None:
        if self._step >= 0 and self._sums[3] > 0 and self._step % self.config.log_every == 0:
            n: float = self._sums[3]
            for name, value in zip(("depth", "normal", "valid_fraction"), self._sums[:3], strict=True):
                self.rows.setdefault(f"prior/{name}", []).append((self._step, value / n))
            self.rows.setdefault("prior/weight", []).append((self._step, self.config.ramp(self._step)))
        self._sums = [0.0, 0.0, 0.0, 0.0]

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
        if cam_idx not in self.positions:
            return depth_rendered.sum() * 0.0
        slot: int = self.positions.index(cam_idx)
        device: torch.device = depth_rendered.device
        prior_depth: Tensor = self.priors.depth[slot][frame_idx].to(device, non_blocking=True).float()
        prior_normals, _ = decode_normals(self.priors.normals[slot][frame_idx].to(device, non_blocking=True))
        mask: Tensor = self.masks[slot][frame_idx].to(device, non_blocking=True).float()
        terms: PriorTerms = prior_terms(
            depth_rendered, alphas[0, ..., 0], mask, prior_depth, prior_normals, self.K[cam_idx].to(device), step, self.config
        )
        if terms.valid_fraction > 0.0:
            self._sums[0] += float(terms.depth.detach())
            self._sums[1] += float(terms.normal.detach())
            self._sums[2] += terms.valid_fraction
            self._sums[3] += 1.0
        weight: float = self.config.ramp(step)
        return weight * (self.config.lambda_depth * terms.depth + self.config.lambda_normal * terms.normal)


# ---------------------------------------------------------------------------------------------------------------------
# Computing the maps (MoGe-2) and the catalog layer
# ---------------------------------------------------------------------------------------------------------------------


def moge_geometry(
    predictor: object, frames: UInt8[Tensor, "b h w 3"], fx: float
) -> tuple[Float32[Tensor, "b h w"], Float32[Tensor, "b h w 3"], Bool[Tensor, "b h w"]]:
    """MoGe-2 metric depth, RDF normals and validity at the frames' size, with the calibrated focal ``fx`` (pixels).

    ``monopriors`` ``postprocess_moge_v2_geometry`` passes ``fov_x=None`` (MoGe guesses the focal); here the shift
    recovery gets ``fov_x = 2 atan(w / 2 fx)`` instead (metrictrail ``recover_metric_depth`` pattern). MoGe assumes a
    centred principal point.
    """
    from monopriors.third_party.moge.model._inference import recover_shift_and_intrinsics

    height, width = frames.shape[1], frames.shape[2]
    graph = predictor.forward_geometry(frames)  # type: ignore[attr-defined]
    points: Tensor = graph.points_bhw3.float()
    scale: Tensor = graph.metric_scale_b.float()
    mask_net: Tensor = graph.mask_bhw.float()
    fov_x: float = math.degrees(2.0 * math.atan(width / (2.0 * fx)))
    fov: Tensor = torch.tensor(fov_x, device=points.device)
    camera = recover_shift_and_intrinsics(points, mask_net > 0.5, fov_x=fov, aspect_ratio=width / height)
    z: Tensor = (points[..., 2] + camera.shift_b.float()[:, None, None]) * scale[:, None, None]
    z_full: Tensor = F.interpolate(z[:, None], size=(height, width), mode="bilinear", align_corners=False)[:, 0]
    mask_full: Tensor = F.interpolate(mask_net[:, None], size=(height, width), mode="bilinear", align_corners=False)[:, 0]
    normals: Tensor = F.interpolate(graph.normals_bhw3.float().permute(0, 3, 1, 2), size=(height, width), mode="bilinear", align_corners=False)
    normals = F.normalize(normals.permute(0, 2, 3, 1), dim=-1)
    valid: Tensor = (mask_full > 0.5) & (z_full > 0)
    return torch.where(valid, z_full, torch.zeros_like(z_full)), normals, valid


def compute_clip_priors(
    clip: MultiViewClip, camera_names: Sequence[str], predictor: object, batch_size: int = 8, device: torch.device | None = None
) -> ClipPriors:
    """MoGe-2 maps for ``camera_names`` at every clip frame, from the clip's own (undistorted, resized) frames."""
    dev: torch.device = device if device is not None else torch.device("cuda")
    depths: list[Tensor] = []
    normals: list[Tensor] = []
    for name in camera_names:
        cam: int = clip.camera_names.index(name)
        fx: float = float(clip.K[cam, 0, 0])
        d_parts: list[Tensor] = []
        n_parts: list[Tensor] = []
        for start in range(0, clip.frames[cam].shape[0], batch_size):
            batch: Tensor = clip.frames[cam][start : start + batch_size].to(dev)
            depth, normal, valid = moge_geometry(predictor, batch, fx)
            d_parts.append(depth.half().cpu())
            n_parts.append(encode_normals(normal, valid).cpu())
        depths.append(torch.cat(d_parts))
        normals.append(torch.cat(n_parts))
    return ClipPriors(camera_names=tuple(camera_names), frame_indices=clip.frame_indices.clone(), depth=tuple(depths), normals=tuple(normals))


def prior_entity(camera_name: str, kind: Literal["depth", "normals"]) -> str:
    """``/world/rig_NN/cam_00/pinhole/moge2/depth`` or ``.../normals``."""
    return f"/world/{camera_name}/pinhole/moge2/{kind}"


def write_prior_layer(path: Path, segment_id: str, priors: ClipPriors, properties: dict[str, str | int | float]) -> None:
    """Write the ``moge2_prior`` layer recording (recording id = ``segment_id``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = rr.RecordingStream("open4dgs-moge2-prior", recording_id=segment_id)
    rec.save(path)
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
    rec.flush()
    path.chmod(0o644)


def register_prior_layer(catalog_url: str, dataset: str, path: Path) -> None:
    """Register ``path`` as layer ``moge2_prior`` of its segment on ``dataset`` (get-or-create with one retry; replaces)."""
    client = CatalogClient(catalog_url)
    for attempt in range(2):
        try:
            client.create_dataset(dataset, exist_ok=True)
            break
        except BeartypeException:
            raise
        except Exception:
            if attempt == 1:  # the first failure may be another worker creating the same dataset
                raise
    entry: DatasetEntry = client.get_dataset(dataset)
    entry.register([path.resolve().as_uri()], layer_name=PRIOR_LAYER, on_duplicate=OnDuplicateSegmentLayer.REPLACE).wait()


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
    entry: DatasetEntry = CatalogClient(catalog_url).get_dataset(dataset)
    entities: list[str] = [prior_entity(name, kind) for name in names for kind in ("depth", "normals")]
    frames: list[int] = clip.frame_indices.tolist()
    columns: list[str] = [f"{prior_entity(name, 'depth')}:EncodedDepthImage:blob" for name in names] + [
        f"{prior_entity(name, 'normals')}:EncodedImage:blob" for name in names
    ]
    view = entry.filter_segments(segment_id).filter_contents(entities)
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
