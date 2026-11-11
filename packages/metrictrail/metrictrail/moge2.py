"""MetricTrail's shipped MoGe-2 prior: ``moge-2-vitl-normal`` through monoprior's TensorRT engine, with surface normals.

ViPE's ``moge2-l`` (:class:`vipe.metrictrail.depth.MetricTrailMoGe2`) runs the ``moge-2-vitl`` checkpoint in torch.
This module registers ``moge2-<variant>-normal`` in ViPE's ``DEPTH_PROVIDERS`` (recipes/vipe-src patch 0004, a
hook only) and selects it through the Hydra overrides :data:`PROVIDER_OVERRIDES`; the provider makes MetricTrail's
MoGe-2 call:

- the network is monoprior's MoGe-2 on the ``Ruicheng/moge-2-vitl-normal`` checkpoint (depth AND surface normals from
  one forward pass), run as monoprior's TensorRT engine (``MoGeV2TrtPredictor``, heads ``geometry``) when TensorRT
  and trtkit import, else monoprior's torch predictor with FP16 autocast on the same checkpoint
  (:func:`select_moge2_backend`; ``VIPE_MOGE2_BACKEND`` forces one);
- a selectable detail level on the input's own token grid (one engine per grid), never an aspect bucket;
- ``fov_x`` computed from the bound focal in float64 Python arithmetic (ViPE's ``fov_x_degrees``);
- MoGe-2's fov-conditioned post-processing on the raw graph outputs:
  ``recover_shift_and_intrinsics(points, mask > 0.5, fov_x, aspect_ratio)``, the shift added to z, times
  ``metric_scale``, then resized to the input raster;
- the metric validity rule ``MoGe mask & finite & 0.2 m <= depth <= 500 m`` (ViPE's ``estimate`` zeroes the rest).

The forward does not depend on ``fov_x``: the engine caches each source frame's raw outputs (points, mask, metric
scale, normals; keyed by the 8-bit image content) and a new focal only re-runs the post-processing.
``forward_count`` and ``postprocess_count`` go into the run record (:meth:`MoGe2NormalProvider.describe`).
"""

import functools
import hashlib
import importlib.util
import os
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias
from weakref import WeakSet

import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float, Float32, Float64, Int64, UInt8
from numpy import ndarray
from torch import Tensor
from vipe.metrictrail.depth import (
    METRIC_DEPTH_MAX_M,
    METRIC_DEPTH_MIN_M,
    MetricDepthPrediction,
    MetricTrailMoGe2,
    fov_x_degrees,
    normalized_to_pixel_k4,
)
from vipe.priors.depth.base import DepthEstimationModel, DepthType
from vipe.utils.model_cache import ModelCache

MoGe2Backend: TypeAlias = Literal["tensorrt", "torch"]
MoGe2Encoder: TypeAlias = Literal["vits", "vitb", "vitl"]
"""monoprior's MoGe-2 encoders (``monopriors.models.moge_v2.export.Encoder``)."""
ImageKey: TypeAlias = tuple[int, int, bytes]
"""A source frame's cache key: height, width and a 16-byte digest of its 8-bit bytes."""

MOGE2_BACKEND_ENV: str = "VIPE_MOGE2_BACKEND"
"""Environment override of the backend choice (``auto`` when unset)."""
MOGE2_RESOLUTION_LEVEL: int = 9
"""Original MoGe-2 detail level (3600 tokens); the run settings can select a smaller grid."""
MOGE2_ENCODERS: dict[str, MoGe2Encoder] = {"l": "vitl", "b": "vitb", "s": "vits"}
"""Registry variant -> monoprior encoder of the pinned ``moge-2-vit*-normal`` checkpoint."""
MOGE2_FORWARD_CACHE_FRAMES: int = 256
"""Source frames whose raw forward outputs stay cached (least recently used are evicted)."""
PROVIDER_NAMES: dict[str, str] = {f"moge2-{variant}-normal": variant for variant in MOGE2_ENCODERS}
"""ViPE depth-provider name -> variant, as :func:`register_providers` registers them."""
DEFAULT_PROVIDER: str = "moge2-l-normal"
"""The provider our port runs."""
PROVIDER_OVERRIDES: list[str] = [
    f"pipeline.init.calibration.depth_model={DEFAULT_PROVIDER}",
    # Upstream's FP32 default is for ViPE's own moge2-l; our provider runs FP16 on every backend.
    "pipeline.init.calibration.depth_use_fp16=true",
    f"pipeline.slam.keyframe_depth={DEFAULT_PROVIDER}",
]
"""Hydra overrides that select :data:`DEFAULT_PROVIDER` (and its FP16 precision) for SharedK calibration and for tracking."""


def tensorrt_importable() -> bool:
    """Whether monoprior's TensorRT lane (``tensorrt`` and ``trtkit``) is installed."""
    return importlib.util.find_spec("tensorrt") is not None and importlib.util.find_spec("trtkit") is not None


def select_moge2_backend(choice: str, *, tensorrt_available: bool) -> MoGe2Backend:
    """The MoGe-2 backend for ``choice``: ``auto`` takes TensorRT when it is available, else torch.

    ``tensorrt`` without TensorRT is an error (no silent fallback when it was asked for).
    """
    if choice == "torch":
        return "torch"
    if choice == "tensorrt":
        if not tensorrt_available:
            raise RuntimeError(f"{MOGE2_BACKEND_ENV}=tensorrt, but tensorrt/trtkit do not import")
        return "tensorrt"
    if choice == "auto":
        return "tensorrt" if tensorrt_available else "torch"
    raise ValueError(f"unknown MoGe-2 backend {choice!r}; choose auto, tensorrt or torch")


@dataclass(frozen=True, slots=True)
class MoGe2Forward:
    """One source frame's raw MoGe-2 outputs: everything a focal change does not touch."""

    points: Float32[Tensor, "nh nw 3"]
    """Affine point-head output at network resolution."""
    mask: Bool[Tensor, "nh nw"]
    """Mask probability > 0.5, at network resolution."""
    metric_scale: Float32[Tensor, ""]
    """Metric-scale head output."""
    normals: Float[Tensor, "h w 3"]
    """Unit camera-space normals (x right, y down, z forward) at input resolution."""


@dataclass(frozen=True, slots=True)
class MoGe2Prediction:
    """One image's post-processed MoGe-2 output for one FoV."""

    depth: Float32[Tensor, "h w"]
    """Metric z-depth, metres."""
    mask: Bool[Tensor, "h w"]
    """MoGe validity (mask > 0.5 and positive depth)."""
    intrinsics: Float[Tensor, "3 3"]
    """Normalised K."""
    normals: Float[Tensor, "h w 3"]
    """Unit camera-space normals."""


@dataclass(frozen=True)
class MetricDepthNormalsPrediction(MetricDepthPrediction):
    """ViPE's metric depth prediction (depth, validity, FoV) plus the normals of the same forward."""

    normals: Float[Tensor, "b h w 3"]
    """Unit camera-space normals, shaped like the input batch."""


def recover_metric_depth(forward: MoGe2Forward, *, fov_x: float | None, image_hw: tuple[int, int]) -> MoGe2Prediction:
    """MoGe-2's post-processing of one forward on the input raster; ``fov_x=None`` recovers the focal as well.

    The cached forward is never modified.
    """
    from monopriors.third_party.moge.model._inference import recover_shift_and_intrinsics

    points: Float32[Tensor, "1 nh nw 3"] = forward.points[None].clone()
    mask: Bool[Tensor, "1 nh nw"] = forward.mask[None].clone()
    # MoGe's stub types fov_x as numbers.Number, which a float is only at runtime.
    camera: Any = recover_shift_and_intrinsics(points, mask, fov_x=fov_x, aspect_ratio=points.shape[2] / points.shape[1])  # pyrefly: ignore[bad-argument-type]
    depth: Float32[Tensor, "1 nh nw"] = (points[..., 2] + camera.shift_b[:, None, None]) * forward.metric_scale
    mask &= depth > 0
    resized: Float32[Tensor, "1 h w"] = F.interpolate(depth[:, None], size=image_hw, mode="bilinear", align_corners=False)[:, 0]
    resized_mask: Bool[Tensor, "1 h w"] = F.interpolate(mask[:, None].float(), size=image_hw, mode="nearest")[:, 0] > 0.5
    return MoGe2Prediction(depth=resized[0], mask=resized_mask[0], intrinsics=camera.intrinsics_b33[0], normals=forward.normals)


DEVICE_KEY_MAX_VALUES: int = 8_000_000
"""Largest image (values) keyed on the GPU; its two weight vectors take 16 bytes per value."""
_device_key_weights: dict[tuple[int, str], Int64[Tensor, "2 n"]] = {}


def image_key(rgb_u8_hw3: UInt8[Tensor, "h w 3"]) -> ImageKey:
    """Cache key of one 8-bit image: its size and a 16-byte digest of its bytes.

    A CUDA image is keyed where it lives: two dot products with fixed pseudo-random 64-bit weights (wrapping modulo
    2**64), so 16 bytes cross to the host instead of the frame; two different images collide with probability
    ~2**-128. A host image (or one too large for the weight vectors) gets a BLAKE2b digest.
    """
    height, width = int(rgb_u8_hw3.shape[0]), int(rgb_u8_hw3.shape[1])
    values: int = rgb_u8_hw3.numel()
    if rgb_u8_hw3.is_cuda and values <= DEVICE_KEY_MAX_VALUES:
        slot: tuple[int, str] = (values, str(rgb_u8_hw3.device))
        weights: Int64[Tensor, "2 n"] | None = _device_key_weights.get(slot)
        if weights is None:
            generator: torch.Generator = torch.Generator().manual_seed(values)
            weights = torch.randint(-(2**62), 2**62, (2, values), generator=generator, dtype=torch.int64).to(rgb_u8_hw3.device)
            _device_key_weights[slot] = weights
        digest: Int64[Tensor, "2"] = (weights * rgb_u8_hw3.reshape(1, -1).to(torch.int64)).sum(dim=1)
        return height, width, digest.cpu().numpy().tobytes()
    data: UInt8[ndarray, "h w 3"] = rgb_u8_hw3.contiguous().cpu().numpy()
    return height, width, hashlib.blake2b(data.tobytes(), digest_size=16).digest()


_ENGINES: WeakSet["MoGe2Engine"] = WeakSet()
"""Every live engine, so :func:`clear_forward_caches` can reach the ones a long-lived ``ModelCache`` holds."""


def clear_forward_caches() -> None:
    """Release every engine's cached forwards (GPU tensors, up to :data:`MOGE2_FORWARD_CACHE_FRAMES` per engine) and
    keep the networks: a run reuses its own frames' forwards, but an app that keeps its engines between runs must not
    hold the last clip's outputs on the GPU until the next click."""
    for engine in list(_ENGINES):
        engine._cache.clear()


class MoGe2Engine:
    """MoGe-2 (normal checkpoint) forwards with a per-source-frame cache of the raw outputs.

    ``predictor_factory(network_hw)`` returns an object with monoprior's ``forward_geometry(rgb_u8_bhw3)``; the
    default builds, per token grid, a TensorRT engine (batch 1) or shares one torch FP16-autocast model. Tests inject
    fakes.
    """

    def __init__(
        self,
        backend: MoGe2Backend,
        encoder: MoGe2Encoder = "vitl",
        *,
        resolution_level: int = MOGE2_RESOLUTION_LEVEL,
        cache_frames: int = MOGE2_FORWARD_CACHE_FRAMES,
        predictor_factory: Callable[[tuple[int, int]], Any] | None = None,
    ) -> None:
        if not 0 <= resolution_level <= 9:
            raise ValueError(f"resolution_level must be within [0, 9], got {resolution_level}")
        if cache_frames < 0:
            raise ValueError(f"cache_frames must be >= 0, got {cache_frames}")
        self.backend: MoGe2Backend = backend
        self.encoder: MoGe2Encoder = encoder
        self.resolution_level: int = resolution_level
        self.cache_frames: int = int(cache_frames)
        self._factory: Callable[[tuple[int, int]], Any] = predictor_factory if predictor_factory is not None else self._default_predictor
        self._predictors: dict[tuple[int, int], Any] = {}
        self._torch_predictor: Any = None
        self._cache: OrderedDict[ImageKey, MoGe2Forward] = OrderedDict()
        self.forward_count: int = 0
        self.postprocess_count: int = 0
        _ENGINES.add(self)

    @staticmethod
    def network_hw(image_hw: tuple[int, int], *, resolution_level: int = MOGE2_RESOLUTION_LEVEL) -> tuple[int, int]:
        """The input's own 14-pixel-aligned token grid at the selected detail level."""
        from monopriors.models.moge_v2 import resolve_num_tokens, token_grid_hw

        return token_grid_hw(image_hw[1] / image_hw[0], resolve_num_tokens(resolution_level))

    def _default_predictor(self, network_hw: tuple[int, int]) -> Any:
        from monopriors.models.moge_v2 import MoGeV2TorchPredictor, MoGeV2TrtPredictor

        if self.backend == "tensorrt":
            from monopriors.models.moge_v2 import export_moge_v2_onnx

            # The ONNX (cached per grid, batch-dynamic) is exported for batches up to 8 as monoprior's default
            # predictor does: a batch-1 export fails (torch.export rejects Dim(min=1, max=1)); the engine is batch 1.
            export_moge_v2_onnx(
                heads="geometry", encoder=self.encoder, image_hw=network_hw, resolution_level=self.resolution_level, max_batch_size=8
            )
            return MoGeV2TrtPredictor(
                encoder=self.encoder, heads="geometry", network_hw_options=(network_hw,), resolution_level=self.resolution_level, batch_size=1
            )
        if self._torch_predictor is None:
            # network_hw_options=None: the exact token grid of each input, like the engines.
            self._torch_predictor = MoGeV2TorchPredictor(
                encoder=self.encoder, heads="geometry", network_hw_options=None, resolution_level=self.resolution_level
            )
        return self._torch_predictor

    def forward(self, rgb_u8_hw3: UInt8[Tensor, "h w 3"]) -> MoGe2Forward:
        """The raw outputs of one image: from the cache, or one network forward."""
        from monopriors.models.moge_v2 import MoGeV2NormalGraphOutput, postprocess_moge_v2_normal

        key: ImageKey = image_key(rgb_u8_hw3)
        cached: MoGe2Forward | None = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        image_hw: tuple[int, int] = (int(rgb_u8_hw3.shape[0]), int(rgb_u8_hw3.shape[1]))
        network_hw: tuple[int, int] = self.network_hw(image_hw, resolution_level=self.resolution_level)
        predictor: Any = self._predictors.get(network_hw)
        if predictor is None:
            predictor = self._predictors[network_hw] = self._factory(network_hw)
        with torch.inference_mode():
            graph: Any = predictor.forward_geometry(rgb_u8_hw3[None])
            if tuple(graph.points_bhw3.shape[1:3]) != network_hw:
                raise ValueError(f"MoGe-2 ran at {tuple(graph.points_bhw3.shape[1:3])}, not the token grid {network_hw}")
            normals: Float[Tensor, "1 h w 3"] = postprocess_moge_v2_normal(
                MoGeV2NormalGraphOutput(normals_bhw3=graph.normals_bhw3, mask_bhw=graph.mask_bhw), image_hw
            ).normals_bhw3
            # TensorRT outputs may alias the runtime's buffers: every cached tensor is a copy.
            result = MoGe2Forward(
                points=graph.points_bhw3[0].float().clone(),
                mask=graph.mask_bhw[0].float() > 0.5,
                metric_scale=graph.metric_scale_b[0].float().clone(),
                normals=normals[0].contiguous(),
            )
        self.forward_count += 1
        if self.cache_frames > 0:
            self._cache[key] = result
            while len(self._cache) > self.cache_frames:
                self._cache.popitem(last=False)
        return result

    def predict(self, rgb_u8_hw3: UInt8[Tensor, "h w 3"], *, fov_x: float | None) -> MoGe2Prediction:
        """One image's depth, mask, normalised K and normals for ``fov_x`` (None: MoGe-2's own focal)."""
        forward: MoGe2Forward = self.forward(rgb_u8_hw3)
        with torch.inference_mode():
            prediction: MoGe2Prediction = recover_metric_depth(forward, fov_x=fov_x, image_hw=(int(rgb_u8_hw3.shape[0]), int(rgb_u8_hw3.shape[1])))
        self.postprocess_count += 1
        return prediction

    def describe(self) -> dict[str, Any]:
        """The engine block of the run record."""
        return {
            "backend": self.backend,
            "network_hw": sorted([int(h), int(w)] for h, w in self._predictors),
            "forward_count": self.forward_count,
            "postprocess_count": self.postprocess_count,
            "cache_frames": self.cache_frames,
        }


def rgb_u8_bhw3(rgb_bchw: Float[Tensor, "b 3 h w"]) -> UInt8[Tensor, "b h w 3"]:
    """RGB in [0, 1] holding 8-bit values -> 8-bit channels-last."""
    return (rgb_bchw.float() * 255.0).round().clamp(0.0, 255.0).to(torch.uint8).permute(0, 2, 3, 1).contiguous()


class MoGe2NormalProvider(MetricTrailMoGe2):
    """MoGe-2 (``moge-2-vit*-normal``) prior with MetricTrail's inference call and validity rule.

    It subclasses ViPE's :class:`MetricTrailMoGe2` so ViPE's ``isinstance`` branches (``predict_metric`` in tracking,
    the run record's ``describe``, the calibration initializer's name) take it like ``moge2-l``; the parent's
    ``__init__`` is skipped, as it loads the ``moge-2-vitl`` torch weights this provider does not use.

    ``engine`` lets wrappers share one set of networks and one forward cache (the calibration and the tracking
    wrapper); when omitted it is built for the backend :func:`select_moge2_backend` picks. Both backends compute in FP16
    (the TensorRT engine is strongly typed from an FP16 ONNX export, the torch predictor autocasts), so ``use_fp16=False``
    is refused rather than ignored.
    """

    def __init__(
        self,
        variant: str = "l",
        *,
        use_fp16: bool = True,
        min_depth_m: float = METRIC_DEPTH_MIN_M,
        max_depth_m: float = METRIC_DEPTH_MAX_M,
        resolution_level: int = MOGE2_RESOLUTION_LEVEL,
        engine: MoGe2Engine | None = None,
    ) -> None:
        DepthEstimationModel.__init__(self)
        if not use_fp16:
            raise ValueError(f"moge2-{variant}-normal runs FP16 on every backend (TensorRT FP16 engine, torch FP16 autocast); FP32 is not available")
        if variant not in MOGE2_ENCODERS:
            raise ValueError(f"unknown MoGe-2 variant {variant!r}; choose one of {sorted(MOGE2_ENCODERS)}")
        if not 0.0 < float(min_depth_m) < float(max_depth_m):
            raise ValueError(f"invalid metric depth range: [{min_depth_m}, {max_depth_m}]")
        if engine is None:
            backend: MoGe2Backend = select_moge2_backend(os.environ.get(MOGE2_BACKEND_ENV, "auto"), tensorrt_available=tensorrt_importable())
            engine = MoGe2Engine(backend, MOGE2_ENCODERS[variant], resolution_level=resolution_level)
        self.engine: MoGe2Engine = engine
        self.variant: str = variant
        self.min_depth_m: float = float(min_depth_m)
        self.max_depth_m: float = float(max_depth_m)

    @property
    def depth_type(self) -> DepthType:
        return DepthType.MODEL_METRIC_DEPTH

    @property
    def precision(self) -> str:
        return "TensorRT-FP16" if self.engine.backend == "tensorrt" else "FP16-autocast"

    def predict_metric(self, rgb_bchw: Float[Tensor, "b 3 h w"], *, focal_px: float) -> MetricDepthNormalsPrediction:
        """K-conditioned metric depth, validity and normals for a batch."""
        if rgb_bchw.ndim != 4:
            raise ValueError(f"expected a (B,3,H,W) batch, got {tuple(rgb_bchw.shape)}")
        fov: float = fov_x_degrees(focal_px, int(rgb_bchw.shape[-1]))
        rows: list[MoGe2Prediction] = [self.engine.predict(image, fov_x=fov) for image in rgb_u8_bhw3(rgb_bchw)]
        depth: Float32[Tensor, "b h w"] = torch.stack([row.depth for row in rows])
        valid: Bool[Tensor, "b h w"] = torch.isfinite(depth) & torch.stack([row.mask for row in rows])
        valid &= depth >= self.min_depth_m
        valid &= depth <= self.max_depth_m
        normals: Float[Tensor, "b h w 3"] = torch.stack([row.normals for row in rows])
        return MetricDepthNormalsPrediction(depth=depth, valid=valid, fov_x_degrees=fov, normals=normals)

    def estimate_free_fov_k4(self, rgb_bchw: Float[Tensor, "1 3 h w"]) -> Float64[ndarray, "4"]:
        """MoGe-2's own pinhole K (no FoV given) for one image, in pixels."""
        if rgb_bchw.ndim != 4 or rgb_bchw.shape[0] != 1:
            raise ValueError(f"expected a (1,3,H,W) image, got {tuple(rgb_bchw.shape)}")
        intrinsics: Float[Tensor, "3 3"] = self.engine.predict(rgb_u8_bhw3(rgb_bchw)[0], fov_x=None).intrinsics
        image_hw: tuple[int, int] = (int(rgb_bchw.shape[-2]), int(rgb_bchw.shape[-1]))
        return np.asarray(normalized_to_pixel_k4(intrinsics.detach().float().cpu().numpy(), image_hw), dtype=np.float64)

    def describe(self) -> dict[str, Any]:
        """The depth-provider block of the run record."""
        from monopriors.models.moge_v2 import MOGE_V2_CHECKPOINTS

        repo, revision = MOGE_V2_CHECKPOINTS[MOGE2_ENCODERS[self.variant]]
        return {
            "model": "MoGe-2",
            "registry_name": f"moge2-{self.variant}-normal",
            "checkpoint": f"{repo}@{revision}",
            "precision": self.precision,
            "depth_type": self.depth_type.name,
            "resolution_level": self.engine.resolution_level,
            "fov_x": "float64 math.atan of the bound focal",
            "post_processing": "recover_shift_and_intrinsics(points, mask > 0.5, fov_x) at network resolution; "
            "(z + shift) * metric_scale; depth bilinear, mask nearest to the input raster",
            "valid_rule": f"mask & finite & {self.min_depth_m} <= depth <= {self.max_depth_m} m",
            "engine": self.engine.describe(),
        }


def make_provider(
    variant: str, *, use_fp16: bool = True, resolution_level: int = MOGE2_RESOLUTION_LEVEL, model_cache: ModelCache | None = None
) -> MoGe2NormalProvider:
    """ViPE's depth factory: share an engine per variant and detail level through ``model_cache``."""
    if model_cache is None:
        return MoGe2NormalProvider(variant, use_fp16=use_fp16, resolution_level=resolution_level)
    engine: MoGe2Engine = model_cache.get(
        f"metrictrail/depth/moge2-{variant}-normal/level-{resolution_level}/engine",
        lambda: MoGe2NormalProvider(variant, use_fp16=use_fp16, resolution_level=resolution_level).engine,
    )
    return MoGe2NormalProvider(variant, use_fp16=use_fp16, engine=engine)


def register_providers(*, resolution_level: int = MOGE2_RESOLUTION_LEVEL) -> None:
    """Register every :data:`PROVIDER_NAMES` entry in ViPE's depth-provider registry (idempotent)."""
    from vipe.metrictrail.depth import DEPTH_PROVIDERS

    for name, variant in PROVIDER_NAMES.items():
        DEPTH_PROVIDERS[name] = functools.partial(make_provider, variant, resolution_level=resolution_level)
