"""Replay fixed feature extractors with CUDA graphs during one MetricTrail run.

The frozen DINO backbone and the two DPVO convolutional encoders submit thousands of small kernels
per frame. Their weights and rasters stay fixed during tracking. Graph replay preserves those kernels
while removing Python dispatch; patch sampling, graph updates, and capture logging remain outside it.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from typing import TypeVar
from weakref import WeakKeyDictionary

import torch
from dpvo.extractor import BasicEncoder4
from jaxtyping import Float
from torch import Tensor, nn
from vipe.metrictrail.dino_fusion import FrozenDinoV3

FeatureModel = TypeVar("FeatureModel", bound=nn.Module)


@dataclass(frozen=True, slots=True)
class FeatureGraphKey:
    """Input and precision state fixed by one model's feature graph."""
    shape: torch.Size
    """Input tensor dimensions, including batch and raster."""
    dtype: torch.dtype
    """Input storage dtype."""
    device: torch.device
    """Device that owns the input and graph buffers."""
    autocast: bool
    """Whether CUDA autocast is active."""
    autocast_dtype: torch.dtype
    """Autocast target, separate for the FP32 recovery path."""
    matmul_tf32: bool
    """Matrix multiplication precision captured by the graph."""
    cudnn_tf32: bool
    """Convolution precision captured by the graph."""


@dataclass(frozen=True, slots=True)
class FeatureGraph:
    """Persistent buffers and captured work for one model, raster, and precision policy."""

    image: Float[Tensor, "..."]
    """Persistent input copied before replay."""
    features: Float[Tensor, "..."]
    """Persistent output copied before returning to the tracker."""
    graph: torch.cuda.CUDAGraph
    """The extractor's captured CUDA work."""


@contextmanager
def feature_cuda_graphs() -> Iterator[None]:
    """Replay inference feature extraction for this run, restoring ordinary forwards on exit.

    Each returned tensor owns its storage. A later replay cannot overwrite a feature map retained by
    DPVO. Raster and autocast changes (including FP32 recovery) select separate captured graphs.
    Weak model keys release graph pools when recovery discards the old network.
    """
    cache: WeakKeyDictionary[nn.Module, dict[FeatureGraphKey, FeatureGraph]] = WeakKeyDictionary()

    def replay(model: FeatureModel, images: Float[Tensor, "..."], original: Callable[[FeatureModel, Tensor], Tensor]) -> Float[Tensor, "..."]:
        if not images.is_cuda or model.training or torch.is_grad_enabled():
            return original(model, images)
        key: FeatureGraphKey = FeatureGraphKey(
            images.shape, images.dtype, images.device,
            torch.is_autocast_enabled("cuda"), torch.get_autocast_dtype("cuda"),
            bool(torch.backends.cuda.matmul.allow_tf32), bool(torch.backends.cudnn.allow_tf32),
        )
        model_cache: dict[FeatureGraphKey, FeatureGraph] = cache.setdefault(model, {})
        saved: FeatureGraph | None = model_cache.get(key)
        if saved is None:
            static: Float[Tensor, "..."] = images.clone()
            current: torch.cuda.Stream = torch.cuda.current_stream(images.device)
            warmup: torch.cuda.Stream = torch.cuda.Stream(device=images.device)
            warmup.wait_stream(current)
            # Autocast's temporary weight cache must not outlive allocations made outside the graph.
            with torch.autocast("cuda", enabled=key.autocast, dtype=key.autocast_dtype, cache_enabled=False):
                with torch.cuda.stream(warmup):
                    for _ in range(3):
                        original(model, static)
                current.wait_stream(warmup)
                graph: torch.cuda.CUDAGraph = torch.cuda.CUDAGraph()
                # The independent depth producer submits work from another thread during capture.
                with torch.cuda.graph(graph, capture_error_mode="thread_local"):
                    features: Float[Tensor, "..."] = original(model, static)
            saved = FeatureGraph(static, features, graph)
            model_cache[key] = saved
        saved.image.copy_(images)
        saved.graph.replay()
        return saved.features.clone()

    # DINO's forward observer does PCA and host logging; only its tensor extractor belongs in the graph.
    dino_original = FrozenDinoV3._extract_chunk
    encoder_original = BasicEncoder4.forward

    @wraps(dino_original)
    def dino(self: FrozenDinoV3, images: Float[Tensor, "b 3 h w"]) -> Float[Tensor, "b c ph pw"]:
        return replay(self, images, dino_original)

    @wraps(encoder_original)
    def encoder(self: BasicEncoder4, x: Float[Tensor, "b n c h w"]) -> Float[Tensor, "b n features oh ow"]:
        return replay(self, x, encoder_original)

    FrozenDinoV3._extract_chunk = dino
    BasicEncoder4.forward = encoder
    try:
        yield
    finally:
        FrozenDinoV3._extract_chunk = dino_original
        BasicEncoder4.forward = encoder_original
