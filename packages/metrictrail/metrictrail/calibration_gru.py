"""Weight-keyed TensorRT ConvGRU for SharedK's DROID network.

Only network compilation artifacts and runtimes are cached. Every call consumes the current
hidden state, and returns independent output storage. Training, CPU, FP32 and edge counts outside
1..128 use the original PyTorch forward. The ONNX files and engines live in trtkit's cache
(``TRTKIT_TRT_CACHE``, default ``~/.cache/trtkit/trt``) under ``droid-gru/``; a Space ships them prebuilt.
"""

import copy
import hashlib
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Literal, TypeAlias
from weakref import WeakKeyDictionary

import torch
from jaxtyping import Float16
from serde import serde
from torch import Tensor
from vipe.slam.networks.droid_net import ConvGRU

from metrictrail.calibration_graphs import CalibrationAccelerationError

GruBackend: TypeAlias = Literal["torch", "tensorrt"]
HIDDEN_CHANNELS: int = 128
"""Channels of the GRU's hidden state."""
CONTEXT_CHANNELS: int = 320
"""Channels of its concatenated inputs."""
MIN_EDGES, OPT_EDGES, MAX_EDGES = 1, 48, 128
"""The engine's dynamic edge-count profile; other counts run the PyTorch forward."""


@dataclass(frozen=True, slots=True)
class GruEngineKey:
    """Identity of a frozen FP16 GRU graph and its dynamic edge-count profile."""

    weights: str
    height: int
    width: int
    min_edges: int = MIN_EDGES
    opt_edges: int = OPT_EDGES
    max_edges: int = MAX_EDGES
    precision: Literal["fp16"] = "fp16"

    @property
    def stem(self) -> str:
        """Portable ONNX name; trtkit adds the profile, ONNX hash, TRT version and GPU to engine names."""
        return f"droid-gru-v2-{self.weights[:16]}-{self.height}x{self.width}-{self.precision}"


@serde(deny_unknown_fields=True)
@dataclass
class GruStats:
    """Per-run counters: compilation time is explicit and remains included in end-to-end wall time."""

    backend: GruBackend
    export_seconds: float = 0.0
    build_seconds: float = 0.0
    load_seconds: float = 0.0
    engine_calls: int = 0
    eager_calls: int = 0
    engine_files: list[str] = field(default_factory=list)


def weights_digest(model: ConvGRU) -> str:
    """Hash the actual frozen weights, names, dtypes and shapes, once on first use."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(f"{name}:{tensor.dtype}:{tuple(tensor.shape)}:".encode())
        digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@dataclass
class ModelEngines:
    """Weak model key's value: no model reference, so clearing ModelCache releases it."""

    versions: tuple[tuple[int, int], ...]
    digest: str
    engines: dict[tuple[GruEngineKey, str], "GruRuntime"] = field(default_factory=dict)


_models: WeakKeyDictionary[ConvGRU, ModelEngines] = WeakKeyDictionary()
_original_forward = ConvGRU.forward


class GruRuntime:
    """One trtkit runtime and its shipping artifact, retaining no source model."""

    def __init__(self, model: ConvGRU, key: GruEngineKey, root: Path, stats: GruStats) -> None:
        from trtkit import TensorRtRuntime, TrtBuildConfig, cached_engine_path, ensure_engine, export_onnx
        from trtkit.artifacts import build_once

        root.mkdir(parents=True, exist_ok=True)
        onnx = root / f"{key.stem}.onnx"

        def export() -> None:
            started = time.perf_counter()
            frozen = copy.deepcopy(model).half().eval()
            frozen.forward = _original_forward.__get__(frozen, ConvGRU)
            device = next(frozen.parameters()).device
            hidden = torch.zeros(2, HIDDEN_CHANNELS, key.height, key.width, device=device, dtype=torch.float16)
            context = torch.zeros(2, CONTEXT_CHANNELS, key.height, key.width, device=device, dtype=torch.float16)
            with torch.autocast("cuda", enabled=False):
                export_onnx(frozen, (hidden, context), onnx, input_names=["hidden", "context"], output_names=["updated"],
                            dynamic_batch_max=key.max_edges, exporter="torchscript")
            stats.export_seconds += time.perf_counter() - started

        build_once(onnx, export)
        config = TrtBuildConfig(max_batch_size=key.max_edges, opt_batch_size=key.opt_edges, allow_tf32=False, workspace_gib=2.0)
        path = cached_engine_path(onnx, config, cache_dir=root)
        missing = not path.is_file()
        started = time.perf_counter()
        self.path = ensure_engine(onnx, config, cache_dir=root)
        if missing:
            stats.build_seconds += time.perf_counter() - started
        started = time.perf_counter()
        self.runtime = TensorRtRuntime(self.path)
        stats.load_seconds += time.perf_counter() - started

    def __call__(self, hidden: Float16[Tensor, "edges hidden h w"], context: Float16[Tensor, "edges context h w"]) -> Float16[Tensor, "edges hidden h w"]:
        return self.runtime({"hidden": hidden, "context": context})["updated"].clone()


@contextmanager
def calibration_gru(backend: GruBackend = "torch", *, cache_dir: Path | None = None) -> Iterator[GruStats]:
    """Select the backend for a run. A runtime lives as long as its model (weak key): an app that keeps its models
    (``ModelCache``) reuses them across runs, and a run that builds a new model builds or loads its runtime again.

    A model's parameter storage/version changes invalidate its digest and runtimes. Engines are keyed
    by actual weights and raster, never by camera calibration. A runtime's output is cloned before return.
    """
    stats = GruStats(backend)
    if backend == "torch":
        yield stats
        return
    from trtkit.trt_builder import DEFAULT_TRT_CACHE_DIR

    root = (cache_dir or DEFAULT_TRT_CACHE_DIR / "droid-gru").expanduser().resolve()
    original = ConvGRU.forward

    @wraps(original)
    def forward(self: ConvGRU, net: Tensor, *inputs: Tensor) -> Tensor:
        if (self.training or torch.is_grad_enabled() or not net.is_cuda or net.dtype != torch.float16
                or not MIN_EDGES <= net.shape[0] <= MAX_EDGES or net.shape[1] != HIDDEN_CHANNELS
                or sum(value.shape[1] for value in inputs) != CONTEXT_CHANNELS):
            stats.eager_calls += 1
            return original(self, net, *inputs)
        versions = tuple((parameter.data_ptr(), parameter._version) for parameter in self.parameters())
        saved = _models.get(self)
        if saved is None or saved.versions != versions:
            saved = ModelEngines(versions, weights_digest(self))
            _models[self] = saved
        key = GruEngineKey(saved.digest, net.shape[-2], net.shape[-1])
        slot = (key, str(root))
        runtime = saved.engines.get(slot)
        if runtime is None:
            try:
                runtime = GruRuntime(self, key, root, stats)
            except Exception as error:  # ViPE's sweep would turn it into the MoGe-2 focal fallback
                raise CalibrationAccelerationError(f"the TensorRT calibration GRU failed to build or load: {error!r}") from error
            saved.engines[slot] = runtime
        path = str(runtime.path)
        if path not in stats.engine_files:
            stats.engine_files.append(path)
        stats.engine_calls += 1
        return runtime(net, torch.cat(inputs, dim=1))

    ConvGRU.forward = forward
    try:
        yield stats
    finally:
        ConvGRU.forward = original
