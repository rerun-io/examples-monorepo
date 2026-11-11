"""Weight-keyed TensorRT ConvGRU for SharedK's DROID network.

Only network compilation artifacts and runtimes are cached. Every call consumes the current
hidden state, and returns independent output storage. Training, CPU, FP32 and edge counts outside
1..128 use the original PyTorch forward. Set METRICTRAIL_TRT_CACHE to ship prebuilt artifacts.
"""

import copy
import hashlib
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal, TypeAlias
from weakref import WeakKeyDictionary

import torch
from jaxtyping import Float16
from serde import serde
from torch import Tensor
from vipe.slam.networks.droid_net import ConvGRU
from vipe.utils.model_cache import ModelCache

GruBackend: TypeAlias = Literal["torch", "tensorrt"]


@dataclass(frozen=True, slots=True)
class GruEngineKey:
    """Identity of a frozen FP16 GRU graph and its dynamic edge-count profile."""

    weights: str
    height: int
    width: int
    min_edges: int = 1
    opt_edges: int = 48
    max_edges: int = 128
    precision: Literal["fp16"] = "fp16"

    @property
    def stem(self) -> str:
        """Portable ONNX name; trtkit adds the profile, ONNX hash, TRT version and GPU to engine names."""
        return f"droid-gru-v1-{self.weights[:16]}-{self.height}x{self.width}-{self.precision}"


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
        from trtkit import TensorRtRuntime, TrtBuildConfig, cached_engine_path, ensure_engine
        from trtkit.artifacts import build_once

        root.mkdir(parents=True, exist_ok=True)
        onnx = root / f"{key.stem}.onnx"

        def export() -> None:
            started = time.perf_counter()
            frozen = copy.deepcopy(model).half().eval()
            frozen.forward = _original_forward.__get__(frozen, ConvGRU)
            device = next(frozen.parameters()).device
            with torch.inference_mode(), torch.autocast("cuda", enabled=False), TemporaryDirectory(dir=root) as temporary:
                hidden = torch.zeros(2, 128, key.height, key.width, device=device, dtype=torch.float16)
                context = torch.zeros(2, 320, key.height, key.width, device=device, dtype=torch.float16)
                staged = Path(temporary) / onnx.name
                torch.onnx.export(frozen, (hidden, context), staged, input_names=["hidden", "context"], output_names=["updated"],
                                  dynamic_axes={"hidden": {0: "edges"}, "context": {0: "edges"}, "updated": {0: "edges"}},
                                  opset_version=17, dynamo=False, external_data=False)
                staged.replace(onnx)
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

    def __call__(self, hidden: Float16[Tensor, "edges 128 h w"], context: Float16[Tensor, "edges 320 h w"]) -> Float16[Tensor, "edges 128 h w"]:
        return self.runtime({"hidden": hidden, "context": context})["updated"].clone()


@contextmanager
def calibration_gru(backend: GruBackend = "torch", *, model_cache: ModelCache | None = None, cache_dir: Path | None = None) -> Iterator[GruStats]:
    """Select the backend for a run, reusing runtimes with app-owned models across runs.

    A model's parameter storage/version changes invalidate its digest and runtimes. Engines are keyed
    by actual weights and raster, never by camera calibration. A runtime's output is cloned before return.
    """
    stats = GruStats(backend)
    if backend == "torch":
        yield stats
        return
    if backend != "tensorrt":
        raise ValueError(f"unknown calibration GRU backend: {backend}")
    root = (cache_dir or Path(os.environ.get("METRICTRAIL_TRT_CACHE", "~/.cache/metrictrail/droid-gru"))).expanduser().resolve()
    original = ConvGRU.forward

    @wraps(original)
    def forward(self: ConvGRU, net: Tensor, *inputs: Tensor) -> Tensor:
        hidden = net
        if (self.training or torch.is_grad_enabled() or not hidden.is_cuda or hidden.dtype != torch.float16
                or not 1 <= hidden.shape[0] <= 128 or hidden.shape[1] != 128 or sum(value.shape[1] for value in inputs) != 320):
            stats.eager_calls += 1
            return original(self, hidden, *inputs)
        versions = tuple((parameter.data_ptr(), parameter._version) for parameter in self.parameters())
        saved = _models.get(self)
        if saved is None or saved.versions != versions:
            saved = ModelEngines(versions, weights_digest(self))
            _models[self] = saved
        key = GruEngineKey(saved.digest, hidden.shape[-2], hidden.shape[-1])
        slot = (key, str(root))
        runtime = saved.engines.get(slot)
        if runtime is None:
            def build() -> GruRuntime:
                return GruRuntime(self, key, root, stats)
            runtime = build() if model_cache is None else model_cache.get(f"metrictrail/gru/{key!r}/{root}/{hidden.device}", build)
            saved.engines[slot] = runtime
        path = str(runtime.path)
        if path not in stats.engine_files:
            stats.engine_files.append(path)
        stats.engine_calls += 1
        return runtime(hidden, torch.cat(inputs, dim=1))

    ConvGRU.forward = forward
    try:
        yield stats
    finally:
        ConvGRU.forward = original
