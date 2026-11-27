"""Backend selection: one tensor-function contract over PyTorch, ONNX Runtime, and TensorRT.

Backend choice is a config value, not a code path. Models accept a
``BackendConfig`` (a tyro subcommand union) and build their runtime through
:func:`create_runtime_from_onnx` — ONNX is the interchange format, so the same
artifact feeds both the ONNX Runtime session and the TensorRT engine builder.
The torch backend wraps a model-provided ``nn.Module`` directly and is offered
by model families that ship PyTorch weights.

``auto`` (:class:`AutoBackendConfig`) is the default of the ONNX-artifact
models: TensorRT where an NVIDIA GPU and the TensorRT bindings are present,
ONNX Runtime on CUDA where only torch has CUDA, ONNX Runtime on the CPU
elsewhere (:func:`resolve_backend`). :func:`create_runtime_from_onnx` resolves
it; a model that reads its backend before then resolves it first.
"""

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeAlias

import torch
import tyro

from trtkit.base import TensorRuntime
from trtkit.onnx_cuda import OnnxRuntime, OrtDevice
from trtkit.tensorrt_runtime import TensorRtRuntime
from trtkit.trt_builder import DEFAULT_TRT_CACHE_DIR, TrtBuildConfig, ensure_engine

AutocastName = Literal["fp32", "fp16", "bf16"]
_AUTOCAST_DTYPES: dict[AutocastName, torch.dtype | None] = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}


@dataclass(frozen=True, slots=True)
class TorchBackendConfig:
    """Run the model's native PyTorch module (eager)."""

    autocast: AutocastName = "fp16"
    """CUDA autocast precision; ``fp32`` disables autocast."""
    max_batch_size: int = 8
    """Largest batch a single runtime call may submit."""

    @property
    def autocast_dtype(self) -> torch.dtype | None:
        """Torch dtype for autocast, or ``None`` when disabled."""
        return _AUTOCAST_DTYPES[self.autocast]


@dataclass(frozen=True, slots=True)
class OnnxBackendConfig:
    """Run the model's ONNX artifact through ONNX Runtime (IOBinding onto torch tensors)."""

    device: OrtDevice = "cuda"
    """``cuda`` (GPU-resident; not on the GB10, whose conda-forge build has no kernels) or ``cpu``."""
    device_id: int = 0
    """CUDA device ordinal; unused on the CPU."""
    max_batch_size: int = 32
    """Batch cap for dynamic-batch graphs; static graphs use their baked batch."""


@dataclass(frozen=True, slots=True)
class TensorRtBackendConfig:
    """Build (once, cached per machine) and run a dynamic-batch TensorRT engine from the model's ONNX artifact."""

    max_batch_size: int = 32
    """Largest batch the engine accepts (dynamic profile max); static-batch ONNX graphs use their baked batch."""
    opt_batch_size: int = 8
    """Batch size TensorRT tunes kernels for (dynamic profile optimum)."""
    use_cuda_graph: bool = False
    """Capture and replay CUDA graphs around the engine launch (one per batch size)."""
    cache_dir: Path = field(default_factory=lambda: DEFAULT_TRT_CACHE_DIR)
    """Machine-local engine cache directory."""
    fp8_attention: bool = False
    """Quantize native attention normalization output to FP8 with scale 1/448."""
    extra_output_patterns: tuple[str, ...] = ()
    """Tensor-name substrings to materialize as extra engine outputs (defeats
    miscompiled fusions; see :class:`trtkit.trt_builder.TrtBuildConfig`)."""


@dataclass(frozen=True, slots=True)
class AutoBackendConfig:
    """TensorRT on an NVIDIA GPU with the TensorRT bindings, else ONNX Runtime: on CUDA where torch has it, else on the CPU."""

    max_batch_size: int = 32
    """Largest batch a single call may submit, on either backend."""
    opt_batch_size: int = 8
    """Batch size TensorRT tunes kernels for; unused on ONNX Runtime."""


def tensorrt_available() -> bool:
    """Whether this process can build and run TensorRT engines: a CUDA device and the ``tensorrt`` bindings."""
    return torch.cuda.is_available() and importlib.util.find_spec("tensorrt") is not None


ResolvedBackendConfig: TypeAlias = OnnxBackendConfig | TensorRtBackendConfig
"""What an ONNX-artifact model runs on once ``auto`` is resolved."""


def resolve_backend(backend: "OnnxOrTrtBackendConfig") -> ResolvedBackendConfig:
    """The concrete backend ``backend`` stands for: :class:`AutoBackendConfig` resolved here, any other one as given.

    Args:
        backend: A backend config.

    Returns:
        ``backend`` itself, or what ``auto`` is on this host: TensorRT; else, where torch has CUDA (the models then feed
        CUDA frames), ONNX Runtime on CUDA; else ONNX Runtime on the CPU.
    """
    if not isinstance(backend, AutoBackendConfig):
        return backend
    if tensorrt_available():
        return TensorRtBackendConfig(max_batch_size=backend.max_batch_size, opt_batch_size=backend.opt_batch_size)
    return OnnxBackendConfig(device="cuda" if torch.cuda.is_available() else "cpu", max_batch_size=backend.max_batch_size)


if TYPE_CHECKING:
    BackendConfig = TorchBackendConfig | OnnxBackendConfig | TensorRtBackendConfig
    OnnxOrTrtBackendConfig = AutoBackendConfig | OnnxBackendConfig | TensorRtBackendConfig
else:
    BackendConfig = tyro.extras.subcommand_type_from_defaults(
        {"torch": TorchBackendConfig(), "onnx": OnnxBackendConfig(), "tensorrt": TensorRtBackendConfig()}, prefix_names=False
    )
    OnnxOrTrtBackendConfig = tyro.extras.subcommand_type_from_defaults(
        {"auto": AutoBackendConfig(), "onnx": OnnxBackendConfig(), "tensorrt": TensorRtBackendConfig()}, prefix_names=False
    )


def create_runtime_from_onnx(onnx_path: Path, backend: "OnnxOrTrtBackendConfig") -> TensorRuntime:
    """Create an ONNX Runtime or TensorRT runtime from one ONNX interchange file.

    Args:
        onnx_path: The model's ONNX artifact (exported once from torch, or
            downloaded from a model zoo).
        backend: Which accelerated backend to run it on; ``auto`` is resolved here.

    Returns:
        A ready runtime satisfying the trtkit tensor-function contract.
    """
    backend = resolve_backend(backend)
    if isinstance(backend, OnnxBackendConfig):
        return OnnxRuntime(onnx_path, device=backend.device, device_id=backend.device_id, max_batch_size=backend.max_batch_size)
    from trtkit.onnx_graph import onnx_static_batch_size

    # Graphs exported with a fixed batch dictate the engine batch themselves;
    # dynamic-batch graphs get a 1..max profile tuned at the opt batch.
    static_batch: int | None = onnx_static_batch_size(onnx_path)
    build_config = TrtBuildConfig(
        max_batch_size=static_batch if static_batch is not None else backend.max_batch_size,
        opt_batch_size=static_batch if static_batch is not None else backend.opt_batch_size,
        extra_output_patterns=backend.extra_output_patterns,
        fp8_attention=backend.fp8_attention,
    )
    engine_path: Path = ensure_engine(onnx_path, build_config, cache_dir=backend.cache_dir)
    return TensorRtRuntime(engine_path, use_cuda_graph=backend.use_cuda_graph)
