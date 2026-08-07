"""Backend selection: one tensor-function contract over PyTorch, ONNX Runtime, and TensorRT.

Backend choice is a config value, not a code path. Models accept a
``BackendConfig`` (a tyro subcommand union) and build their runtime through
:func:`create_runtime_from_onnx` — ONNX is the interchange format, so the same
artifact feeds both the ONNX Runtime CUDA session and the TensorRT engine
builder. The torch backend wraps a model-provided ``nn.Module`` directly and is
offered by model families that ship PyTorch weights.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import torch
import tyro

from trtkit.base import TensorRuntime
from trtkit.onnx_cuda import OnnxCudaRuntime
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
    """Run the model's ONNX artifact through ONNX Runtime's CUDA provider (IOBinding, GPU-resident)."""

    device_id: int = 0
    """CUDA device ordinal."""
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


if TYPE_CHECKING:
    BackendConfig = TorchBackendConfig | OnnxBackendConfig | TensorRtBackendConfig
    OnnxOrTrtBackendConfig = OnnxBackendConfig | TensorRtBackendConfig
else:
    BackendConfig = tyro.extras.subcommand_type_from_defaults(
        {"torch": TorchBackendConfig(), "onnx": OnnxBackendConfig(), "tensorrt": TensorRtBackendConfig()}, prefix_names=False
    )
    OnnxOrTrtBackendConfig = tyro.extras.subcommand_type_from_defaults(
        {"onnx": OnnxBackendConfig(), "tensorrt": TensorRtBackendConfig()}, prefix_names=False
    )


def create_runtime_from_onnx(onnx_path: Path, backend: "OnnxBackendConfig | TensorRtBackendConfig") -> TensorRuntime:
    """Create an ONNX Runtime or TensorRT runtime from one ONNX interchange file.

    Args:
        onnx_path: The model's ONNX artifact (exported once from torch, or
            downloaded from a model zoo).
        backend: Which accelerated backend to run it on.

    Returns:
        A ready runtime satisfying the trtkit tensor-function contract.
    """
    if isinstance(backend, OnnxBackendConfig):
        return OnnxCudaRuntime(onnx_path, device_id=backend.device_id, max_batch_size=backend.max_batch_size)
    from trtkit.onnx_graph import onnx_static_batch_size

    # Graphs exported with a fixed batch dictate the engine batch themselves;
    # dynamic-batch graphs get a 1..max profile tuned at the opt batch.
    static_batch: int | None = onnx_static_batch_size(onnx_path)
    build_config = TrtBuildConfig(
        max_batch_size=static_batch if static_batch is not None else backend.max_batch_size,
        opt_batch_size=static_batch if static_batch is not None else backend.opt_batch_size,
    )
    engine_path: Path = ensure_engine(onnx_path, build_config, cache_dir=backend.cache_dir)
    return TensorRtRuntime(engine_path, use_cuda_graph=backend.use_cuda_graph)
