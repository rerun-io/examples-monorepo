"""One PyTorch → ONNX → TensorRT home for the monorepo.

trtkit consolidates the TensorRT runners and builders previously copied
between posekit, wilor-nano, sapiens2-pose / sapiens-coco133-pose, prompt-da,
and mamma. It splits into two layers (vision-rt's crate layering, in Python):

- **runtime** — the backend-neutral tensor-function contract
  (:class:`trtkit.base.TensorRuntime`) with torch / ONNX Runtime CUDA /
  TensorRT implementations: :mod:`trtkit.base`, :mod:`trtkit.torch_runtime`,
  :mod:`trtkit.onnx_cuda`, :mod:`trtkit.tensorrt_runtime`,
  :mod:`trtkit.backends`.
- **hub** — artifact identity and machine-local engine caching:
  :mod:`trtkit.trt_builder` (build + cache key + manifest) and
  :mod:`trtkit.onnx_graph` (generic ONNX graph surgery).

ONNX files are the universally portable artifacts. TensorRT engines remain
version-specific and are device-specific unless explicitly built for compatible
hardware. Model-specific concerns — export wrappers, checkpoint resolution,
pre/postprocessing — stay in the model packages.
"""

import os

if os.environ.get("PIXI_DEV_MODE") == "1":
    from beartype.claw import beartype_this_package

    beartype_this_package()

from trtkit.backends import (
    BackendConfig,
    OnnxBackendConfig,
    OnnxOrTrtBackendConfig,
    TensorRtBackendConfig,
    TorchBackendConfig,
    create_runtime_from_onnx,
)
from trtkit.base import RuntimeSpec, TensorRuntime, TensorSpec, run_chunked, validate_runtime_inputs
from trtkit.onnx_cuda import OnnxCudaRuntime
from trtkit.onnx_export import DynamicDim, DynamicDims, export_onnx, shallow_module_copy, sweep_stale_onnx_exports
from trtkit.onnx_graph import onnx_static_batch_size
from trtkit.tensorrt_runtime import TensorRtDynamicRuntime, TensorRtRuntime
from trtkit.torch_runtime import TorchRuntime
from trtkit.trt_builder import (
    DEFAULT_TRT_CACHE_DIR,
    HardwareCompatibility,
    InputShapeProfile,
    TrtBuildConfig,
    build_engine,
    cached_engine_path,
    ensure_engine,
    onnx_content_hash,
)

__all__ = (
    "BackendConfig",
    "DEFAULT_TRT_CACHE_DIR",
    "DynamicDim",
    "DynamicDims",
    "HardwareCompatibility",
    "InputShapeProfile",
    "OnnxBackendConfig",
    "OnnxCudaRuntime",
    "OnnxOrTrtBackendConfig",
    "RuntimeSpec",
    "TensorRtBackendConfig",
    "TensorRtDynamicRuntime",
    "TensorRtRuntime",
    "TensorRuntime",
    "TensorSpec",
    "TorchBackendConfig",
    "TorchRuntime",
    "TrtBuildConfig",
    "build_engine",
    "cached_engine_path",
    "create_runtime_from_onnx",
    "ensure_engine",
    "export_onnx",
    "onnx_content_hash",
    "onnx_static_batch_size",
    "run_chunked",
    "shallow_module_copy",
    "sweep_stale_onnx_exports",
    "validate_runtime_inputs",
)
