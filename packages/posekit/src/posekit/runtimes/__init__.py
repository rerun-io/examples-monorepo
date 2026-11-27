"""Runtime backends, re-exported from ``trtkit`` (their consolidated home).

posekit's runtime layer — the tensor-function contract, the torch/ONNX
Runtime/TensorRT backends, and the engine builder/cache — moved to the shared
``trtkit`` package so every model package in the monorepo uses one
implementation. This module keeps the historical ``posekit.runtimes`` import
surface as a plain re-export.
"""

from trtkit import (
    AutoBackendConfig,
    BackendConfig,
    OnnxBackendConfig,
    OnnxCudaRuntime,
    OnnxOrTrtBackendConfig,
    OnnxRuntime,
    OrtDevice,
    ResolvedBackendConfig,
    RuntimeSpec,
    TensorRtBackendConfig,
    TensorRtRuntime,
    TensorRuntime,
    TensorSpec,
    TorchBackendConfig,
    TorchRuntime,
    create_runtime_from_onnx,
    require_cuda_kernels,
    resolve_backend,
    run_chunked,
    tensorrt_available,
    validate_runtime_inputs,
)
from trtkit.trt_builder import DEFAULT_TRT_CACHE_DIR, TrtBuildConfig, ensure_engine

__all__ = (
    "AutoBackendConfig",
    "BackendConfig",
    "DEFAULT_TRT_CACHE_DIR",
    "OnnxBackendConfig",
    "OnnxCudaRuntime",
    "OnnxOrTrtBackendConfig",
    "OnnxRuntime",
    "OrtDevice",
    "RuntimeSpec",
    "TensorRtBackendConfig",
    "TensorRtRuntime",
    "TensorRuntime",
    "TensorSpec",
    "TorchBackendConfig",
    "TorchRuntime",
    "TrtBuildConfig",
    "create_runtime_from_onnx",
    "ensure_engine",
    "require_cuda_kernels",
    "ResolvedBackendConfig",
    "resolve_backend",
    "run_chunked",
    "tensorrt_available",
    "validate_runtime_inputs",
)
