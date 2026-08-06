"""Runtime backends, re-exported from ``trtkit`` (their consolidated home).

posekit's runtime layer — the tensor-function contract, the torch/ONNX
Runtime/TensorRT backends, and the engine builder/cache — moved to the shared
``trtkit`` package so every model package in the monorepo uses one
implementation. This module keeps the historical ``posekit.runtimes`` import
surface as a plain re-export.
"""

from trtkit import (
    BackendConfig,
    OnnxBackendConfig,
    OnnxCudaRuntime,
    OnnxOrTrtBackendConfig,
    RuntimeSpec,
    TensorRtBackendConfig,
    TensorRtRuntime,
    TensorRuntime,
    TensorSpec,
    TorchBackendConfig,
    TorchRuntime,
    create_runtime_from_onnx,
    run_chunked,
    validate_runtime_inputs,
)
from trtkit.trt_builder import DEFAULT_TRT_CACHE_DIR, TrtBuildConfig, TrtPrecision, ensure_engine

__all__ = (
    "BackendConfig",
    "DEFAULT_TRT_CACHE_DIR",
    "OnnxBackendConfig",
    "OnnxCudaRuntime",
    "OnnxOrTrtBackendConfig",
    "RuntimeSpec",
    "TensorRtBackendConfig",
    "TensorRtRuntime",
    "TensorRuntime",
    "TensorSpec",
    "TorchBackendConfig",
    "TorchRuntime",
    "TrtBuildConfig",
    "TrtPrecision",
    "create_runtime_from_onnx",
    "ensure_engine",
    "run_chunked",
    "validate_runtime_inputs",
)
