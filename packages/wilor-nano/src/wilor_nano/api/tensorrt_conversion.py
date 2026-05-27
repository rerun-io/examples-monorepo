"""ONNX export and TensorRT build helpers for WiLoR deployment artifacts."""

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, NamedTuple, cast

import torch
from jaxtyping import Float
from torch import Tensor, nn

from wilor_nano.api.tensorrt_runtime import (
    DETECTOR_INPUT_NAME,
    DETECTOR_INPUT_SHAPE,
    DETECTOR_OUTPUT_NAME,
    FULL_WILOR_INPUT_NAME,
    FULL_WILOR_INPUT_SHAPE,
    FULL_WILOR_OUTPUT_NAMES,
)
from wilor_nano.runtime import get_torch_device

WiLorOnnxTarget = Literal["full_postcrop", "detector_raw"]
TensorRtPrecision = Literal["fp32", "fp16", "bf16", "fp8"]
ExportFn = Callable[..., object]
FullWilorExportOutput = tuple[Float[Tensor, "batch 1 3"], Float[Tensor, "batch 15 3"], Float[Tensor, "batch 10"], Float[Tensor, "batch 3"], Float[Tensor, "batch 21 3"], Float[Tensor, "batch 778 3"]]

DEFAULT_TRT_DIR: Path = Path(__file__).resolve().parents[3] / "pretrained_models" / "tensorrt"
DEFAULT_FULL_WILOR_ONNX_PATH: Path = DEFAULT_TRT_DIR / "wilor_full_postcrop_static_b224.onnx"
DEFAULT_FULL_WILOR_ENGINE_PATH: Path = DEFAULT_TRT_DIR / "wilor_full_postcrop_static_b224_fp16.trt"
DEFAULT_DETECTOR_ONNX_PATH: Path = DEFAULT_TRT_DIR / "detector_raw_static_b110_512x416.onnx"
DEFAULT_DETECTOR_ENGINE_PATH: Path = DEFAULT_TRT_DIR / "detector_raw_static_b110_512x416_tf32.trt"


class _TargetSpec(NamedTuple):
    input_name: str
    input_shape: tuple[int, ...]
    output_names: tuple[str, ...]
    batch_size: int
    dtype: Literal["float16", "float32"]
    opset_version: int
    onnx_path: Path
    engine_path: Path
    precision: TensorRtPrecision
    allow_tf32: bool


_TARGETS: dict[WiLorOnnxTarget, _TargetSpec] = {
    "full_postcrop": _TargetSpec(FULL_WILOR_INPUT_NAME, FULL_WILOR_INPUT_SHAPE, FULL_WILOR_OUTPUT_NAMES, 224, "float16", 17, DEFAULT_FULL_WILOR_ONNX_PATH, DEFAULT_FULL_WILOR_ENGINE_PATH, "fp16", False),
    "detector_raw": _TargetSpec(DETECTOR_INPUT_NAME, DETECTOR_INPUT_SHAPE, (DETECTOR_OUTPUT_NAME,), 110, "float32", 18, DEFAULT_DETECTOR_ONNX_PATH, DEFAULT_DETECTOR_ENGINE_PATH, "fp32", True),
}


@dataclass(frozen=True, slots=True)
class WiLorTensorRtArtifactConfig:
    """Shared ONNX/TensorRT artifact identity used by export and build CLIs."""

    target: WiLorOnnxTarget = "full_postcrop"
    onnx_path: Path | None = None
    batch_size: int | None = None


@dataclass(frozen=True, slots=True)
class WiLorOnnxExportConfig:
    """Configuration for exporting one WiLoR ONNX graph."""

    artifact: WiLorTensorRtArtifactConfig = field(default_factory=WiLorTensorRtArtifactConfig)
    device: str = "cuda"
    dtype: Literal["float16", "float32"] | None = None
    opset_version: int | None = None
    dynamo: bool = False


class WiLorOnnxExportSummary(NamedTuple):
    """Summary returned after exporting a WiLoR ONNX graph."""

    target: WiLorOnnxTarget
    onnx_path: Path
    input_name: str
    input_shape: tuple[int, ...]
    output_names: tuple[str, ...]
    opset_version: int
    dynamo: bool


@dataclass(frozen=True, slots=True)
class TensorRtBuildConfig:
    """Configuration for building a machine-local TensorRT engine from WiLoR ONNX."""

    artifact: WiLorTensorRtArtifactConfig = field(default_factory=WiLorTensorRtArtifactConfig)
    engine_path: Path | None = None
    precision: TensorRtPrecision | None = None
    allow_tf32: bool | None = None
    workspace_gib: float = 24.0
    builder_optimization_level: int = 3

    def to_manifest(self, *, tensorrt_version: str, cuda_device_name: str) -> dict[str, object]:
        """Return reproducibility metadata for the non-portable TensorRT engine."""
        artifact: WiLorTensorRtArtifactConfig = self.artifact
        spec: _TargetSpec = _TARGETS[artifact.target]
        batch_size: int = _batch_size(artifact)
        onnx_path: Path = artifact.onnx_path or spec.onnx_path
        return {
            "target": artifact.target,
            "precision": self.precision or spec.precision,
            "allow_tf32": spec.allow_tf32 if self.allow_tf32 is None else self.allow_tf32,
            "onnx_path": str(onnx_path),
            "onnx_sha256": _sha256_file(onnx_path),
            "engine_path": str(self.engine_path or spec.engine_path),
            "portable_engine": False,
            "rebuild_from_onnx_on_target_machine": True,
            "batch_profile_preset": f"static-b{batch_size}",
            "batch_profile": {"min": batch_size, "optimal": batch_size, "max": batch_size},
            "workspace_gib": self.workspace_gib,
            "builder_optimization_level": self.builder_optimization_level,
            "runtime_recommendation": "static_batch_padding",
            "tensorrt_version": tensorrt_version,
            "cuda_device_name": cuda_device_name,
            "model_io": {"input_name": spec.input_name, "input_shape": [batch_size, *spec.input_shape], "output_names": list(spec.output_names)},
        }


class TensorRtBuildSummary(NamedTuple):
    """Summary returned after building a TensorRT engine."""

    engine_path: Path
    manifest_path: Path
    target: WiLorOnnxTarget
    precision: TensorRtPrecision
    static_batch_size: int


class _OnnxWrapper(nn.Module):
    """Small adapter that gives WiLoR subgraphs stable tensor outputs."""

    def __init__(self, model: nn.Module, target: WiLorOnnxTarget) -> None:
        super().__init__()
        self.model: nn.Module = model
        self.target: WiLorOnnxTarget = target

    def forward(self, inputs: Tensor) -> FullWilorExportOutput | Float[Tensor, "batch fields anchors"]:
        output: Any = self.model(inputs)
        if self.target == "full_postcrop":
            return cast(FullWilorExportOutput, tuple(output[name] for name in FULL_WILOR_OUTPUT_NAMES))
        return output[0] if isinstance(output, (tuple, list)) else output


def export_wilor_onnx(config: WiLorOnnxExportConfig, *, pipeline_factory: Callable[..., Any] | None = None, export_fn: ExportFn = torch.onnx.export) -> WiLorOnnxExportSummary:
    """Export either the accepted full-WiLoR or raw-detector ONNX graph."""
    from wilor_nano.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline

    artifact: WiLorTensorRtArtifactConfig = config.artifact
    spec: _TargetSpec = _TARGETS[artifact.target]
    batch_size: int = _batch_size(artifact)
    input_shape: tuple[int, ...] = (batch_size, *spec.input_shape)
    dtype: torch.dtype = torch.float16 if (config.dtype or spec.dtype) == "float16" else torch.float32
    device: torch.device = torch.device(str(get_torch_device()) if config.device == "auto" else config.device)
    pipeline: Any = (pipeline_factory or WiLorHandPose3dEstimationPipeline)(device=device, dtype=dtype, verbose=False)
    export_model: nn.Module = cast(nn.Module, pipeline.wilor_model if artifact.target == "full_postcrop" else pipeline.hand_detector.model)
    model: nn.Module = _OnnxWrapper(export_model, artifact.target).to(device=device, dtype=dtype).eval()
    onnx_path: Path = artifact.onnx_path or spec.onnx_path
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        export_fn(
            model,
            (torch.zeros(input_shape, dtype=dtype, device=device),),
            onnx_path,
            export_params=True,
            opset_version=config.opset_version or spec.opset_version,
            do_constant_folding=True,
            input_names=[spec.input_name],
            output_names=list(spec.output_names),
            dynamic_axes=None,
            dynamo=config.dynamo,
        )
    return WiLorOnnxExportSummary(artifact.target, onnx_path, spec.input_name, input_shape, spec.output_names, config.opset_version or spec.opset_version, config.dynamo)


def build_wilor_tensorrt_engine(config: TensorRtBuildConfig) -> TensorRtBuildSummary:
    """Build a machine-local TensorRT engine from a WiLoR ONNX graph."""
    if config.workspace_gib <= 0.0:
        raise ValueError("workspace_gib must be positive.")
    if not (0 <= config.builder_optimization_level <= 5):
        raise ValueError("builder_optimization_level must be between 0 and 5.")
    trt: Any = _import_tensorrt()
    logger: Any = trt.Logger(trt.Logger.INFO)
    builder: Any = trt.Builder(logger)
    network: Any = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser: Any = trt.OnnxParser(network, logger)
    artifact: WiLorTensorRtArtifactConfig = config.artifact
    spec: _TargetSpec = _TARGETS[artifact.target]
    batch_size: int = _batch_size(artifact)
    onnx_path: Path = artifact.onnx_path or spec.onnx_path
    if not bool(parser.parse_from_file(str(onnx_path))):
        errors: list[str] = [str(parser.get_error(idx)) for idx in range(parser.num_errors)]
        raise RuntimeError("TensorRT failed to parse ONNX graph:\n" + "\n".join(errors))

    builder_config: Any = builder.create_builder_config()
    builder_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(config.workspace_gib * 1024**3))
    if hasattr(builder_config, "builder_optimization_level"):
        builder_config.builder_optimization_level = config.builder_optimization_level
    _set_precision_flags(builder_config, trt, config, spec)

    serialized_engine: Any = builder.build_serialized_network(network, builder_config)
    if serialized_engine is None:
        raise RuntimeError("TensorRT returned no serialized engine.")

    engine_path: Path = config.engine_path or spec.engine_path
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(serialized_engine))
    manifest_path: Path = engine_path.with_suffix(engine_path.suffix + ".json")
    manifest_path.write_text(json.dumps(config.to_manifest(tensorrt_version=str(trt.__version__), cuda_device_name=_cuda_device_name()), indent=2, sort_keys=True) + "\n")
    return TensorRtBuildSummary(engine_path, manifest_path, artifact.target, config.precision or spec.precision, batch_size)


def _batch_size(artifact: WiLorTensorRtArtifactConfig) -> int:
    batch_size: int = artifact.batch_size or _TARGETS[artifact.target].batch_size
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    return batch_size


def _set_precision_flags(builder_config: Any, trt: Any, config: TensorRtBuildConfig, spec: _TargetSpec) -> None:
    if hasattr(trt.BuilderFlag, "TF32") and not (spec.allow_tf32 if config.allow_tf32 is None else config.allow_tf32):
        builder_config.clear_flag(trt.BuilderFlag.TF32)
    precision: TensorRtPrecision = config.precision or spec.precision
    if precision != "fp32":
        builder_config.set_flag(getattr(trt.BuilderFlag, precision.upper()))


def _cuda_device_name() -> str:
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unknown"


def _sha256_file(path: Path) -> str:
    digest: Any = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _import_tensorrt() -> Any:
    try:
        import tensorrt as trt
    except ImportError as exc:
        raise RuntimeError("TensorRT Python bindings are not installed in this Pixi environment.") from exc
    return trt
