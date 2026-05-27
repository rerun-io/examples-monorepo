"""ONNX export and TensorRT build helpers for Sapiens COCO-133 deployment."""

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, NamedTuple

import torch
from jaxtyping import Float
from torch import Tensor

from sapiens2_pose.api.runtime import DEFAULT_MODEL_SIZE, DeviceChoice, ModelSize, resolve_device
from sapiens2_pose.api.tensorrt_pose import make_sapiens_pose_onnx_exportable
from sapiens2_pose.sapiens_lite.pose import MODEL_SPECS, init_pose_model

TensorRtTarget = Literal["detector", "pose", "rtmlib-pose"]
TensorRtPrecision = Literal["fp16"]
ModelLoader = Callable[[str, str | Path, str], torch.nn.Module]
ExportFn = Callable[..., object]


@dataclass(frozen=True, slots=True)
class SapiensCoco133PoseOnnxExportConfig:
    """Configuration for exporting batched Sapiens2 pose ONNX."""

    checkpoint_path: Path
    """Input Sapiens checkpoint path."""
    onnx_path: Path
    """Output ONNX path."""
    model_size: ModelSize = DEFAULT_MODEL_SIZE
    """Sapiens model size to export."""
    batch_size: int = 4
    """Static batch size baked into the exported ONNX graph."""
    device: DeviceChoice = "cuda"
    """Device used while exporting the ONNX graph."""
    opset_version: int = 17
    """ONNX opset version passed to ``torch.onnx.export``."""
    dynamo: bool = True
    """Whether to use the torch.export-based ONNX exporter."""

    @property
    def dtype(self) -> torch.dtype:
        """Tensor dtype used by the exported pose graph.

        Returns:
            The fixed FP16 export dtype.
        """
        return torch.float16


class SapiensCoco133PoseOnnxExportSummary(NamedTuple):
    """Summary returned after exporting Sapiens COCO-133 pose ONNX."""

    checkpoint_path: Path
    """Input Sapiens checkpoint path."""
    onnx_path: Path
    """Written ONNX path."""
    model_size: ModelSize
    """Exported Sapiens model size."""
    batch_size: int
    """Static batch size baked into the ONNX graph."""
    input_shape: tuple[int, int, int, int]
    """Exported input tensor shape."""
    output_shape: tuple[int, int, int, int]
    """Exported output tensor shape."""
    opset_version: int
    """ONNX opset version."""
    dynamo: bool
    """Whether the torch.export-based ONNX exporter was used."""


@dataclass(frozen=True, slots=True)
class TensorRtEngineBuildConfig:
    """Configuration for building one static-batch TensorRT engine."""

    target: TensorRtTarget
    """Deployment target represented by the ONNX graph."""
    onnx_path: Path
    """Input ONNX model path."""
    engine_path: Path
    """Output TensorRT engine path."""
    input_name: str
    """TensorRT network input tensor name."""
    output_names: tuple[str, ...]
    """TensorRT network output tensor names."""
    input_shape: tuple[int, ...]
    """Static input shape without the batch dimension."""
    batch_size: int
    """Static batch size for the TensorRT optimization profile."""
    workspace_gib: float = 24.0
    """TensorRT builder workspace size in GiB."""
    builder_optimization_level: int = 3
    """TensorRT builder optimization level from 0 to 5."""

    @property
    def precision(self) -> TensorRtPrecision:
        """TensorRT precision preset.

        Returns:
            The fixed FP16 precision string.
        """
        return "fp16"

    def validate(self) -> None:
        """Validate TensorRT build settings.

        Raises:
            ValueError: If batch size, workspace, or optimization level is invalid.
        """
        if self.batch_size <= 0 or self.workspace_gib <= 0.0 or not (0 <= self.builder_optimization_level <= 5):
            raise ValueError("Invalid TensorRT build settings.")

    def to_manifest(self, *, tensorrt_version: str, cuda_device_name: str) -> dict[str, object]:
        """Build reproducibility metadata for a machine-local TensorRT engine.

        Args:
            tensorrt_version: TensorRT package version used for the build.
            cuda_device_name: CUDA device name used for the build.

        Returns:
            JSON-serializable manifest content.
        """
        self.validate()
        return {"target": self.target, "precision": self.precision, "onnx_path": str(self.onnx_path), "onnx_sha256": _sha256_file(self.onnx_path), "engine_path": str(self.engine_path), "portable_engine": False, "rebuild_from_onnx_on_target_machine": True, "batch_profile_preset": f"static-b{self.batch_size}", "batch_profile": {"min": self.batch_size, "optimal": self.batch_size, "max": self.batch_size}, "workspace_gib": self.workspace_gib, "builder_optimization_level": self.builder_optimization_level, "runtime_recommendation": "static_batch_padding", "tensorrt_version": tensorrt_version, "cuda_device_name": cuda_device_name, "model_io": {"input_name": self.input_name, "input_shape": [self.batch_size, *self.input_shape], "output_names": list(self.output_names)}}


class TensorRtEngineBuildSummary(NamedTuple):
    """Summary returned after building a TensorRT engine."""

    engine_path: Path
    """Written TensorRT engine path."""
    manifest_path: Path
    """Written engine reproducibility manifest path."""
    target: TensorRtTarget
    """Engine target type."""
    precision: TensorRtPrecision
    """TensorRT precision preset."""
    static_batch_size: int
    """Static batch size baked into the engine optimization profile."""


def export_sapiens_coco133_pose_onnx(config: SapiensCoco133PoseOnnxExportConfig, *, model_loader: ModelLoader = init_pose_model, export_fn: ExportFn = torch.onnx.export) -> SapiensCoco133PoseOnnxExportSummary:
    """Export a static-batch FP16 Sapiens2 pose network to ONNX.

    Args:
        config: Sapiens checkpoint, output path, model size, batch, device, and exporter settings.
        model_loader: Model loader override used by tests.
        export_fn: ONNX export function override used by tests.

    Returns:
        Export summary containing paths, model size, batch size, input/output shapes, and exporter settings.

    Raises:
        ValueError: If ``batch_size`` is not positive.
    """
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    device = resolve_device(config.device)
    spec = MODEL_SPECS[config.model_size]
    input_shape = (config.batch_size, 3, int(spec.image_size[0]), int(spec.image_size[1]))
    output_shape = (config.batch_size, int(spec.num_keypoints), int(spec.heatmap_size[1]), int(spec.heatmap_size[0]))
    model = make_sapiens_pose_onnx_exportable(model_loader(config.model_size, config.checkpoint_path, device)).to(device=device, dtype=config.dtype).eval()
    dummy_inputs: Float[Tensor, "batch 3 1024 768"] = torch.zeros(input_shape, dtype=config.dtype, device=device)
    config.onnx_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        export_fn(model, (dummy_inputs,), config.onnx_path, export_params=True, opset_version=config.opset_version, do_constant_folding=True, input_names=["inputs"], output_names=["heatmaps"], dynamic_axes=None, dynamo=config.dynamo)
    return SapiensCoco133PoseOnnxExportSummary(config.checkpoint_path, config.onnx_path, config.model_size, config.batch_size, input_shape, output_shape, config.opset_version, config.dynamo)


def build_tensorrt_engine(config: TensorRtEngineBuildConfig) -> TensorRtEngineBuildSummary:
    """Build a static-batch FP16 TensorRT engine from ONNX and write a manifest.

    Args:
        config: TensorRT target, ONNX path, output engine path, IO names, shape, and builder settings.

    Returns:
        Engine path, manifest path, target, precision, and static batch size.

    Raises:
        RuntimeError: If TensorRT cannot parse the ONNX graph or returns no serialized engine.
    """
    config.validate()
    trt = _import_tensorrt()
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not bool(parser.parse_from_file(str(config.onnx_path))):
        raise RuntimeError("TensorRT failed to parse ONNX graph:\n" + "\n".join(str(parser.get_error(idx)) for idx in range(parser.num_errors)))
    if config.target in ("pose", "rtmlib-pose"):
        network.get_input(0).dtype = trt.float16
        for output_idx in range(int(network.num_outputs)):
            network.get_output(output_idx).dtype = trt.float16
    builder_config = builder.create_builder_config()
    builder_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(config.workspace_gib * 1024**3))
    if hasattr(builder_config, "builder_optimization_level"):
        builder_config.builder_optimization_level = config.builder_optimization_level
    if hasattr(trt.BuilderFlag, "TF32"):
        builder_config.clear_flag(trt.BuilderFlag.TF32)
    builder_config.set_flag(trt.BuilderFlag.FP16)
    profile = builder.create_optimization_profile()
    profile.set_shape(str(network.get_input(0).name), (config.batch_size, *config.input_shape), (config.batch_size, *config.input_shape), (config.batch_size, *config.input_shape))
    builder_config.add_optimization_profile(profile)
    serialized_engine = builder.build_serialized_network(network, builder_config)
    if serialized_engine is None:
        raise RuntimeError("TensorRT returned no serialized engine.")
    config.engine_path.parent.mkdir(parents=True, exist_ok=True)
    config.engine_path.write_bytes(bytes(serialized_engine))
    manifest_path = config.engine_path.with_suffix(config.engine_path.suffix + ".json")
    manifest_path.write_text(json.dumps(config.to_manifest(tensorrt_version=str(trt.__version__), cuda_device_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unknown"), indent=2, sort_keys=True) + "\n")
    return TensorRtEngineBuildSummary(config.engine_path, manifest_path, config.target, config.precision, config.batch_size)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return str(digest.hexdigest())


def _import_tensorrt() -> Any:
    try:
        import tensorrt as trt
    except ImportError as exc:
        raise RuntimeError("TensorRT Python bindings are not installed in this Pixi environment.") from exc
    return trt
