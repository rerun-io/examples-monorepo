"""BF16 TensorRT conversion and inference helpers for Sapiens2 pose models."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from jaxtyping import Float32, UInt8
from numpy import ndarray

from sapiens2_pose.api.image_pose import (
    ImagePoseConfig,
    ImagePoseSummary,
    run_image_pose,
)
from sapiens2_pose.api.pose_artifact import PosePredictionArtifact
from sapiens2_pose.api.runtime import (
    POSE_MODELS,
    DeviceChoice,
    ModelSize,
    resolve_device,
)
from sapiens2_pose.sapiens_lite.backbones.sapiens2 import RopePositionEmbedding
from sapiens2_pose.sapiens_lite.pose import (
    MODEL_SPECS,
    ImagePreprocessor,
    UDPHeatmap,
    init_pose_model,
    prepare_pose_sample,
)

TensorRtPrecision = Literal["bf16"]
"""The retained TensorRT precision. BF16 was the fastest strict-accuracy floating-point engine in the 0.4B sweep."""

STATIC_B1_PROFILE: tuple[int, int, int] = (1, 1, 1)
"""Static batch-1 profile used by the retained fastest engine."""

PoseHeatmapRunner = Callable[[torch.Tensor], torch.Tensor | Float32[ndarray, "n k h w"]]
"""Callable backend that maps preprocessed Sapiens pose tensors to heatmaps."""

ModelLoader = Callable[[str, str | Path, str], torch.nn.Module]
ExportFn = Callable[..., object]


class ExportableRMSNorm(torch.nn.Module):
    """RMSNorm implementation composed from ONNX-exportable tensor operations."""

    def __init__(self, normalized_shape: tuple[int, ...], eps: float | None, weight: torch.Tensor | None) -> None:
        """Create an ONNX-friendly RMSNorm layer."""
        super().__init__()
        self.normalized_shape: tuple[int, ...] = normalized_shape
        self.eps: float | None = eps
        if weight is None:
            self.register_parameter("weight", None)
        else:
            self.weight = torch.nn.Parameter(weight.detach().clone())

    @classmethod
    def from_rms_norm(cls, module: torch.nn.RMSNorm) -> ExportableRMSNorm:
        """Create an equivalent exportable module from `torch.nn.RMSNorm`."""
        normalized_shape: tuple[int, ...] = tuple(int(dim) for dim in module.normalized_shape)
        weight: torch.Tensor | None = module.weight
        exportable: ExportableRMSNorm = cls(normalized_shape=normalized_shape, eps=module.eps, weight=weight)
        exportable.training = module.training
        return exportable

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization using primitive tensor math."""
        norm_dims: tuple[int, ...] = tuple(range(-len(self.normalized_shape), 0))
        eps: float = self.eps if self.eps is not None else float(torch.finfo(inputs.dtype).eps)
        variance: torch.Tensor = inputs.to(torch.float32).pow(2).mean(dim=norm_dims, keepdim=True)
        output: torch.Tensor = inputs * torch.rsqrt(variance.to(dtype=inputs.dtype) + eps)
        weight: torch.Tensor | None = self.weight
        if weight is not None:
            output = output * weight.to(dtype=output.dtype)
        return output


def make_sapiens_pose_onnx_exportable(model: torch.nn.Module) -> torch.nn.Module:
    """Replace Sapiens2 RMSNorm modules with equivalent ONNX-friendly modules."""
    child_items: list[tuple[str, torch.nn.Module]] = list(model.named_children())
    for child_name, child_module in child_items:
        if isinstance(child_module, torch.nn.RMSNorm):
            setattr(model, child_name, ExportableRMSNorm.from_rms_norm(child_module))
        elif isinstance(child_module, RopePositionEmbedding):
            rope_module: Any = child_module
            rope_module.dtype = torch.float32
            rope_module.periods.data = rope_module.periods.data.to(dtype=torch.float32)
        else:
            make_sapiens_pose_onnx_exportable(child_module)
    return model


@dataclass(frozen=True, slots=True)
class SapiensPoseOnnxExportConfig:
    """Configuration for exporting one static batch-1 Sapiens2 pose network to ONNX."""

    checkpoint_path: Path
    """Path to the Sapiens2 `.safetensors` checkpoint."""
    onnx_path: Path
    """Path where the exported ONNX graph should be written."""
    model_size: ModelSize = "0.4B"
    """Sapiens2 pose model size to export."""
    device: DeviceChoice = "cuda"
    """Device used while tracing; CUDA is preferred for the real 0.4B export."""


@dataclass(frozen=True, slots=True)
class SapiensPoseOnnxExportSummary:
    """Summary returned after exporting a Sapiens2 pose model to ONNX."""

    checkpoint_path: Path
    """Checkpoint used for export."""
    onnx_path: Path
    """Written ONNX graph path."""
    model_size: ModelSize
    """Sapiens2 model size exported."""
    input_shape: tuple[int, int, int, int]
    """NCHW input tensor shape used for export."""
    output_shape: tuple[int, int, int, int]
    """NCHW heatmap output tensor shape expected from the graph."""


@dataclass(frozen=True, slots=True)
class TensorRtBuildConfig:
    """Configuration for building the retained BF16 static batch-1 TensorRT engine from ONNX."""

    onnx_path: Path
    """Path to the portable ONNX graph."""
    engine_path: Path
    """Path where the machine-local TensorRT engine should be written."""
    model_size: ModelSize = "0.4B"
    """Sapiens2 model size represented by the ONNX graph."""
    workspace_gib: float = 24.0
    """TensorRT workspace memory limit in GiB."""
    builder_optimization_level: int = 3
    """TensorRT builder optimization level, from 0 through 5."""

    @property
    def precision(self) -> TensorRtPrecision:
        """Return the retained TensorRT precision."""
        return "bf16"

    def to_manifest(self, *, tensorrt_version: str, cuda_device_name: str) -> dict[str, object]:
        """Return reproducibility metadata for a non-portable TensorRT engine."""
        self.validate()
        onnx_sha256: str = _sha256_file(self.onnx_path)
        deploy_metadata: dict[str, object] = make_sapiens_pose_deploy_metadata(self.model_size)
        manifest: dict[str, object] = {
            "model_size": self.model_size,
            "precision": self.precision,
            "onnx_path": str(self.onnx_path),
            "onnx_sha256": onnx_sha256,
            "engine_path": str(self.engine_path),
            "portable_engine": False,
            "rebuild_from_onnx_on_target_machine": True,
            "batch_profile_preset": "static-b1",
            "batch_profile": {"min": 1, "optimal": 1, "max": 1},
            "workspace_gib": self.workspace_gib,
            "builder_optimization_level": self.builder_optimization_level,
            "runtime_recommendation": "cuda_graph_replay",
            "tensorrt_version": tensorrt_version,
            "cuda_device_name": cuda_device_name,
            **deploy_metadata,
        }
        return manifest

    def validate(self) -> None:
        """Validate the TensorRT build profile before allocating resources."""
        if self.workspace_gib <= 0.0:
            raise ValueError("TensorRT workspace_gib must be positive.")
        if not (0 <= self.builder_optimization_level <= 5):
            raise ValueError("TensorRT builder_optimization_level must be between 0 and 5.")

    def resolved_batch_profile(self) -> tuple[int, int, int]:
        """Return the retained static batch-1 TensorRT batch profile."""
        return STATIC_B1_PROFILE


@dataclass(frozen=True, slots=True)
class TensorRtBuildSummary:
    """Summary returned after building a TensorRT engine."""

    engine_path: Path
    """Written TensorRT engine path."""
    manifest_path: Path
    """JSON manifest path describing the engine build."""
    precision: TensorRtPrecision
    """Precision mode requested for the build."""
    batch_profile: tuple[int, int, int]
    """Minimum, optimal, and maximum batch sizes used by the profile."""


@dataclass(frozen=True, slots=True)
class TensorRtImagePoseConfig:
    """Configuration for running single-image Sapiens2 pose with a TensorRT engine."""

    image_path: Path
    """Path to the input image."""
    engine_path: Path
    """Path to the TensorRT engine for the Sapiens2 pose model."""
    rrd_path: Path
    """Path to the output Rerun recording."""
    artifact_path: Path
    """Path to the output numeric `.npz` pose artifact."""
    model_size: ModelSize = "0.4B"
    """Sapiens2 pose model size represented by the TensorRT engine."""
    bbox_thr: float = 0.3
    """DETR person-detection confidence threshold."""
    nms_thr: float = 0.3
    """Person-box NMS IoU threshold."""
    kpt_thr: float = 0.3
    """Pose keypoint visibility threshold for Rerun rendering."""
    device: DeviceChoice = "cuda"
    """Compute device selection; TensorRT inference requires CUDA."""


def resolve_sapiens_pose_checkpoint(model_size: ModelSize = "0.4B", checkpoint_path: Path | None = None) -> Path:
    """Return an explicit checkpoint path or download the requested Sapiens2 checkpoint."""
    if checkpoint_path is not None:
        return checkpoint_path
    spec: dict[str, str] = POSE_MODELS[model_size]
    resolved_path: str = hf_hub_download(repo_id=spec["repo"], filename=spec["filename"])
    return Path(resolved_path)


def make_sapiens_pose_deploy_metadata(model_size: ModelSize) -> dict[str, object]:
    """Return model I/O, preprocessing, and decode metadata needed to audit a TensorRT engine."""
    spec: Any = MODEL_SPECS[model_size]
    preprocessor: ImagePreprocessor = ImagePreprocessor()
    mean_tensor: torch.Tensor = cast(torch.Tensor, preprocessor.mean)
    std_tensor: torch.Tensor = cast(torch.Tensor, preprocessor.std)
    mean: list[float] = [round(float(value), 3) for value in mean_tensor.reshape(-1).tolist()]
    std: list[float] = [round(float(value), 3) for value in std_tensor.reshape(-1).tolist()]
    metadata: dict[str, object] = {
        "model_io": {
            "input_name": "inputs",
            "output_name": "heatmaps",
            "input_shape": [1, 3, int(spec.image_size[0]), int(spec.image_size[1])],
            "output_shape": [1, int(spec.num_keypoints), int(spec.heatmap_size[1]), int(spec.heatmap_size[0])],
        },
        "preprocessing": {
            "color_order_before_normalize": "RGB",
            "mean": mean,
            "std": std,
        },
        "decode": {
            "codec": "UDPHeatmap",
            "input_size": [int(spec.input_size[0]), int(spec.input_size[1])],
            "heatmap_size": [int(spec.heatmap_size[0]), int(spec.heatmap_size[1])],
            "sigma": float(spec.sigma),
        },
    }
    return metadata


def export_sapiens_pose_onnx(
    config: SapiensPoseOnnxExportConfig,
    *,
    model_loader: ModelLoader = init_pose_model,
    export_fn: ExportFn = torch.onnx.export,
) -> SapiensPoseOnnxExportSummary:
    """Export a Sapiens2 pose model checkpoint to a static batch-1 ONNX graph."""
    resolved_device: str = resolve_device(config.device)
    spec: Any = MODEL_SPECS[config.model_size]
    input_shape: tuple[int, int, int, int] = (1, 3, spec.image_size[0], spec.image_size[1])
    output_shape: tuple[int, int, int, int] = (1, spec.num_keypoints, spec.heatmap_size[1], spec.heatmap_size[0])

    from trtkit import export_onnx

    model: torch.nn.Module = model_loader(config.model_size, config.checkpoint_path, resolved_device)
    model = make_sapiens_pose_onnx_exportable(model)
    model.eval()
    dummy_inputs: torch.Tensor = torch.zeros(input_shape, dtype=torch.float32, device=resolved_device)
    # bf16 is the retained strict-accuracy precision from the 0.4B sweep
    # (fp16-typed Sapiens graphs overflow, ~70 px error); trtkit owns the
    # autocast wrapping, opset policy, and atomic publish.
    export_onnx(
        model,
        (dummy_inputs,),
        config.onnx_path,
        input_names=["inputs"],
        output_names=["heatmaps"],
        compute_dtype=torch.bfloat16 if resolved_device == "cuda" else None,
        export_fn=export_fn,
    )

    return SapiensPoseOnnxExportSummary(
        checkpoint_path=config.checkpoint_path,
        onnx_path=config.onnx_path,
        model_size=config.model_size,
        input_shape=input_shape,
        output_shape=output_shape,
    )


def estimate_sapiens_pose_with_heatmap_runner(
    image_rgb: UInt8[ndarray, "h w 3"],
    bboxes: Float32[ndarray, "n 4"],
    *,
    model_size: ModelSize = "0.4B",
    device: DeviceChoice = "cuda",
    heatmap_runner: PoseHeatmapRunner,
) -> PosePredictionArtifact:
    """Estimate Sapiens2 pose by reusing preprocessing and decoding around a heatmap backend."""
    bboxes_f32: Float32[ndarray, "n 4"] = np.asarray(bboxes, dtype=np.float32).reshape(-1, 4)
    spec: Any = MODEL_SPECS[model_size]
    if bboxes_f32.shape[0] == 0:
        empty_keypoints: Float32[ndarray, "0 308 2"] = np.empty((0, spec.num_keypoints, 2), dtype=np.float32)
        empty_scores: Float32[ndarray, "0 308"] = np.empty((0, spec.num_keypoints), dtype=np.float32)
        return PosePredictionArtifact(bboxes=bboxes_f32, keypoints=empty_keypoints, scores=empty_scores)

    resolved_device: str = resolve_device(device)
    preprocessor: ImagePreprocessor = ImagePreprocessor().to(resolved_device)
    codec: UDPHeatmap = UDPHeatmap(input_size=spec.input_size, heatmap_size=spec.heatmap_size, sigma=spec.sigma)
    image_bgr: UInt8[ndarray, "h w 3"] = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    inputs_list: list[torch.Tensor] = []
    samples_list: list[dict[str, Any]] = []
    for bbox in bboxes_f32:
        data: dict[str, Any] = prepare_pose_sample(image_bgr, bbox, input_size=spec.input_size)
        processed: dict[str, Any] = preprocessor(data)
        inputs_list.append(processed["inputs"])
        samples_list.append(processed["data_samples"])

    inputs: torch.Tensor = torch.cat(inputs_list, dim=0).to(resolved_device).contiguous()
    with torch.no_grad():
        heatmaps_value: torch.Tensor | Float32[ndarray, "n k h w"] = heatmap_runner(inputs)
    if isinstance(heatmaps_value, torch.Tensor):
        heatmaps: Float32[ndarray, "n k h w"] = (
            heatmaps_value.detach().float().cpu().numpy().astype(np.float32, copy=False)
        )
    else:
        heatmaps = np.asarray(heatmaps_value, dtype=np.float32)

    keypoints_list: list[Float32[ndarray, "k 2"]] = []
    scores_list: list[Float32[ndarray, "k"]] = []
    for idx, sample in enumerate(samples_list):
        decoded_keypoints: Float32[ndarray, "1 k 2"]
        decoded_scores: Float32[ndarray, "1 k"]
        decoded_keypoints, decoded_scores = codec.decode(heatmaps[idx])
        meta: dict[str, Any] = sample["meta"]
        keypoints_i: Float32[ndarray, "1 k 2"] = (
            decoded_keypoints / meta["input_size"] * meta["bbox_scale"] + meta["bbox_center"] - 0.5 * meta["bbox_scale"]
        ).astype(np.float32, copy=False)
        keypoints_list.append(keypoints_i[0])
        scores_list.append(decoded_scores[0].astype(np.float32, copy=False))

    keypoints: Float32[ndarray, "n k 2"] = np.stack(keypoints_list, axis=0).astype(np.float32, copy=False)
    scores: Float32[ndarray, "n k"] = np.stack(scores_list, axis=0).astype(np.float32, copy=False)
    return PosePredictionArtifact(bboxes=bboxes_f32, keypoints=keypoints, scores=scores)


def build_tensorrt_engine(config: TensorRtBuildConfig) -> TensorRtBuildSummary:
    """Build the retained BF16 static batch-1 TensorRT engine via trtkit and write a manifest."""
    config.validate()
    from trtkit import TrtBuildConfig as TrtKitBuildConfig
    from trtkit import build_engine as trtkit_build_engine

    trtkit_build_engine(
        config.onnx_path,
        config.engine_path,
        TrtKitBuildConfig(
            max_batch_size=1,
            opt_batch_size=1,
            allow_tf32=False,
            workspace_gib=config.workspace_gib,
            builder_optimization_level=config.builder_optimization_level,
        ),
    )
    import tensorrt

    # Replace trtkit's generic manifest with the Sapiens deploy manifest.
    manifest_path: Path = config.engine_path.with_suffix(config.engine_path.suffix + ".json")
    manifest: dict[str, object] = config.to_manifest(tensorrt_version=tensorrt.__version__, cuda_device_name=torch.cuda.get_device_name())
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    return TensorRtBuildSummary(
        engine_path=config.engine_path,
        manifest_path=manifest_path,
        precision=config.precision,
        batch_profile=config.resolved_batch_profile(),
    )


class TensorRtPoseHeatmapRunner:
    """TensorRT backend for static batch-1 Sapiens2 pose heatmap inference.

    Thin wrapper over :class:`trtkit.TensorRtRuntime` with CUDA-graph replay.
    """

    def __init__(
        self,
        engine_path: Path,
        *,
        device: DeviceChoice = "cuda",
    ) -> None:
        """Load a TensorRT engine for CUDA inference."""
        resolved_device: str = resolve_device(device)
        if resolved_device != "cuda":
            raise ValueError("TensorRT pose inference requires device='cuda'.")
        from trtkit import TensorRtRuntime

        self._runtime = TensorRtRuntime(engine_path, use_cuda_graph=True)
        if self._runtime.spec.max_batch_size != 1:
            raise ValueError(f"The retained TensorRT engine expects batch size 1, got {self._runtime.spec.max_batch_size}.")
        self._input_name: str = self._runtime.spec.inputs[0].name
        self._output_name: str = self._runtime.spec.outputs[0].name

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run TensorRT heatmap inference for one normalized static batch-1 input."""
        if inputs.device.type != "cuda":
            inputs = inputs.to("cuda")
        return self._runtime({self._input_name: inputs})[self._output_name]


def estimate_sapiens_pose_tensorrt(
    image_rgb: UInt8[ndarray, "h w 3"],
    bboxes: Float32[ndarray, "n 4"],
    *,
    engine_path: Path,
    model_size: ModelSize = "0.4B",
    device: DeviceChoice = "cuda",
    heatmap_runner: PoseHeatmapRunner | None = None,
) -> PosePredictionArtifact:
    """Estimate Sapiens2 pose by running each person crop through the static batch-1 TensorRT engine."""
    bboxes_f32: Float32[ndarray, "n 4"] = np.asarray(bboxes, dtype=np.float32).reshape(-1, 4)
    spec: Any = MODEL_SPECS[model_size]
    if bboxes_f32.shape[0] == 0:
        empty_keypoints: Float32[ndarray, "0 308 2"] = np.empty((0, spec.num_keypoints, 2), dtype=np.float32)
        empty_scores: Float32[ndarray, "0 308"] = np.empty((0, spec.num_keypoints), dtype=np.float32)
        return PosePredictionArtifact(bboxes=bboxes_f32, keypoints=empty_keypoints, scores=empty_scores)

    runner: PoseHeatmapRunner = heatmap_runner or TensorRtPoseHeatmapRunner(engine_path, device=device)
    keypoints_list: list[Float32[ndarray, "sapiens_k 2"]] = []
    scores_list: list[Float32[ndarray, "sapiens_k"]] = []
    for bbox in bboxes_f32:
        one_bbox: Float32[ndarray, "1 4"] = np.asarray(bbox, dtype=np.float32).reshape(1, 4)
        artifact: PosePredictionArtifact = estimate_sapiens_pose_with_heatmap_runner(
            image_rgb,
            one_bbox,
            model_size=model_size,
            device=device,
            heatmap_runner=runner,
        )
        keypoints_list.append(np.asarray(artifact.keypoints[0], dtype=np.float32))
        scores_list.append(np.asarray(artifact.scores[0], dtype=np.float32).reshape(-1))

    keypoints: Float32[ndarray, "n sapiens_k 2"] = np.stack(keypoints_list, axis=0).astype(np.float32, copy=False)
    scores: Float32[ndarray, "n sapiens_k"] = np.stack(scores_list, axis=0).astype(np.float32, copy=False)
    return PosePredictionArtifact(bboxes=bboxes_f32, keypoints=keypoints, scores=scores)


def run_tensorrt_image_pose(config: TensorRtImagePoseConfig) -> ImagePoseSummary:
    """Run single-image Sapiens2 pose with TensorRT and write RRD plus numeric artifacts."""
    image_config: ImagePoseConfig = ImagePoseConfig(
        image_path=config.image_path,
        rrd_path=config.rrd_path,
        artifact_path=config.artifact_path,
        model_size=config.model_size,
        bbox_thr=config.bbox_thr,
        nms_thr=config.nms_thr,
        kpt_thr=config.kpt_thr,
        device=config.device,
    )

    def estimate_pose_fn(
        image_rgb: UInt8[ndarray, "h w 3"],
        bboxes: Float32[ndarray, "n 4"],
        **_kwargs: object,
    ) -> PosePredictionArtifact:
        return estimate_sapiens_pose_tensorrt(
            image_rgb,
            bboxes,
            engine_path=config.engine_path,
            model_size=config.model_size,
            device=config.device,
        )

    return run_image_pose(image_config, estimate_pose_fn=estimate_pose_fn)


def _sha256_file(path: Path) -> str:
    digest: Any = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
