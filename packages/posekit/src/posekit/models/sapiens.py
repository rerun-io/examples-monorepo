"""Sapiens2 top-down 2D pose adapter — the reference three-backend model.

Sapiens2 ships PyTorch weights (HF ``facebook/sapiens2-pose-*``), so this
adapter demonstrates the full posekit story on one network:

- ``torch`` backend wraps the native ``TopDownPoseModel`` module directly.
- ``onnx``/``tensorrt`` backends consume a one-time ONNX export of that module
  (cached under the posekit ONNX cache), sharing the exact same GPU crop
  generation and UDP decode.

The 308 Goliath keypoints are projected to COCO-133 (unmapped indices are NaN
with score 0), matching the sapiens-coco133-pose pipeline.

Requires the ``sapiens2-pose`` package (model definition + checkpoints);
imports are lazy so the rest of posekit works without it.
"""

from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import torch
from jaxtyping import Float, UInt8
from torch import Tensor

from posekit.artifacts import DEFAULT_ONNX_CACHE_DIR
from posekit.models.base import TopDownPose2d
from posekit.ops.crops import IMAGENET_MEAN_255, IMAGENET_STD_255, CropBatch, CropSpec, crop_coords_to_image, crop_frames
from posekit.ops.decode import decode_udp_heatmaps
from posekit.predictions import BoxDetections, Keypoints2d
from posekit.runtimes import (
    BackendConfig,
    TensorRtBackendConfig,
    TensorRuntime,
    TensorSpec,
    TorchBackendConfig,
    TorchRuntime,
    create_runtime_from_onnx,
)
from posekit.runtimes.base import run_chunked
from posekit.skeletons import COCO_133

SapiensModelSize = Literal["0.4B", "0.8B", "1B"]


@dataclass(frozen=True, slots=True)
class SapiensPoseConfig:
    """Sapiens2 top-down 2D pose configuration."""

    model_size: SapiensModelSize = "0.4B"
    """Sapiens2 checkpoint size."""
    backend: BackendConfig = field(default_factory=TorchBackendConfig)
    """Backend running the network: native torch module, or its ONNX export on ONNX Runtime/TensorRT."""
    padding: float = 1.25
    """Bbox padding multiplier before cropping."""

    def setup(self) -> "SapiensPose2d":
        """Load weights (and export/build artifacts if needed), return a ready estimator."""
        return SapiensPose2d(self)


class SapiensPose2d(TopDownPose2d):
    """Batched GPU Sapiens2 estimator over any posekit backend."""

    def __init__(self, config: SapiensPoseConfig) -> None:
        """Create the selected backend runtime for the Sapiens2 pose network.

        Args:
            config: Model size, backend, and crop padding.
        """
        from sapiens2_pose.sapiens_lite.pose import MODEL_SPECS

        self.config: SapiensPoseConfig = config
        self.skeleton = COCO_133
        spec: Any = MODEL_SPECS[config.model_size]
        self._input_size: tuple[int, int] = (int(spec.input_size[0]), int(spec.input_size[1]))
        self._heatmap_size: tuple[int, int] = (int(spec.heatmap_size[0]), int(spec.heatmap_size[1]))
        input_spec = TensorSpec(name="inputs", shape=(3, self._input_size[1], self._input_size[0]), dtype=torch.float32)
        heatmap_spec = TensorSpec(
            name="heatmaps", shape=(int(spec.num_keypoints), self._heatmap_size[1], self._heatmap_size[0]), dtype=torch.float32
        )
        if isinstance(config.backend, TorchBackendConfig):
            from sapiens2_pose.api.runtime import get_pose_model

            self.runtime: TensorRuntime = TorchRuntime(
                get_pose_model(config.model_size, "cuda"),
                input_specs=(input_spec,),
                output_specs=(heatmap_spec,),
                max_batch_size=config.backend.max_batch_size,
                autocast_dtype=config.backend.autocast_dtype,
            )
        else:
            backend: Any = config.backend
            if isinstance(backend, TensorRtBackendConfig) and backend.precision == "fp16":
                # fp16 Sapiens ViT engines overflow (validated: ~70 px error weakly
                # typed, still broken strongly typed). The sapiens2-pose 0.4B
                # precision sweep settled on BF16 as the fastest strict-accuracy
                # precision — rewrite the default silently-broken config.
                print("[posekit] Sapiens TensorRT with precision='fp16' overflows; using precision='bf16' instead.")
                backend = replace(backend, precision="bf16")
            # The dynamo export bakes a fixed batch, so both accelerated
            # backends run a static graph sized at the TRT opt batch.
            static_batch: int = backend.opt_batch_size if isinstance(backend, TensorRtBackendConfig) else backend.max_batch_size
            self.runtime = create_runtime_from_onnx(_ensure_sapiens_onnx(config.model_size, static_batch), backend)
        self.crop_spec: CropSpec = CropSpec(
            input_size=self._input_size, padding=config.padding, align="udp", bgr=False, mean_rgb=IMAGENET_MEAN_255, std_rgb=IMAGENET_STD_255
        )

    @torch.inference_mode()
    def __call__(self, frames_rgb: UInt8[Tensor, "b h w 3"], detections: BoxDetections) -> Keypoints2d:
        """Estimate COCO-133 keypoints for every detection.

        Args:
            frames_rgb: uint8 RGB NHWC frame batch on the inference device.
            detections: Instance boxes referencing ``frames_rgb`` by index.

        Returns:
            Image-space COCO-133 keypoints (NaN where Sapiens has no mapped keypoint).
        """
        crop_batch: CropBatch = crop_frames(
            frames_rgb,
            frame_indices=detections.frame_indices,
            bboxes_xyxy=detections.xyxy,
            spec=self.crop_spec,
            dtype=self.runtime.spec.inputs[0].dtype,
        )
        num_instances: int = int(crop_batch.inputs.shape[0])
        num_keypoints: int = self.skeleton.num_keypoints
        if num_instances == 0:
            return Keypoints2d.empty(self.skeleton, frames_rgb.device)
        outputs: dict[str, Tensor] = run_chunked(self.runtime, {self.runtime.spec.inputs[0].name: crop_batch.inputs})
        heatmaps: Float[Tensor, "n 308 hm_h hm_w"] = outputs[self.runtime.spec.outputs[0].name]
        coco_indices, goliath_indices = _coco133_projection_indices(heatmaps.device)
        xy_crop, mapped_scores = decode_udp_heatmaps(
            heatmaps[:, goliath_indices], input_size=self._input_size, heatmap_size=self._heatmap_size, blur_kernel_size=11
        )
        xy_image: Float[Tensor, "n mapped 2"] = crop_coords_to_image(
            xy_crop, centers=crop_batch.centers, scales=crop_batch.scales, input_size=self._input_size
        )
        xy_coco: Float[Tensor, "n 133 2"] = torch.full(
            (num_instances, num_keypoints, 2), float("nan"), dtype=torch.float32, device=heatmaps.device
        )
        scores_coco: Float[Tensor, "n 133"] = torch.zeros((num_instances, num_keypoints), dtype=torch.float32, device=heatmaps.device)
        xy_coco[:, coco_indices] = xy_image
        scores_coco[:, coco_indices] = mapped_scores
        return Keypoints2d(xy_coco, scores_coco, detections.frame_indices, self.skeleton)


def _ensure_sapiens_onnx(model_size: SapiensModelSize, static_batch: int) -> Path:
    """Export the Sapiens2 pose module to a cached static-batch ONNX file.

    Args:
        model_size: Sapiens2 checkpoint size.
        static_batch: Batch size baked into the exported graph.

    Returns:
        Path to the cached ONNX export.
    """
    from sapiens2_pose.api.runtime import get_pose_model
    from sapiens2_pose.api.tensorrt_pose import make_sapiens_pose_onnx_exportable
    from sapiens2_pose.sapiens_lite.pose import MODEL_SPECS

    # fp32 export on purpose: an fp16-typed graph is numerically fine on ONNX
    # Runtime but overflows in every TensorRT precision mode (fused ViT kernels).
    # The fp32 interchange runs accurately on ORT and builds accurate BF16 engines.
    onnx_path: Path = DEFAULT_ONNX_CACHE_DIR / f"sapiens2_{model_size.lower()}_pose_b{static_batch}_fp32.onnx"
    if onnx_path.exists():
        return onnx_path
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[posekit] exporting Sapiens2 {model_size} pose to ONNX (one-time): {onnx_path.name}")
    spec: Any = MODEL_SPECS[model_size]
    model: Any = make_sapiens_pose_onnx_exportable(get_pose_model(model_size, "cuda")).eval()
    dummy: Tensor = torch.zeros((static_batch, 3, int(spec.image_size[0]), int(spec.image_size[1])), dtype=torch.float32, device="cuda")
    with torch.no_grad():
        # The dynamo exporter is required: the TorchScript tracer fails on the
        # Sapiens head ("instance_norm for unknown channel size").
        torch.onnx.export(
            model,
            (dummy,),
            str(onnx_path),
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["inputs"],
            output_names=["heatmaps"],
            dynamo=True,
        )
    return onnx_path


@lru_cache(maxsize=8)
def _coco133_projection_indices(device: torch.device) -> tuple[Tensor, Tensor]:
    """COCO-133 <- Goliath-308 index pairs as device tensors.

    Args:
        device: Device the pose tensors live on.

    Returns:
        COCO-133 indices and the Goliath-308 indices they map from.
    """
    from sapiens2_pose.api.metadata import get_sapiens_metainfo

    mapping: dict[int, int] = {int(k): int(v) for k, v in get_sapiens_metainfo()["coco_wholebody_to_goliath_mapping"].items()}
    pairs: list[tuple[int, int]] = sorted((coco, goliath) for coco, goliath in mapping.items() if 0 <= coco < 133)
    coco_indices: Tensor = torch.tensor([pair[0] for pair in pairs], dtype=torch.long, device=device)
    goliath_indices: Tensor = torch.tensor([pair[1] for pair in pairs], dtype=torch.long, device=device)
    return coco_indices, goliath_indices
