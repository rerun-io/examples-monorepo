"""EgoHand's GPU-resident hand pose role: calibrated low-precision TensorRT networks and full SOMA meshes on CUDA.

``FastEgoHandPredictor`` is a :class:`~posekit.models.base.TopDownHandPose3d` over the
same released weights as :class:`~posekit.models.egohand.EgoHandPredictor`: a
calibrated NVFP4 DINOv3 with an FP8 tail and FP8 attention, BF16 TensorRT heads,
captured CUDA crops, and per-frame SOMA reconstruction. Identity, all 26 segment
scales, joint rotations and every hand vertex remain dynamic outputs.
"""

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeAlias

import torch
from jaxtyping import Int64, UInt8
from torch import Tensor
from trtkit import TensorRtBackendConfig, export_onnx
from trtkit.artifacts import build_once
from trtkit.trt_builder import DEFAULT_TRT_CACHE_DIR

from posekit.cameras import CalibratedCamera, CameraKey, camera_key
from posekit.models.base import TopDownHandPose3d
from posekit.models.egohand import (
    CACHED_BACKBONE_ONNX,
    EGOHAND_REVISION,
    SOMA_REVISION,
    EgoHandAssetPaths,
    EgoHandConfig,
    EgoHandDetectorConfig,
    EgoHandReconstruction,
    EgoHandStage,
    TensorRtBackbone,
    asset_paths,
    egohand_weights_path,
    export_backbone_once,
    require_camera,
    soma_assets_path,
    soma_faces,
)
from posekit.models.egohand_head import EgoHandHead
from posekit.models.egohand_onnx import quantize_backbone
from posekit.models.egohand_quantization import EgoHandCalibration, calibrated_backbone_path, load_calibration
from posekit.predictions import BoxDetections, HandDetections, HandSide, Keypoints3d, validate_frames_rgb
from posekit.runtimes import create_runtime_from_onnx
from posekit.skeletons import SOMA_HAND, KeypointSkeleton

# Upstream's EgoHand imports Hydra at module load; environments that only reach posekit.models for other
# roles (mv-api, exo-calib) do not install it. Annotations name the class; at runtime it is an nn.Module.
if TYPE_CHECKING:
    from egohand.models.egohand import EgoHand
else:
    EgoHand = torch.nn.Module

CACHED_FAST_BACKBONE_ONNX: Path = DEFAULT_TRT_CACHE_DIR / f"egohand-dinov3-nvfp4-{EGOHAND_REVISION[:8]}-v1.onnx"
"""Base path of the calibrated NVFP4 DINOv3; the calibration digest and FP8 tail count are appended to its stem."""
CACHED_HEAD_ONNX: Path = DEFAULT_TRT_CACHE_DIR / f"egohand-head-bf16-{EGOHAND_REVISION[:8]}-v1.onnx"
"""Both learned heads with their virtual-camera conditioning, exported in BF16."""


@dataclass(frozen=True, slots=True)
class EgoHandFastConfig:
    """Calibrated NVFP4/FP8 EgoHand on CUDA; released weights and SOMA files download at pinned revisions unless local paths override them."""

    weights: Path | None = None
    """EgoHand safetensors path; None downloads it."""
    soma_assets: Path | None = None
    """Directory containing SOMAHand.npz; None downloads it."""
    batch_size: int = 16
    """Maximum hand crops per captured execution."""
    padding: float = 0.2
    """Added fraction of the square detector crop side."""
    backbone_onnx: Path = CACHED_FAST_BACKBONE_ONNX
    """Base path of the quantized DINOv3; the calibration digest and FP8 tail count are appended to its stem."""
    backbone_source_onnx: Path = CACHED_BACKBONE_ONNX
    """bf16 DINOv3 that the quantized backbone is made from; exported from these weights when absent."""
    head_onnx: Path = CACHED_HEAD_ONNX
    """BF16 heads artifact; exported from these weights when absent."""
    backbone_backend: TensorRtBackendConfig = field(default_factory=lambda: TensorRtBackendConfig(max_batch_size=32, opt_batch_size=2))
    """TensorRT runtime of the backbone; the heads use the same profile without FP8 attention fusion."""
    fp8_tail_blocks: int = 8
    """Final DINO blocks whose linears stay FP8 instead of NVFP4 (0..32)."""
    calibration: Path | None = None
    """Activation ranges; None uses the committed disjoint SelfCap split."""
    hf_egohand_repo_id: str = "pablovela5620/egohand"
    """Hugging Face repo holding the safetensors."""
    hf_egohand_revision: str = EGOHAND_REVISION
    """Pinned commit of that repo."""
    hf_soma_repo_id: str = "nvidia/SOMA-X"
    """Hugging Face repo holding the SOMA hand model."""
    hf_soma_revision: str = SOMA_REVISION
    """Pinned commit of that repo."""

    backbone_precision: Literal["bf16", "nvfp4"] = "nvfp4"
    """The calibrated NVFP4 backbone with its FP8 tail, or unquantized BF16 for GPUs without NVFP4 (Ampere)."""

    def setup(self) -> "FastEgoHandPredictor":
        """Export or reuse the calibrated artifacts and construct the CUDA predictor."""
        return FastEgoHandPredictor(self)


def export_quantized_backbone_once(config: EgoHandFastConfig, weights: Path, soma: Path, calibration: EgoHandCalibration) -> Path:
    """Quantize ``config.backbone_source_onnx`` once, exporting that bf16 source first when it is missing.

    Returns:
        The quantized backbone, named by its calibration digest and FP8 tail count.
    """
    onnx: Path = calibrated_backbone_path(config.backbone_onnx, calibration, fp8_tail_blocks=config.fp8_tail_blocks)

    def quantize() -> None:
        export_backbone_once(
            weights, soma, config.backbone_source_onnx, batch_size=config.batch_size, max_batch_size=config.backbone_backend.max_batch_size
        )
        quantize_backbone(config.backbone_source_onnx, onnx, calibration, fp8_tail_blocks=config.fp8_tail_blocks)

    build_once(onnx, quantize)
    return onnx


def _export_head_once(model: EgoHand, onnx: Path, max_batch_size: int) -> None:
    """Export both learned heads and the analytic ray conditioning once, with a dynamic batch."""

    def export() -> None:
        export_onnx(
            EgoHandHead(model).eval(),
            (torch.zeros((2, 1280, 16, 16), device="cuda"), torch.full((2,), 300.0, device="cuda")),
            onnx,
            input_names=["features", "focal"],
            output_names=["poses", "identity", "scales", "camera", "direct", "presence"],
            compute_dtype=torch.bfloat16,
            dynamic_batch_max=max_batch_size,
        )

    build_once(onnx, export)


class FastEgoHandPredictor(TopDownHandPose3d):
    """RGB frames + anatomical boxes + calibrated camera -> metric SOMA hands, computed and returned on CUDA."""

    skeleton: KeypointSkeleton = SOMA_HAND

    def __init__(self, config: EgoHandFastConfig) -> None:
        from egohand_inference.pipeline.predictor import load_egohand_model

        from posekit.models.egohand_cuda_crops import CudaHandCropper
        from posekit.models.egohand_soma import prepare_cuda_soma
        from posekit.models.egohand_tensor import EgoHandTensorExecutor

        if config.backbone_precision == "nvfp4" and torch.cuda.get_device_capability()[0] < 10:
            raise ValueError('NVFP4 needs a Blackwell GPU (compute capability 10+); without one use catalog_egohand_configs(backbone_precision="bf16").')
        self.config: EgoHandFastConfig = config
        self.profile_stages: bool = False
        """Record per-frame stage times in ``stage_ms``; off, inference adds no synchronization."""
        self.stage_ms: dict[EgoHandStage, list[float]] = {}
        weights: Path = egohand_weights_path(config.weights, config.hf_egohand_repo_id, config.hf_egohand_revision)
        soma: Path = soma_assets_path(config.soma_assets, config.hf_soma_repo_id, config.hf_soma_revision)
        if config.backbone_precision == "bf16":
            export_backbone_once(weights, soma, config.backbone_source_onnx, batch_size=config.batch_size, max_batch_size=config.backbone_backend.max_batch_size)
            backbone_onnx: Path = config.backbone_source_onnx
        else:
            backbone_onnx = export_quantized_backbone_once(config, weights, soma, load_calibration(config.calibration))
        loaded, _ = load_egohand_model(weights, soma, backbone=TensorRtBackbone(create_runtime_from_onnx(backbone_onnx, config.backbone_backend)))
        model: EgoHand = loaded.eval().to("cuda")
        self.faces: dict[HandSide, Tensor] = soma_faces(model)
        _export_head_once(model, config.head_onnx, config.backbone_backend.max_batch_size)
        # The BF16 heads have no Q/DQ attention inputs; FP8 fusion applies only to DINO.
        self.executor: EgoHandTensorExecutor = EgoHandTensorExecutor(
            model, config.batch_size, create_runtime_from_onnx(config.head_onnx, replace(config.backbone_backend, fp8_attention=False))
        )
        prepare_cuda_soma(model.soma_hand)
        # The TensorRT heads replace the torch heads, which nothing else reads.
        del model.hand_head, model.kp2d_head
        torch.cuda.empty_cache()
        # One lens cropper, with its captured graphs, per calibrated camera.
        self.croppers: dict[tuple[CameraKey, str], CudaHandCropper] = {}

    def __call__(
        self, frames_rgb: UInt8[Tensor, "b h w 3"], detections: HandDetections, camera: CalibratedCamera | None = None
    ) -> EgoHandReconstruction:
        """Reconstruct every detected hand in its source camera frame.

        Raises:
            ValueError: If ``camera`` is missing; EgoHand's perspective crops undo the lens.
        """
        from posekit.models.egohand_cuda_crops import CudaHandCropper, PreparedHandCrops
        from posekit.models.egohand_tensor import EgoHandTensors

        camera = require_camera(camera)
        validate_frames_rgb(frames_rgb)
        if not detections.num_detections:
            return EgoHandReconstruction.empty(frames_rgb.device, self.faces)
        key: tuple[CameraKey, str] = (camera_key(camera), str(frames_rgb.device))
        if key not in self.croppers:
            self.croppers[key] = CudaHandCropper(camera, frames_rgb.device)
        cropper: CudaHandCropper = self.croppers[key]
        profile: bool = self.profile_stages
        crop_events: tuple[torch.cuda.Event, torch.cuda.Event] | None = (
            (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) if profile else None
        )
        if crop_events is not None:
            crop_events[0].record()
        # The executor runs left hands first; sort on the host so that the crops come out in that order.
        left_first: list[int] = sorted(range(len(detections.sides)), key=lambda index: detections.sides[index] == "right")
        order: Int64[Tensor, "n"] | None = None
        hands: HandDetections = detections
        if left_first != list(range(len(left_first))):
            order = torch.tensor(left_first, dtype=torch.int64).pin_memory().to(frames_rgb.device, non_blocking=True)
            boxes: BoxDetections = detections.boxes
            hands = HandDetections(
                BoxDetections(boxes.xyxy[order], boxes.scores[order], boxes.frame_indices[order]), tuple(detections.sides[index] for index in left_first)
            )
        prepared: PreparedHandCrops = cropper.prepare(frames_rgb, hands, self.config.padding)
        if crop_events is not None:
            crop_events[1].record()
        output: EgoHandTensors = self.executor(prepared, hands.sides, cropper.project, profile=profile)
        if order is not None:
            output = output.select(torch.argsort(order))
        if crop_events is not None:
            # The executor synchronized after its last chunk, so the crop events have completed too.
            self.stage_ms.setdefault("preprocessing", []).append(crop_events[0].elapsed_time(crop_events[1]) / len(frames_rgb))
            for stage, elapsed in self.executor.stage_ms.items():
                self.stage_ms.setdefault(stage, []).append(elapsed / len(frames_rgb))
        return EgoHandReconstruction(
            keypoints=Keypoints3d(
                output.pixels,
                output.joints[:, :, 2] - output.joints[:, :1, 2],
                detections.boxes.scores[:, None].expand(-1, SOMA_HAND.num_keypoints).contiguous(),
                detections.boxes.frame_indices,
                SOMA_HAND,
                (0,),
                output.joints,
            ),
            sides=detections.sides,
            vertices=output.vertices,
            faces=self.faces,
            rotations=output.rotations,
            identity=output.identity,
            scale=output.scale,
            translation=output.translation,
            scale_parameters=output.scale_parameters,
        )


EgoHandConfigs: TypeAlias = tuple[EgoHandDetectorConfig, EgoHandConfig | EgoHandFastConfig]
"""One EgoHand stack: the detector and either the reference or the fast hand pose role."""


def fast_egohand_configs(assets: Path | None = None) -> tuple[EgoHandDetectorConfig, EgoHandFastConfig]:
    """SelfCap's EgoHand detector with GPU resizing, an FP16 trunk and CUDA-graph replay, and the fast hand pose role.

    Args:
        assets: Local asset tree (see :func:`~posekit.models.egohand.asset_paths`); ``None`` takes the
            pinned downloads and the trtkit engine cache.

    Returns:
        The detector and predictor configurations.
    """
    paths: EgoHandAssetPaths = asset_paths(assets)
    return (
        EgoHandDetectorConfig(
            paths.detector,
            backend=TensorRtBackendConfig(max_batch_size=8, opt_batch_size=1, cache_dir=paths.engines, use_cuda_graph=True),
            fp16_trunk=True,
            gpu_preprocessing=True,
        ),
        EgoHandFastConfig(
            paths.weights,
            paths.soma,
            backbone_onnx=paths.engines / CACHED_FAST_BACKBONE_ONNX.name,
            backbone_source_onnx=paths.bf16_backbone,
            head_onnx=paths.engines / CACHED_HEAD_ONNX.name,
            backbone_backend=TensorRtBackendConfig(max_batch_size=32, opt_batch_size=2, cache_dir=paths.engines),
        ),
    )


def catalog_egohand_configs(
    assets: Path | None = None, *, backbone_precision: Literal["bf16", "nvfp4"] = "nvfp4"
) -> tuple[EgoHandDetectorConfig, EgoHandFastConfig]:
    """The five-dataset calibrated stack: batch-one detector and two-hand pose batches, full outputs.

    NVFP4 (Blackwell): first 8 DINO blocks NVFP4, last 24 FP8, and the detector's three
    encoder input projections FP8. BF16 (Ampere): unquantized pose and an FP16 detector.
    """
    detector, pose = fast_egohand_configs(assets)
    assert isinstance(detector.backend, TensorRtBackendConfig)
    calibration: Path = Path(__file__).resolve().parents[1] / "assets"
    quantize: bool = backbone_precision == "nvfp4"
    return (
        replace(
            detector,
            backend=replace(detector.backend, max_batch_size=1, opt_batch_size=1),
            fp16_trunk=not quantize,
            fp8_calibration=calibration / "egohand_detector_catalog.rrd" if quantize else None,
        ),
        replace(
            pose,
            backbone_precision=backbone_precision,
            fp8_tail_blocks=24,
            calibration=calibration / "egohand_catalog.rrd" if quantize else None,
            batch_size=2,
            backbone_backend=replace(pose.backbone_backend, max_batch_size=2, opt_batch_size=2),
        ),
    )
