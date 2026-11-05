"""EgoHand's GPU-resident hand pose role: calibrated low-precision TensorRT networks and full SOMA meshes on CUDA.

``FastEgoHandPredictor`` is a :class:`~posekit.models.base.TopDownHandPose3d` over the
same released weights as :class:`~posekit.models.egohand.EgoHandPredictor`: a
calibrated NVFP4 DINOv3 with an FP8 tail and FP8 attention, BF16 TensorRT heads,
captured CUDA crops, and per-frame SOMA reconstruction. Identity, all 26 segment
scales, joint rotations and every hand vertex remain dynamic outputs.
"""

import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TypeAlias

import numpy as np
import torch
from jaxtyping import UInt8
from torch import Tensor
from trtkit import TensorRtBackendConfig, export_onnx
from trtkit.artifacts import build_once
from trtkit.trt_builder import DEFAULT_TRT_CACHE_DIR

from egohand.models.egohand import EgoHand
from egohand_inference.perf import PerfStage
from posekit.cameras import CalibratedCamera
from posekit.models.base import TopDownHandPose3d
from posekit.models.egohand import (
    CACHED_BACKBONE_ONNX,
    EGOHAND_REVISION,
    SOMA_REVISION,
    EgoHandConfig,
    EgoHandDetectorConfig,
    EgoHandReconstruction,
    TensorRtBackbone,
    egohand_weights_path,
    export_backbone_once,
    soma_assets_path,
)
from posekit.models.egohand_head import EgoHandHead
from posekit.models.egohand_onnx import quantize_backbone
from posekit.models.egohand_quantization import EgoHandCalibration, calibrated_backbone_path, load_calibration
from posekit.predictions import HandDetections, HandSide, Keypoints3d, validate_frames_rgb
from posekit.runtimes import create_runtime_from_onnx
from posekit.skeletons import SOMA_HAND, KeypointSkeleton

CACHED_FAST_BACKBONE_ONNX: Path = DEFAULT_TRT_CACHE_DIR / f"egohand-dinov3-nvfp4-{EGOHAND_REVISION[:8]}-v1.onnx"
"""Base path of the calibrated NVFP4 DINOv3; the calibration digest and FP8 tail count are appended to its stem."""
CACHED_HEAD_ONNX: Path = DEFAULT_TRT_CACHE_DIR / f"egohand-head-bf16-{EGOHAND_REVISION[:8]}-v1.onnx"
"""Both learned heads with their virtual-camera conditioning, exported in BF16."""

CameraKey: TypeAlias = tuple[str, bytes, bytes, str]
"""Lens model, intrinsics bytes, distortion bytes and device of one calibrated camera."""


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
        from egohand_inference.pipeline.predictor import EgoHandPredictor as UpstreamPredictor
        from egohand_inference.pipeline.predictor import load_egohand_model
        from posekit.models.egohand_cuda_crops import CudaHandCropper
        from posekit.models.egohand_tensor import EgoHandTensorExecutor

        self.config: EgoHandFastConfig = config
        self.stage_ms: dict[str, list[float]] = {}
        weights: Path = egohand_weights_path(config.weights, config.hf_egohand_repo_id, config.hf_egohand_revision)
        soma: Path = soma_assets_path(config.soma_assets, config.hf_soma_repo_id, config.hf_soma_revision)
        calibration: EgoHandCalibration = load_calibration(config.calibration)
        backbone_onnx: Path = export_quantized_backbone_once(config, weights, soma, calibration)
        model, _ = load_egohand_model(weights, soma, backbone=TensorRtBackbone(create_runtime_from_onnx(backbone_onnx, config.backbone_backend)))
        predictor: UpstreamPredictor = UpstreamPredictor(model, config.batch_size, device="cuda", dtype=torch.bfloat16, use_perspective_cropping=True)
        self.faces: dict[HandSide, Tensor] = {
            side: torch.from_numpy(np.asarray(model.soma_hand.faces(side), dtype=np.int64)).cuda() for side in ("left", "right")
        }
        _export_head_once(model, config.head_onnx, config.backbone_backend.max_batch_size)
        # The BF16 heads have no Q/DQ attention inputs; FP8 fusion applies only to DINO.
        self.executor: EgoHandTensorExecutor = EgoHandTensorExecutor(
            predictor, create_runtime_from_onnx(config.head_onnx, replace(config.backbone_backend, fp8_attention=False))
        )
        # One lens cropper, with its captured graphs, per calibrated camera.
        self.croppers: dict[CameraKey, CudaHandCropper] = {}

    def __call__(
        self, frames_rgb: UInt8[Tensor, "b h w 3"], detections: HandDetections, camera: CalibratedCamera | None = None
    ) -> EgoHandReconstruction:
        """Reconstruct every detected hand in its source camera frame.

        Raises:
            ValueError: If ``camera`` is missing; EgoHand's perspective crops undo the lens.
        """
        from posekit.models.egohand_cuda_crops import CudaHandCropper, PreparedHandCrops
        from posekit.models.egohand_tensor import EgoHandTensors

        if camera is None:
            raise ValueError("EgoHand needs the CalibratedCamera of the frames: its perspective crops undo the lens.")
        validate_frames_rgb(frames_rgb)
        if not detections.num_detections:
            return EgoHandReconstruction.empty(frames_rgb.device, self.faces)
        camera_key: CameraKey = (camera.lens, camera.intrinsics.tobytes(), camera.distortion.tobytes(), str(frames_rgb.device))
        if camera_key not in self.croppers:
            self.croppers[camera_key] = CudaHandCropper(camera, frames_rgb.device)
        cropper: CudaHandCropper = self.croppers[camera_key]
        started: float = time.perf_counter()
        prepared: PreparedHandCrops = cropper.prepare(frames_rgb, detections, self.config.padding)
        torch.cuda.synchronize()
        self.stage_ms.setdefault(PerfStage.PREPROCESSING.value, []).append((time.perf_counter() - started) * 1000.0 / len(frames_rgb))
        output: EgoHandTensors = self.executor(prepared, detections.sides, cropper)
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
        assets: Local weights/engines tree (``weights/...``, ``dinov3-bf16.onnx``, ``engines/``);
            ``None`` takes the pinned downloads and the trtkit engine cache.

    Returns:
        The detector and predictor configurations.
    """
    engines: Path = DEFAULT_TRT_CACHE_DIR if assets is None else assets / "engines"
    return (
        EgoHandDetectorConfig(
            None if assets is None else assets / "weights/detector/hand_detector.onnx",
            backend=TensorRtBackendConfig(max_batch_size=8, opt_batch_size=1, cache_dir=engines, use_cuda_graph=True),
            fp16_trunk=True,
            gpu_preprocessing=True,
        ),
        EgoHandFastConfig(
            None if assets is None else assets / "weights/egohand/egohand_s060000.safetensors",
            None if assets is None else assets / "weights/soma",
            backbone_onnx=engines / CACHED_FAST_BACKBONE_ONNX.name,
            backbone_source_onnx=CACHED_BACKBONE_ONNX if assets is None else assets / "dinov3-bf16.onnx",
            head_onnx=engines / CACHED_HEAD_ONNX.name,
            backbone_backend=TensorRtBackendConfig(max_batch_size=32, opt_batch_size=2, cache_dir=engines),
        ),
    )
