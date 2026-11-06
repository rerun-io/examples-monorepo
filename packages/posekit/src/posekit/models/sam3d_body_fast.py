"""SAM 3D Body with captured decoder execution and the complete MHR output."""

from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from jaxtyping import UInt8
from serde import serde
from torch import Tensor, nn
from trtkit import TensorRtBackendConfig
from trtkit.trt_builder import DEFAULT_TRT_CACHE_DIR

from posekit.cameras import CalibratedCamera, CameraKey, camera_key
from posekit.models.dinov3_calibration import DINO_BLOCKS
from posekit.models.sam3d_body import ROOT_INDICES, SAM3D_BODY_REVISION, Sam3dBodyConfig, Sam3dBodyPredictor, Sam3dBodyReconstruction
from posekit.models.sam3d_body_quantization import BodyBackbonePrecision, BodyCalibration, BodyPrecisionPlan
from posekit.predictions import BoxDetections, Keypoints3d, validate_frames_rgb

CACHED_BODY_BACKBONE: Path = DEFAULT_TRT_CACHE_DIR / "sam3d-body-backbone-nvfp4.onnx"
"""Calibration-identified quantized backbone; the digest also identifies its checkpoint revision."""
CACHED_BODY_SOURCE: Path = DEFAULT_TRT_CACHE_DIR / f"sam3d-body-backbone-bf16-{SAM3D_BODY_REVISION[:8]}-v3.onnx"
"""BF16 source exported from the pinned body checkpoint at 512 pixels."""
CACHED_BODY_DECODERS: Path = DEFAULT_TRT_CACHE_DIR / f"sam3d-body-decoders-{SAM3D_BODY_REVISION[:8]}-v1"
"""The twelve learned decoder layers from the pinned checkpoint."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class Sam3dBodyFastConfig(Sam3dBodyConfig):
    """Released SAM 3D Body weights with CUDA graph execution."""

    gpu_preprocessing: bool = True
    """Prepare crops and merge body/hand results entirely on CUDA."""
    capture_pipeline: bool = True
    """Capture the complete GPU pipeline; off retains per-decoder graphs for ablations."""
    backbone_onnx: Path | None = CACHED_BODY_BACKBONE
    """Quantized backbone base path (or its resolved calibration path); None keeps the Torch backbone for ablations."""
    compile_torch_backbone: bool = False
    """Compile the Torch backbone while retaining BF16 intermediate rounding; requires backbone_onnx=None."""
    backbone_source_onnx: Path = CACHED_BODY_SOURCE
    """BF16 source backbone, exported from these weights when missing."""
    backbone_precision: Literal["hybrid", "nvfp4", "bf16"] = "hybrid"
    """Default: BF16 body and calibrated FP8 hands. NVFP4 and wholly BF16 are explicit alternatives."""
    decoder_onnx_dir: Path | None = CACHED_BODY_DECODERS
    """Learned BF16 decoder layers, exported when missing; None keeps Torch layers for ablations."""
    backbone_backend: TensorRtBackendConfig = field(default_factory=lambda: TensorRtBackendConfig(max_batch_size=16, opt_batch_size=2, use_cuda_graph=True))
    """Backbone runtime profile; decoder layers use this profile with internal graph capture and FP8 normalization disabled."""
    calibration: Path | None = None
    """SAM-specific 512-pixel ranges; None loads the committed wide calibration and disjoint evaluation split."""
    fp8_tail_blocks: int = 8
    """Final backbone blocks that use FP8 linears instead of NVFP4."""
    backbone_plan: BodyPrecisionPlan = field(default_factory=BodyPrecisionPlan)
    """Per-module body backbone precision overrides, applied after the FP8 tail selection."""
    hand_backbone_plan: BodyPrecisionPlan = field(default_factory=BodyPrecisionPlan)
    """Additional hand backbone precision overrides, applied after the body plan."""

    def __post_init__(self) -> None:
        alternate_checkpoint: bool = (
            self.checkpoint_dir is not None or self.hf_repo_id != "facebook/sam-3d-body-dinov3"
            or self.hf_revision != SAM3D_BODY_REVISION
        )
        if alternate_checkpoint and (
            (self.backbone_onnx is not None and (self.backbone_source_onnx == CACHED_BODY_SOURCE or self.backbone_onnx == CACHED_BODY_BACKBONE))
            or self.decoder_onnx_dir == CACHED_BODY_DECODERS
        ):
            raise ValueError("An alternate checkpoint requires explicit export paths, or backbone_onnx=None and decoder_onnx_dir=None.")
        if self.compile_torch_backbone and self.backbone_onnx is not None:
            raise ValueError("compile_torch_backbone requires backbone_onnx=None.")
        if (
            (self.backbone_onnx is None or self.backbone_precision == "bf16")
            and (self.backbone_plan.linears or self.backbone_plan.attention or self.hand_backbone_plan.linears or self.hand_backbone_plan.attention)
        ):
            raise ValueError("Module precision plans require backbone_onnx and backbone_precision='nvfp4'. Use BF16 overrides within the plan for mixed precision.")
        if not self.gpu_preprocessing and (self.hand_backbone_plan.linears or self.hand_backbone_plan.attention):
            raise ValueError("A separate hand backbone plan requires GPU preprocessing.")
        if not self.gpu_preprocessing and self.backbone_onnx is not None and self.backbone_precision == "hybrid":
            raise ValueError("The hybrid backbone requires GPU preprocessing for separate body and hand plans.")

    def precision_plans(self, calibration: BodyCalibration) -> tuple[BodyPrecisionPlan, BodyPrecisionPlan]:
        """Resolve the complete body and hand plans, including explicit module overrides."""
        body: BodyPrecisionPlan = BodyPrecisionPlan()
        hands: BodyPrecisionPlan = BodyPrecisionPlan()
        if self.backbone_precision == "hybrid":
            body = BodyPrecisionPlan({name: "bf16" for name in calibration.input_amax}, {index: "bf16" for index in range(DINO_BLOCKS)})
            hands = BodyPrecisionPlan({name: "fp8" for name in calibration.input_amax}, {index: "fp8" for index in range(DINO_BLOCKS)})
        body = BodyPrecisionPlan(body.linears | self.backbone_plan.linears, body.attention | self.backbone_plan.attention)
        hands = BodyPrecisionPlan(
            body.linears | hands.linears | self.hand_backbone_plan.linears,
            body.attention | hands.attention | self.hand_backbone_plan.attention,
        )
        return body, hands

    def setup(self) -> "FastSam3dBodyPredictor":
        """Load the model and prepare capture-safe decoders."""
        return FastSam3dBodyPredictor(self)


class FastSam3dBodyPredictor(Sam3dBodyPredictor):
    """The complete body role with one graph owner per decoder input shape."""

    def __init__(self, config: Sam3dBodyFastConfig) -> None:
        import warp as wp
        from sam_3d_body.models.heads.mhr_head import MHRHead

        from posekit.models.sam3d_body_capture import CapturedBodyDecoder, CapturePromptEncoder, prepare_decoder_projection
        from posekit.models.sam3d_body_executor import BodyTensorExecutor
        from posekit.models.sam3d_body_head import CaptureMhrHead
        from posekit.models.sam3d_body_quantization import load_body_calibration

        calibration = load_body_calibration(config.calibration) if config.backbone_onnx is not None and config.backbone_precision != "bf16" else None
        if calibration is not None and calibration.revision != config.hf_revision:
            raise ValueError("SAM body calibration revision differs from the selected checkpoint revision")
        wp.init()
        super().__init__(config)
        self.camera_intrinsics: dict[tuple[CameraKey, torch.device], Tensor] = {}
        model = self.estimator.model
        capture_pipeline: bool = config.gpu_preprocessing and config.capture_pipeline
        hand_backbone: nn.Module | None = None
        if config.decoder_onnx_dir is not None:
            from sam_3d_body.models.modules.transformer import TransformerDecoderLayer

            from posekit.models.sam3d_body_onnx import build_decoder_layer

            for name, decoder in (("body", model.decoder), ("hand", model.decoder_hand)):
                for index, layer in enumerate(decoder.layers):
                    assert isinstance(layer, TransformerDecoderLayer)
                    decoder.layers[index] = build_decoder_layer(
                        layer, config.decoder_onnx_dir / f"{name}-{index}.onnx",
                        replace(config.backbone_backend, use_cuda_graph=False, fp8_attention=False),
                    )
        if config.backbone_onnx is not None:
            from posekit.models.sam3d_body_onnx import TensorRtBodyBackbone, build_body_backbone

            torch_backbone: nn.Module = model.backbone
            plans: tuple[BodyPrecisionPlan, BodyPrecisionPlan] = (
                config.precision_plans(calibration) if calibration is not None else (BodyPrecisionPlan(), BodyPrecisionPlan())
            )
            precision: BodyBackbonePrecision = "bf16" if config.backbone_precision == "bf16" else "nvfp4"
            backend: TensorRtBackendConfig = replace(config.backbone_backend, use_cuda_graph=False) if capture_pipeline else config.backbone_backend
            backbone = TensorRtBodyBackbone(build_body_backbone(
                torch_backbone, config.backbone_source_onnx, config.backbone_onnx, calibration,
                backend, config.fp8_tail_blocks, precision, plan=plans[0],
            ))
            if config.gpu_preprocessing and config.mode == "full":
                # TensorRT performs a host synchronization after a shape change.
                # Separate contexts keep body and hand shapes stable while the
                # enclosing graph records both dependent passes.
                hand_backbone = TensorRtBodyBackbone(build_body_backbone(
                    torch_backbone, config.backbone_source_onnx, config.backbone_onnx, calibration,
                    backend, config.fp8_tail_blocks, precision, plan=plans[1],
                ))
            model.add_module("backbone", backbone)
        elif config.compile_torch_backbone:
            # Keep BF16 operator boundaries: ordinary fusion changes rounding
            # enough to fail the OAK accuracy band. The outer pipeline owns capture.
            model.backbone.forward = torch.compile(
                model.backbone.forward, fullgraph=True, dynamic=False,
                options={"triton.cudagraphs": False, "emulate_precision_casts": True},
            )
        assert isinstance(model.head_pose, MHRHead) and isinstance(model.head_pose_hand, MHRHead)
        heads: tuple[CaptureMhrHead, CaptureMhrHead] = (CaptureMhrHead(model.head_pose), CaptureMhrHead(model.head_pose_hand))
        model.add_module("head_pose", heads[0])
        model.add_module("head_pose_hand", heads[1])
        if config.gpu_preprocessing:
            prepare_decoder_projection(model)
        model.prompt_encoder = CapturePromptEncoder(model.prompt_encoder)
        pool: torch.cuda._POOL_HANDLE = torch.cuda.graph_pool_handle()
        self.body_decoder: CapturedBodyDecoder = CapturedBodyDecoder(
            model.forward_decoder, pool, self.clock, full_mesh=heads[0] if config.mode == "body" else None,
            prompt_decoder=model.decoder if config.gpu_preprocessing else None, capture=not capture_pipeline,
        )
        self.hand_decoder: CapturedBodyDecoder = CapturedBodyDecoder(model.forward_decoder_hand, pool, self.clock, hand=True, capture=not capture_pipeline)
        model.forward_decoder = self.body_decoder
        model.forward_decoder_hand = self.hand_decoder
        self.tensor_executor: BodyTensorExecutor | None = BodyTensorExecutor(
            model, heads[0], self.clock, capture=capture_pipeline,
            max_people=max(1, config.backbone_backend.max_batch_size // 2), hand_backbone=hand_backbone,
        ) if config.gpu_preprocessing else None

    @torch.inference_mode()
    def __call__(
        self, frames_rgb: UInt8[Tensor, "b h w 3"], detections: BoxDetections, camera: CalibratedCamera | None = None,
    ) -> Sam3dBodyReconstruction:
        """Reconstruct complete camera-space bodies while retaining frame and output tensors on CUDA."""
        if self.tensor_executor is None:
            return super().__call__(frames_rgb, detections, camera)
        validate_frames_rgb(frames_rgb)
        if detections.num_detections == 0:
            return self._to_reconstruction([], [], np.empty(0, dtype=np.float32), frames_rgb.device)
        intrinsic: Tensor | None = None
        if camera is not None:
            camera_identity: tuple[CameraKey, torch.device] = (camera_key(camera), frames_rgb.device)
            if camera_identity not in self.camera_intrinsics:
                self.camera_intrinsics[camera_identity] = torch.as_tensor(camera.intrinsics, dtype=torch.float32, device=frames_rgb.device)
            intrinsic = self.camera_intrinsics[camera_identity]
        self.clock.start_frame()
        events: tuple[torch.cuda.Event, torch.cuda.Event] | None = (
            (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) if self.profile_stages else None
        )
        if events is not None:
            events[0].record()
        with self.clock.recording(self.tensor_executor.hand_backbone) if self.profile_stages else nullcontext():
            body, left_boxes, right_boxes = self.tensor_executor.run(frames_rgb, detections, intrinsic, full=self.config.mode == "full")
        translation: Tensor = body.pred_cam_t
        xyz: Tensor = body.pred_keypoints_3d + translation[:, None]
        root_z: Tensor = (xyz[:, ROOT_INDICES[0], 2:3] + xyz[:, ROOT_INDICES[1], 2:3]) / 2
        result = Sam3dBodyReconstruction(
            keypoints=Keypoints3d(body.pred_keypoints_2d, xyz[:, :, 2] - root_z, detections.scores[:, None].expand(-1, 70).contiguous(),
                                 detections.frame_indices, self.skeleton, ROOT_INDICES, xyz),
            boxes=detections.xyxy, vertices=body.pred_vertices + translation[:, None], faces=self.faces,
            translation=translation, focal_length=body.focal_length, global_rotation=body.global_rot,
            body_pose=body.body_pose, hand_pose=body.hand, scale=body.scale, shape=body.shape, expression=body.face,
            joint_coordinates=body.pred_joint_coords + translation[:, None], joint_rotations=body.joint_global_rots,
            left_hand_boxes=left_boxes, right_hand_boxes=right_boxes,
        )
        if events is not None:
            events[1].record()
            events[1].synchronize()
            for stage, value in self.clock.stage_ms(events[0].elapsed_time(events[1])).items():
                self.stage_ms.setdefault(stage, []).append(value)
        return result

    @property
    def graph_count(self) -> int:
        """Number of whole-frame and decoder graphs owned by this predictor."""
        frames: int = 0 if self.tensor_executor is None else len(self.tensor_executor.graphs)
        return frames + len(self.body_decoder.graphs) + len(self.hand_decoder.graphs)
