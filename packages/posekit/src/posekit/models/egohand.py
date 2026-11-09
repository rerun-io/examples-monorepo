"""NVIDIA EgoHand as posekit hand roles: RT-DETR hand detector + calibrated top-down SOMA hands.

``EgoHandDetector`` is a :class:`~posekit.models.base.HandDetector` and
``EgoHandPredictor`` a :class:`~posekit.models.base.TopDownHandPose3d` over the
IsaacCapture inference source (``egohand``/``egohand_inference``, recipes/egohand-src).
The eager reference and accelerated path share upstream crop and mirroring
geometry. Optional GPU sampling and TensorRT preserve full SOMA meshes and
parameters. Remaining CPU output conversion is included in stage timings.
The GPU-resident fast role, ``FastEgoHandPredictor``, shares the weights,
detector and backbone export helpers below (see ``posekit.models.egohand_fast``).
"""

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeAlias

import numpy as np
import torch
from egohand_inference.hand_detector_2d.rtdetr.tracking import select_new
from egohand_inference.perf import PerfStage
from huggingface_hub import hf_hub_download, snapshot_download
from jaxtyping import Bool, Float32, Int, Int64, UInt8
from numpy import ndarray
from PIL import Image
from serde import from_dict, serde
from torch import Tensor
from trtkit import TensorRtBackendConfig, export_onnx
from trtkit.artifacts import build_once
from trtkit.trt_builder import DEFAULT_TRT_CACHE_DIR

from posekit.cameras import CalibratedCamera
from posekit.models.base import HandDetector, TopDownHandPose3d
from posekit.ops.pil_resize import resize_bilinear_uint8
from posekit.predictions import BoxDetections, HandDetections, HandPoses3d, HandSide, Keypoints3d, validate_frames_rgb
from posekit.runtimes import OnnxBackendConfig, OnnxOrTrtBackendConfig, TensorRuntime, create_runtime_from_onnx, run_chunked
from posekit.skeletons import SOMA_HAND, KeypointSkeleton

# Upstream's EgoHand imports Hydra at module load; environments that only reach posekit.models for other
# roles (mv-api, exo-calib) do not install it. Annotations name the class; at runtime it is an nn.Module.
if TYPE_CHECKING:
    from egohand.models.egohand import EgoHand
else:
    EgoHand = torch.nn.Module


def resolve_hand_identities(detections: HandDetections, frame_count: int) -> HandDetections:
    """Assign an upright ego sequence before deciding which crops to mirror.

    This reuses upstream's candidate deduplication and IoU identity assignment.
    Two-hand bootstrap uses image geometry; later frames use track continuity,
    including crossings. Detector class IDs are only a lone-bootstrap fallback.
    The original selected boxes and scores are retained: no padding or smoothing.
    """
    # One device-to-host copy: frame indices are exact in float32.
    host: Float32[ndarray, "n 6"] = (
        torch.cat((detections.boxes.xyxy, detections.boxes.scores[:, None], detections.boxes.frame_indices[:, None].float()), dim=1).cpu().numpy()
    )
    boxes: Float32[ndarray, "n 4"] = host[:, :4]
    scores: list[float] = host[:, 4].tolist()
    frames: list[int] = host[:, 5].astype(np.int64).tolist()
    records: list[dict[str, list[list[float]]]] = [{"L": [], "R": []} for _ in range(frame_count)]
    candidates_by_frame: list[list[int]] = [[] for _ in range(frame_count)]
    for row, (box, frame, score, side) in enumerate(zip(boxes, frames, scores, detections.sides, strict=True)):
        candidates_by_frame[frame].append(row)
        records[frame]["L" if side == "left" else "R"].append(
            [
                float((box[0] + box[2]) / 2),
                float((box[1] + box[3]) / 2),
                float(box[2] - box[0]),
                float(box[3] - box[1]),
                score,
                1.0 if side == "left" else 2.0,
            ]
        )
    tracks, _ = select_new(records)
    selected: list[int] = []
    sides: list[HandSide] = []
    for frame in range(frame_count):
        candidates: list[int] = candidates_by_frame[frame]
        for side, key in (("left", "L"), ("right", "R")):
            if not tracks[key][1][frame]:
                continue
            center: Float32[ndarray, "2"] = tracks[key][0][frame, :2]
            row: int = min(candidates, key=lambda i: float(np.linalg.norm((boxes[i, :2] + boxes[i, 2:]) / 2 - center)))
            selected.append(row)
            sides.append(side)
    return HandDetections(
        BoxDetections(
            detections.boxes.xyxy[selected],
            detections.boxes.scores[selected],
            detections.boxes.frame_indices[selected],
        ),
        tuple(sides),
    )


EgoHandStage: TypeAlias = Literal["preprocessing", "batch_assembly", "model_forward", "post_forward", "adaptive_smoothing", "backbone", "head", "soma"]
"""Stage names of ``stage_ms``: upstream's per-frame stages plus the CUDA-event-timed backbone, head and SOMA."""
UPSTREAM_STAGES: dict[PerfStage, EgoHandStage] = {
    PerfStage.PREPROCESSING: "preprocessing",
    PerfStage.BATCH_ASSEMBLY: "batch_assembly",
    PerfStage.MODEL_FORWARD: "model_forward",
    PerfStage.POST_FORWARD: "post_forward",
    PerfStage.ADAPTIVE_SMOOTHING: "adaptive_smoothing",
}
"""The upstream reporter stages EgoHandPredictor keeps."""

EGOHAND_REVISION: str = "24a325fde719fdb985c7fb4606115b0c1912ab7d"
"""Weights of nvidia/EgoHand-v1 a153616c egohand_s060000.ckpt, without its pickle."""
CACHED_BACKBONE_ONNX: Path = DEFAULT_TRT_CACHE_DIR / f"egohand-dinov3-bf16-{EGOHAND_REVISION[:8]}.onnx"
"""The bf16 DINOv3 exported from those weights, beside the engines built from it."""
SOMA_REVISION: str = "f81a01796f652260ff930d5318d6c6b845a505ed"
"""Pinned nvidia/SOMA-X commit holding the SOMA hand model."""
SOMA_HAND_FILES: tuple[str, ...] = (
    "SOMAHand.npz", "SOMA_neutral.npz", "SOMA_procedural_transforms.json", "SOMA_template_rig.usda",
    "MHR/mhr_model_lod1.pt", "MHR/base_body_lod1.obj", "MHR/SOMA_wrap_lod1.obj",
)
"""What the SOMA hand model reads."""


@dataclass(frozen=True, slots=True)
class EgoHandDetectorConfig:
    """Released ONNX hand detector configuration."""

    weights: Path | None = None
    """Path to hand_detector.onnx; None downloads it."""
    threshold: float = 0.5
    """Minimum detector confidence, unrelated to landmark visibility."""
    backend: OnnxOrTrtBackendConfig = field(default_factory=OnnxBackendConfig)
    """ONNX Runtime reference or cached TensorRT engine, with a dynamic batch."""
    hf_detector_repo_id: str = "nvidia/RT-DETR-Hand-Detector-v1"
    """Hugging Face repo holding hand_detector.onnx."""
    hf_detector_revision: str = "55572ddc93a5916c3e486811dc6e62bb2b7bdb9d"
    """Pinned commit of that repo."""
    fp16_trunk: bool = False
    """Run the detector convolutional trunk in FP16, with its outputs retained in FP32."""
    gpu_preprocessing: bool = False
    """Resize decoded CUDA frames in place of the reference PIL host round trip."""

    fp8_calibration: Path | None = None
    """Disjoint detector activation ranges; enables FP8 convolutions with an FP32 decoding tail."""

    fp8_scope: Literal["all", "backbone", "late_backbone", "projections"] = "all"
    """Quantize all, residual backbone, final residual stage, or three encoder feature projections."""

    def weights_path(self) -> Path:
        """The ONNX file: ``weights``, or the pinned download in the Hugging Face cache."""
        if self.weights is not None:
            return self.weights
        return Path(hf_hub_download(repo_id=self.hf_detector_repo_id, filename="hand_detector.onnx", revision=self.hf_detector_revision))

    def setup(self) -> "EgoHandDetector":
        """Create the CUDA detector."""
        return EgoHandDetector(self)


class EgoHandDetector(HandDetector):
    """CUDA detector with upstream sequence-level anatomical identity assignment."""

    def __init__(self, config: EgoHandDetectorConfig) -> None:
        self.config: EgoHandDetectorConfig = config
        weights: Path = config.weights_path()
        if config.fp8_calibration is not None:
            from posekit.models.egohand_detector_quantization import quantized_detector_path

            if not isinstance(config.backend, TensorRtBackendConfig):
                raise ValueError("The FP8 detector requires a TensorRT backend.")
            if torch.cuda.get_device_capability()[0] < 9:
                raise ValueError("The FP8 detector requires an FP8-capable CUDA GPU.")
            weights = quantized_detector_path(weights, config.fp8_calibration, config.backend.cache_dir, scope=config.fp8_scope)
        elif config.fp16_trunk:
            from trtkit.onnx_precision import fp16_trunk

            if not isinstance(config.backend, TensorRtBackendConfig):
                raise ValueError("The FP16 detector trunk requires a TensorRT backend.")
            weights = fp16_trunk(weights, config.backend.cache_dir)
        self.runtime: TensorRuntime = create_runtime_from_onnx(weights, config.backend)

    def __call__(self, frames_rgb: UInt8[Tensor, "b h w 3"]) -> HandDetections:
        """Detect an ordered RGB sequence; resolve identity before hand mirroring."""
        from posekit.models.egohand_geometry import detector_hand_side

        validate_frames_rgb(frames_rgb)
        boxes: list[Tensor] = []
        scores: list[Tensor] = []
        indices: list[Tensor] = []
        sides: list[HandSide] = []
        height: int = int(frames_rgb.shape[1])
        width: int = int(frames_rgb.shape[2])
        for start in range(0, len(frames_rgb), self.runtime.spec.max_batch_size):
            stop: int = min(start + self.runtime.spec.max_batch_size, len(frames_rgb))
            pixels: Float32[Tensor, "b 3 640 640"] = (
                _resize_on_gpu(frames_rgb[start:stop]) if self.config.gpu_preprocessing else _resize_with_pil(frames_rgb[start:stop])
            )
            # Filled on the device: a host tensor here would be a blocking upload on every call.
            sizes: Int64[Tensor, "b 2"] = torch.empty((stop - start, 2), dtype=torch.int64, device=frames_rgb.device)
            sizes[:, 0] = width
            sizes[:, 1] = height
            outputs: dict[str, Tensor] = run_chunked(
                self.runtime,
                {
                    "images": pixels.contiguous(),
                    "orig_target_sizes": sizes,
                },
            )
            # Keep candidates across class IDs for upstream's identity assignment.
            # Top-1 per class can discard a hand when both are misclassified alike.
            labels: Int[Tensor, "b q"] = outputs["labels"]
            keep: Bool[Tensor, "b 2 q"] = torch.stack([labels == 1, labels == 2], dim=1) & (outputs["scores"] > self.config.threshold)[:, None]
            # Row-major order: frame, then label 1 before 2, then query.
            hits: tuple[Tensor, ...] = torch.where(keep)
            frame: Int64[Tensor, "n"] = hits[0]
            query: Int64[Tensor, "n"] = hits[2]
            boxes.append(outputs["boxes"][frame, query].float())
            scores.append(outputs["scores"][frame, query].float())
            indices.append(frame + start)
            sides.extend(detector_hand_side(label + 1) for label in hits[1].tolist())
        if not sides:
            return HandDetections.empty(frames_rgb.device)
        raw: HandDetections = HandDetections(
            BoxDetections(torch.cat(boxes), torch.cat(scores), torch.cat(indices).to(device=frames_rgb.device, dtype=torch.int64)), tuple(sides)
        )
        return resolve_hand_identities(raw, len(frames_rgb))


def _resize_with_pil(frames_rgb: UInt8[Tensor, "b h w 3"]) -> Float32[Tensor, "b 3 640 640"]:
    """The reference detector input: PIL bilinear resizes of host copies, on up to eight threads."""
    host_frames: UInt8[ndarray, "b h w 3"] = frames_rgb.cpu().numpy()
    with ThreadPoolExecutor(max_workers=min(8, len(host_frames))) as pool:
        resized: list[UInt8[ndarray, "640 640 3"]] = list(
            pool.map(lambda frame: np.asarray(Image.fromarray(frame).resize((640, 640), Image.Resampling.BILINEAR)).copy(), host_frames)
        )
    return torch.from_numpy(np.stack(resized)).permute(0, 3, 1, 2).to(device=frames_rgb.device, dtype=torch.float32) / 255.0


def _resize_on_gpu(frames_rgb: UInt8[Tensor, "b h w 3"]) -> Float32[Tensor, "b 3 640 640"]:
    """Match the reference PIL bilinear input without transferring image pixels."""
    return resize_bilinear_uint8(frames_rgb, (640, 640))


@serde
@dataclass(frozen=True, slots=True)
class _UpstreamHand:
    """Typed boundary for a single upstream model result (unknown fields allowed)."""

    pred_keypoints_3d: Float32[ndarray, "25 3"]
    """Wrist-relative joints in source camera axes, metres."""
    transl: Float32[ndarray, "3"]
    """Wrist camera translation, metres."""
    poses: Float32[ndarray, "25 3 3"]
    """SOMA local joint rotations."""
    identity_coeffs: Float32[ndarray, "n"]
    """SOMA or MHR identity vector."""
    global_scale: float
    """Uniform hand scale."""
    scale_params: Float32[ndarray, "n"] | None
    """Optional MHR scale coefficients."""
    pred_vertices: Float32[ndarray, "v 3"]
    """Wrist-relative mesh in source camera axes, metres."""


@dataclass(frozen=True, slots=True, kw_only=True)
class EgoHandReconstruction(HandPoses3d):
    """Metric SOMA hands: camera-space joints and meshes plus the SOMA parameters behind them."""

    rotations: Float32[Tensor, "n 25 3 3"]
    """SOMA local joint rotations with source-camera wrist orientation."""
    identity: Float32[Tensor, "n k"]
    """Learned identity coefficients."""
    scale: Float32[Tensor, "n"]
    """Uniform model scale."""
    translation: Float32[Tensor, "n 3"]
    """Camera-space wrist translation in metres."""
    scale_parameters: Float32[Tensor, "n s"] | None
    """Per-finger MHR scales; None for checkpoints without this head."""

    @classmethod
    def empty(cls, device: torch.device, faces: dict[HandSide, Tensor]) -> "EgoHandReconstruction":
        """No hands: zero-row outputs on ``device``."""
        joints: int = SOMA_HAND.num_keypoints
        return cls(
            keypoints=Keypoints3d.empty(SOMA_HAND, device, root_indices=(0,), with_xyz_camera=True),
            sides=(),
            vertices=None,
            faces=faces,
            rotations=torch.empty((0, joints, 3, 3), device=device),
            identity=torch.empty((0, 0), device=device),
            scale=torch.empty((0,), device=device),
            translation=torch.empty((0, 3), device=device),
            scale_parameters=None,
        )


@dataclass(frozen=True, slots=True)
class EgoHandConfig:
    """Released weights and SOMA files, downloaded at pinned revisions unless local paths override them."""

    weights: Path | None = None
    """EgoHand safetensors path; None downloads it."""
    soma_assets: Path | None = None
    """Directory containing SOMAHand.npz; None downloads it."""
    batch_size: int = 2
    """Maximum simultaneous hand crops."""
    padding: float = 0.2
    """Added fraction of the square detector crop side."""
    gpu_crops: bool = False
    """Sample calibrated crops on the GPU with the reference lens model."""
    backbone_onnx: Path | None = None
    """Optional bf16 DINOv3 artifact; export from these weights when absent."""
    backbone_backend: TensorRtBackendConfig = field(default_factory=lambda: TensorRtBackendConfig(max_batch_size=32, opt_batch_size=32))
    """Cached TensorRT runtime for backbone_onnx; ignored for the eager path."""
    hf_egohand_repo_id: str = "pablovela5620/egohand"
    """Hugging Face repo holding the safetensors."""
    hf_egohand_revision: str = EGOHAND_REVISION
    """Pinned commit of that repo."""
    hf_soma_repo_id: str = "nvidia/SOMA-X"
    """Hugging Face repo holding the SOMA hand model."""
    hf_soma_revision: str = SOMA_REVISION
    """Pinned commit of that repo."""

    def weights_path(self) -> Path:
        """The safetensors file: ``weights``, or the pinned download in the Hugging Face cache."""
        return egohand_weights_path(self.weights, self.hf_egohand_repo_id, self.hf_egohand_revision)

    def soma_path(self) -> Path:
        """The SOMA directory: ``soma_assets``, or the pinned download in the Hugging Face cache."""
        return soma_assets_path(self.soma_assets, self.hf_soma_repo_id, self.hf_soma_revision)

    def setup(self) -> "EgoHandPredictor":
        """Load the weights and construct the calibrated predictor."""
        return EgoHandPredictor(self)


def egohand_weights_path(weights: Path | None, repo_id: str, revision: str) -> Path:
    """``weights``, or the pinned EgoHand safetensors download in the Hugging Face cache."""
    if weights is not None:
        return weights
    return Path(hf_hub_download(repo_id=repo_id, filename="egohand_s060000.safetensors", revision=revision))


def soma_assets_path(soma_assets: Path | None, repo_id: str, revision: str) -> Path:
    """``soma_assets``, or the pinned download of the SOMA hand files in the Hugging Face cache."""
    if soma_assets is not None:
        return soma_assets
    return Path(snapshot_download(repo_id=repo_id, revision=revision, allow_patterns=list(SOMA_HAND_FILES)))


def export_backbone_once(weights: Path, soma: Path, onnx: Path, *, batch_size: int, max_batch_size: int) -> None:
    """Export the released bf16 DINOv3 to ``onnx`` with a dynamic batch up to ``max_batch_size``, unless it exists."""

    def export() -> None:
        from egohand_inference.pipeline.predictor import EgoHandPredictor as UpstreamPredictor
        from egohand_inference.pipeline.predictor import load_egohand_model

        model, _ = load_egohand_model(weights, soma)
        export_onnx(
            UpstreamPredictor(model, batch_size, device="cuda", dtype=torch.bfloat16, use_perspective_cropping=True).model.backbone,
            (torch.zeros((2, 3, 256, 256), device="cuda"),),
            onnx,
            input_names=["images"],
            output_names=["features"],
            compute_dtype=torch.bfloat16,
            dynamic_batch_max=max_batch_size,
        )
        del model
        torch.cuda.empty_cache()

    build_once(onnx, export)


def soma_faces(model: EgoHand) -> dict[HandSide, Int64[Tensor, "f 3"]]:
    """Each side's SOMA hand triangles on CUDA, from a released EgoHand model."""
    return {side: torch.from_numpy(np.asarray(model.soma_hand.faces(side), dtype=np.int64)).cuda() for side in ("left", "right")}


def require_camera(camera: CalibratedCamera | None) -> CalibratedCamera:
    """The frames' camera, which EgoHand's perspective crops need to undo the lens.

    Raises:
        ValueError: If ``camera`` is missing.
    """
    if camera is None:
        raise ValueError("EgoHand needs the CalibratedCamera of the frames: its perspective crops undo the lens.")
    return camera


class _ModuleTimer:
    """CUDA-event pairs around every forward of one module, recorded only while enabled."""

    def __init__(self, module: torch.nn.Module) -> None:
        self.enabled: bool = False
        self.events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        module.register_forward_pre_hook(self._before)
        module.register_forward_hook(self._after)

    def _before(self, *_: object) -> None:
        if self.enabled:
            start = torch.cuda.Event(enable_timing=True)
            start.record()
            self.events.append((start, torch.cuda.Event(enable_timing=True)))

    def _after(self, *_: object) -> None:
        if self.enabled:
            self.events[-1][1].record()

    def elapsed_ms(self) -> float:
        """GPU time of the recorded forwards; their work must have completed."""
        return sum(start.elapsed_time(end) for start, end in self.events)


class EgoHandPredictor(TopDownHandPose3d):
    """RGB frames + anatomical boxes + calibrated camera -> metric SOMA hands."""

    skeleton: KeypointSkeleton = SOMA_HAND

    def __init__(self, config: EgoHandConfig) -> None:
        from egohand_inference.pipeline.predictor import EgoHandPredictor as UpstreamPredictor
        from egohand_inference.pipeline.predictor import load_egohand_model

        self.config: EgoHandConfig = config
        self.profile_stages: bool = False
        """Record per-frame stage times in ``stage_ms``; off, inference adds no synchronization."""
        self.stage_ms: dict[EgoHandStage, list[float]] = {}
        # A cached backbone artifact replaces the 3 GB PyTorch DINOv3, which is then never built or loaded.
        engine: TensorRtBackbone | None = None
        weights: Path = config.weights_path()
        soma: Path = config.soma_path()
        if config.backbone_onnx is not None:
            export_backbone_once(weights, soma, config.backbone_onnx, batch_size=config.batch_size, max_batch_size=config.backbone_backend.max_batch_size)
            engine = TensorRtBackbone(create_runtime_from_onnx(config.backbone_onnx, config.backbone_backend))
        model, _ = load_egohand_model(weights, soma, backbone=engine)
        self.predictor: UpstreamPredictor = UpstreamPredictor(
            model, config.batch_size, device="cuda", dtype=torch.bfloat16, use_perspective_cropping=True, perf_reporter=self
        )
        self.faces: dict[HandSide, Tensor] = soma_faces(model)
        self.module_timers: dict[EgoHandStage, _ModuleTimer] = {"backbone": _ModuleTimer(model.backbone), "head": _ModuleTimer(model.hand_head)}

    def record_latency(self, stage: PerfStage, latency_ms: float) -> None:
        """Collect upstream's per-frame stage timings while profiling; SOMA uses nested CUDA-event timing."""
        if stage in UPSTREAM_STAGES:
            self.stage_ms.setdefault(UPSTREAM_STAGES[stage], []).append(latency_ms)

    def __call__(
        self, frames_rgb: UInt8[Tensor, "b h w 3"], detections: HandDetections, camera: CalibratedCamera | None = None
    ) -> EgoHandReconstruction:
        """Reconstruct each valid detected hand in its source camera frame.

        Raises:
            ValueError: If ``camera`` is missing; EgoHand's perspective crops undo the lens.
        """
        from posekit.models.egohand_geometry import lens_model, project

        camera = require_camera(camera)

        validate_frames_rgb(frames_rgb)
        ids: list[str] = [str(index) for index in range(len(frames_rgb))]
        boxes: dict[str, dict[HandSide, ndarray]] = {key: {} for key in ids}
        for box, index, side in zip(detections.boxes.xyxy_numpy(), detections.boxes.frame_indices.tolist(), detections.sides, strict=True):
            center: Float32[ndarray, "2"] = (box[:2] + box[2:]) / 2.0
            size: float = float(np.max(box[2:] - box[:2])) * (1.0 + self.config.padding)
            boxes[str(index)][side] = np.array([*center, size], dtype=np.float32)
        prepared: list[dict] | None = None
        frames: list[UInt8[ndarray, "h w 3"]] = []
        profile: bool = self.profile_stages
        # Upstream synchronizes around its timed stages only while it has a reporter.
        self.predictor.perf_reporter = self if profile else None
        if self.config.gpu_crops:
            from posekit.models.egohand_crops import prepare_gpu_crops

            started: float = time.perf_counter()
            prepared = prepare_gpu_crops(frames_rgb, boxes, camera)
            if profile:
                torch.cuda.synchronize()
                self.record_latency(PerfStage.PREPROCESSING, (time.perf_counter() - started) * 1000.0 / len(frames_rgb))
        else:
            frames = [np.ascontiguousarray(frame.cpu().numpy()[:, :, ::-1]) for frame in frames_rgb]
        soma_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] | None = [] if profile else None
        self.predictor.model.soma_hand.profile_events = soma_events
        for timer in self.module_timers.values():
            timer.events.clear()
            timer.enabled = profile
        raw: dict[str, dict[str, dict]] = self.predictor.run(
            frames, ids, boxes, ("left", "right"), intrinsics_per_image={key: lens_model(camera) for key in ids}, prepared_crops=prepared
        )

        if soma_events is not None:
            self.stage_ms.setdefault("soma", []).append(sum(start.elapsed_time(end) for start, end in soma_events) / len(frames_rgb))
            for stage, timer in self.module_timers.items():
                self.stage_ms.setdefault(stage, []).append(timer.elapsed_ms() / len(frames_rgb))
        rows: list[_UpstreamHand] = []
        sides: list[HandSide] = []
        indices: list[int] = []
        confidence: list[float] = []
        for row, (index, side) in enumerate(zip(detections.boxes.frame_indices.tolist(), detections.sides, strict=True)):
            result: dict | None = raw[side].get(str(index))
            if result is not None:
                rows.append(from_dict(_UpstreamHand, result))
                sides.append(side)
                indices.append(index)
                confidence.append(float(detections.boxes.scores[row]))
        device: torch.device = frames_rgb.device
        if not rows:
            return EgoHandReconstruction.empty(device, self.faces)
        xyz: Float32[ndarray, "n 25 3"] = np.stack([row.pred_keypoints_3d + row.transl for row in rows]).astype(np.float32)
        xy: Float32[ndarray, "n 25 2"] = np.stack([project(camera, points.astype(np.float64)) for points in xyz]).astype(np.float32)
        xyz_tensor: Float32[Tensor, "n 25 3"] = torch.from_numpy(xyz).to(device)
        return EgoHandReconstruction(
            keypoints=Keypoints3d(
                torch.from_numpy(xy).to(device),
                xyz_tensor[:, :, 2] - xyz_tensor[:, :1, 2],
                torch.tensor(confidence, dtype=torch.float32, device=device)[:, None].expand(-1, 25).contiguous(),
                torch.tensor(indices, dtype=torch.int64, device=device),
                SOMA_HAND,
                (0,),
                xyz_tensor,
            ),
            sides=tuple(sides),
            vertices=torch.from_numpy(np.stack([row.pred_vertices + row.transl for row in rows]).astype(np.float32)).to(device),
            faces=self.faces,
            rotations=torch.from_numpy(np.stack([row.poses for row in rows])).to(device),
            identity=torch.from_numpy(np.stack([row.identity_coeffs for row in rows])).to(device),
            scale=torch.tensor([row.global_scale for row in rows], dtype=torch.float32, device=device),
            translation=torch.from_numpy(np.stack([row.transl for row in rows])).to(device),
            scale_parameters=torch.from_numpy(np.stack([row.scale_params for row in rows if row.scale_params is not None])).to(device)
            if rows[0].scale_params is not None
            else None,
        )


class TensorRtBackbone(torch.nn.Module):
    """DINOv3 feature-map contract over a cached dynamic-batch runtime."""

    def __init__(self, runtime: TensorRuntime) -> None:
        super().__init__()
        self.runtime: TensorRuntime = runtime

    def forward(self, images: Float32[Tensor, "b 3 256 256"]) -> Float32[Tensor, "b 1280 16 16"]:
        """Return the final spatial token map, without class/storage tokens."""
        return run_chunked(self.runtime, {"images": images.float()})["features"]


@dataclass(frozen=True, slots=True)
class EgoHandAssetPaths:
    """Where an EgoHand stack's files live: a local asset tree, or the pinned downloads and the trtkit cache."""

    detector: Path | None
    """hand_detector.onnx; None downloads it."""
    weights: Path | None
    """EgoHand safetensors; None downloads it."""
    soma: Path | None
    """SOMA hand model directory; None downloads it."""
    bf16_backbone: Path
    """bf16 DINOv3 export."""
    engines: Path
    """TensorRT engine and derived-artifact cache."""


def asset_paths(assets: Path | None) -> EgoHandAssetPaths:
    """Paths inside an asset tree (``weights/...``, ``dinov3-bf16.onnx``, ``engines/``); ``None`` takes the pinned downloads and the trtkit cache."""
    if assets is None:
        return EgoHandAssetPaths(None, None, None, CACHED_BACKBONE_ONNX, DEFAULT_TRT_CACHE_DIR)
    return EgoHandAssetPaths(
        assets / "weights/detector/hand_detector.onnx",
        assets / "weights/egohand/egohand_s060000.safetensors",
        assets / "weights/soma",
        assets / "dinov3-bf16.onnx",
        assets / "engines",
    )


def selfcap_egohand_configs(assets: Path | None) -> tuple[EgoHandDetectorConfig, EgoHandConfig]:
    """SelfCap's accelerated EgoHand: TRT detector, GPU crops, TRT bf16 backbone, 16 crops per batch.

    Args:
        assets: Local asset tree (see :func:`asset_paths`); ``None`` takes the pinned downloads and the trtkit engine cache.

    Returns:
        The detector and predictor configurations.
    """
    paths: EgoHandAssetPaths = asset_paths(assets)
    detector = EgoHandDetectorConfig(paths.detector, backend=TensorRtBackendConfig(max_batch_size=8, opt_batch_size=4, cache_dir=paths.engines))
    predictor = EgoHandConfig(
        paths.weights,
        paths.soma,
        batch_size=16,
        gpu_crops=True,
        backbone_onnx=paths.bf16_backbone,
        backbone_backend=TensorRtBackendConfig(max_batch_size=32, opt_batch_size=32, cache_dir=paths.engines),
    )
    return detector, predictor
