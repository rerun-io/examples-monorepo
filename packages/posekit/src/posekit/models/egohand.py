"""Calibrated, top-down SOMA hand reconstruction from NVIDIA EgoHand.

The first port retains upstream CPU perspective cropping and mirroring for
parity. Model input/output follows posekit's GPU tensor contracts; the CPU
boundary is explicit and included in the stage timings.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from egohand_inference.hand_detector_2d.rtdetr.tracking import select_new
from egohand_inference.perf import PerfStage
from jaxtyping import Bool, Float32, Int64, UInt8
from numpy import ndarray
from PIL import Image
from serde import from_dict, serde
from torch import Tensor

from posekit.models.egohand_geometry import HandSide, OAKCamera, detector_hand_side
from posekit.predictions import BoxDetections, Keypoints3d, validate_frames_rgb
from posekit.runtimes import OnnxCudaRuntime
from posekit.skeletons import SOMA_HAND


@dataclass(frozen=True, slots=True)
class HandDetections:
    """Boxes paired with anatomical handedness; row ordering is shared."""

    boxes: BoxDetections
    """Source-image boxes and batch indices."""
    sides: tuple[HandSide, ...]
    """Assigned side, or raw class name when detector identity assignment is disabled."""


def resolve_hand_identities(detections: HandDetections, frame_count: int) -> HandDetections:
    """Assign an upright ego sequence before deciding which crops to mirror.

    This reuses upstream's candidate deduplication and IoU identity assignment.
    Two-hand bootstrap uses image geometry; later frames use track continuity,
    including crossings. Detector class IDs are only a lone-bootstrap fallback.
    The original selected boxes and scores are retained: no padding or smoothing.
    """
    boxes: Float32[ndarray, "n 4"] = detections.boxes.xyxy_numpy()
    frames: list[int] = detections.boxes.frame_indices.tolist()
    scores: list[float] = detections.boxes.scores.tolist()
    records: list[dict[str, list[list[float]]]] = [{"L": [], "R": []} for _ in range(frame_count)]
    for box, frame, score, side in zip(boxes, frames, scores, detections.sides, strict=True):
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
        candidates: list[int] = [i for i, value in enumerate(frames) if value == frame]
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


@dataclass(frozen=True, slots=True)
class EgoHandDetectorConfig:
    """Released ONNX hand detector configuration."""

    weights: Path
    """Path to hand_detector.onnx."""
    threshold: float = 0.5
    """Minimum detector confidence, unrelated to landmark visibility."""

    assign_identity: bool = True
    """Use upstream geometry/IoU assignment on an ordered, upright ego sequence."""

    def setup(self) -> "EgoHandDetector":
        """Create the CUDA detector."""
        return EgoHandDetector(self)


class EgoHandDetector:
    """CUDA detector with upstream sequence-level anatomical identity assignment."""

    def __init__(self, config: EgoHandDetectorConfig) -> None:
        self.config: EgoHandDetectorConfig = config
        self.runtime: OnnxCudaRuntime = OnnxCudaRuntime(config.weights, max_batch_size=1)

    def __call__(self, frames_rgb: UInt8[Tensor, "b h w 3"]) -> HandDetections:
        """Detect an ordered RGB sequence; resolve identity before hand mirroring."""
        validate_frames_rgb(frames_rgb)
        boxes: list[Tensor] = []
        scores: list[Tensor] = []
        indices: list[int] = []
        sides: list[HandSide] = []
        height: int = int(frames_rgb.shape[1])
        width: int = int(frames_rgb.shape[2])
        for frame_index, frame in enumerate(frames_rgb):
            # Match the released detector's Pillow resampling exactly.
            image: Image.Image = Image.fromarray(frame.cpu().numpy()).resize((640, 640), Image.Resampling.BILINEAR)
            pixels: Float32[Tensor, "1 3 640 640"] = (
                torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).permute(2, 0, 1).unsqueeze(0).to(frames_rgb.device) / 255.0
            )
            outputs: dict[str, Tensor] = self.runtime(
                {"images": pixels.contiguous(), "orig_target_sizes": torch.tensor([[width, height]], device=frames_rgb.device, dtype=torch.int64)}
            )
            for label in (1, 2):
                valid: Bool[Tensor, "q"] = (outputs["labels"][0] == label) & (outputs["scores"][0] > self.config.threshold)
                candidates: Int64[Tensor, "n"] = torch.where(valid)[0]
                if not len(candidates):
                    continue
                # Keep candidates across class IDs for upstream's identity assignment.
                # Top-1 per class can discard a hand when both are misclassified alike.
                for candidate in candidates:
                    boxes.append(outputs["boxes"][0, candidate].clone().float())
                    scores.append(outputs["scores"][0, candidate].clone().float())
                    indices.append(frame_index)
                    sides.append(detector_hand_side(label))
        if not boxes:
            return HandDetections(BoxDetections.empty(frames_rgb.device), ())
        raw: HandDetections = HandDetections(
            BoxDetections(torch.stack(boxes), torch.stack(scores), torch.tensor(indices, device=frames_rgb.device, dtype=torch.int64)), tuple(sides)
        )
        return resolve_hand_identities(raw, len(frames_rgb)) if self.config.assign_identity else raw


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
    pred_projection_valid: Bool[ndarray, "25"]
    """Geometric projection validity, not visibility."""


@dataclass(frozen=True, slots=True)
class HandReconstruction:
    """Row-aligned metric hand observations and parametric outputs."""

    keypoints: Keypoints3d
    """Camera-space joints and distorted source-image projection."""
    sides: tuple[HandSide, ...]
    """Anatomical handedness per result row."""
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
    vertices: tuple[Float32[Tensor, "v 3"], ...]
    """Camera-space mesh vertices in metres."""


@dataclass(frozen=True, slots=True)
class EgoHandConfig:
    """Local released assets; construction never downloads weights."""

    weights: Path
    """EgoHand checkpoint path."""
    soma_assets: Path
    """Directory containing SOMAHand.npz."""
    batch_size: int = 2
    """Maximum simultaneous hand crops."""
    device: str = "cuda"
    """Inference device."""
    padding: float = 0.2
    """Added fraction of the square detector crop side."""

    def setup(self) -> "EgoHandPredictor":
        """Load local weights and construct the calibrated predictor."""
        return EgoHandPredictor(self)


class EgoHandPredictor:
    """RGB frames + anatomical boxes + calibrated camera -> metric hands."""

    def __init__(self, config: EgoHandConfig) -> None:
        from egohand_inference.pipeline.predictor import EgoHandPredictor as UpstreamPredictor

        self.config: EgoHandConfig = config
        self.stage_ms: dict[str, list[float]] = {}
        self.predictor: UpstreamPredictor = UpstreamPredictor.from_checkpoint(
            config.weights, config.soma_assets, config.batch_size, device=config.device, dtype=torch.bfloat16, use_perspective_cropping=True
        )
        self.predictor.perf_reporter = self

    def record_latency(self, stage: PerfStage, latency_ms: float) -> None:
        """Collect upstream CUDA-synchronized stage timings."""
        self.stage_ms.setdefault(stage.value, []).append(latency_ms)

    def __call__(self, frames_rgb: UInt8[Tensor, "b h w 3"], detections: HandDetections, camera: OAKCamera) -> HandReconstruction:
        """Reconstruct each valid detected hand in its source camera frame."""
        validate_frames_rgb(frames_rgb)
        ids: list[str] = [str(index) for index in range(len(frames_rgb))]
        boxes: dict[str, dict[HandSide, ndarray]] = {key: {} for key in ids}
        for box, index, side in zip(detections.boxes.xyxy_numpy(), detections.boxes.frame_indices.tolist(), detections.sides, strict=True):
            center: Float32[ndarray, "2"] = (box[:2] + box[2:]) / 2.0
            size: float = float(np.max(box[2:] - box[:2])) * (1.0 + self.config.padding)
            boxes[str(index)][side] = np.array([*center, size], dtype=np.float32)
        frames: list[UInt8[ndarray, "h w 3"]] = [np.ascontiguousarray(frame.cpu().numpy()[:, :, ::-1]) for frame in frames_rgb]
        self.predictor.model.soma_hand.profile_ms = []
        raw: dict[str, dict[str, dict]] = self.predictor.run(
            frames, ids, boxes, ("left", "right"), intrinsics_per_image={key: camera.model for key in ids}
        )
        self.stage_ms.setdefault("soma", []).append(sum(self.predictor.model.soma_hand.profile_ms))
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
            return HandReconstruction(
                Keypoints3d(
                    torch.empty((0, 25, 2), dtype=torch.float32, device=device),
                    torch.empty((0, 25), dtype=torch.float32, device=device),
                    torch.empty((0, 25), dtype=torch.float32, device=device),
                    torch.empty((0,), dtype=torch.int64, device=device),
                    SOMA_HAND,
                    (0,),
                    torch.empty((0, 25, 3), dtype=torch.float32, device=device),
                ),
                (),
                torch.empty((0, 25, 3, 3), device=device),
                torch.empty((0, 0), device=device),
                torch.empty((0,), device=device),
                torch.empty((0, 3), device=device),
                None,
                (),
            )
        xyz: Float32[ndarray, "n 25 3"] = np.stack([row.pred_keypoints_3d + row.transl for row in rows]).astype(np.float32)
        xy: Float32[ndarray, "n 25 2"] = np.stack([camera.project(points.astype(np.float64)) for points in xyz]).astype(np.float32)
        xyz_tensor: Float32[Tensor, "n 25 3"] = torch.from_numpy(xyz).to(device)
        return HandReconstruction(
            Keypoints3d(
                torch.from_numpy(xy).to(device),
                xyz_tensor[:, :, 2] - xyz_tensor[:, :1, 2],
                torch.tensor(confidence, dtype=torch.float32, device=device)[:, None].expand(-1, 25).contiguous(),
                torch.tensor(indices, dtype=torch.int64, device=device),
                SOMA_HAND,
                (0,),
                xyz_tensor,
            ),
            tuple(sides),
            torch.from_numpy(np.stack([row.poses for row in rows])).to(device),
            torch.from_numpy(np.stack([row.identity_coeffs for row in rows])).to(device),
            torch.tensor([row.global_scale for row in rows], dtype=torch.float32, device=device),
            torch.from_numpy(np.stack([row.transl for row in rows])).to(device),
            torch.from_numpy(np.stack([row.scale_params for row in rows if row.scale_params is not None])).to(device)
            if rows[0].scale_params is not None
            else None,
            tuple(torch.from_numpy(row.pred_vertices + row.transl).to(device) for row in rows),
        )
