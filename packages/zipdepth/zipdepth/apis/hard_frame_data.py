"""Hard-frame manifests, archives, previews, and the batched hole-aware prompt upsampler."""

from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float32, UInt8
from numpy import ndarray
from serde import field as serde_field
from serde import serde
from serde.json import from_json, to_json
from torch import Tensor

ERROR_MAX_MM: float = 250.0
"""Upper bound of every hard-frame preview error heatmap."""

DepthMap: TypeAlias = Float32[ndarray, "h w"]
ImageRGB: TypeAlias = UInt8[ndarray, "h w 3"]


@serde
@dataclass(frozen=True, slots=True)
class FrameMetrics:
    """Teacher-relative errors retained for one mined capture frame."""

    frame_index: int
    """Zero-based frame index in capture iteration order."""
    student_overall_dev_m: float
    """Student-teacher mean absolute deviation over valid pixels, in metres."""
    student_edge_dev_m: float
    """Student-teacher mean absolute deviation on teacher edges, in metres."""
    student_flat_dev_m: float
    """Student-teacher mean absolute deviation off teacher edges, in metres."""
    baseline_overall_dev_m: float
    """Baseline-teacher mean absolute deviation over valid pixels, in metres."""
    baseline_edge_dev_m: float
    """Baseline-teacher mean absolute deviation on teacher edges, in metres."""
    baseline_flat_dev_m: float
    """Baseline-teacher mean absolute deviation off teacher edges, in metres."""


@serde
@dataclass(frozen=True, slots=True)
class RankedFrameRecord:
    """One full-ranking entry."""

    rank: int
    """One-based rank by descending student edge deviation."""
    metrics: FrameMetrics = serde_field(flatten=True)
    """Teacher-relative metrics flattened into this record."""


@serde
@dataclass(frozen=True, slots=True)
class KeptFrameRecord:
    """One written hard frame and its artifact paths."""

    rank: int
    """One-based rank by descending student edge deviation."""
    frame_path: str
    """Frame NPZ path relative to the output directory."""
    preview_path: str
    """Preview PNG path relative to the output directory."""
    metrics: FrameMetrics = serde_field(flatten=True)
    """Teacher-relative metrics flattened into this record."""


@serde
@dataclass(frozen=True, slots=True)
class RunMetadata:
    """Provenance of one hard-frame mining run."""

    capture_path: str
    """Polycam capture the frames came from."""
    checkpoint_path: str | None
    """Student checkpoint, when the student is ZipDepth-PromptDA."""
    edge_quantile: float
    """Per-frame teacher-gradient quantile defining edges."""
    capture_hw: tuple[int, int]
    """Capture resolution as height and width."""
    teacher_config_class: str
    """Completion config class that produced the labels."""
    student_config_class: str
    """Completion config class whose deviations ranked frames."""
    student_reference_version: str
    """Human label for the stored student checkpoint generation."""
    student_output_role: str
    """How downstream evaluation should use stored student depth."""
    eval_label_field: str
    """NPZ field holding the evaluation label."""
    batch_size: int
    """Frames per model batch."""
    max_frames: int | None
    """Frame cap applied to the capture, if any."""
    max_keep: int
    """Maximum number of highest-error frames written."""
    processed_frames: int
    """Frames processed by the miner."""


@serde
@dataclass(frozen=True, slots=True)
class HardFramesReport:
    """Typed hard-frame ``metrics.json`` document."""

    run: RunMetadata
    """Provenance of this mining run."""
    kept_frames: list[KeptFrameRecord]
    """Written frames in rank order."""
    full_ranking: list[RankedFrameRecord]
    """Every processed frame in rank order."""


@dataclass(frozen=True, slots=True)
class HardFrameArchive:
    """Arrays stored for one selected hard frame."""

    frame_index: int
    """Zero-based source capture frame index."""
    rgb_hwc: ImageRGB
    """Capture-resolution uint8 RGB image."""
    prompt_hw: DepthMap
    """Raw 192x256 metric prompt; zero denotes invalid."""
    teacher_hw: DepthMap
    """Frozen capture-resolution teacher depth in metres."""
    student_hw: DepthMap
    """Frozen capture-resolution reference-student depth in metres."""


def prompt_upsample_depth(
    prompt_depth_bhw: Float32[Tensor, "b ph pw"],
    prompt_valid_bhw: Bool[Tensor, "b ph pw"],
    *,
    height: int,
    width: int,
) -> Float32[Tensor, "b h w"]:
    """Bilinearly upsample batched LiDAR prompts without bleeding holes.

    Args:
        prompt_depth_bhw: Float32 metric prompts with shape ``(B, PH, PW)``.
        prompt_valid_bhw: Boolean prompt validity with shape ``(B, PH, PW)``.
        height: Output height.
        width: Output width.

    Returns:
        Dense Float32 metric depth with shape ``(B, height, width)``.
    """
    if prompt_depth_bhw.ndim != 3 or prompt_valid_bhw.shape != prompt_depth_bhw.shape:
        raise ValueError("prompt depth and validity must share one BxHxW shape.")
    valid_weight_bhw: Float32[Tensor, "b ph pw"] = prompt_valid_bhw.to(dtype=torch.float32)
    weighted_bhw: Float32[Tensor, "b ph pw"] = prompt_depth_bhw * valid_weight_bhw
    weighted_up_bhw: Float32[Tensor, "b h w"] = F.interpolate(
        weighted_bhw[:, None], size=(height, width), mode="bilinear", align_corners=False
    )[:, 0]
    weight_up_bhw: Float32[Tensor, "b h w"] = F.interpolate(
        valid_weight_bhw[:, None], size=(height, width), mode="bilinear", align_corners=False
    )[:, 0]
    fallback_values: list[Float32[Tensor, ""]] = [
        torch.quantile(depth_hw[valid_hw], 0.5) if bool(valid_hw.any()) else torch.tensor(1.0, device=depth_hw.device, dtype=depth_hw.dtype)
        for depth_hw, valid_hw in zip(prompt_depth_bhw, prompt_valid_bhw, strict=True)
    ]
    fallback_b: Float32[Tensor, "b 1 1"] = torch.stack(fallback_values)[:, None, None]
    supported_bhw: Bool[Tensor, "b h w"] = weight_up_bhw > 1.0e-3
    return torch.where(supported_bhw, weighted_up_bhw / weight_up_bhw.clamp_min(1.0e-3), fallback_b)


def read_hard_frames_report(path: Path) -> HardFramesReport:
    """Deserialize one typed hard-frame manifest."""
    if not path.is_file():
        raise FileNotFoundError(f"hard-frame metrics file is missing: {path}")
    return from_json(HardFramesReport, path.read_text(encoding="utf-8"))


def write_hard_frames_report(path: Path, report: HardFramesReport) -> None:
    """Serialize one typed hard-frame manifest."""
    path.write_text(to_json(report) + "\n", encoding="utf-8")


def read_hard_frame_archive(path: Path) -> HardFrameArchive:
    """Load and validate one hard-frame NPZ archive."""
    if not path.is_file():
        raise FileNotFoundError(f"hard-frame archive is missing: {path}")
    with np.load(path) as archive:
        required_keys: set[str] = {"rgb", "prompt", "teacher", "student", "frame_index"}
        if not required_keys.issubset(archive.files):
            raise ValueError(f"{path} is missing keys {sorted(required_keys - set(archive.files))}")
        frame = HardFrameArchive(
            frame_index=int(archive["frame_index"].item()),
            rgb_hwc=np.ascontiguousarray(archive["rgb"].astype(np.uint8, copy=False)),
            prompt_hw=np.ascontiguousarray(archive["prompt"].astype(np.float32, copy=False)),
            teacher_hw=np.ascontiguousarray(archive["teacher"].astype(np.float32, copy=False)),
            student_hw=np.ascontiguousarray(archive["student"].astype(np.float32, copy=False)),
        )
    if frame.rgb_hwc.ndim != 3 or frame.rgb_hwc.shape[-1] != 3:
        raise ValueError(f"{path}: rgb must have shape HxWx3")
    if frame.prompt_hw.shape != (192, 256):
        raise ValueError(f"{path}: prompt must have shape 192x256")
    if frame.teacher_hw.shape != frame.rgb_hwc.shape[:2] or frame.student_hw.shape != frame.teacher_hw.shape:
        raise ValueError(f"{path}: teacher and stored student must match RGB spatial shape")
    return frame


def write_hard_frame_archive(path: Path, frame: HardFrameArchive) -> None:
    """Write one hard-frame NPZ archive."""
    np.savez_compressed(
        path,
        rgb=frame.rgb_hwc,
        prompt=frame.prompt_hw,
        teacher=frame.teacher_hw,
        student=frame.student_hw,
        frame_index=frame.frame_index,
    )


def write_hard_frame_preview(
    path: Path,
    rgb_hwc: ImageRGB,
    teacher_hw: DepthMap,
    student_hw: DepthMap,
) -> None:
    """Write RGB, Turbo depths, and fixed 0--250 mm Turbo error as one strip."""
    min_depth_m: float = float(min(teacher_hw.min(), student_hw.min()))
    max_depth_m: float = float(max(teacher_hw.max(), student_hw.max()))
    depth_span_m: float = max(max_depth_m - min_depth_m, float(np.finfo(np.float32).eps))
    teacher_scaled_hw: UInt8[ndarray, "h w"] = np.clip((teacher_hw - min_depth_m) * (255.0 / depth_span_m), 0.0, 255.0).astype(np.uint8)
    student_scaled_hw: UInt8[ndarray, "h w"] = np.clip((student_hw - min_depth_m) * (255.0 / depth_span_m), 0.0, 255.0).astype(np.uint8)
    error_mm_hw: DepthMap = np.abs(student_hw - teacher_hw) * 1000.0
    error_scaled_hw: UInt8[ndarray, "h w"] = np.clip(error_mm_hw * (255.0 / ERROR_MAX_MM), 0.0, 255.0).astype(np.uint8)
    preview_bgr_hwc: UInt8[ndarray, "h four_w 3"] = np.concatenate(
        (
            cv2.cvtColor(rgb_hwc, cv2.COLOR_RGB2BGR),
            cv2.applyColorMap(teacher_scaled_hw, cv2.COLORMAP_TURBO),
            cv2.applyColorMap(student_scaled_hw, cv2.COLORMAP_TURBO),
            cv2.applyColorMap(error_scaled_hw, cv2.COLORMAP_TURBO),
        ),
        axis=1,
    )
    if not cv2.imwrite(str(path), preview_bgr_hwc):
        raise OSError(f"failed to write preview: {path}")
