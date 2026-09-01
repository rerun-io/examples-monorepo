"""Mine Polycam frames where ZipDepth-PromptDA deviates most at depth edges."""

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float32, UInt8
from monopriors.models.depth_completion import AnnotatedCompletionConfig, PromptDAConfig, ZipDepthPromptConfig
from monopriors.models.depth_completion.base_completion_depth import BaseCompletionPredictor
from monopriors.models.depth_completion.prompt_da import DEFAULT_PROMPTDA_CACHE_DIR
from numpy import ndarray
from serde import field as serde_field
from serde import serde, to_dict
from simplecv.data.polycam import PolycamData, PolycamDataset, load_polycam_data
from torch import Tensor
from trtkit import TensorRtBackendConfig, TorchBackendConfig

from rerun_prompt_da.polycam_batching import PolycamBatchPlan, PolycamTensorBatch, prepare_polycam_batches, stack_polycam_batch

ERROR_MAX_MM: float = 250.0
"""Upper bound of every preview error heatmap."""


@dataclass
class PolycamHardFramesConfig:
    """Runtime configuration for Polycam hard-frame mining."""

    polycam_zip_path: Path
    """Polycam capture zip (or extracted directory) to process."""
    teacher: AnnotatedCompletionConfig = field(
        default_factory=lambda: PromptDAConfig(
            backend=TensorRtBackendConfig(max_batch_size=8, opt_batch_size=8, cache_dir=DEFAULT_PROMPTDA_CACHE_DIR / "trt")
        )
    )
    """Teacher completion model and backend."""
    student: AnnotatedCompletionConfig = field(
        default_factory=lambda: ZipDepthPromptConfig(checkpoint=Path(), backend=TorchBackendConfig())
    )
    """Student completion model and backend."""
    zipdepth_checkpoint: Path | None = None
    """Compatibility alias for ``student=zipdepth-promptda --checkpoint``."""
    out_dir: Path = Path("data/polycam-hard-frames")
    """Directory receiving the standalone hard-frame eval set."""
    max_keep: int = 20
    """Maximum number of highest-error frames to write."""
    batch_size: int = 8
    """Frames per model batch."""
    edge_quantile: float = 0.90
    """Per-frame teacher-gradient quantile defining the edge region."""
    max_frames: int | None = None
    """Optional cap on processed frames."""


@serde
@dataclass(frozen=True, slots=True)
class FrameMetrics:
    """Teacher-relative errors for one capture frame."""

    frame_index: int
    """Zero-based frame index in capture iteration order."""
    student_overall_dev_m: float
    """Student-teacher mean absolute deviation over the full frame, in metres."""
    student_edge_dev_m: float
    """Student-teacher mean absolute deviation in the teacher edge region, in metres."""
    student_flat_dev_m: float
    """Student-teacher mean absolute deviation outside the teacher edge region, in metres."""
    baseline_overall_dev_m: float
    """Bilinear-teacher mean absolute deviation over the full frame, in metres."""
    baseline_edge_dev_m: float
    """Bilinear-teacher mean absolute deviation in the teacher edge region, in metres."""
    baseline_flat_dev_m: float
    """Bilinear-teacher mean absolute deviation outside the teacher edge region, in metres."""
    baseline_to_student_overall_ratio: float
    """Bilinear deviation divided by student deviation over the full frame."""
    baseline_to_student_edge_ratio: float
    """Bilinear deviation divided by student deviation in the teacher edge region."""
    baseline_to_student_flat_ratio: float
    """Bilinear deviation divided by student deviation outside the teacher edge region."""


@dataclass(frozen=True, slots=True)
class HardFrame:
    """Host arrays retained for one current top-N frame."""

    metrics: FrameMetrics
    """Teacher-relative scalar metrics used for ranking."""
    rgb_hw3: UInt8[ndarray, "h w 3"]
    """Capture-resolution RGB image."""
    prompt_hw: Float32[ndarray, "192 256"]
    """Raw Polycam prompt depth in metres."""
    teacher_hw: Float32[ndarray, "h w"]
    """Capture-resolution PromptDA teacher depth in metres."""
    student_hw: Float32[ndarray, "h w"]
    """Capture-resolution ZipDepth-PromptDA reference depth in metres."""


def compute_frame_metrics(
    *,
    frame_index: int,
    teacher_hw: Float32[Tensor, "h w"],
    student_hw: Float32[Tensor, "h w"],
    baseline_hw: Float32[Tensor, "h w"],
    edge_quantile: float,
) -> FrameMetrics:
    """Measure teacher-relative errors in one frame's edge and flat regions.

    Args:
        frame_index: Zero-based frame index in capture iteration order.
        teacher_hw: Float32 teacher depth in metres with shape ``(height, width)``.
        student_hw: Float32 student depth in metres with shape ``(height, width)``.
        baseline_hw: Float32 bilinear prompt depth in metres with shape ``(height, width)``.
        edge_quantile: Per-frame teacher-gradient quantile included in the edge mask.

    Returns:
        Scalar deviations and baseline-to-student ratios. Tensor reductions stay on
        the input device; only the result scalars are transferred to the host.
    """
    if teacher_hw.ndim != 2 or student_hw.shape != teacher_hw.shape or baseline_hw.shape != teacher_hw.shape:
        raise ValueError("teacher, student, and baseline depths must share one 2D shape.")
    if not 0.0 <= edge_quantile <= 1.0:
        raise ValueError("edge_quantile must be in [0, 1].")

    teacher_gradient_hw: Float32[Tensor, "h w"] = torch.zeros_like(teacher_hw)
    gradient_x: Float32[Tensor, "h w_minus_one"] = (teacher_hw[:, 1:] - teacher_hw[:, :-1]).abs()
    gradient_y: Float32[Tensor, "h_minus_one w"] = (teacher_hw[1:, :] - teacher_hw[:-1, :]).abs()
    teacher_gradient_hw[:, 1:] += gradient_x
    teacher_gradient_hw[1:, :] += gradient_y

    edge_threshold: Float32[Tensor, ""] = torch.quantile(teacher_gradient_hw, edge_quantile)
    edge_mask_hw: Bool[Tensor, "h w"] = teacher_gradient_hw >= edge_threshold
    flat_mask_hw: Bool[Tensor, "h w"] = ~edge_mask_hw
    student_abs_dev_hw: Float32[Tensor, "h w"] = (student_hw - teacher_hw).abs()
    baseline_abs_dev_hw: Float32[Tensor, "h w"] = (baseline_hw - teacher_hw).abs()

    student_overall_dev: Float32[Tensor, ""] = student_abs_dev_hw.mean()
    student_edge_dev: Float32[Tensor, ""] = student_abs_dev_hw[edge_mask_hw].mean()
    student_flat_dev: Float32[Tensor, ""] = student_abs_dev_hw[flat_mask_hw].mean()
    baseline_overall_dev: Float32[Tensor, ""] = baseline_abs_dev_hw.mean()
    baseline_edge_dev: Float32[Tensor, ""] = baseline_abs_dev_hw[edge_mask_hw].mean()
    baseline_flat_dev: Float32[Tensor, ""] = baseline_abs_dev_hw[flat_mask_hw].mean()

    return FrameMetrics(
        frame_index=frame_index,
        student_overall_dev_m=float(student_overall_dev.item()),
        student_edge_dev_m=float(student_edge_dev.item()),
        student_flat_dev_m=float(student_flat_dev.item()),
        baseline_overall_dev_m=float(baseline_overall_dev.item()),
        baseline_edge_dev_m=float(baseline_edge_dev.item()),
        baseline_flat_dev_m=float(baseline_flat_dev.item()),
        baseline_to_student_overall_ratio=float((baseline_overall_dev / student_overall_dev).item()),
        baseline_to_student_edge_ratio=float((baseline_edge_dev / student_edge_dev).item()),
        baseline_to_student_flat_ratio=float((baseline_flat_dev / student_flat_dev).item()),
    )


@serde
@dataclass(frozen=True, slots=True)
class RankedFrameRecord:
    """One full-ranking entry: a rank and its frame's flattened metrics."""

    rank: int
    """One-based rank by descending edge-region student deviation."""
    metrics: FrameMetrics = serde_field(flatten=True)
    """Teacher-relative scalar metrics, flattened into this record."""


@serde
@dataclass(frozen=True, slots=True)
class KeptFrameRecord:
    """One written hard frame: rank, artifact paths, and flattened metrics."""

    rank: int
    """One-based rank by descending edge-region student deviation."""
    frame_path: str
    """Frame npz path relative to the output directory."""
    preview_path: str
    """Preview PNG path relative to the output directory."""
    metrics: FrameMetrics = serde_field(flatten=True)
    """Teacher-relative scalar metrics, flattened into this record."""


@serde
@dataclass(frozen=True, slots=True)
class RunMetadata:
    """Provenance of one mining run."""

    capture_path: str
    """Polycam capture the frames came from."""
    checkpoint_path: str | None
    """Student checkpoint, when the student is ZipDepth-PromptDA."""
    edge_quantile: float
    """Per-frame teacher-gradient quantile defining the edge region."""
    capture_hw: tuple[int, int]
    """Capture resolution as (height, width)."""
    teacher_config_class: str
    """Completion config class that produced the eval labels."""
    student_config_class: str
    """Completion config class whose deviations ranked the frames."""
    student_reference_version: str
    """Human label for the student checkpoint generation."""
    student_output_role: str
    """How the stored student depth should be used downstream."""
    eval_label_field: str
    """npz field holding the eval label."""
    batch_size: int
    """Frames per model batch."""
    max_frames: int | None
    """Frame cap applied to the capture, if any."""
    max_keep: int
    """Maximum number of highest-error frames written."""
    processed_frames: int
    """Frames actually processed."""


@serde
@dataclass(frozen=True, slots=True)
class HardFramesReport:
    """The complete metrics.json document."""

    run: RunMetadata
    """Provenance of this mining run."""
    kept_frames: list[KeptFrameRecord]
    """Written frames, ranked."""
    full_ranking: list[RankedFrameRecord]
    """Every processed frame, ranked."""


def _colorize_depth(
    depth_hw: Float32[ndarray, "h w"],
    *,
    min_depth_m: float,
    max_depth_m: float,
) -> UInt8[ndarray, "h w 3"]:
    """Color one float32 metre-depth image over an explicit shared range."""
    depth_span_m: float = max(max_depth_m - min_depth_m, float(np.finfo(np.float32).eps))
    normalized_hw: UInt8[ndarray, "h w"] = np.clip((depth_hw - min_depth_m) * (255.0 / depth_span_m), 0.0, 255.0).astype(np.uint8)
    return cv2.applyColorMap(normalized_hw, cv2.COLORMAP_TURBO)


def _write_preview(frame: HardFrame, preview_path: Path) -> None:
    """Write RGB, shared-range depths, and fixed-range error as one BGR strip."""
    min_depth_m: float = float(min(frame.teacher_hw.min(), frame.student_hw.min()))
    max_depth_m: float = float(max(frame.teacher_hw.max(), frame.student_hw.max()))
    rgb_bgr_hw3: UInt8[ndarray, "h w 3"] = cv2.cvtColor(frame.rgb_hw3, cv2.COLOR_RGB2BGR)
    teacher_bgr_hw3: UInt8[ndarray, "h w 3"] = _colorize_depth(
        frame.teacher_hw,
        min_depth_m=min_depth_m,
        max_depth_m=max_depth_m,
    )
    student_bgr_hw3: UInt8[ndarray, "h w 3"] = _colorize_depth(
        frame.student_hw,
        min_depth_m=min_depth_m,
        max_depth_m=max_depth_m,
    )
    abs_diff_mm_hw: Float32[ndarray, "h w"] = np.abs(frame.student_hw - frame.teacher_hw) * 1000.0
    abs_diff_u8_hw: UInt8[ndarray, "h w"] = np.clip(abs_diff_mm_hw * (255.0 / ERROR_MAX_MM), 0.0, 255.0).astype(np.uint8)
    abs_diff_bgr_hw3: UInt8[ndarray, "h w 3"] = cv2.applyColorMap(abs_diff_u8_hw, cv2.COLORMAP_TURBO)
    preview_bgr_h4w3: UInt8[ndarray, "h four_w 3"] = np.concatenate(
        (rgb_bgr_hw3, teacher_bgr_hw3, student_bgr_hw3, abs_diff_bgr_hw3),
        axis=1,
    )
    if not cv2.imwrite(str(preview_path), preview_bgr_h4w3):
        raise OSError(f"Failed to write preview: {preview_path}")


def polycam_hard_frames(config: PolycamHardFramesConfig) -> None:
    """Rank one Polycam capture and write its highest edge-deviation frames."""
    if config.max_keep <= 0:
        raise ValueError("max_keep must be positive.")
    if not 0.0 <= config.edge_quantile <= 1.0:
        raise ValueError("edge_quantile must be in [0, 1].")

    dataset: PolycamDataset = load_polycam_data(polycam_zip_or_directory_path=config.polycam_zip_path)
    batch_plan: PolycamBatchPlan = prepare_polycam_batches(
        dataset,
        batch_size=config.batch_size,
        max_frames=config.max_frames,
        capture_path=config.polycam_zip_path,
        description="Hard frames",
    )
    capture_hw: tuple[int, int] = batch_plan.first_batch[0].rgb_hw3.shape[:2]

    student_config: AnnotatedCompletionConfig = config.student
    if config.zipdepth_checkpoint is not None:
        if not isinstance(student_config, ZipDepthPromptConfig):
            raise ValueError("--zipdepth-checkpoint requires the ZipDepth-PromptDA student config.")
        student_config = replace(student_config, checkpoint=config.zipdepth_checkpoint)
    teacher: BaseCompletionPredictor = config.teacher.setup()
    student: BaseCompletionPredictor = student_config.setup()

    all_metrics: list[FrameMetrics] = []
    kept_frames: list[HardFrame] = []
    n_frames: int = 0
    batch: tuple[PolycamData, ...]
    for batch in batch_plan:
        batch_start: int = n_frames
        n_frames += len(batch)
        tensor_batch: PolycamTensorBatch = stack_polycam_batch(batch)
        rgb_bhw3: UInt8[Tensor, "b h w 3"] = tensor_batch.rgb_bhw3
        prompt_bhw: Float32[Tensor, "b 192 256"] = tensor_batch.prompt_bhw
        teacher_bhw: Float32[Tensor, "b h w"] = teacher(rgb_bhw3, prompt_bhw)
        student_bhw: Float32[Tensor, "b h w"] = student(rgb_bhw3, prompt_bhw)
        baseline_bhw: Float32[Tensor, "b h w"] = F.interpolate(
            prompt_bhw[:, None],
            size=capture_hw,
            mode="bilinear",
            align_corners=False,
        )[:, 0]

        frame_offset: int
        polycam_data: PolycamData
        for frame_offset, polycam_data in enumerate(batch):
            frame_index: int = batch_start + frame_offset
            metrics: FrameMetrics = compute_frame_metrics(
                frame_index=frame_index,
                teacher_hw=teacher_bhw[frame_offset],
                student_hw=student_bhw[frame_offset],
                baseline_hw=baseline_bhw[frame_offset],
                edge_quantile=config.edge_quantile,
            )
            all_metrics.append(metrics)

            candidate_is_kept: bool = len(kept_frames) < config.max_keep
            if not candidate_is_kept:
                worst_metrics: FrameMetrics = kept_frames[-1].metrics
                candidate_is_kept = (metrics.student_edge_dev_m, -metrics.frame_index) > (
                    worst_metrics.student_edge_dev_m,
                    -worst_metrics.frame_index,
                )
            if candidate_is_kept:
                prompt_hw: Float32[ndarray, "192 256"] = prompt_bhw[frame_offset].detach().cpu().numpy().copy()
                teacher_hw: Float32[ndarray, "h w"] = teacher_bhw[frame_offset].detach().cpu().numpy().copy()
                student_hw: Float32[ndarray, "h w"] = student_bhw[frame_offset].detach().cpu().numpy().copy()
                kept_frames.append(
                    HardFrame(
                        metrics=metrics,
                        rgb_hw3=polycam_data.rgb_hw3.copy(),
                        prompt_hw=prompt_hw,
                        teacher_hw=teacher_hw,
                        student_hw=student_hw,
                    )
                )
                kept_frames.sort(key=lambda frame: (-frame.metrics.student_edge_dev_m, frame.metrics.frame_index))
                if len(kept_frames) > config.max_keep:
                    kept_frames.pop()

    ranked_metrics: list[FrameMetrics] = sorted(
        all_metrics,
        key=lambda metrics: (-metrics.student_edge_dev_m, metrics.frame_index),
    )
    frames_dir: Path = config.out_dir / "frames"
    previews_dir: Path = config.out_dir / "previews"
    frames_dir.mkdir(parents=True, exist_ok=True)
    previews_dir.mkdir(parents=True, exist_ok=True)

    kept_records: list[KeptFrameRecord] = []
    frame: HardFrame
    for rank, frame in enumerate(kept_frames, start=1):
        frame_name: str = f"frame_{frame.metrics.frame_index:05d}"
        frame_relative_path: Path = Path("frames") / f"{frame_name}.npz"
        preview_relative_path: Path = Path("previews") / f"{frame_name}.png"
        np.savez_compressed(
            config.out_dir / frame_relative_path,
            rgb=frame.rgb_hw3,
            prompt=frame.prompt_hw,
            teacher=frame.teacher_hw,
            student=frame.student_hw,
            frame_index=frame.metrics.frame_index,
        )
        _write_preview(frame, config.out_dir / preview_relative_path)
        kept_records.append(
            KeptFrameRecord(
                rank=rank,
                frame_path=frame_relative_path.as_posix(),
                preview_path=preview_relative_path.as_posix(),
                metrics=frame.metrics,
            )
        )

    checkpoint_path: Path | None = student_config.checkpoint if isinstance(student_config, ZipDepthPromptConfig) else None
    report = HardFramesReport(
        run=RunMetadata(
            capture_path=str(config.polycam_zip_path),
            checkpoint_path=str(checkpoint_path) if checkpoint_path is not None else None,
            edge_quantile=config.edge_quantile,
            capture_hw=capture_hw,
            teacher_config_class=type(config.teacher).__name__,
            student_config_class=type(student_config).__name__,
            student_reference_version="v4",
            student_output_role="reference_only",
            eval_label_field="teacher",
            batch_size=config.batch_size,
            max_frames=config.max_frames,
            max_keep=config.max_keep,
            processed_frames=n_frames,
        ),
        kept_frames=kept_records,
        full_ranking=[RankedFrameRecord(rank=rank, metrics=metrics) for rank, metrics in enumerate(ranked_metrics, start=1)],
    )
    metrics_path: Path = config.out_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(to_dict(report), metrics_file, indent=2, allow_nan=False)
        metrics_file.write("\n")

    print(f"{'frame':>7}  {'edge dev mm':>11}  {'overall dev mm':>14}  {'baseline/student edge':>21}")
    for frame in kept_frames:
        print(
            f"{frame.metrics.frame_index:7d}  "
            f"{1000.0 * frame.metrics.student_edge_dev_m:11.2f}  "
            f"{1000.0 * frame.metrics.student_overall_dev_m:14.2f}  "
            f"{frame.metrics.baseline_to_student_edge_ratio:21.3f}"
        )


def main(config: PolycamHardFramesConfig) -> None:
    """Entry point for Polycam hard-frame mining."""
    polycam_hard_frames(config)
