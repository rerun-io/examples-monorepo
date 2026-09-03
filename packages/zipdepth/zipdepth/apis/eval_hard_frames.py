"""Score fixed saved hard frames without rerunning the teacher or mining frames."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from itertools import batched
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float, Float32, UInt8
from monopriors.models.depth_completion.base_completion_depth import preprocess_completion_batch
from monopriors.models.depth_completion.zipdepth_prompt import ZipDepthPrompt, load_zipdepth_prompt
from numpy import ndarray
from serde import serde
from serde.json import to_json
from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle

from zipdepth.apis.hard_frame_data import (
    HardFrameArchive,
    KeptFrameRecord,
    prompt_upsample_depth,
    read_hard_frame_archive,
    read_hard_frames_report,
    write_hard_frame_preview,
)
from zipdepth.evaluation.edge_metrics import EDGE_QUANTILE, DepthMetrics, EdgeStratifiedResult, edge_stratified_mae

DEFAULT_EVAL_DIR: Path = Path("data/eval/polycam-hard20")
"""Saved hard-20 set produced once by the miner."""
DEFAULT_CHECKPOINT: Path = Path("data/checkpoints/zdpda-v4/final_model.pth")
"""Production v4 checkpoint whose outputs are stored in the hard-20 archives."""
MODEL_BATCH_SIZE: int = 8
"""Static production batch used to reproduce the stored v4 arrays."""
DepthMap: TypeAlias = Float32[ndarray, "h w"]
ImageRGB: TypeAlias = UInt8[ndarray, "h w 3"]
DeviceChoice: TypeAlias = Literal["auto", "cuda", "cpu"]


@dataclass(frozen=True, slots=True)
class Config:
    """Fixed hard-frame scoring configuration."""

    eval_dir: Path = DEFAULT_EVAL_DIR
    """Directory containing ``metrics.json`` and the frozen frame archives."""
    checkpoint: Path = DEFAULT_CHECKPOINT
    """Student checkpoint to score; the teacher is never loaded."""
    output: Path | None = None
    """JSON output path; None writes beside the checkpoint."""
    device: DeviceChoice = "cuda"
    """Inference device; production parity uses CUDA fp16 autocast."""


@serde
@dataclass(frozen=True, slots=True)
class H2Diagnostic:
    """Student H/2 depth quality relative to the 2x2-mean teacher."""

    edge_mae_m: float
    """H/2 mean absolute error on teacher H/2 edges."""
    flat_mae_m: float
    """H/2 mean absolute error outside teacher H/2 edges."""
    overall_mae_m: float
    """H/2 mean absolute error over all valid teacher H/2 pixels."""
    half_gradient_retention: float
    """Teacher H/2 edge fraction where the student keeps at least half its gradient."""


@dataclass(frozen=True, slots=True)
class HardFrameData:
    """One immutable saved hard-frame input and reference record."""

    rank: int
    """Fixed rank recorded by the miner."""
    frame_index: int
    """Source capture frame index."""
    rgb_hwc: ImageRGB
    """Capture-resolution uint8 RGB image."""
    prompt_hw: DepthMap
    """Raw 192x256 metric prompt; zero denotes invalid."""
    teacher_hw: DepthMap
    """Frozen capture-resolution teacher depth in metres."""
    stored_student_hw: DepthMap
    """Frozen v4 output used only for the v4 preprocessing parity check."""
    miner_student_edge_mae_m: float | None
    """Original miner edge deviation, when present in ``metrics.json``."""


@dataclass(frozen=True, slots=True)
class ModelPredictions:
    """Full-resolution and captured H/2 outputs in input-frame order."""

    student_depths: list[DepthMap]
    """Current student predictions at capture resolution."""
    student_half_depths: list[DepthMap]
    """Current student metric depths decoded directly from H/2 logits."""


@serde
@dataclass(frozen=True, slots=True)
class FrameResult:
    """All fixed-scorer measurements for one saved frame."""

    rank: int
    """Fixed miner rank."""
    frame_index: int
    """Source capture frame index."""
    student: DepthMetrics
    """Current student versus frozen teacher."""
    baseline: DepthMetrics
    """Hole-aware bilinear prompt baseline versus frozen teacher."""
    h2: H2Diagnostic
    """Current student's own H/2 diagnostic."""
    parity_max_abs_m: float | None
    """Current versus stored v4 maximum difference, only for the reference checkpoint."""
    miner_student_edge_mae_m: float | None
    """Student edge MAE saved by the original miner."""


@serde
@dataclass(frozen=True, slots=True)
class EvaluationReportConfig:
    """Resolved fixed hard-frame evaluation configuration."""

    eval_dir: Path
    """Directory containing the frozen evaluation frames."""
    checkpoint: Path
    """Student checkpoint that was scored."""
    output: Path
    """Written JSON report path."""
    device: DeviceChoice
    """Requested inference device policy."""


@serde
@dataclass(frozen=True, slots=True)
class MacroAverage:
    """Macro-average scorer metrics across the fixed frames."""

    student: DepthMetrics
    """Current student metrics relative to the frozen teacher."""
    baseline: DepthMetrics
    """Hole-aware prompt-upsample metrics relative to the frozen teacher."""
    h2: H2Diagnostic
    """Current student H/2 diagnostic."""
    parity_max_abs_m: float | None
    """Maximum current-versus-reference difference, when checked."""


@serde
@dataclass(frozen=True, slots=True)
class HardFramesEvaluationReport:
    """Typed fixed hard-frame scorer document."""

    config: EvaluationReportConfig
    """Resolved evaluation configuration."""
    per_frame: list[FrameResult]
    """Metrics for each fixed frame in miner rank order."""
    macro_average: MacroAverage
    """Macro-average metrics over every fixed frame."""


ModelRunner: TypeAlias = Callable[[list[HardFrameData], Config], ModelPredictions]


def _resolve_device(device: DeviceChoice) -> str:
    """Resolve a requested Torch device and reject an unavailable explicit CUDA device."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not visible")
    return device


def score_depth_metrics(prediction_hw: DepthMap, teacher_hw: DepthMap) -> DepthMetrics:
    """Score one full-resolution depth prediction with the fixed hard-frame rule.

    Args:
        prediction_hw: Float32 predicted metric depth with shape ``(H, W)``.
        teacher_hw: Float32 frozen teacher metric depth with shape ``(H, W)``.

    Returns:
        Edge, flat, overall, and teacher-gradient-weighted absolute errors.
    """
    if prediction_hw.shape != teacher_hw.shape:
        raise ValueError("prediction and teacher must have equal 2D shapes")
    teacher_valid_hw: Bool[ndarray, "h w"] = np.isfinite(teacher_hw) & (teacher_hw > 0.0)
    result: EdgeStratifiedResult | None = edge_stratified_mae(
        prediction_hw,
        teacher_hw,
        teacher_valid_hw,
        edge_quantile=EDGE_QUANTILE,
    )
    if result is None:
        raise ValueError("teacher must produce nonempty valid edge and flat strata")
    return result.prediction


def score_h2_diagnostic(student_half_hw: DepthMap, teacher_hw: DepthMap) -> H2Diagnostic:
    """Compare decoded student H/2 depth with the teacher's exact 2x2 mean."""
    teacher: Float32[Tensor, "h w"] = torch.from_numpy(np.ascontiguousarray(teacher_hw)).to(dtype=torch.float32)
    if teacher.ndim != 2 or teacher.shape[0] % 2 != 0 or teacher.shape[1] % 2 != 0:
        raise ValueError("teacher height and width must both be even for the H/2 diagnostic")
    teacher_half: Float32[Tensor, "half_h half_w"] = F.avg_pool2d(teacher[None, None], kernel_size=2, stride=2)[0, 0]
    teacher_half_hw: DepthMap = teacher_half.numpy().astype(np.float32, copy=False)
    if student_half_hw.shape != teacher_half_hw.shape:
        raise ValueError(f"student H/2 shape {student_half_hw.shape} does not match teacher H/2 shape {teacher_half_hw.shape}")
    teacher_valid_hw: Bool[ndarray, "h w"] = np.isfinite(teacher_half_hw) & (teacher_half_hw > 0.0)
    result: EdgeStratifiedResult | None = edge_stratified_mae(
        student_half_hw,
        teacher_half_hw,
        teacher_valid_hw,
        edge_quantile=EDGE_QUANTILE,
    )
    if result is None:
        raise ValueError("H/2 teacher must produce nonempty valid edge and flat strata")
    student: Float32[Tensor, "half_h half_w"] = torch.from_numpy(np.ascontiguousarray(student_half_hw)).to(dtype=torch.float32)
    student_gradient: Float32[Tensor, "half_h half_w"] = torch.zeros_like(student)
    student_gradient[:, 1:] += (student[:, 1:] - student[:, :-1]).abs()
    student_gradient[1:, :] += (student[1:, :] - student[:-1, :]).abs()
    retained: Bool[Tensor, "half_h half_w"] = result.edge_hw & (student_gradient >= 0.5 * result.gradient_hw)
    return H2Diagnostic(
        edge_mae_m=result.prediction.edge_mae_m,
        flat_mae_m=result.prediction.flat_mae_m,
        overall_mae_m=result.prediction.overall_mae_m,
        half_gradient_retention=float(retained.sum().div(result.edge_hw.sum()).item()),
    )


def _load_frames(eval_dir: Path) -> list[HardFrameData]:
    """Load the immutable frame order from the miner report and validate every archive."""
    metrics_path: Path = eval_dir / "metrics.json"
    report = read_hard_frames_report(metrics_path)
    frames: list[HardFrameData] = []
    item: KeptFrameRecord
    for item in report.kept_frames:
        archive: HardFrameArchive = read_hard_frame_archive(eval_dir / item.frame_path)
        frames.append(
            HardFrameData(
                rank=item.rank,
                frame_index=archive.frame_index,
                rgb_hwc=archive.rgb_hwc,
                prompt_hw=archive.prompt_hw,
                teacher_hw=archive.teacher_hw,
                stored_student_hw=archive.student_hw,
                miner_student_edge_mae_m=item.metrics.student_edge_dev_m,
            )
        )
    if not frames:
        raise ValueError("metrics.json kept_frames is empty")
    return frames


def _run_model(frames: list[HardFrameData], config: Config) -> ModelPredictions:
    """Run the student in static batches and capture metric depth at H/2."""
    if not config.checkpoint.is_file():
        raise FileNotFoundError(f"student checkpoint is missing: {config.checkpoint}")
    device: str = _resolve_device(config.device)
    model: ZipDepthPrompt = load_zipdepth_prompt(config.checkpoint).to(device=device, dtype=torch.float32).fuse_for_inference()
    captured_logits: Float[Tensor, "b 1 half_h half_w"] | None = None

    def capture_head_half(
        _module: nn.Module,
        _inputs: tuple[Float[Tensor, "b 32 half_h half_w"], ...],
        output: Float[Tensor, "b 1 half_h half_w"],
    ) -> None:
        nonlocal captured_logits
        captured_logits = output.detach()

    hook_handle: RemovableHandle = model.backbone.decoder.head_half.register_forward_hook(capture_head_half)
    student_depths: list[DepthMap] = []
    student_half_depths: list[DepthMap] = []
    try:
        frame_batch: tuple[HardFrameData, ...]
        for frame_batch in batched(frames, MODEL_BATCH_SIZE):
            real_batch_size: int = len(frame_batch)
            padded_batch: tuple[HardFrameData, ...] = frame_batch + (frame_batch[-1],) * (MODEL_BATCH_SIZE - real_batch_size)
            rgb_bhwc: UInt8[Tensor, "b h w 3"] = torch.from_numpy(np.stack([frame.rgb_hwc for frame in padded_batch])).to(device=device)
            prompt_bhw: Float32[Tensor, "b 192 256"] = torch.from_numpy(np.stack([frame.prompt_hw for frame in padded_batch])).to(
                device=device, dtype=torch.float32
            )
            image_hw: tuple[int, int] = frame_batch[0].rgb_hwc.shape[:2]
            image_bchw: Float32[Tensor, "b 3 h w"]
            prompt_bchw: Float32[Tensor, "b 1 192 256"]
            image_bchw, prompt_bchw = preprocess_completion_batch(rgb_bhwc, prompt_bhw, image_hw)
            captured_logits = None
            with torch.inference_mode():
                output: tuple[Float32[Tensor, "b 1 h w"], Float32[Tensor, "b 1 1 1"], Float32[Tensor, "b 1 1 1"]]
                if device == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        output = model.forward_with_range(image_bchw, prompt_bchw)
                else:
                    output = model.forward_with_range(image_bchw, prompt_bchw)
            if captured_logits is None:
                raise RuntimeError("H/2 logits were not captured from decoder.head_half")
            student_bchw: Float32[Tensor, "b 1 h w"] = output[0]
            min_depth_b: Float32[Tensor, "b 1 1 1"] = output[1]
            max_depth_b: Float32[Tensor, "b 1 1 1"] = output[2]
            span_b: Float32[Tensor, "b 1 1 1"] = (max_depth_b - min_depth_b).clamp_min(1.0e-6)
            student_half_bchw: Float32[Tensor, "b 1 half_h half_w"] = torch.sigmoid(captured_logits) * span_b + min_depth_b
            for batch_index in range(real_batch_size):
                student_depths.append(student_bchw[batch_index, 0].float().cpu().numpy().astype(np.float32, copy=False))
                student_half_depths.append(student_half_bchw[batch_index, 0].float().cpu().numpy().astype(np.float32, copy=False))
    finally:
        hook_handle.remove()
    return ModelPredictions(student_depths=student_depths, student_half_depths=student_half_depths)


def _mean_depth_metrics(results: list[FrameResult], field_name: Literal["student", "baseline"]) -> DepthMetrics:
    """Macro-average one full-resolution metric group over fixed frames."""
    values: list[DepthMetrics] = [getattr(result, field_name) for result in results]
    return DepthMetrics(
        edge_mae_m=float(np.mean([value.edge_mae_m for value in values])),
        flat_mae_m=float(np.mean([value.flat_mae_m for value in values])),
        overall_mae_m=float(np.mean([value.overall_mae_m for value in values])),
        ewmae_m=float(np.mean([value.ewmae_m for value in values])),
    )


def _mean_h2(results: list[FrameResult]) -> H2Diagnostic:
    """Macro-average the H/2 diagnostics over fixed frames."""
    return H2Diagnostic(
        edge_mae_m=float(np.mean([result.h2.edge_mae_m for result in results])),
        flat_mae_m=float(np.mean([result.h2.flat_mae_m for result in results])),
        overall_mae_m=float(np.mean([result.h2.overall_mae_m for result in results])),
        half_gradient_retention=float(np.mean([result.h2.half_gradient_retention for result in results])),
    )


def _print_table(results: list[FrameResult]) -> None:
    """Print one compact millimetre table plus the macro-average row."""
    print("frame  student E/F/O mm       baseline E/F/O mm      EW/B-EW mm   H2 E/F mm    H2 retain  parity mm")
    for result in results:
        parity_mm: str = "-" if result.parity_max_abs_m is None else f"{1000.0 * result.parity_max_abs_m:.4f}"
        print(
            f"{result.frame_index:5d}  "
            f"{1000.0 * result.student.edge_mae_m:6.1f}/{1000.0 * result.student.flat_mae_m:6.1f}/{1000.0 * result.student.overall_mae_m:6.1f}  "
            f"{1000.0 * result.baseline.edge_mae_m:6.1f}/{1000.0 * result.baseline.flat_mae_m:6.1f}/{1000.0 * result.baseline.overall_mae_m:6.1f}  "
            f"{1000.0 * result.student.ewmae_m:6.1f}/{1000.0 * result.baseline.ewmae_m:6.1f}  "
            f"{1000.0 * result.h2.edge_mae_m:6.1f}/{1000.0 * result.h2.flat_mae_m:6.1f}  "
            f"{100.0 * result.h2.half_gradient_retention:8.2f}%  {parity_mm:>9}"
        )
    student: DepthMetrics = _mean_depth_metrics(results, "student")
    baseline: DepthMetrics = _mean_depth_metrics(results, "baseline")
    h2: H2Diagnostic = _mean_h2(results)
    parity_values: list[float] = [result.parity_max_abs_m for result in results if result.parity_max_abs_m is not None]
    parity_mm: str = "-" if not parity_values else f"{1000.0 * max(parity_values):.4f}"
    print(
        f"macro  {1000.0 * student.edge_mae_m:6.1f}/{1000.0 * student.flat_mae_m:6.1f}/{1000.0 * student.overall_mae_m:6.1f}  "
        f"{1000.0 * baseline.edge_mae_m:6.1f}/{1000.0 * baseline.flat_mae_m:6.1f}/{1000.0 * baseline.overall_mae_m:6.1f}  "
        f"{1000.0 * student.ewmae_m:6.1f}/{1000.0 * baseline.ewmae_m:6.1f}  "
        f"{1000.0 * h2.edge_mae_m:6.1f}/{1000.0 * h2.flat_mae_m:6.1f}  "
        f"{100.0 * h2.half_gradient_retention:8.2f}%  {parity_mm:>9}"
    )


def evaluate_hard_frames(config: Config, *, model_runner: ModelRunner = _run_model) -> Path:
    """Score the fixed saved frames, write JSON and plant previews, and print a table."""
    frames: list[HardFrameData] = _load_frames(config.eval_dir)
    predictions: ModelPredictions = model_runner(frames, config)
    if len(predictions.student_depths) != len(frames) or len(predictions.student_half_depths) != len(frames):
        raise ValueError("model runner must return one full-resolution and H/2 prediction per frame")
    reference_checkpoint: bool = config.checkpoint.resolve() == DEFAULT_CHECKPOINT.resolve()
    results: list[FrameResult] = []
    output: Path = config.output or config.checkpoint.parent / f"hard20_{config.checkpoint.stem}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame: HardFrameData
    student_hw: DepthMap
    student_half_hw: DepthMap
    for frame, student_hw, student_half_hw in zip(
        frames, predictions.student_depths, predictions.student_half_depths, strict=True
    ):
        prompt_hw: Float32[Tensor, "1 192 256"] = torch.from_numpy(np.ascontiguousarray(frame.prompt_hw))[None]
        prompt_valid_hw: Bool[Tensor, "1 192 256"] = torch.isfinite(prompt_hw) & (prompt_hw > 0.0)
        baseline_hw: DepthMap = prompt_upsample_depth(
            prompt_hw,
            prompt_valid_hw,
            height=frame.teacher_hw.shape[0],
            width=frame.teacher_hw.shape[1],
        )[0].numpy().astype(np.float32, copy=False)
        parity_max_abs_m: float | None = None
        if reference_checkpoint:
            parity_max_abs_m = float(np.max(np.abs(student_hw - frame.stored_student_hw)))
            if parity_max_abs_m > 1.0e-5:
                raise AssertionError(
                    f"frame {frame.frame_index} v4 preprocessing parity failed: maximum difference is {1000.0 * parity_max_abs_m:.4f} mm"
                )
        teacher_valid_hw: Bool[ndarray, "h w"] = np.isfinite(frame.teacher_hw) & (frame.teacher_hw > 0.0)
        stratified: EdgeStratifiedResult | None = edge_stratified_mae(
            student_hw,
            frame.teacher_hw,
            teacher_valid_hw,
            baseline_hw,
            edge_quantile=EDGE_QUANTILE,
        )
        if stratified is None or stratified.baseline is None:
            raise ValueError(f"frame {frame.frame_index} must produce valid student and baseline edge metrics")
        result: FrameResult = FrameResult(
            rank=frame.rank,
            frame_index=frame.frame_index,
            student=stratified.prediction,
            baseline=stratified.baseline,
            h2=score_h2_diagnostic(student_half_hw, frame.teacher_hw),
            parity_max_abs_m=parity_max_abs_m,
            miner_student_edge_mae_m=frame.miner_student_edge_mae_m,
        )
        results.append(result)
        if frame.frame_index in (6, 7):
            write_hard_frame_preview(
                output.with_name(f"{output.stem}_frame_{frame.frame_index:05d}.png"),
                frame.rgb_hwc,
                frame.teacher_hw,
                student_hw,
            )

    macro_student: DepthMetrics = _mean_depth_metrics(results, "student")
    macro_baseline: DepthMetrics = _mean_depth_metrics(results, "baseline")
    macro_h2: H2Diagnostic = _mean_h2(results)
    parity_values: list[float] = [result.parity_max_abs_m for result in results if result.parity_max_abs_m is not None]
    document: HardFramesEvaluationReport = HardFramesEvaluationReport(
        config=EvaluationReportConfig(
            eval_dir=config.eval_dir,
            checkpoint=config.checkpoint,
            output=output,
            device=config.device,
        ),
        per_frame=results,
        macro_average=MacroAverage(
            student=macro_student,
            baseline=macro_baseline,
            h2=macro_h2,
            parity_max_abs_m=max(parity_values) if parity_values else None,
        ),
    )
    output.write_text(to_json(document) + "\n", encoding="utf-8")
    _print_table(results)
    print(f"wrote {output}")
    return output


def main(config: Config) -> Path:
    """Run the fixed hard-frame scorer."""
    return evaluate_hard_frames(config)
