"""Mine Polycam frames where ZipDepth-PromptDA deviates most at depth edges."""

from dataclasses import dataclass, field, replace
from pathlib import Path

import torch
from jaxtyping import Bool, Float32, UInt8
from monopriors.models.depth_completion import AnnotatedCompletionConfig, PromptDAConfig, ZipDepthPromptConfig
from monopriors.models.depth_completion.base_completion_depth import BaseCompletionPredictor
from monopriors.models.depth_completion.prompt_da import DEFAULT_PROMPTDA_CACHE_DIR
from numpy import ndarray
from simplecv.data.polycam import (
    PolycamBatchPlan,
    PolycamData,
    PolycamDataset,
    load_polycam_data,
    prepare_polycam_batches,
    stack_polycam_batch,
)
from torch import Tensor
from trtkit import TensorRtBackendConfig, TorchBackendConfig

from zipdepth.apis.hard_frame_data import (
    FrameMetrics,
    HardFrameArchive,
    HardFramesReport,
    KeptFrameRecord,
    RankedFrameRecord,
    RunMetadata,
    prompt_upsample_depth,
    write_hard_frame_archive,
    write_hard_frame_preview,
    write_hard_frames_report,
)
from zipdepth.evaluation.edge_metrics import EdgeStratifiedResult, edge_stratified_mae


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


@dataclass(frozen=True, slots=True)
class HardFrame:
    """Host arrays retained for one current top-N frame."""

    metrics: FrameMetrics
    """Teacher-relative scalar metrics used for ranking."""
    archive: HardFrameArchive
    """Arrays written for this retained frame."""


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
        tensor_batch: tuple[UInt8[Tensor, "b h w 3"], Float32[Tensor, "b 192 256"]] = stack_polycam_batch(batch)
        rgb_bhwc: UInt8[Tensor, "b h w 3"] = tensor_batch[0]
        prompt_bhw: Float32[Tensor, "b 192 256"] = tensor_batch[1]
        teacher_bhw: Float32[Tensor, "b h w"] = teacher(rgb_bhwc, prompt_bhw)
        student_bhw: Float32[Tensor, "b h w"] = student(rgb_bhwc, prompt_bhw)
        prompt_valid_bhw: Bool[Tensor, "b 192 256"] = torch.isfinite(prompt_bhw) & (prompt_bhw > 0.0)
        baseline_bhw: Float32[Tensor, "b h w"] = prompt_upsample_depth(
            prompt_bhw,
            prompt_valid_bhw,
            height=capture_hw[0],
            width=capture_hw[1],
        )

        frame_offset: int
        polycam_data: PolycamData
        for frame_offset, polycam_data in enumerate(batch):
            frame_index: int = batch_start + frame_offset
            teacher_hw: Float32[Tensor, "h w"] = teacher_bhw[frame_offset]
            teacher_valid_hw: Bool[Tensor, "h w"] = torch.isfinite(teacher_hw) & (teacher_hw > 0.0)
            stratified: EdgeStratifiedResult | None = edge_stratified_mae(
                student_bhw[frame_offset],
                teacher_hw,
                teacher_valid_hw,
                baseline_bhw[frame_offset],
                edge_quantile=config.edge_quantile,
            )
            if stratified is None or stratified.baseline is None:
                raise ValueError(f"frame {frame_index} must produce valid student and baseline edge metrics")
            metrics = FrameMetrics(
                frame_index=frame_index,
                student_overall_dev_m=stratified.prediction.overall_mae_m,
                student_edge_dev_m=stratified.prediction.edge_mae_m,
                student_flat_dev_m=stratified.prediction.flat_mae_m,
                baseline_overall_dev_m=stratified.baseline.overall_mae_m,
                baseline_edge_dev_m=stratified.baseline.edge_mae_m,
                baseline_flat_dev_m=stratified.baseline.flat_mae_m,
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
                teacher_array_hw: Float32[ndarray, "h w"] = teacher_hw.detach().cpu().numpy().copy()
                student_hw: Float32[ndarray, "h w"] = student_bhw[frame_offset].detach().cpu().numpy().copy()
                kept_frames.append(
                    HardFrame(
                        metrics=metrics,
                        archive=HardFrameArchive(
                            frame_index=frame_index,
                            rgb_hwc=polycam_data.rgb_hw3.copy(),
                            prompt_hw=prompt_hw,
                            teacher_hw=teacher_array_hw,
                            student_hw=student_hw,
                        ),
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
        write_hard_frame_archive(config.out_dir / frame_relative_path, frame.archive)
        write_hard_frame_preview(
            config.out_dir / preview_relative_path,
            frame.archive.rgb_hwc,
            frame.archive.teacher_hw,
            frame.archive.student_hw,
        )
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
    write_hard_frames_report(metrics_path, report)

    print(f"{'frame':>7}  {'edge dev mm':>11}  {'overall dev mm':>14}  {'baseline/student edge':>21}")
    for frame in kept_frames:
        print(
            f"{frame.metrics.frame_index:7d}  "
            f"{1000.0 * frame.metrics.student_edge_dev_m:11.2f}  "
            f"{1000.0 * frame.metrics.student_overall_dev_m:14.2f}  "
            f"{frame.metrics.baseline_edge_dev_m / max(frame.metrics.student_edge_dev_m, 1.0e-9):21.3f}"
        )


def main(config: PolycamHardFramesConfig) -> None:
    """Entry point for Polycam hard-frame mining."""
    polycam_hard_frames(config)
