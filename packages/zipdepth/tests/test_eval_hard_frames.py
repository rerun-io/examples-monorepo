"""Fixed hard-frame scorer tests."""

from pathlib import Path

import numpy as np
import pytest
import torch
from jaxtyping import Float32
from monopriors.models.depth_completion.zipdepth_prompt import ZipDepthPrompt
from numpy import ndarray
from serde.json import from_json
from torch import Tensor, nn

from zipdepth.apis.eval_hard_frames import (
    Config,
    DepthMetrics,
    H2Diagnostic,
    HardFrameData,
    HardFramesEvaluationReport,
    ModelPredictions,
    evaluate_hard_frames,
    score_depth_metrics,
    score_h2_diagnostic,
)
from zipdepth.apis.hard_frame_data import (
    FrameMetrics,
    HardFrameArchive,
    HardFramesReport,
    KeptFrameRecord,
    RankedFrameRecord,
    RunMetadata,
    write_hard_frame_archive,
    write_hard_frames_report,
)


def _write_eval_set(
    eval_dir: Path,
    frame_indices: tuple[int, ...],
    teacher_hw: Float32[ndarray, "h w"],
    prompt_hw: Float32[ndarray, "192 256"],
    stored_student_hw: Float32[ndarray, "h w"],
) -> None:
    """Write a small typed hard-frame set for scorer integration tests."""
    frames_dir: Path = eval_dir / "frames"
    frames_dir.mkdir(parents=True)
    kept_records: list[KeptFrameRecord] = []
    ranked_records: list[RankedFrameRecord] = []
    for rank, frame_index in enumerate(frame_indices, start=1):
        metrics = FrameMetrics(
            frame_index=frame_index,
            student_overall_dev_m=0.1,
            student_edge_dev_m=0.1,
            student_flat_dev_m=0.1,
            baseline_overall_dev_m=0.2,
            baseline_edge_dev_m=0.2,
            baseline_flat_dev_m=0.2,
        )
        frame_relative_path: Path = Path("frames") / f"frame_{frame_index:05d}.npz"
        write_hard_frame_archive(
            eval_dir / frame_relative_path,
            HardFrameArchive(
                frame_index=frame_index,
                rgb_hwc=np.full((*teacher_hw.shape, 3), frame_index, dtype=np.uint8),
                prompt_hw=prompt_hw,
                teacher_hw=teacher_hw,
                student_hw=stored_student_hw,
            ),
        )
        kept_records.append(
            KeptFrameRecord(
                rank=rank,
                frame_path=frame_relative_path.as_posix(),
                preview_path=f"previews/frame_{frame_index:05d}.png",
                metrics=metrics,
            )
        )
        ranked_records.append(RankedFrameRecord(rank=rank, metrics=metrics))
    write_hard_frames_report(
        eval_dir / "metrics.json",
        HardFramesReport(
            run=RunMetadata(
                capture_path="capture.zip",
                checkpoint_path="student.pth",
                edge_quantile=0.9,
                capture_hw=teacher_hw.shape,
                teacher_config_class="PromptDAConfig",
                student_config_class="ZipDepthPromptConfig",
                student_reference_version="v4",
                student_output_role="reference_only",
                eval_label_field="teacher",
                batch_size=8,
                max_frames=None,
                max_keep=len(frame_indices),
                processed_frames=len(frame_indices),
            ),
            kept_frames=kept_records,
            full_ranking=ranked_records,
        ),
    )


def test_edge_weighted_metrics_match_a_worked_two_step_frame() -> None:
    """Score errors at teacher steps of magnitude one and three exactly."""
    teacher_hw: Float32[ndarray, "10 10"] = np.ones((10, 10), dtype=np.float32)
    teacher_hw[:, 5:] += 1.0
    teacher_hw[:, 8:] += 3.0
    prediction_hw: Float32[ndarray, "10 10"] = teacher_hw.copy()
    prediction_hw[:, 5] += 2.0
    prediction_hw[:, 8] += 4.0

    metrics: DepthMetrics = score_depth_metrics(prediction_hw, teacher_hw)

    assert metrics.edge_mae_m == pytest.approx(4.0)
    assert metrics.flat_mae_m == pytest.approx(20.0 / 90.0)
    assert metrics.overall_mae_m == pytest.approx(0.6)
    assert metrics.ewmae_m == pytest.approx((1.0 * 2.0 + 3.0 * 4.0) / (1.0 + 3.0))


def test_h2_diagnostic_uses_two_by_two_teacher_mean_and_gradient_retention() -> None:
    """Give an exact H/2 student zero error and full teacher-gradient retention."""
    teacher_half_hw: Float32[ndarray, "10 10"] = np.ones((10, 10), dtype=np.float32)
    teacher_half_hw[:, 5:] = 2.0
    teacher_hw: Float32[ndarray, "20 20"] = np.repeat(np.repeat(teacher_half_hw, 2, axis=0), 2, axis=1)

    diagnostic: H2Diagnostic = score_h2_diagnostic(teacher_half_hw, teacher_hw)

    assert diagnostic.edge_mae_m == pytest.approx(0.0)
    assert diagnostic.flat_mae_m == pytest.approx(0.0)
    assert diagnostic.half_gradient_retention == pytest.approx(1.0)


def test_fixed_scorer_writes_json_and_only_requested_plant_preview(tmp_path: Path) -> None:
    """Consume saved frames without mining and route fake model outputs to artifacts."""
    eval_dir: Path = tmp_path / "hard20"
    teacher_hw: Float32[ndarray, "20 20"] = np.ones((20, 20), dtype=np.float32)
    teacher_hw[:, 8:] += 1.0
    teacher_hw[:, 16:] += 3.0
    stored_student_hw: Float32[ndarray, "20 20"] = teacher_hw + 0.1
    prompt_hw: Float32[ndarray, "192 256"] = np.full((192, 256), 1.5, dtype=np.float32)
    prompt_hw[:, :32] = 0.0
    _write_eval_set(eval_dir, (6,), teacher_hw, prompt_hw, stored_student_hw)
    checkpoint: Path = tmp_path / "candidate.pth"
    checkpoint.touch()
    output: Path = tmp_path / "candidate-hard20.json"

    def fake_model_runner(frames: list[HardFrameData], _: Config) -> ModelPredictions:
        return ModelPredictions(
            student_depths=[frames[0].stored_student_hw],
            student_half_depths=[teacher_hw.reshape(10, 2, 10, 2).mean(axis=(1, 3)).astype(np.float32)],
        )

    result: Path = evaluate_hard_frames(
        Config(eval_dir=eval_dir, checkpoint=checkpoint, output=output),
        model_runner=fake_model_runner,
    )

    assert result == output
    report: HardFramesEvaluationReport = from_json(HardFramesEvaluationReport, output.read_text())
    assert report.per_frame[0].frame_index == 6
    assert report.macro_average.student.edge_mae_m == pytest.approx(0.1)
    assert report.macro_average.baseline.edge_mae_m > 0.0
    assert (tmp_path / "candidate-hard20_frame_00006.png").is_file()
    assert not (tmp_path / "candidate-hard20_frame_00007.png").exists()


def test_fixed_scorer_batches_distinct_frames_and_captures_only_h2_logits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run eight distinct frames, pad only the tail, and decode captured H/2 logits."""
    eval_dir: Path = tmp_path / "hard20"
    teacher_hw: Float32[ndarray, "20 20"] = np.ones((20, 20), dtype=np.float32)
    teacher_hw[:, 8:] += 1.0
    teacher_hw[:, 16:] += 3.0
    prompt_hw: Float32[ndarray, "192 256"] = np.full((192, 256), 1.5, dtype=np.float32)
    _write_eval_set(eval_dir, tuple(range(8, 17)), teacher_hw, prompt_hw, teacher_hw)
    checkpoint: Path = tmp_path / "signed-head.pth"
    checkpoint.touch()

    model: ZipDepthPrompt = ZipDepthPrompt()
    head_half: nn.Conv2d = model.backbone.decoder.head_half
    nn.init.zeros_(head_half.weight)
    assert head_half.bias is not None
    nn.init.zeros_(head_half.bias)
    observed_first_pixels: list[Float32[Tensor, "b"]] = []

    def forward_with_range(
        image: Float32[Tensor, "b 3 h w"],
        prompt_depth: Float32[Tensor, "b 1 192 256"],
    ) -> tuple[Float32[Tensor, "b 1 h w"], Float32[Tensor, "b 1 1 1"], Float32[Tensor, "b 1 1 1"]]:
        """Run head_half but bypass the vendored convex helper."""
        batch_size: int = image.shape[0]
        observed_first_pixels.append(image[:, 0, 0, 0].detach().cpu())
        features_bchw: Float32[Tensor, "b 32 10 10"] = torch.zeros((batch_size, 32, 10, 10), device=image.device)
        head_half(features_bchw)
        full_depth_bchw: Float32[Tensor, "b 1 20 20"] = torch.full((batch_size, 1, 20, 20), 1.5, device=image.device)
        min_depth_b: Float32[Tensor, "b 1 1 1"] = torch.ones((batch_size, 1, 1, 1), device=prompt_depth.device)
        max_depth_b: Float32[Tensor, "b 1 1 1"] = torch.full((batch_size, 1, 1, 1), 2.0, device=prompt_depth.device)
        return full_depth_bchw, min_depth_b, max_depth_b

    monkeypatch.setattr(model, "fuse_for_inference", lambda: model)
    monkeypatch.setattr(model, "forward_with_range", forward_with_range)
    monkeypatch.setattr("zipdepth.apis.eval_hard_frames.load_zipdepth_prompt", lambda *_args, **_kwargs: model)

    output: Path = evaluate_hard_frames(
        Config(eval_dir=eval_dir, checkpoint=checkpoint, device="cpu"),
    )

    assert output.is_file()
    report: HardFramesEvaluationReport = from_json(HardFramesEvaluationReport, output.read_text())
    assert report.macro_average.h2.overall_mae_m == pytest.approx(1.1)
    assert len(observed_first_pixels) == 2
    torch.testing.assert_close(observed_first_pixels[0], torch.arange(8, 16, dtype=torch.float32) / 255.0)
    torch.testing.assert_close(observed_first_pixels[1], torch.full((8,), 16.0 / 255.0))
