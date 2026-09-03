"""Tests for shared Polycam hard-frame artifacts."""

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from jaxtyping import Bool, Float32, UInt8
from numpy import ndarray
from torch import Tensor

from zipdepth.apis.hard_frame_data import (
    FrameMetrics,
    HardFrameArchive,
    HardFramesReport,
    KeptFrameRecord,
    RankedFrameRecord,
    RunMetadata,
    prompt_upsample_depth,
    read_hard_frame_archive,
    read_hard_frames_report,
    write_hard_frame_archive,
    write_hard_frame_preview,
    write_hard_frames_report,
)


def test_prompt_upsample_depth_fills_holes_without_zero_bleed() -> None:
    """Use valid support where available and the frame median elsewhere."""
    prompt_bhw: Float32[Tensor, "2 4 4"] = torch.zeros((2, 4, 4), dtype=torch.float32)
    valid_bhw: Bool[Tensor, "2 4 4"] = torch.zeros((2, 4, 4), dtype=torch.bool)
    prompt_bhw[0, 1:3, 1:3] = 2.0
    valid_bhw[0, 1:3, 1:3] = True

    upsampled_bhw: Float32[Tensor, "2 8 8"] = prompt_upsample_depth(prompt_bhw, valid_bhw, height=8, width=8)

    assert tuple(upsampled_bhw.shape) == (2, 8, 8)
    assert torch.equal(upsampled_bhw[0], torch.full((8, 8), 2.0))
    assert torch.equal(upsampled_bhw[1], torch.ones((8, 8)))


def test_hard_frame_manifest_and_archive_round_trip_without_ratios(tmp_path: Path) -> None:
    """Share one typed manifest and validated NPZ format between miner and scorer."""
    metrics = FrameMetrics(
        frame_index=6,
        student_overall_dev_m=0.1,
        student_edge_dev_m=0.2,
        student_flat_dev_m=0.05,
        baseline_overall_dev_m=0.15,
        baseline_edge_dev_m=0.3,
        baseline_flat_dev_m=0.075,
    )
    report = HardFramesReport(
        run=RunMetadata(
            capture_path="capture.zip",
            checkpoint_path="student.pth",
            edge_quantile=0.9,
            capture_hw=(20, 20),
            teacher_config_class="PromptDAConfig",
            student_config_class="ZipDepthPromptConfig",
            student_reference_version="v4",
            student_output_role="reference_only",
            eval_label_field="teacher",
            batch_size=8,
            max_frames=None,
            max_keep=20,
            processed_frames=1,
        ),
        kept_frames=[
            KeptFrameRecord(
                rank=1,
                frame_path="frames/frame_00006.npz",
                preview_path="previews/frame_00006.png",
                metrics=metrics,
            )
        ],
        full_ranking=[RankedFrameRecord(rank=1, metrics=metrics)],
    )
    report_path: Path = tmp_path / "metrics.json"
    write_hard_frames_report(report_path, report)

    document: dict[str, object] = json.loads(report_path.read_text())
    assert "baseline_to_student_edge_ratio" not in document["kept_frames"][0]  # type: ignore[index]
    assert read_hard_frames_report(report_path) == report

    archive = HardFrameArchive(
        frame_index=6,
        rgb_hwc=np.full((20, 20, 3), 127, dtype=np.uint8),
        prompt_hw=np.full((192, 256), 1.5, dtype=np.float32),
        teacher_hw=np.ones((20, 20), dtype=np.float32),
        student_hw=np.full((20, 20), 1.1, dtype=np.float32),
    )
    archive_path: Path = tmp_path / "frame.npz"
    write_hard_frame_archive(archive_path, archive)
    loaded: HardFrameArchive = read_hard_frame_archive(archive_path)

    assert loaded.frame_index == archive.frame_index
    np.testing.assert_array_equal(loaded.rgb_hwc, archive.rgb_hwc)
    np.testing.assert_array_equal(loaded.prompt_hw, archive.prompt_hw)
    np.testing.assert_array_equal(loaded.teacher_hw, archive.teacher_hw)
    np.testing.assert_array_equal(loaded.student_hw, archive.student_hw)

    preview_path: Path = tmp_path / "preview.png"
    write_hard_frame_preview(preview_path, loaded.rgb_hwc, loaded.teacher_hw, loaded.student_hw)
    preview_bgr_hwc: UInt8[ndarray, "h four_w 3"] | None = cv2.imread(str(preview_path))
    assert preview_bgr_hwc is not None
    assert preview_bgr_hwc.shape == (20, 80, 3)
