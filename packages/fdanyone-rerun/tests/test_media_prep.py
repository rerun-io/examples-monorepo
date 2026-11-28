"""Source and skeleton clips go through one chunked crop: the pose batch's device tensor equals the source's host one (CPU)."""

from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
import torch
from jaxtyping import UInt8
from numpy import ndarray

from fdanyone.errors import FourDAnyoneError
from fdanyone.geometry.crop import Crop
from fdanyone.skeleton import pipeline
from fdanyone.skeleton.pipeline import SkeletonVideo
from fdanyone.video import write_video

CROP: Crop = Crop(top=8, left=16, height=48, width=64, original_height=64, original_width=96, output_height=24, output_width=32)


def _video(path: Path, count: int) -> Path:
    generator: np.random.Generator = np.random.default_rng(0)
    frames: list[UInt8[ndarray, "64 96 3"]] = [generator.integers(0, 256, (64, 96, 3), dtype=np.uint8) for _ in range(count)]
    return write_video(frames, path, Fraction(30), crf=18, preset="medium")


def test_chunked_skeleton_prep_equals_the_source_path_and_one_whole_clip_batch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FDANYONE_CPU_MEDIA_PREP", "1")
    monkeypatch.setattr(pipeline, "MEDIA_PREP_CHUNK_FRAMES", 4)
    video: Path = _video(tmp_path / "clip.mp4", 10)
    source = pipeline._video_tensor(video, 10, crop=CROP)
    frames = pipeline._decode_video_frames(video, 10)
    skeleton = pipeline.Conditioning.prepare_skeleton(None, frames, SkeletonVideo(path=video, crop=CROP))  # pyrefly: ignore[bad-argument-type]
    whole = pipeline._crop_and_normalize(torch.stack(frames).permute(0, 3, 1, 2), CROP).permute(1, 0, 2, 3)
    assert source.shape == (1, 3, 10, 24, 32)
    assert torch.equal(source[0], skeleton)
    assert torch.equal(skeleton, whole)


def test_a_clip_with_the_wrong_frame_count_is_rejected(tmp_path: Path) -> None:
    video: Path = _video(tmp_path / "clip.mp4", 6)
    with pytest.raises(FourDAnyoneError, match="6 frames, expected 7"):
        pipeline._decode_video_frames(video, 7)
