"""Frame-threaded decoding returns the same frames as a single-threaded decode (CPU, synthetic video)."""

from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from jaxtyping import UInt8
from numpy import ndarray

from fdanyone.video import decode_canonical_clip, iter_rgb_video, write_video


def _reference_frames(path: Path) -> list[UInt8[ndarray, "h w 3"]]:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "NONE"
        return [frame.to_ndarray(format="rgb24") for frame in container.decode(stream)]


def _noise_video(path: Path, count: int) -> Path:
    generator: np.random.Generator = np.random.default_rng(0)
    frames: list[UInt8[ndarray, "64 96 3"]] = [generator.integers(0, 256, (64, 96, 3), dtype=np.uint8) for _ in range(count)]
    return write_video(tuple(frames), path, Fraction(30), crf=18, preset="medium")


def test_iter_rgb_video_matches_a_single_threaded_decode(tmp_path: Path) -> None:
    video: Path = _noise_video(tmp_path / "noise.mp4", 40)
    reference = _reference_frames(video)
    decoded = list(iter_rgb_video(video))
    assert len(decoded) == len(reference) == 40
    assert all(np.array_equal(a, b) for a, b in zip(decoded, reference, strict=True))


def test_canonical_clip_frames_match_a_single_threaded_decode(tmp_path: Path) -> None:
    video: Path = _noise_video(tmp_path / "noise.mp4", 40)
    reference = _reference_frames(video)
    clip = decode_canonical_clip(video, num_frames=25, start_time=0.2)
    assert [frame.source_index for frame in clip.frames] == list(range(6, 31))
    assert all(np.array_equal(frame.rgb, reference[frame.source_index]) for frame in clip.frames)
