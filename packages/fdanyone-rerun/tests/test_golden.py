"""The example run against the banked reference recording (``$FDANYONE_GOLDEN_RRD``): per-view SSIM in a band.

SageAttention and Inductor make two runs of the same code differ slightly, so the check is a band, calibrated by a
same-code repeat (README "Goldens"), never bit equality.
"""

import os
from pathlib import Path
from typing import cast

import av
import numpy as np
import pytest
import rerun as rr
import torch
from av.video.codeccontext import VideoCodecContext
from jaxtyping import Float32, UInt8
from numpy import ndarray

from fdanyone_rerun.assets import PACKAGE_ROOT
from fdanyone_rerun.outputs import read_camera_rig
from fdanyone_rerun.rerun_log import camera_entity

pytestmark = pytest.mark.golden

GOLDEN_RRD: Path = Path(os.environ.get("FDANYONE_GOLDEN_RRD", PACKAGE_ROOT / "data" / "golden" / "example-fp8.rrd"))
MEAN_SSIM_MIN: float = 0.91
"""Mean over the six views. Two runs of the default (FP8) build on the RTX 5090 (2026-11-15) scored 0.942 mean,
0.938-0.947 per view (frames 0/30/60/90/120); the band leaves ~0.03 for process-to-process variation."""
VIEW_SSIM_MIN: float = 0.90
"""Worst single view (same calibration)."""
SAMPLE_FRAMES: tuple[int, ...] = (0, 30, 60, 90, 120)


def _frames_from_mp4(path: Path) -> list[UInt8[ndarray, "h w 3"]]:
    with av.open(str(path)) as container:
        return [frame.to_ndarray(format="rgb24") for frame in container.decode(container.streams.video[0])]


def _frames_from_rrd(entity: str) -> list[UInt8[ndarray, "h w 3"]]:
    from simplecv.rerun_log_utils import read_video_stream_from_rrd

    codec, _, samples = read_video_stream_from_rrd(str(GOLDEN_RRD), f"{entity}/video", "time")
    names: dict[rr.VideoCodec, str] = {rr.VideoCodec.H264: "h264", rr.VideoCodec.H265: "hevc", rr.VideoCodec.AV1: "av1"}
    decoder: VideoCodecContext = cast(VideoCodecContext, av.CodecContext.create(names[codec], "r"))
    frames: list[UInt8[ndarray, "h w 3"]] = []
    # One row per sample, each a list of blobs (one blob per VideoStream sample).
    for row in samples.to_pylist():
        for packet in decoder.parse(b"".join(bytes(blob) for blob in row)):
            frames.extend(frame.to_ndarray(format="rgb24") for frame in decoder.decode(packet))
    # Flush the parser (it holds the last access unit until it sees the next), then the decoder.
    for packet in decoder.parse(None):
        frames.extend(frame.to_ndarray(format="rgb24") for frame in decoder.decode(packet))
    frames.extend(frame.to_ndarray(format="rgb24") for frame in decoder.decode(None))
    return frames


def _ssim(first: UInt8[ndarray, "h w 3"], second: UInt8[ndarray, "h w 3"]) -> float:
    from kornia.metrics import ssim

    def tensor(image: UInt8[ndarray, "h w 3"]) -> Float32[torch.Tensor, "1 3 h w"]:
        return torch.from_numpy(image).permute(2, 0, 1)[None].float() / 255.0

    return float(ssim(tensor(first), tensor(second), window_size=11).mean())


def test_views_match_the_banked_reference(example_run: Path) -> None:
    if not GOLDEN_RRD.is_file():
        pytest.skip(f"needs the banked reference recording {GOLDEN_RRD} (README 'Goldens')")
    rig = read_camera_rig(example_run)
    scores: list[float] = []
    for camera in rig.cameras:
        ours: list[UInt8[ndarray, "h w 3"]] = _frames_from_mp4(example_run / camera.video)
        reference: list[UInt8[ndarray, "h w 3"]] = _frames_from_rrd(camera_entity(camera.camera_id))
        assert len(reference) == len(ours) == 121
        scores.append(float(np.mean([_ssim(ours[index], reference[index]) for index in SAMPLE_FRAMES])))
    print("per-view SSIM", [round(score, 4) for score in scores])
    assert float(np.mean(scores)) >= MEAN_SSIM_MIN, scores
    assert min(scores) >= VIEW_SSIM_MIN, scores
