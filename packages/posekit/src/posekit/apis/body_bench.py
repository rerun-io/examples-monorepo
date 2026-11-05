"""Per-frame speed of SAM 3D Body, from a decoded GPU frame and a person box to the full output (joints, mesh, MHR parameters).

Fixed input: ``assets/examples/hands_video.mp4`` (one person, both hands visible), decoded to the GPU before timing.
SAM3 finds the person in every frame once, before timing; the highest-scoring box per frame is the input (no mask, the
model's default camera). Video decode, detection and model set-up are excluded. Every frame runs alone (batch 1),
CUDA-synchronised, in ``body`` and ``full`` mode.

Stages come from the predictor's CUDA events (``profile_stages``): the backbone and decoder of each pass (body, mirrored
left-hand crop, right-hand crop, keypoint-prompted body decoder), ``mhr`` (forward kinematics outside the decoders: the
hand merge), and ``other`` (CPU crops and batch building, uploads, hand boxes, merge math, device-to-host copies, output
conversion). ``total`` is the event span of the model call; ``wall`` adds the adapter's frame copy and output tensors.
This is the eager baseline: crops run on the CPU with OpenCV, everything else in PyTorch.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import numpy as np
import torch
from jaxtyping import UInt8
from serde import serde, to_dict
from torch import Tensor

from posekit.apis.hands_bench import EXAMPLES, StageSummary, StageTimes
from posekit.models.base import SegmentationPrompts
from posekit.models.sam3_segmenter import Sam3SegmenterConfig
from posekit.models.sam3d_body import Sam3dBodyConfig, Sam3dBodyMode, Sam3dBodyPredictor
from posekit.predictions import BoxDetections

BodyBenchMode: TypeAlias = Sam3dBodyMode
DETECTION_BATCH: int = 8
"""Frames per SAM3 call when finding the input boxes (untimed)."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class BodyBenchResult:
    """One mode's per-frame timing; ``body_bench.json`` holds a list of these."""

    mode: BodyBenchMode
    gpu: str
    frames: int
    people_per_frame: float
    stages: dict[str, StageSummary]


@dataclass(frozen=True, slots=True, kw_only=True)
class BodyBenchConfig:
    """Per-frame SAM 3D Body benchmark on a fixed clip."""

    video_path: Path = EXAMPLES / "hands_video.mp4"
    """Fixed input clip."""
    max_frames: int = 60
    """Frames timed (from the start of the clip)."""
    modes: tuple[BodyBenchMode, ...] = ("body", "full")
    """Inference modes to time."""
    warmup: int = 5
    """Untimed frames first (lazy init; the first full-mode call is also numerically different)."""
    output: Path = Path("/tmp/posekit-body-bench/body_bench.json")
    """Where the results go."""


def person_boxes(frames: UInt8[Tensor, "n h w 3"]) -> BoxDetections:
    """The highest-scoring SAM3 person box of each frame, without masks."""
    segmenter = Sam3SegmenterConfig().setup()
    rows: list[BoxDetections] = []
    for start in range(0, len(frames), DETECTION_BATCH):
        found: BoxDetections = segmenter(frames[start : start + DETECTION_BATCH], SegmentationPrompts(frame_indices=torch.zeros(0, dtype=torch.int64), text="person"))
        for local in range(min(DETECTION_BATCH, len(frames) - start)):
            candidates: Tensor = torch.nonzero(found.frame_indices == local).flatten()
            if len(candidates):
                best: Tensor = candidates[found.scores[candidates].argmax()][None]
                rows.append(BoxDetections(found.xyxy[best], found.scores[best], torch.full((1,), start + local, dtype=torch.int64, device=frames.device)))
    del segmenter
    torch.cuda.empty_cache()
    return BoxDetections(torch.cat([r.xyxy for r in rows]), torch.cat([r.scores for r in rows]), torch.cat([r.frame_indices for r in rows]))


def bench_mode(frames: UInt8[Tensor, "n h w 3"], boxes: BoxDetections, mode: BodyBenchMode, warmup: int) -> tuple[StageTimes, float]:
    """One mode, one frame at a time; the predictor reports its own stage times."""
    predictor: Sam3dBodyPredictor = Sam3dBodyConfig(mode=mode).setup()
    times = StageTimes()
    people: list[int] = []
    for step, index in enumerate(list(range(min(warmup, len(frames)))) + list(range(len(frames)))):
        rows: Tensor = torch.nonzero(boxes.frame_indices == index).flatten()
        frame_boxes = BoxDetections(boxes.xyxy[rows], boxes.scores[rows], torch.zeros(len(rows), dtype=torch.int64, device=frames.device))
        predictor.profile_stages = step >= min(warmup, len(frames))
        predictor.stage_ms.clear()
        torch.cuda.synchronize()
        started: float = time.perf_counter()
        result = predictor(frames[index : index + 1], frame_boxes)
        torch.cuda.synchronize()
        wall_ms: float = (time.perf_counter() - started) * 1000.0
        if not predictor.profile_stages:
            continue
        people.append(result.num_people)
        times.add("wall", wall_ms)
        for stage, values in predictor.stage_ms.items():
            times.add(stage, sum(values))
    del predictor
    torch.cuda.empty_cache()
    return times, float(np.mean(people))


def main(config: BodyBenchConfig) -> list[BodyBenchResult]:
    """Time every configured mode and write the results JSON.

    Returns:
        One result per mode.
    """
    from torchcodec.decoders import VideoDecoder

    decoder = VideoDecoder(config.video_path, dimension_order="NHWC", device="cuda")
    frames: UInt8[Tensor, "n h w 3"] = decoder.get_frames_in_range(0, min(config.max_frames, len(decoder))).data.contiguous()
    results: list[BodyBenchResult] = []
    with torch.inference_mode():
        boxes: BoxDetections = person_boxes(frames)
        for mode in config.modes:
            times, people = bench_mode(frames, boxes, mode, config.warmup)
            result = BodyBenchResult(mode, torch.cuda.get_device_name(), len(times.values.get("wall", [])), people, times.summary())
            results.append(result)
            print(f"[body-bench] {mode}: " + ", ".join(f"{stage} {s.median_ms:.2f}" for stage, s in result.stages.items()) + f" ms/frame (median), {people:.2f} people/frame")
    config.output.parent.mkdir(parents=True, exist_ok=True)
    config.output.write_text(json.dumps([to_dict(result) for result in results], indent=2) + "\n")
    return results
