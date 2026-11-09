"""End-to-end per-frame speed of the posekit hand models, from a decoded GPU frame to joints and meshes.

Fixed input: the SelfCap OAK left clip in ``assets/examples`` (1280x720, 100 frames), decoded
to the GPU before timing; video decode and model set-up are excluded. Every frame runs alone
(batch 1): hand detection, then the 3D hand model on that frame's hands, CUDA-synchronised.

EgoHand runs SelfCap's accelerated configuration (``EgoHandStack.selfcap``): TensorRT
detector, GPU perspective crops, TensorRT bf16 DINOv3 backbone, PyTorch head, full SOMA mesh;
``egohand-fast`` runs ``EgoHandStack.fast``. Each predictor reports its stages in
``stage_ms``: detection (wall), crops (GPU remap), batch assembly, backbone, head and SOMA (CUDA
events), and ``other`` = total minus those (forward glue, the right-hand
mirror, device-to-host copies, upstream result dicts and the adapter's output conversion).
WiLoR is timed as detection + pose on both backends; its TensorRT engines are built for
batch 1 (detector) and ``wilor_pose_batch`` crops here, since the default static batches
(110 frames, 224 crops) are sized for offline throughput and would pad every single frame.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
import torch
from jaxtyping import UInt8
from serde import serde, to_dict
from torch import Tensor

from posekit.cameras import CalibratedCamera, load_calibrated_camera
from posekit.models.base import HandDetector, TopDownHandPose3d
from posekit.models.egohand import EgoHandDetector, EgoHandPredictor, EgoHandStage
from posekit.models.egohand_fast import EgoHandStack, FastEgoHandPredictor
from posekit.predictions import HandDetections, HandPoses3d

EXAMPLES: Path = Path(__file__).resolve().parents[3] / "assets" / "examples"
BenchModel: TypeAlias = Literal["egohand", "egohand-fast", "wilor-tensorrt", "wilor-torch"]


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class StageSummary:
    """One stage over the measured frames."""

    median_ms: float
    mean_ms: float
    p90_ms: float
    frames: int


@dataclass(slots=True)
class StageTimes:
    """Per-frame milliseconds of every stage."""

    values: dict[str, list[float]] = field(default_factory=dict)

    def add(self, stage: str, milliseconds: float) -> None:
        """Record one frame of one stage."""
        self.values.setdefault(stage, []).append(milliseconds)

    def summary(self) -> dict[str, StageSummary]:
        """Median, mean and p90 per stage."""
        return {
            stage: StageSummary(float(np.median(v)), float(np.mean(v)), float(np.quantile(v, 0.9)), len(v)) for stage, v in self.values.items()
        }


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class BenchResult:
    """One model's per-frame timing; ``hands_bench.json`` holds a list of these."""

    model: BenchModel
    gpu: str
    frames: int
    hands_per_frame: float
    stages: dict[str, StageSummary]


@dataclass(frozen=True, slots=True, kw_only=True)
class HandsBenchConfig:
    """Per-frame end-to-end hand benchmark on a fixed clip."""

    video_path: Path = EXAMPLES / "selfcap_oak_left.mp4"
    """Fixed input clip."""
    camera_path: Path = EXAMPLES / "selfcap_oak_left.rrd"
    """Its CalibratedCamera."""
    models: tuple[BenchModel, ...] = ("egohand", "wilor-tensorrt", "wilor-torch")
    """Stacks to time."""
    wilor_pose_batch: int = 4
    """Static crop batch of the WiLoR TensorRT pose engine in this benchmark (more hands run in chunks)."""
    warmup: int = 10
    """Untimed frames first (CUDA graphs, autotune, lazy init)."""
    repeats: int = 1
    """Passes over the clip after warm-up."""
    output: Path = Path("/tmp/posekit-hands-bench/hands_bench.json")
    """Where the results go."""


def _synced_ms(start: float) -> float:
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0


def bench_egohand(frames: UInt8[Tensor, "n h w 3"], camera: CalibratedCamera, warmup: int, repeats: int, configs: EgoHandStack) -> tuple[StageTimes, float]:
    """One EgoHand stack, one frame at a time; its predictor reports its own stage times."""
    detector: EgoHandDetector = configs.detector.setup()
    predictor: EgoHandPredictor | FastEgoHandPredictor = configs.pose.setup()
    predictor.profile_stages = True
    times = StageTimes()
    hands: list[int] = []
    order: list[int] = list(range(len(frames))) * repeats
    for step, index in enumerate(list(range(min(warmup, len(frames)))) + order):
        frame: UInt8[Tensor, "1 h w 3"] = frames[index : index + 1]
        predictor.stage_ms.clear()
        torch.cuda.synchronize()
        started: float = time.perf_counter()
        detections: HandDetections = detector(frame)
        detection_ms: float = _synced_ms(started)
        result: HandPoses3d = predictor(frame, detections, camera)
        total_ms: float = _synced_ms(started)
        if step < min(warmup, len(frames)):
            continue
        assert result.vertices is not None or result.num_hands == 0
        hands.append(result.num_hands)
        stage: dict[EgoHandStage, float] = {name: sum(values) for name, values in predictor.stage_ms.items()}
        parts: dict[str, float] = {
            "detection": detection_ms,
            "crops": stage.get("preprocessing", 0.0),
            "batch_assembly": stage.get("batch_assembly", 0.0),
            "backbone": stage.get("backbone", 0.0),
            "head": stage.get("head", 0.0),
            "soma": stage.get("soma", 0.0),
        }
        times.add("total", total_ms)
        for name, milliseconds in parts.items():
            times.add(name, milliseconds)
        times.add("other", total_ms - sum(parts.values()))
    return times, float(np.mean(hands))


def bench_pipeline(frames: UInt8[Tensor, "n h w 3"], detector: HandDetector, pose: TopDownHandPose3d, warmup: int, repeats: int) -> tuple[StageTimes, float]:
    """A detector + pose stack, one frame at a time, timed per stage."""
    times = StageTimes()
    hands: list[int] = []
    order: list[int] = list(range(len(frames))) * repeats
    for step, index in enumerate(list(range(min(warmup, len(frames)))) + order):
        frame: UInt8[Tensor, "1 h w 3"] = frames[index : index + 1]
        torch.cuda.synchronize()
        started: float = time.perf_counter()
        detections: HandDetections = detector(frame)
        detection_ms: float = _synced_ms(started)
        result: HandPoses3d = pose(frame, detections)
        total_ms: float = _synced_ms(started)
        if step < min(warmup, len(frames)):
            continue
        hands.append(result.num_hands)
        times.add("total", total_ms)
        times.add("detection", detection_ms)
        times.add("pose", total_ms - detection_ms)
    return times, float(np.mean(hands))


def main(config: HandsBenchConfig) -> list[BenchResult]:
    """Time every configured stack and write the results JSON.

    Returns:
        One result per model.
    """
    from torchcodec.decoders import VideoDecoder
    from trtkit import TensorRtBackendConfig, TorchBackendConfig

    from posekit.models.wilor import WilorHandDetectorConfig, WilorHandPoseConfig

    egohand_stacks: dict[BenchModel, EgoHandStack] = {"egohand": EgoHandStack.selfcap(), "egohand-fast": EgoHandStack.fast()}
    decoder = VideoDecoder(config.video_path, dimension_order="NHWC", device="cuda")
    frames: UInt8[Tensor, "n h w 3"] = decoder.get_frames_in_range(0, len(decoder)).data.contiguous()
    camera: CalibratedCamera = load_calibrated_camera(config.camera_path)
    results: list[BenchResult] = []
    for model in config.models:
        with torch.inference_mode():
            if model in egohand_stacks:
                times, hands = bench_egohand(frames, camera, config.warmup, config.repeats, egohand_stacks[model])
            else:
                backend = TensorRtBackendConfig() if model == "wilor-tensorrt" else TorchBackendConfig(autocast="fp16")
                detector = WilorHandDetectorConfig(backend=backend, static_batch_size=1).setup()
                pose = WilorHandPoseConfig(backend=backend, static_batch_size=config.wilor_pose_batch).setup()
                times, hands = bench_pipeline(frames, detector, pose, config.warmup, config.repeats)
        result = BenchResult(model, torch.cuda.get_device_name(), len(frames) * config.repeats, hands, times.summary())
        results.append(result)
        print(f"[hands-bench] {model}: " + ", ".join(f"{stage} {s.median_ms:.2f}" for stage, s in result.stages.items()) + f" ms/frame (median), {hands:.2f} hands/frame")
        torch.cuda.empty_cache()
    config.output.parent.mkdir(parents=True, exist_ok=True)
    config.output.write_text(json.dumps([to_dict(result) for result in results], indent=2) + "\n")
    return results
