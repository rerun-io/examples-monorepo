"""Accuracy and batch-one latency of an EgoHand stack on a fixed, locally cached frame set.

Each cached segment is a camera-scoped native RRD from the catalog (video, calibration, GT):
``<cache>/<dataset>/eval/*.rrd``. Every frame first runs once untimed and is scored against GT;
``repeats`` timed passes follow, each timing one synchronized detector + full reconstruction call
per frame. With ``--baseline`` (an earlier ``results.json``, e.g. of ``--stack egohand``), the run
fails unless every dataset passes all nine ≤5 % accuracy bands.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from jaxtyping import Bool, Float32, UInt8
from numpy import ndarray
from serde import serde
from serde.json import from_json, to_json
from torch import Tensor

from posekit.apis.hands_eval import decode_samples, joint_box
from posekit.eval.cached_hand_gt import load_cached_hand_gt
from posekit.eval.hand_accuracy_bands import error_quantiles, within_accuracy_band
from posekit.eval.hand_gt import SIDES, HandGtSegment
from posekit.eval.hand_metrics import box_iou
from posekit.models.egohand import EgoHandAssetPaths, EgoHandConfig, EgoHandDetector, EgoHandDetectorConfig, EgoHandPredictor, asset_paths
from posekit.models.egohand_fast import EgoHandFastConfig, FastEgoHandPredictor, fast_egohand_configs
from posekit.predictions import HandDetections, HandPoses3d

BANDED_METRICS: dict[str, bool] = {
    "recall": True,
    "mean_iou": True,
    "side_accuracy": True,
    "epe_2d_median_px": False,
    "epe_2d_p95_px": False,
    "absolute_3d_median_mm": False,
    "absolute_3d_p95_mm": False,
    "relative_3d_median_mm": False,
    "relative_3d_p95_mm": False,
}
"""The nine accuracy bands per dataset, each mapped to whether higher is better."""


@dataclass(frozen=True, slots=True)
class Config:
    """One run over the fixed cache, without catalog access."""

    cache: Path
    """Native camera-scoped RRD cache: ``<cache>/<dataset>/eval/*.rrd``."""
    output: Path
    """Directory for ``results.json``."""
    stack: Literal["egohand", "egohand-fast", "egohand-fast-bf16"] = "egohand-fast"
    """``egohand``: the unquantized reference classes; ``egohand-fast``/``egohand-fast-bf16``: ``fast_egohand_configs``."""
    assets: Path | None = None
    """Local asset tree; None uses the pinned Hugging Face downloads."""
    datasets: tuple[str, ...] = ("dataforge-umetrack", "dataforge-hot3d-quest3", "dataforge-show3d", "gsi-six-view", "dataforge-gen-humanego-subset")
    """Cached datasets to run."""
    max_frames: int = 100
    """Uniformly chosen visible-GT frames per segment."""
    measure_timing: bool = True
    """Run the timed passes; false records accuracy only."""
    repeats: int = 2
    """Timed passes after the untimed scoring pass."""
    baseline: Path | None = None
    """Reference ``results.json``; set, every dataset must pass all nine accuracy bands against it."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class Result:
    """One dataset's latency and ground-truth accuracy."""

    model: str
    """The stack name."""
    gpu: str
    """CUDA device name reported by PyTorch."""
    dataset: str
    """Cached dataset name."""
    frames: int
    """Number of unique frames; repeats do not increase it."""
    hands_per_frame: float
    """Mean reconstructed hands per frame."""
    gt_hands: int
    """Visible GT hands: a visible wrist and at least 15 joints."""
    matched_hands: int
    """GT hands matched at IoU >= 0.3 with the correct side and a pose."""
    recall: float | None
    """Side-correct matched hands divided by visible GT hands."""
    mean_iou: float | None
    """Best detector IoU per GT hand, including zero for misses."""
    side_accuracy: float | None
    """Correct sides among one-to-one IoU matches of either side."""
    epe_2d_median_px: float | None
    """Pooled visible-joint median error in source pixels."""
    epe_2d_p95_px: float | None
    """Pooled visible-joint 95th percentile error in source pixels."""
    absolute_3d_median_mm: float | None
    """Median camera-space joint error in millimeters."""
    absolute_3d_p95_mm: float | None
    """95th percentile camera-space joint error in millimeters."""
    relative_3d_median_mm: float | None
    """Median wrist-relative joint error in millimeters."""
    relative_3d_p95_mm: float | None
    """95th percentile wrist-relative joint error in millimeters."""
    mean_ms: float | None
    """Mean end-to-end latency per frame over the timed passes; None without timing."""
    median_ms: float | None
    """Median end-to-end latency per frame."""
    p95_ms: float | None
    """95th percentile end-to-end latency per frame."""


@dataclass(slots=True)
class DatasetScore:
    """Ground-truth accuracy of one dataset, accumulated over its fixed frames."""

    frames: int = 0
    hands: int = 0
    gt_hands: int = 0
    matched_hands: int = 0
    side_correct: int = 0
    any_side_matches: int = 0
    ious: list[float] = field(default_factory=list)
    pixels: list[float] = field(default_factory=list)
    absolute_mm: list[float] = field(default_factory=list)
    relative_mm: list[float] = field(default_factory=list)

    def add_frame(self, segment: HandGtSegment, index: int, detections: HandDetections, hands: HandPoses3d, num_hands: int) -> None:
        """Match visible GT hands one-to-one at IoU >= 0.3 and pool the side-correct joint errors."""
        assert hands.keypoints.xyz_camera is not None
        boxes: Float32[ndarray, "n 4"] = detections.boxes.xyxy_numpy()
        pred_uv: Float32[ndarray, "m 21 2"] = hands.keypoints.xy.cpu().numpy()
        pred_xyz: Float32[ndarray, "m 21 3"] = hands.keypoints.xyz_camera.cpu().numpy()
        self.frames += 1
        self.hands += num_hands
        valid: Bool[ndarray, "2 21"] = segment.valid[index]
        gt_sides: list[int] = [s for s in range(2) if valid[s, 0] and valid[s].sum() >= 15]
        self.gt_hands += len(gt_sides)
        gt_boxes: dict[int, Float32[ndarray, "4"]] = {s: joint_box(segment.uv[index, s], valid[s], 1.0) for s in gt_sides}
        pairs: list[tuple[float, int, int]] = sorted(((box_iou(box, gt_boxes[s]), s, j) for s in gt_sides for j, box in enumerate(boxes)), reverse=True)
        used_gt: set[int] = set()
        used_pred: set[int] = set()
        matches: list[int] = []
        for iou, s, j in pairs:
            if iou < 0.3 or s in used_gt or j in used_pred:
                continue
            used_gt.add(s)
            used_pred.add(j)
            self.any_side_matches += 1
            if detections.sides[j] == SIDES[s]:
                self.side_correct += 1
                matches.append(s)
        self.ious.extend(max((box_iou(b, gt_boxes[s]) for b in boxes), default=0.0) for s in gt_sides)
        for s in matches:
            if SIDES[s] not in hands.sides:
                continue
            row: int = hands.sides.index(SIDES[s])
            self.matched_hands += 1
            mask: Bool[ndarray, "21"] = valid[s]
            gt: Float32[ndarray, "21 3"] = segment.xyz_camera[index, s]
            self.pixels.extend(np.linalg.norm(pred_uv[row] - segment.uv[index, s], axis=-1)[mask].tolist())
            self.absolute_mm.extend((1000.0 * np.linalg.norm(pred_xyz[row] - gt, axis=-1)[mask]).tolist())
            self.relative_mm.extend((1000.0 * np.linalg.norm((pred_xyz[row] - pred_xyz[row, 0]) - (gt - gt[0]), axis=-1)[mask]).tolist())


def failed_bands(baseline: list[Result], results: list[Result]) -> list[str]:
    """Every dataset metric outside its ≤5 % relative band; a missing dataset fails."""
    candidates: dict[str, Result] = {row.dataset: row for row in results}
    failed: list[str] = []
    for reference in baseline:
        candidate: Result | None = candidates.get(reference.dataset)
        if candidate is None:
            failed.append(f"{reference.dataset}: missing")
            continue
        failed += [
            f"{reference.dataset}: {name}"
            for name, higher_is_better in BANDED_METRICS.items()
            if not within_accuracy_band(getattr(reference, name), getattr(candidate, name), higher_is_better=higher_is_better)
        ]
    return failed


def main(config: Config) -> None:
    """Score every fixed frame once, time ``repeats`` passes, write ``results.json`` and check the bands."""
    config.output.mkdir(parents=True, exist_ok=True)
    if config.stack == "egohand":
        paths: EgoHandAssetPaths = asset_paths(config.assets)
        detector_config: EgoHandDetectorConfig = EgoHandDetectorConfig(weights=paths.detector)
        pose_config: EgoHandConfig | EgoHandFastConfig = EgoHandConfig(weights=paths.weights, soma_assets=paths.soma)
    else:
        detector_config, pose_config = fast_egohand_configs(config.assets, backbone_precision="bf16" if config.stack == "egohand-fast-bf16" else "nvfp4")
    detector: EgoHandDetector = detector_config.setup()
    pose: EgoHandPredictor | FastEgoHandPredictor = pose_config.setup()
    print("GPU", torch.cuda.get_device_name(), "STACK", config.stack, flush=True)
    results: list[Result] = []
    with torch.inference_mode():
        for dataset in config.datasets:
            score: DatasetScore = DatasetScore()
            latencies: list[float] = []
            for path in sorted((config.cache / dataset / "eval").glob("*.rrd")):
                segment: HandGtSegment = load_cached_hand_gt(path, max_frames=config.max_frames)
                frames: UInt8[Tensor, "f h w 3"] = decode_samples(segment)
                for index, frame in enumerate(frames):
                    detections: HandDetections = detector(frame[None])
                    reconstruction = pose(frame[None], detections, segment.camera)
                    score.add_frame(segment, index, detections, reconstruction.as_hand21(), reconstruction.num_hands)
                for _ in range(config.repeats if config.measure_timing else 0):
                    for frame in frames:
                        torch.cuda.synchronize()
                        start: float = time.perf_counter()
                        pose(frame[None], detector(frame[None]), segment.camera)
                        torch.cuda.synchronize()
                        latencies.append((time.perf_counter() - start) * 1000.0)
                print("SEGMENT", dataset, path.stem, len(frames), flush=True)
                del frames
                torch.cuda.empty_cache()
            epe_2d_median, epe_2d_p95 = error_quantiles(score.pixels)
            absolute_median, absolute_p95 = error_quantiles(score.absolute_mm)
            relative_median, relative_p95 = error_quantiles(score.relative_mm)
            result: Result = Result(
                model=config.stack,
                gpu=torch.cuda.get_device_name(),
                dataset=dataset,
                frames=score.frames,
                hands_per_frame=score.hands / score.frames,
                gt_hands=score.gt_hands,
                matched_hands=score.matched_hands,
                recall=score.matched_hands / score.gt_hands if score.gt_hands else None,
                mean_iou=float(np.mean(score.ious)) if score.ious else None,
                side_accuracy=score.side_correct / score.any_side_matches if score.any_side_matches else None,
                epe_2d_median_px=epe_2d_median,
                epe_2d_p95_px=epe_2d_p95,
                absolute_3d_median_mm=absolute_median,
                absolute_3d_p95_mm=absolute_p95,
                relative_3d_median_mm=relative_median,
                relative_3d_p95_mm=relative_p95,
                mean_ms=float(np.mean(latencies)) if latencies else None,
                median_ms=float(np.median(latencies)) if latencies else None,
                p95_ms=float(np.quantile(latencies, 0.95)) if latencies else None,
            )
            results.append(result)
            (config.output / "results.json").write_text(to_json(results) + "\n")
            print("RESULT", to_json(result), flush=True)
    if config.baseline is not None:
        failed: list[str] = failed_bands(from_json(list[Result], config.baseline.read_text()), results)
        print("BANDS", f"{9 * len(results) - len(failed)}/{9 * len(results)}", "passed", flush=True)
        if failed:
            raise RuntimeError("Accuracy bands failed: " + "; ".join(failed))
