"""Accuracy and batch-one latency of the EgoHand detector + pose stack on a fixed, locally cached frame set.

Each cached segment is a camera-scoped native RRD from the catalog (video, calibration, GT).
Every frame first runs once untimed: that pass scores against GT and logs Rerun evidence.
``repeats`` timed passes follow; each times one synchronized detector + full reconstruction
call per frame. With ``--baseline`` (an earlier ``results.json``, e.g. of ``--stack egohand``),
the run fails unless every dataset passes all nine ≤5 % accuracy bands.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
from jaxtyping import Bool, Float32, UInt8
from numpy import ndarray
from serde import serde
from serde.json import from_json, to_json
from torch import Tensor

from posekit.apis.hands_eval import eval_model, joint_box, log_context
from posekit.eval.cached_hand_gt import load_cached_hand_gt
from posekit.eval.hand_accuracy_bands import error_quantiles, within_accuracy_band
from posekit.eval.hand_gt import SIDES, HandGtSegment, decode_catalog_frames
from posekit.eval.hand_metrics import box_iou
from posekit.models.egohand import EgoHandConfig, EgoHandDetector, EgoHandPredictor, EgoHandReconstruction
from posekit.models.egohand_fast import EGOHAND_STACKS, EgoHandFastConfig, EgoHandStack, EgoHandStackName, FastEgoHandPredictor
from posekit.predictions import HandDetections, HandPoses3d, HandSide

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
    """Directory for ``results.json`` and one Rerun recording per dataset."""
    stack: EgoHandStackName = "egohand-fast"
    """EgoHand stack: ``egohand`` is main's unquantized reference, ``egohand-fast`` the accepted fast stack (NVFP4/FP8,
    Blackwell) and ``egohand-fast-bf16`` the same stack on GPUs without NVFP4."""
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
    """The stack name (results written before the stack registry say ``default`` or ``fast-<precision>``)."""
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


@dataclass(frozen=True, slots=True)
class FramePrediction:
    """One frame's detector boxes and 21-joint hands, copied to the host for scoring and evidence."""

    boxes: Float32[ndarray, "n 4"]
    """Detector boxes in source pixels."""
    box_sides: tuple[HandSide, ...]
    """Detector side of each box."""
    sides: tuple[HandSide, ...]
    """Side of each reconstructed hand."""
    uv: Float32[ndarray, "m 21 2"]
    """Reconstructed joints in source pixels."""
    xyz: Float32[ndarray, "m 21 3"]
    """Reconstructed joints in camera metres."""
    num_hands: int
    """Reconstructed hands, as the reconstruction counts them."""


def frame_prediction(detections: HandDetections, reconstruction: EgoHandReconstruction) -> FramePrediction:
    """Copy the outputs the accuracy protocol reads; the timed passes never call this."""
    hands: HandPoses3d = reconstruction.as_hand21()
    assert hands.keypoints.xyz_camera is not None
    return FramePrediction(
        detections.boxes.xyxy_numpy(),
        detections.sides,
        hands.sides,
        hands.keypoints.xy.cpu().numpy(),
        hands.keypoints.xyz_camera.cpu().numpy(),
        reconstruction.num_hands,
    )


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

    def add_frame(self, segment: HandGtSegment, index: int, prediction: FramePrediction) -> dict[HandSide, tuple[float, float]]:
        """Match visible GT hands one-to-one at IoU >= 0.3 and pool the side-correct joint errors.

        Returns the median pixel and millimetre error of each scored side.
        """
        self.frames += 1
        self.hands += prediction.num_hands
        valid: Bool[ndarray, "2 21"] = segment.valid[index]
        gt_sides: list[int] = [s for s in range(2) if valid[s, 0] and valid[s].sum() >= 15]
        self.gt_hands += len(gt_sides)
        gt_boxes: dict[int, Float32[ndarray, "4"]] = {s: joint_box(segment.uv[index, s], valid[s], 1.0) for s in gt_sides}
        pairs: list[tuple[float, int, int]] = sorted(
            ((box_iou(box, gt_boxes[s]), s, j) for s in gt_sides for j, box in enumerate(prediction.boxes)), reverse=True
        )
        used_gt: set[int] = set()
        used_pred: set[int] = set()
        matches: dict[int, int] = {}
        for iou, s, j in pairs:
            if iou < 0.3 or s in used_gt or j in used_pred:
                continue
            used_gt.add(s)
            used_pred.add(j)
            self.any_side_matches += 1
            if prediction.box_sides[j] == SIDES[s]:
                self.side_correct += 1
                matches[s] = j
        self.ious.extend(max((box_iou(b, gt_boxes[s]) for b in prediction.boxes), default=0.0) for s in gt_sides)
        side_errors: dict[HandSide, tuple[float, float]] = {}
        for s in matches:
            side: HandSide = SIDES[s]
            if side not in prediction.sides:
                continue
            row: int = prediction.sides.index(side)
            self.matched_hands += 1
            mask: Bool[ndarray, "21"] = valid[s]
            gt: Float32[ndarray, "21 3"] = segment.xyz_camera[index, s]
            pixels: Float32[ndarray, "v"] = np.linalg.norm(prediction.uv[row] - segment.uv[index, s], axis=-1)[mask]
            absolute: Float32[ndarray, "v"] = 1000.0 * np.linalg.norm(prediction.xyz[row] - gt, axis=-1)[mask]
            relative: Float32[ndarray, "v"] = 1000.0 * np.linalg.norm((prediction.xyz[row] - prediction.xyz[row, 0]) - (gt - gt[0]), axis=-1)[mask]
            self.pixels.extend(pixels.tolist())
            self.absolute_mm.extend(absolute.tolist())
            self.relative_mm.extend(relative.tolist())
            side_errors[side] = (float(np.median(pixels)), float(np.median(absolute)))
        return side_errors


def log_frame_evidence(
    segment: HandGtSegment, index: int, frame: UInt8[Tensor, "h w 3"], prediction: FramePrediction, *, image: bool, left_class: int
) -> None:
    """Log the source boxes and GT vs predicted joints (in the stack's ``log_context`` class) in 2D and 3D; ``image`` adds the decoded frame."""
    rr.log("source", rr.AnyValues(segment=segment.segment_id, index=int(segment.sample_indices[index])))
    rr.log("camera/image/detection", rr.Boxes2D(array=prediction.boxes, array_format=rr.Box2DFormat.XYXY, labels=list(prediction.box_sides)))
    rr.log("camera/image", rr.Image(frame.cpu().numpy()).compress(jpeg_quality=90) if image else rr.Clear(recursive=False))
    valid: Bool[ndarray, "2 21"] = segment.valid[index]
    for s, side in enumerate(SIDES):
        uv: Float32[ndarray, "21 2"] = np.where(valid[s, :, None], segment.uv[index, s], np.nan)
        rr.log(f"camera/image/gt/{side}", rr.Points2D(uv, class_ids=0, keypoint_ids=np.arange(21), radii=2.0, show_labels=False))
        rr.log(
            f"camera/gt/{side}", rr.Points3D(segment.xyz_camera[index, s], class_ids=0, keypoint_ids=np.arange(21), radii=0.002, show_labels=False)
        )
        if side in prediction.sides:
            row: int = prediction.sides.index(side)
            rr.log(
                f"camera/image/pred/{side}",
                rr.Points2D(prediction.uv[row], class_ids=left_class + s, keypoint_ids=np.arange(21), radii=2.0, show_labels=False),
            )
            rr.log(
                f"camera/pred/{side}", rr.Points3D(prediction.xyz[row], class_ids=left_class + s, keypoint_ids=np.arange(21), radii=0.002, show_labels=False)
            )
        else:
            rr.log(f"camera/image/pred/{side}", rr.Clear(recursive=True))
            rr.log(f"camera/pred/{side}", rr.Clear(recursive=True))


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
    stack: EgoHandStack[EgoHandConfig] | EgoHandStack[EgoHandFastConfig] = EGOHAND_STACKS[config.stack](config.assets)
    label: str = config.stack
    left_class: int = eval_model(config.stack).left_class
    detector: EgoHandDetector = stack.detector.setup()
    pose: EgoHandPredictor | FastEgoHandPredictor = stack.pose.setup()
    print("GPU", torch.cuda.get_device_name(), "MODEL", label, flush=True)
    results: list[Result] = []
    with torch.inference_mode():
        for dataset in config.datasets:
            score: DatasetScore = DatasetScore()
            latencies: list[float] = []
            rr.init(f"{label}-{dataset}", spawn=False)
            rr.save(config.output / f"{dataset}.rrd")
            recording: rr.RecordingStream | None = rr.get_global_data_recording()
            assert recording is not None
            log_context()
            rr.log("camera", rr.ViewCoordinates.RDF, static=True)
            offset: int = 0
            for path in sorted((config.cache / dataset / "eval").glob("*.rrd")):
                segment: HandGtSegment = load_cached_hand_gt(path, max_frames=config.max_frames)
                frames: UInt8[Tensor, "f h w 3"] = decode_catalog_frames(segment.video, segment.sample_indices.tolist())
                image_frames: set[int] = set(np.linspace(0, len(frames) - 1, 10, dtype=int).tolist())
                for index, frame in enumerate(frames):
                    detections: HandDetections = detector(frame[None])
                    prediction: FramePrediction = frame_prediction(detections, pose(frame[None], detections, segment.camera))
                    side_errors: dict[HandSide, tuple[float, float]] = score.add_frame(segment, index, prediction)
                    rr.set_time("eval_frame", sequence=offset + index)
                    for side in SIDES:
                        if side in side_errors:
                            rr.log(f"error/{side}/pixels", rr.Scalars(side_errors[side][0]))
                            rr.log(f"error/{side}/millimeters", rr.Scalars(side_errors[side][1]))
                        else:
                            rr.log(f"error/{side}", rr.Clear(recursive=True))
                    log_frame_evidence(segment, index, frame, prediction, image=index in image_frames, left_class=left_class)
                if config.measure_timing:
                    recording.flush()
                    for _ in range(config.repeats):
                        for frame in frames:
                            torch.cuda.synchronize()
                            start: float = time.perf_counter()
                            pose(frame[None], detector(frame[None]), segment.camera)
                            torch.cuda.synchronize()
                            latencies.append((time.perf_counter() - start) * 1000.0)
                    rr.send_columns(
                        "timing/total",
                        indexes=[rr.TimeColumn("eval_frame", sequence=np.tile(np.arange(offset, offset + len(frames)), config.repeats))],
                        columns=rr.Scalars.columns(scalars=latencies[-config.repeats * len(frames) :]),
                    )
                print("SEGMENT", dataset, path.stem, len(frames), flush=True)
                offset += len(frames)
                del frames
                torch.cuda.empty_cache()
            epe_2d_median, epe_2d_p95 = error_quantiles(score.pixels)
            absolute_median, absolute_p95 = error_quantiles(score.absolute_mm)
            relative_median, relative_p95 = error_quantiles(score.relative_mm)
            result: Result = Result(
                model=label,
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
            rr.send_blueprint(
                rrb.Blueprint(
                    rrb.Horizontal(
                        rrb.Spatial2DView(origin="camera/image", name="Source: GT white, prediction green/pink"),
                        rrb.Spatial3DView(origin="camera", contents=["camera/gt/**", "camera/pred/**"], name="Metric hands"),
                        rrb.Vertical(
                            rrb.TimeSeriesView(origin="error", name="Matched-hand error"), rrb.TimeSeriesView(origin="timing", name="Latency ms")
                        ),
                        column_shares=[2, 2, 1],
                    ),
                    collapse_panels=True,
                )
            )
            recording.flush()
            print("RESULT", to_json(result), flush=True)
    if config.baseline is not None:
        failed: list[str] = failed_bands(from_json(list[Result], config.baseline.read_text()), results)
        print("BANDS", f"{9 * len(results) - len(failed)}/{9 * len(results)}", "passed", flush=True)
        if failed:
            raise RuntimeError("Accuracy bands failed: " + "; ".join(failed))
