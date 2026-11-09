"""Measure detector plus full EgoHand reconstruction on fixed, locally cached catalog frames."""

import time
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
from jaxtyping import Bool, Float32, UInt8
from numpy import ndarray
from serde import serde
from serde.json import to_json
from torch import Tensor
from trtkit import TensorRtBackendConfig

from posekit.apis.hands_bench import StageTimes
from posekit.apis.hands_eval import decode_samples, joint_box, log_context
from posekit.eval.cached_hand_gt import load_cached_hand_gt
from posekit.eval.gpu_timing import GpuActivityMonitor, require_idle_gpu
from posekit.eval.hand_accuracy_bands import error_quantiles
from posekit.eval.hand_evidence import log_hand_intermediates
from posekit.eval.hand_gt import SIDES, HandGtSegment
from posekit.eval.hand_metrics import box_iou
from posekit.models.egohand import (
    DetectorFp8Scope,
    EgoHandAssetPaths,
    EgoHandConfig,
    EgoHandDetector,
    EgoHandDetectorConfig,
    EgoHandPredictor,
    EgoHandReconstruction,
    asset_paths,
    selfcap_egohand_configs,
)
from posekit.models.egohand_fast import (
    EgoHandBackbonePrecision,
    EgoHandFastConfig,
    FastEgoHandPredictor,
    catalog_egohand_configs,
    fast_egohand_configs,
)
from posekit.predictions import HandDetections, HandPoses3d, HandSide


@dataclass(frozen=True, slots=True)
class Config:
    """One reproducible precision experiment, without catalog access."""

    cache: Path = Path.home() / "egohand-3060/cache"
    """Native camera-scoped RRD cache."""
    output: Path = Path.home() / "egohand-3060/results/default"
    """Results and Rerun evidence directory."""
    model: Literal["default", "selfcap", "fast", "catalog"] = "default"
    """Main defaults, SelfCap, manual fast stack, or the measured catalog preset."""
    precision: EgoHandBackbonePrecision = "nvfp4"
    """Fast backbone precision; ignored by the untouched default path."""
    calibration: Path | None = None
    """Disjoint activation calibration for FP8/NVFP4."""
    fp8_prefix_blocks: int = 0
    """Initial NVFP4 backbone blocks retained in FP8."""
    fp8_tail_blocks: int = 8
    """NVFP4 backbone final blocks retained in FP8; all-FP8 precision always uses 32."""
    detector_mode: Literal["default", "fp32", "fp16", "fp8"] = "fp16"
    """Fast-stack detector variant for isolating accuracy changes."""
    gpu_preprocessing: bool = True
    """Fast detector resize on CUDA; false retains PIL resize."""
    detector_calibration: Path | None = None
    """Enable calibrated FP8 detector convolutions for the fast stack."""
    detector_fp8_scope: DetectorFp8Scope = "all"
    """Quantize all, residual backbone, final residual stage, or encoder feature projections."""
    assets: Path | None = None
    """Local asset tree; None uses the pinned HF cache."""
    soma: Path | None = None
    """Local SOMA directory for the default stack."""
    datasets: tuple[str, ...] = ("dataforge-umetrack", "dataforge-hot3d-quest3", "dataforge-show3d", "gsi-six-view", "dataforge-gen-humanego-subset")
    """Fixed cached datasets; unsupported cameras must be handled explicitly."""
    max_frames: int = 100
    """Uniformly chosen visible-GT frames per segment."""
    measure_timing: bool = True
    """Require an idle GPU and record latency; false records accuracy only."""
    allow_idle_residents: bool = False
    """Authorize idle GPU residents, with <=5% before the run and no per-process activity increase."""
    profile_breakdown: bool = True
    """Run a separate instrumented pass; primary latency always has profiling disabled."""
    repeats: int = 2
    """Timed passes after one complete untimed pass."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class Result:
    """One dataset's measured latency and ground-truth accuracy."""

    model: str
    """Precision and detector variant used by this experiment."""
    gpu: str
    """CUDA device name reported by PyTorch."""
    dataset: str
    """Native catalog dataset name."""
    frames: int
    """Number of unique held-out frames; repeats do not increase it."""
    hands_per_frame: float
    """Mean reconstructed hands per unique frame."""
    gt_hands: int
    """Visible GT hands admitted by the fixed selection rule."""
    matched_hands: int
    """GT hands matched at IoU >=0.3 with the correct side and a pose."""
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
    stages_mean_ms: dict[str, float]
    """Mean latency; total is unprofiled, other stages come from a separate pass."""
    stages_median_ms: dict[str, float]
    """Median latency under the same stage protocol."""
    total_p95_ms: float | None
    """95th percentile unprofiled end-to-end latency; None for accuracy-only runs."""
    timing_method: Literal["profiled", "unprofiled", "accuracy_only"] = "profiled"
    """Legacy records used profiled latency; new speed claims use unprofiled end-to-end time."""


def resolve_configs(config: Config) -> tuple[EgoHandDetectorConfig, EgoHandConfig | EgoHandFastConfig, str]:
    """The experiment's detector and pose configurations, and the label its results carry."""
    paths: EgoHandAssetPaths = asset_paths(config.assets)
    if config.model == "default":
        return EgoHandDetectorConfig(weights=paths.detector), EgoHandConfig(weights=paths.weights, soma_assets=config.soma or paths.soma), "default"
    if config.model == "catalog":
        detector_config, pose_config = catalog_egohand_configs(config.assets, backbone_precision=config.precision)
        return detector_config, pose_config, f"catalog-{config.precision}"
    if config.model == "selfcap":
        detector_config, pose_config = selfcap_egohand_configs(config.assets)
        return detector_config, pose_config, "selfcap"
    detector_config, pose_config = fast_egohand_configs(config.assets)
    assert isinstance(detector_config.backend, TensorRtBackendConfig)
    assert isinstance(pose_config, EgoHandFastConfig)
    if config.detector_mode == "fp8" and config.detector_calibration is None:
        raise ValueError("FP8 detector requires --detector-calibration.")
    if config.detector_mode == "default":
        detector_config = EgoHandDetectorConfig(weights=paths.detector)
    else:
        detector_config = replace(
            detector_config,
            fp8_calibration=config.detector_calibration if config.detector_mode == "fp8" else None,
            fp8_scope=config.detector_fp8_scope,
            fp16_trunk=config.detector_mode == "fp16",
            gpu_preprocessing=config.gpu_preprocessing,
            backend=replace(detector_config.backend, max_batch_size=1, opt_batch_size=1),
        )
    pose_config = replace(
        pose_config,
        backbone_precision=config.precision,
        fp8_tail_blocks=config.fp8_tail_blocks,
        fp8_prefix_blocks=config.fp8_prefix_blocks,
        calibration=config.calibration,
        batch_size=2,
        backbone_backend=replace(pose_config.backbone_backend, max_batch_size=2, opt_batch_size=2),
    )
    experiment: str = f"fast-{config.precision}-{config.detector_mode}"
    if config.precision == "nvfp4":
        experiment += f"-prefix{config.fp8_prefix_blocks}-tail{config.fp8_tail_blocks}"
    if config.detector_mode == "fp8":
        experiment += f"-detector-{config.detector_fp8_scope}"
    return detector_config, pose_config, experiment


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
    """Copy the outputs the accuracy protocol reads; the timed pass never calls this."""
    prediction: HandPoses3d = reconstruction.as_hand21()
    assert prediction.keypoints.xyz_camera is not None
    return FramePrediction(
        detections.boxes.xyxy_numpy(),
        detections.sides,
        prediction.sides,
        prediction.keypoints.xy.cpu().numpy(),
        prediction.keypoints.xyz_camera.cpu().numpy(),
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

        A GT hand counts when its wrist and at least 15 joints are visible. Returns the
        median pixel and millimetre error of each scored side.
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
    segment: HandGtSegment,
    index: int,
    batch: UInt8[Tensor, "1 h w 3"],
    detections: HandDetections,
    reconstruction: EgoHandReconstruction,
    prediction: FramePrediction,
    pose: EgoHandPredictor | FastEgoHandPredictor,
    *,
    full: bool,
) -> None:
    """Log the source, boxes, GT and predicted joints; ``full`` adds the image, meshes, crops and parameters."""
    rr.log("source", rr.AnyValues(segment=segment.segment_id, index=int(segment.sample_indices[index])))
    rr.log("camera/image/detection", rr.Boxes2D(array=prediction.boxes, array_format=rr.Box2DFormat.XYXY, labels=list(prediction.box_sides)))
    if full:
        rr.log("camera/image", rr.Image(batch[0].cpu().numpy()).compress(jpeg_quality=90))
        log_hand_intermediates(batch, detections, reconstruction, segment.camera, pose)
    else:
        rr.log("camera/image", rr.Clear(recursive=False))
        rr.log("camera/mesh", rr.Clear(recursive=True))
        rr.log("crops", rr.Clear(recursive=True))
        rr.log("parameters", rr.Clear(recursive=True))
    valid: Bool[ndarray, "2 21"] = segment.valid[index]
    for s, side in enumerate(SIDES):
        rr.log(
            f"camera/image/gt/{side}",
            rr.Points2D(
                np.where(valid[s, :, None], segment.uv[index, s], np.nan), class_ids=0, keypoint_ids=np.arange(21), radii=2.0, show_labels=False
            ),
        )
        rr.log(
            f"camera/gt/{side}",
            rr.Points3D(segment.xyz_camera[index, s], class_ids=0, keypoint_ids=np.arange(21), radii=0.002, show_labels=False),
        )
        if side in prediction.sides:
            row: int = prediction.sides.index(side)
            rr.log(
                f"camera/image/pred/{side}",
                rr.Points2D(prediction.uv[row], class_ids=4 + s, keypoint_ids=np.arange(21), radii=2.0, show_labels=False),
            )
            rr.log(
                f"camera/pred/{side}",
                rr.Points3D(prediction.xyz[row], class_ids=4 + s, keypoint_ids=np.arange(21), radii=0.002, show_labels=False),
            )
        else:
            rr.log(f"camera/image/pred/{side}", rr.Clear(recursive=True))
            rr.log(f"camera/pred/{side}", rr.Clear(recursive=True))


def send_evidence_blueprint() -> None:
    """Source and metric hands above; crops, errors and stage times below."""
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Vertical(
                rrb.Horizontal(
                    rrb.Spatial2DView(origin="camera/image", name="Source: GT white, prediction green/pink"),
                    rrb.Spatial3DView(
                        origin="camera", contents=["camera/gt/**", "camera/pred/**", "camera/mesh/**"], name="Metric hands and full meshes"
                    ),
                ),
                rrb.Horizontal(
                    rrb.Spatial2DView(origin="crops/left", name="Left model crop"),
                    rrb.Spatial2DView(origin="crops/right", name="Right model crop"),
                    rrb.TimeSeriesView(origin="error", name="Matched-hand error"),
                    rrb.TimeSeriesView(origin="timing", name="Stage milliseconds"),
                ),
                row_shares=[3, 1],
            ),
            collapse_panels=True,
        )
    )


def main(config: Config) -> None:
    """Warm every frame, time batch-one inference, score all fixed GT and save evidence.

    Each segment gets one untimed pass that scores and logs every frame, then
    ``repeats`` timed passes with stage profiling off, then (optionally) one
    profiled pass whose stage times form the breakdown.
    """
    config.output.mkdir(parents=True, exist_ok=True)
    detector_config, pose_config, experiment = resolve_configs(config)
    if config.measure_timing:
        require_idle_gpu(allow_idle_residents=config.allow_idle_residents)
    detector: EgoHandDetector = detector_config.setup()
    pose: EgoHandPredictor | FastEgoHandPredictor = pose_config.setup()
    pose.profile_stages = False
    print("GPU", torch.cuda.get_device_name(), "MODEL", experiment, flush=True)
    if hasattr(detector.runtime, "_session"):
        print("ORT_PROVIDERS", detector.runtime._session.get_providers(), flush=True)
    results: list[Result] = []
    passes: int = 1 + config.repeats + int(config.profile_breakdown) if config.measure_timing else 1
    with (
        torch.inference_mode(),
        GpuActivityMonitor(config.output, allow_idle_residents=config.allow_idle_residents) if config.measure_timing else nullcontext(),
    ):
        for dataset in config.datasets:
            times: StageTimes = StageTimes()
            score: DatasetScore = DatasetScore()
            rr.init(f"egohand-{experiment}-{dataset}", spawn=False)
            rr.save(config.output / f"{dataset}.rrd")
            log_context()
            rr.log("camera", rr.ViewCoordinates.RDF, static=True)
            offset: int = 0
            for path in sorted((config.cache / dataset / "eval").glob("*.rrd")):
                segment: HandGtSegment = load_cached_hand_gt(path, max_frames=config.max_frames)
                frames: UInt8[Tensor, "f h w 3"] = decode_samples(segment)
                print("ACCURACY_AND_WARMUP", path.stem, len(frames), flush=True)
                evidence_frames: set[int] = set(np.linspace(0, len(frames) - 1, 10, dtype=int).tolist())
                for repeat in range(passes):
                    measured: bool = config.measure_timing and repeat > 0
                    profile: bool = measured and repeat == config.repeats + 1
                    pose.profile_stages = profile
                    if repeat == 1:
                        recording: rr.RecordingStream | None = rr.get_global_data_recording()
                        assert recording is not None
                        recording.flush()
                        torch.cuda.synchronize()
                        print("MEASURE", path.stem, flush=True)
                    for index, frame in enumerate(frames):
                        batch: UInt8[Tensor, "1 h w 3"] = frame[None]
                        pose.stage_ms.clear()
                        torch.cuda.synchronize()
                        start: float = time.perf_counter() if measured else 0.0
                        detections: HandDetections = detector(batch)
                        if profile:
                            torch.cuda.synchronize()
                        detect_ms: float = (time.perf_counter() - start) * 1000.0 if measured else 0.0
                        reconstruction: EgoHandReconstruction = pose(batch, detections, segment.camera)
                        torch.cuda.synchronize()
                        total_ms: float = (time.perf_counter() - start) * 1000.0 if measured else 0.0
                        if measured:
                            parts: dict[str, float] = {"total": total_ms}
                            if profile:
                                parts = {
                                    "detection": detect_ms,
                                    **{k: sum(pose.stage_ms.get(k, [])) for k in ("preprocessing", "batch_assembly", "backbone", "head", "soma")},
                                }
                                parts["other"] = total_ms - sum(parts.values())
                                parts["total_profiled"] = total_ms
                            for key, value in parts.items():
                                times.add(key, value)
                            if profile:
                                rr.set_time("eval_frame", sequence=offset + index)
                                for stage, ms in parts.items():
                                    rr.log(f"timing/{stage}", rr.Scalars(ms))
                        if repeat:
                            continue
                        prediction: FramePrediction = frame_prediction(detections, reconstruction)
                        side_errors: dict[HandSide, tuple[float, float]] = score.add_frame(segment, index, prediction)
                        rr.set_time("eval_frame", sequence=offset + index)
                        for side in SIDES:
                            if side in side_errors:
                                rr.log(f"error/{side}/pixels", rr.Scalars(side_errors[side][0]))
                                rr.log(f"error/{side}/millimeters", rr.Scalars(side_errors[side][1]))
                            else:
                                rr.log(f"error/{side}", rr.Clear(recursive=True))
                        log_frame_evidence(segment, index, batch, detections, reconstruction, prediction, pose, full=index in evidence_frames)
                        if index % 25 == 0:
                            print("FRAME", dataset, offset + index, total_ms, "hands", reconstruction.num_hands, flush=True)
                if config.measure_timing:
                    totals: list[float] = times.values["total"][-config.repeats * len(frames) :]
                    rr.send_columns(
                        "timing/total",
                        indexes=[rr.TimeColumn("eval_frame", sequence=np.tile(np.arange(offset, offset + len(frames)), config.repeats))],
                        columns=rr.Scalars.columns(scalars=totals),
                    )
                if config.measure_timing and not config.allow_idle_residents:
                    require_idle_gpu()
                offset += len(frames)
                del frames
                torch.cuda.empty_cache()
            epe_2d_median, epe_2d_p95 = error_quantiles(score.pixels)
            absolute_median, absolute_p95 = error_quantiles(score.absolute_mm)
            relative_median, relative_p95 = error_quantiles(score.relative_mm)
            result: Result = Result(
                model=experiment,
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
                stages_mean_ms={k: s.mean_ms for k, s in times.summary().items()},
                stages_median_ms={k: s.median_ms for k, s in times.summary().items()},
                total_p95_ms=float(np.quantile(times.values["total"], 0.95)) if config.measure_timing else None,
                timing_method="unprofiled" if config.measure_timing else "accuracy_only",
            )
            results.append(result)
            (config.output / "results.json").write_text(to_json(results) + "\n")
            send_evidence_blueprint()
            recording = rr.get_global_data_recording()
            assert recording is not None
            recording.flush()
            print("RESULT", to_json(result), flush=True)
