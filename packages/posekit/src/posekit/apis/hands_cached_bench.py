"""Measure detector plus full EgoHand reconstruction on fixed, locally cached catalog frames."""

import time
from contextlib import nullcontext
from dataclasses import dataclass, replace
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
from posekit.eval.hand_gt import HandGtSegment
from posekit.eval.hand_metrics import box_iou
from posekit.models.egohand import (
    EgoHandAssetPaths,
    EgoHandConfig,
    EgoHandDetector,
    EgoHandDetectorConfig,
    EgoHandPredictor,
    EgoHandReconstruction,
    asset_paths,
    selfcap_egohand_configs,
)
from posekit.models.egohand_fast import EgoHandFastConfig, FastEgoHandPredictor, catalog_egohand_configs, fast_egohand_configs
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
    precision: Literal["bf16", "fp8", "nvfp4"] = "nvfp4"
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
    detector_fp8_scope: Literal["all", "backbone", "late_backbone", "projections"] = "all"
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


def main(config: Config) -> None:
    """Warm every frame, time batch-one inference, score all fixed GT and save evidence."""
    config.output.mkdir(parents=True, exist_ok=True)
    paths: EgoHandAssetPaths = asset_paths(config.assets)
    if config.model == "default":
        detector_config: EgoHandDetectorConfig = EgoHandDetectorConfig(weights=paths.detector)
        pose_config: EgoHandConfig | EgoHandFastConfig = EgoHandConfig(weights=paths.weights, soma_assets=config.soma or paths.soma)
    elif config.model == "catalog":
        detector_config, pose_config = catalog_egohand_configs(config.assets, backbone_precision=config.precision)
    else:
        detector_config, pose_config = fast_egohand_configs(config.assets) if config.model == "fast" else selfcap_egohand_configs(config.assets)
    if config.model == "fast":
        assert isinstance(detector_config.backend, TensorRtBackendConfig)
        assert isinstance(pose_config, EgoHandFastConfig)
        detector_config = replace(
            detector_config,
            fp8_calibration=config.detector_calibration if config.detector_mode == "fp8" else None,
            fp8_scope=config.detector_fp8_scope,
            fp16_trunk=config.detector_mode == "fp16",
            gpu_preprocessing=config.gpu_preprocessing,
            backend=replace(detector_config.backend, max_batch_size=1, opt_batch_size=1),
        )
        if config.detector_mode == "fp8" and config.detector_calibration is None:
            raise ValueError("FP8 detector requires --detector-calibration.")
        if config.detector_mode == "default":
            detector_config = EgoHandDetectorConfig(weights=paths.detector)
        pose_config = replace(
            pose_config,
            backbone_precision=config.precision,
            fp8_tail_blocks=config.fp8_tail_blocks,
            fp8_prefix_blocks=config.fp8_prefix_blocks,
            calibration=config.calibration,
            batch_size=2,
            backbone_backend=replace(pose_config.backbone_backend, max_batch_size=2, opt_batch_size=2),
        )
    if config.measure_timing:
        require_idle_gpu(allow_idle_residents=config.allow_idle_residents)
    detector: EgoHandDetector = detector_config.setup()
    pose: EgoHandPredictor | FastEgoHandPredictor = pose_config.setup()
    pose.profile_stages = False
    experiment: str = f"fast-{config.precision}-{config.detector_mode}" if config.model == "fast" else config.model
    if config.model == "catalog":
        experiment = f"catalog-{config.precision}"
    if config.model == "fast" and config.precision == "nvfp4":
        experiment += f"-prefix{config.fp8_prefix_blocks}-tail{config.fp8_tail_blocks}"
    if config.model == "fast" and config.detector_mode == "fp8":
        experiment += f"-detector-{config.detector_fp8_scope}"
    print("GPU", torch.cuda.get_device_name(), "MODEL", experiment, flush=True)
    if hasattr(detector.runtime, "_session"):
        print("ORT_PROVIDERS", detector.runtime._session.get_providers(), flush=True)
    results: list[Result] = []
    side_names: tuple[HandSide, HandSide] = ("left", "right")
    with (
        torch.inference_mode(),
        GpuActivityMonitor(config.output, allow_idle_residents=config.allow_idle_residents) if config.measure_timing else nullcontext(),
    ):
        for dataset in config.datasets:
            times: StageTimes = StageTimes()
            gt_count: int = 0
            matched_count: int = 0
            side_correct: int = 0
            any_side_count: int = 0
            hand_count: int = 0
            frame_count: int = 0
            ious: list[float] = []
            pixels: list[float] = []
            absolute: list[float] = []
            relative: list[float] = []
            rr.init(f"egohand-{experiment}-{dataset}", spawn=False)
            rr.save(config.output / f"{dataset}.rrd")
            log_context()
            rr.log("camera", rr.ViewCoordinates.RDF, static=True)
            offset: int = 0
            for path in sorted((config.cache / dataset / "eval").glob("*.rrd")):
                segment: HandGtSegment = load_cached_hand_gt(path, max_frames=config.max_frames)
                frames: UInt8[Tensor, "f h w 3"] = decode_samples(segment)
                print("ACCURACY_AND_WARMUP", path.stem, len(frames), flush=True)
                passes: int = 1 + config.repeats + int(config.profile_breakdown) if config.measure_timing else 1
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
                        parts: dict[str, float] = {}
                        if measured:
                            if profile:
                                parts = {
                                    "detection": detect_ms,
                                    **{k: sum(pose.stage_ms.get(k, [])) for k in ("preprocessing", "batch_assembly", "backbone", "head", "soma")},
                                }
                                parts["other"] = total_ms - sum(parts.values())
                                parts["total_profiled"] = total_ms
                            else:
                                parts = {"total": total_ms}
                            for key, value in parts.items():
                                times.add(key, value)
                            if profile:
                                rr.set_time("eval_frame", sequence=offset + index)
                                for stage, ms in parts.items():
                                    rr.log(f"timing/{stage}", rr.Scalars(ms))
                        if repeat:
                            continue
                        hand_count += reconstruction.num_hands
                        frame_count += 1
                        prediction: HandPoses3d = reconstruction.as_hand21()
                        boxes: Float32[ndarray, "n 4"] = detections.boxes.xyxy_numpy()
                        valid: Bool[ndarray, "2 21"] = segment.valid[index]
                        gt_sides: list[int] = [s for s in range(2) if valid[s, 0] and valid[s].sum() >= 15]
                        gt_count += len(gt_sides)
                        gt_boxes: dict[int, Float32[ndarray, "4"]] = {s: joint_box(segment.uv[index, s], valid[s], 1.0) for s in gt_sides}
                        pairs: list[tuple[float, int, int]] = sorted(
                            ((box_iou(box, gt_boxes[s]), s, j) for s in gt_sides for j, box in enumerate(boxes)), reverse=True
                        )
                        used_gt: set[int] = set()
                        used_pred: set[int] = set()
                        matches: dict[int, int] = {}
                        for iou, s, j in pairs:
                            if iou < 0.3 or s in used_gt or j in used_pred:
                                continue
                            used_gt.add(s)
                            used_pred.add(j)
                            any_side_count += 1
                            if detections.sides[j] == side_names[s]:
                                side_correct += 1
                                matches[s] = j
                        ious.extend(max((box_iou(b, gt_boxes[s]) for b in boxes), default=0.0) for s in gt_sides)
                        assert prediction.keypoints.xyz_camera is not None
                        pred_xyz: Float32[ndarray, "n 21 3"] = prediction.keypoints.xyz_camera.cpu().numpy()
                        pred_uv: Float32[ndarray, "n 21 2"] = prediction.keypoints.xy.cpu().numpy()
                        for s in matches:
                            side: HandSide = side_names[s]
                            if side not in prediction.sides:
                                continue
                            row: int = prediction.sides.index(side)
                            matched_count += 1
                            mask: Bool[ndarray, "21"] = valid[s]
                            gt: Float32[ndarray, "21 3"] = segment.xyz_camera[index, s]
                            pixels.extend(np.linalg.norm(pred_uv[row] - segment.uv[index, s], axis=-1)[mask].tolist())
                            absolute.extend((1000.0 * np.linalg.norm(pred_xyz[row] - gt, axis=-1)[mask]).tolist())
                            relative.extend((1000.0 * np.linalg.norm((pred_xyz[row] - pred_xyz[row, 0]) - (gt - gt[0]), axis=-1)[mask]).tolist())
                        rr.set_time("eval_frame", sequence=offset + index)
                        for s, side in enumerate(side_names):
                            if s in matches and side in prediction.sides:
                                row = prediction.sides.index(side)
                                rr.log(
                                    f"error/{side}/pixels",
                                    rr.Scalars(float(np.median(np.linalg.norm(pred_uv[row] - segment.uv[index, s], axis=-1)[valid[s]]))),
                                )
                                rr.log(
                                    f"error/{side}/millimeters",
                                    rr.Scalars(
                                        float(np.median(1000.0 * np.linalg.norm(pred_xyz[row] - segment.xyz_camera[index, s], axis=-1)[valid[s]]))
                                    ),
                                )
                            else:
                                rr.log(f"error/{side}", rr.Clear(recursive=True))
                        rr.log("source", rr.AnyValues(segment=path.stem, index=int(segment.sample_indices[index])))
                        rr.log("camera/image/detection", rr.Boxes2D(array=boxes, array_format=rr.Box2DFormat.XYXY, labels=list(detections.sides)))
                        if index in set(np.linspace(0, len(frames) - 1, 10, dtype=int).tolist()):
                            rr.log("camera/image", rr.Image(frame.cpu().numpy()).compress(jpeg_quality=90))
                            log_hand_intermediates(batch, detections, reconstruction, segment.camera, pose)
                        else:
                            rr.log("camera/image", rr.Clear(recursive=False))
                            rr.log("camera/mesh", rr.Clear(recursive=True))
                            rr.log("crops", rr.Clear(recursive=True))
                            rr.log("parameters", rr.Clear(recursive=True))
                        for s, side in enumerate(side_names):
                            rr.log(
                                f"camera/image/gt/{side}",
                                rr.Points2D(
                                    np.where(valid[s, :, None], segment.uv[index, s], np.nan),
                                    class_ids=0,
                                    keypoint_ids=np.arange(21),
                                    radii=2.0,
                                    show_labels=False,
                                ),
                            )
                            rr.log(
                                f"camera/gt/{side}",
                                rr.Points3D(segment.xyz_camera[index, s], class_ids=0, keypoint_ids=np.arange(21), radii=0.002, show_labels=False),
                            )
                            if side in prediction.sides:
                                row = prediction.sides.index(side)
                                rr.log(
                                    f"camera/image/pred/{side}",
                                    rr.Points2D(pred_uv[row], class_ids=4 + s, keypoint_ids=np.arange(21), radii=2.0, show_labels=False),
                                )
                                rr.log(
                                    f"camera/pred/{side}",
                                    rr.Points3D(pred_xyz[row], class_ids=4 + s, keypoint_ids=np.arange(21), radii=0.002, show_labels=False),
                                )
                            else:
                                rr.log(f"camera/image/pred/{side}", rr.Clear(recursive=True))
                                rr.log(f"camera/pred/{side}", rr.Clear(recursive=True))
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
            e2: tuple[float | None, float | None] = error_quantiles(pixels)
            e3: tuple[float | None, float | None] = error_quantiles(absolute)
            er: tuple[float | None, float | None] = error_quantiles(relative)
            result: Result = Result(
                experiment,
                torch.cuda.get_device_name(),
                dataset,
                frame_count,
                hand_count / frame_count,
                gt_count,
                matched_count,
                matched_count / gt_count if gt_count else None,
                float(np.mean(ious)) if ious else None,
                side_correct / any_side_count if any_side_count else None,
                *e2,
                *e3,
                *er,
                {k: s.mean_ms for k, s in times.summary().items()},
                {k: s.median_ms for k, s in times.summary().items()},
                float(np.quantile(times.values["total"], 0.95)) if config.measure_timing else None,
                "unprofiled" if config.measure_timing else "accuracy_only",
            )
            results.append(result)
            (config.output / "results.json").write_text(to_json(results) + "\n")
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
            recording: rr.RecordingStream | None = rr.get_global_data_recording()
            assert recording is not None
            recording.flush()
            print("RESULT", to_json(result), flush=True)
