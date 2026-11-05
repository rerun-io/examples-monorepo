"""Held-out end-to-end accuracy for complete SAM body role outputs."""

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field, replace
from json import JSONDecodeError
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from jaxtyping import Float32, UInt8
from numpy import ndarray
from rerun.catalog import CatalogClient
from serde import SerdeError, serde
from serde.json import from_json, to_json
from simplecv.catalog_video import read_catalog_videos
from simplecv.catalog_video_codec import wrap_mp4
from torch import Tensor
from torchcodec.decoders import VideoDecoder
from torchvision.io import ImageReadMode, decode_image

from posekit.apis.body_bench import person_boxes_per_frame
from posekit.body_run import BodyRunIdentity, body_run_identity
from posekit.models.sam3d_body import Sam3dBodyConfig, Sam3dBodyReconstruction
from posekit.models.sam3d_body_fast import Sam3dBodyFastConfig
from posekit.models.sam3d_body_quantization import (
    BodyCalibration,
    BodyCalibrationSplit,
    CalibrationFrame,
    load_body_calibration,
    load_development_frames,
)
from posekit.predictions import BoxDetections


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class AccuracyBand:
    """Measured Euclidean errors and the required median/p95 limits."""

    median: float
    """Median error over all evaluated output elements."""
    p95: float
    """95th percentile error over all evaluated output elements."""
    maximum: float
    """Largest observed error, including rare outliers."""
    median_limit: float
    """Allowed median in the measure's stated units."""
    p95_limit: float
    """Allowed 95th percentile in the measure's stated units."""
    passed: bool
    """Whether both required percentile limits hold."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class FrameAccuracy:
    """Per-frame maxima that locate rare errors without serializing model outputs."""

    frame: CalibrationFrame
    """The exact input and its decoded pixel hash."""
    maximum: dict[str, float]
    """Fast-versus-eager maxima, one per error measure."""
    eager_repeat_maximum: dict[str, float]
    """Repeated-eager maxima on the same input; empty when not requested."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class BodyAccuracyReport:
    """One evaluated configuration, with every held-out input identified."""

    frames: list[CalibrationFrame]
    """All evaluated input identities."""
    people: int
    """Total reconstructed people."""
    bands: dict[str, AccuracyBand]
    """Fast-versus-eager aggregate accuracy."""
    passed: bool
    """Whether all required percentile limits hold."""
    per_frame: list[FrameAccuracy]
    """Per-frame maxima, including the corresponding repeated-eager baseline."""
    eager_repeat_bands: dict[str, AccuracyBand]
    """Repeated-eager aggregate variation; empty when not requested."""
    source_bands: dict[str, dict[str, AccuracyBand]] = field(default_factory=dict)
    """The same measurements grouped by input video or image."""
    stress_sources: tuple[str, ...] = ()
    """Predeclared sources reported separately from real-person acceptance."""
    stress_bands: dict[str, AccuracyBand] = field(default_factory=dict)
    """Aggregate errors on the declared detector-false-positive stress sources."""
    split: Literal["acceptance", "development"] = "acceptance"
    """Whether these inputs were reserved for acceptance or used for precision-plan selection."""
    run: BodyRunIdentity | None = None
    """Effective measured settings and export identities; None in legacy reports."""


def acceptance_passes(source_bands: dict[str, dict[str, AccuracyBand]], stress_sources: tuple[str, ...]) -> bool:
    """Require every real-person source to pass; a stress-only run cannot pass."""
    required: list[dict[str, AccuracyBand]] = [bands for source, bands in source_bands.items() if source not in stress_sources]
    return bool(required) and all(band.passed for bands in required for band in bands.values())


def load_accuracy_report(path: Path) -> BodyAccuracyReport:
    """Read an accuracy record, naming the source when its schema is invalid."""
    try:
        return from_json(BodyAccuracyReport, path.read_text())
    except (SerdeError, JSONDecodeError) as error:
        raise ValueError(f"Invalid body accuracy report {path}: {error}") from error


@dataclass(frozen=True, slots=True)
class Config:
    """Evaluate the calibration record's held-out phone frames."""

    calibration: Path
    """Measured ranges whose complete evaluation split is read from the catalog."""
    output: Path
    """JSON accuracy report destination."""
    fast: Sam3dBodyFastConfig = field(default_factory=lambda: Sam3dBodyFastConfig(gpu_preprocessing=True))
    """Fast role configuration under evaluation."""
    catalog_url: str = "rerun+http://127.0.0.1:51235"
    """Catalog containing the exact evaluation sources."""
    dataset: str = "dataforge-selfcap"
    """Dataset containing the held-out phone videos."""
    repeat_eager: bool = False
    """Repeat eager inference to measure its own full-mode variation on this split."""
    inputs: tuple[Path, ...] = ()
    """Local images/videos to evaluate instead of the reserved catalog frames."""
    frames_per_video: int = 20
    """Evenly spaced decoded frames from each local video."""
    evaluation_report: Path | None = None
    """Reuse exactly the input identities of a prior acceptance report instead of the entire reserved split."""
    development_frames: Path | None = None
    """Typed development frame set, disjoint from both calibration and reserved acceptance frames."""
    evidence_dir: Path | None = None
    """Save worst-frame Rerun comparisons here, with eager, fast and optional repeated eager."""
    stress_sources: tuple[str, ...] = ()
    """Exact source identities declared as detector-false-positive stress cases before evaluation."""


def save_comparison(
    path: Path, frame: CalibrationFrame, image: UInt8[Tensor, "h w 3"],
    results: dict[str, Sam3dBodyReconstruction],
) -> None:
    """Save the production 2D/3D entities with fixed matching oblique cameras."""
    import rerun as rr
    import rerun.blueprint as rrb

    from posekit.body_rerun import log_people, set_annotation_context

    reference = next(iter(results.values()))
    assert reference.keypoints.xyz_camera is not None
    target: Float32[ndarray, "3"] = reference.keypoints.xyz_camera[0, [9, 10]].mean(0).cpu().numpy()
    columns: list[rrb.Vertical] = []
    with rr.RecordingStream(f"SAM body comparison: {path.stem}") as recording:
        recording.save(path)
        set_annotation_context()
        rr.log("/", rr.ViewCoordinates.RDF, static=True)
        rr.set_time("source_frame", sequence=frame.index)
        rr.log("/input", rr.TextDocument(f"{frame.source}\nFrame {frame.index}\nSHA256 {frame.sha256}"))
        rgb = image.cpu().numpy()
        for name, result in results.items():
            root: str = f"/{name}"
            log_people(result, rgb, Path(root))
            columns.append(rrb.Vertical(
                rrb.Spatial2DView(name=f"{name}: image", origin=f"{root}/cam/pinhole"),
                rrb.Spatial3DView(
                    name=f"{name}: oblique", origin=root, contents=[f"{root}/pred/**"],
                    eye_controls=rrb.EyeControls3D(position=(target + np.array([1.0, -0.4, -1.5])).tolist(),
                                                   look_target=target.tolist(), eye_up=(0.0, -1.0, 0.0)),
                ),
            ))
        rr.send_blueprint(rrb.Blueprint(rrb.Horizontal(*columns), collapse_panels=True))
        recording.flush()


def output_errors(reference: Sam3dBodyReconstruction, fast: Sam3dBodyReconstruction) -> dict[str, Tensor]:
    """Euclidean errors with each body or hand centered on its own corresponding root."""
    expected = reference.keypoints.xyz_camera
    actual = fast.keypoints.xyz_camera
    assert expected is not None and actual is not None
    expected_root = (expected[:, 9:10] + expected[:, 10:11]) / 2
    actual_root = (actual[:, 9:10] + actual[:, 10:11]) / 2
    expected_hands = torch.cat((expected[:, 21:42] - expected[:, 41:42], expected[:, 42:63] - expected[:, 62:63]), 1)
    actual_hands = torch.cat((actual[:, 21:42] - actual[:, 41:42], actual[:, 42:63] - actual[:, 62:63]), 1)
    return {
        "keypoints_2d_px": (reference.keypoints.xy - fast.keypoints.xy).norm(dim=-1).flatten(),
        "body_root_relative_mm": ((expected - expected_root) - (actual - actual_root)).norm(dim=-1).flatten() * 1000,
        "hand_root_relative_mm": (expected_hands - actual_hands).norm(dim=-1).flatten() * 1000,
        "translation_mm": (reference.translation - fast.translation).norm(dim=-1).flatten() * 1000,
        "mesh_root_relative_mm": ((reference.vertices - expected_root) - (fast.vertices - actual_root)).norm(dim=-1).flatten() * 1000,
    }


def read_local_frames(paths: tuple[Path, ...], frames_per_video: int) -> list[tuple[CalibrationFrame, Tensor]]:
    """Decode and hash local images or evenly spaced video frames on the GPU."""
    if frames_per_video < 1:
        raise ValueError("frames_per_video must be positive")
    inputs: list[tuple[CalibrationFrame, Tensor]] = []
    for path in paths:
        source: str = str(path.resolve())
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            pixels = decode_image(str(path), mode=ImageReadMode.RGB).permute(1, 2, 0).contiguous().cuda()
            selected = [(0, pixels)]
        else:
            decoder = VideoDecoder(path, device="cuda", dimension_order="NHWC")
            indices = np.unique(np.linspace(0, len(decoder) - 1, min(frames_per_video, len(decoder)), dtype=np.int64)).tolist()
            frames = decoder.get_frames_at(indices).data.contiguous()
            selected = list(zip(indices, frames, strict=True))
        for index, pixels in selected:
            digest: str = hashlib.sha256(pixels.cpu().numpy().tobytes()).hexdigest()
            inputs.append((CalibrationFrame(source, index, digest), pixels))
    return inputs


def evaluation_inputs(config: Config, calibration: BodyCalibration) -> list[tuple[CalibrationFrame, Tensor]]:
    """Read local assets or the reserved catalog split, then check decoded-frame identity."""
    if sum((bool(config.inputs), config.evaluation_report is not None, config.development_frames is not None)) > 1:
        raise ValueError("Choose one of local inputs, a prior evaluation report, or a development set")
    if config.inputs:
        inputs = read_local_frames(config.inputs, config.frames_per_video)
        BodyCalibrationSplit(calibration.split.calibration, [row for row, _ in inputs])
        return inputs
    selected_frames: list[CalibrationFrame] = calibration.split.evaluation
    if config.evaluation_report is not None:
        prior: BodyAccuracyReport = load_accuracy_report(config.evaluation_report)
        if prior.split != "acceptance":
            raise ValueError("Use development_frames for development inputs; a development report cannot become acceptance evidence.")
        selected_frames = prior.frames
    if config.development_frames is not None:
        selected_frames = load_development_frames(config.development_frames, calibration)
    BodyCalibrationSplit(calibration.split.calibration, selected_frames)
    dataset = CatalogClient(config.catalog_url).get_dataset(name=config.dataset)
    by_segment: dict[str, dict[str, list[CalibrationFrame]]] = defaultdict(lambda: defaultdict(list))
    local_sources: dict[str, list[CalibrationFrame]] = defaultdict(list)
    for frame in selected_frames:
        if Path(frame.source).is_absolute():
            local_sources[frame.source].append(frame)
            continue
        segment, _, entity = frame.source.partition("/world/")
        if not entity:
            raise ValueError(f"Evaluation source is not a catalog camera: {frame.source}")
        by_segment[segment][f"/world/{entity}"].append(frame)
    inputs: list[tuple[CalibrationFrame, Tensor]] = []
    forbidden: set[str] = {frame.sha256 for frame in calibration.split.calibration}
    for source, rows in local_sources.items():
        path: Path = Path(source)
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            frames = decode_image(str(path), mode=ImageReadMode.RGB).permute(1, 2, 0).contiguous().cuda()[None]
        else:
            decoder = VideoDecoder(path, device="cuda", dimension_order="NHWC")
            frames = decoder.get_frames_at([row.index for row in rows]).data.contiguous()
        for row, frame in zip(rows, frames, strict=True):
            digest: str = hashlib.sha256(frame.cpu().numpy().tobytes()).hexdigest()
            if digest != row.sha256 or digest in forbidden:
                raise ValueError(f"Evaluation pixels changed or overlap calibration: {source} frame {row.index}")
            inputs.append((row, frame.clone()))
    for segment, sources in by_segment.items():
        videos = read_catalog_videos(dataset, segment, list(sources), "video_time")
        for (entity, rows), video in zip(sources.items(), videos, strict=True):
            fps: float = 1e9 / float(np.median(np.diff(video.t_ns)))
            decoder = VideoDecoder(wrap_mp4(video.samples, video.keyframes, round(fps), codec=video.codec), device="cuda", dimension_order="NHWC", seek_mode="approximate")
            frames = decoder.get_frames_at([frame.index for frame in rows]).data.contiguous()
            for row, frame in zip(rows, frames, strict=True):
                digest: str = hashlib.sha256(frame.cpu().numpy().tobytes()).hexdigest()
                if digest != row.sha256 or digest in forbidden:
                    raise ValueError(f"Evaluation pixels changed or overlap calibration: {entity} frame {row.index}")
                inputs.append((row, frame.clone()))
    return inputs


@torch.inference_mode()
def main(config: Config) -> BodyAccuracyReport:
    """Compare complete eager/fast outputs on disjoint, decoded and hashed inputs."""
    calibration = load_body_calibration(config.calibration)
    inputs = evaluation_inputs(config, calibration)
    detections_by_frame: list[BoxDetections] = person_boxes_per_frame([frame for _, frame in inputs])
    reference = Sam3dBodyConfig(mode=config.fast.mode).setup()
    effective_fast: Sam3dBodyFastConfig = replace(config.fast, calibration=config.calibration)
    fast = effective_fast.setup()
    errors: dict[str, list[Tensor]] = defaultdict(list)
    stress_errors: dict[str, list[Tensor]] = defaultdict(list)
    repeated_errors: dict[str, list[Tensor]] = defaultdict(list)
    source_errors: dict[str, dict[str, list[Tensor]]] = defaultdict(lambda: defaultdict(list))
    per_frame: list[FrameAccuracy] = []
    worst: dict[str, tuple[float, CalibrationFrame, Tensor, dict[str, Sam3dBodyReconstruction]]] = {}
    people: int = 0
    for position, (row, frame) in enumerate(inputs):
        detections = detections_by_frame[position]
        if position == 0:
            reference(frame[None], detections)
            fast(frame[None], detections)
        expected = reference(frame[None], detections)
        actual = fast(frame[None], detections)
        repeated = reference(frame[None], detections) if config.repeat_eager else None
        if expected.num_people != actual.num_people:
            raise ValueError(f"Different person counts on {row.source} frame {row.index}")
        people += expected.num_people
        if not expected.num_people:
            per_frame.append(FrameAccuracy(row, {}, {}))
            continue
        frame_maximum: dict[str, float] = {}
        for name, values in output_errors(expected, actual).items():
            host_values: Tensor = values.cpu()
            (stress_errors if row.source in config.stress_sources else errors)[name].append(host_values)
            source_errors[row.source][name].append(host_values)
            maximum: float = float(values.max())
            frame_maximum[name] = maximum
            evidence_name: str = f"stress-{name}" if row.source in config.stress_sources else name
            if config.evidence_dir is not None and (evidence_name not in worst or maximum > worst[evidence_name][0]):
                results = {"eager": expected, "fast": actual}
                if repeated is not None:
                    results["eager_repeat"] = repeated
                worst[evidence_name] = (maximum, row, frame, results)
        repeat_maximum: dict[str, float] = {}
        if repeated is not None:
            for name, values in output_errors(expected, repeated).items():
                repeated_errors[name].append(values.cpu())
                repeat_maximum[name] = float(values.max())
        per_frame.append(FrameAccuracy(row, frame_maximum, repeat_maximum))
        print(f"[body-accuracy] {position + 1}/{len(inputs)} {row.source} frame {row.index}", flush=True)
    if not people:
        raise ValueError("No people were detected in the evaluation inputs")
    limits: dict[str, tuple[float, float]] = {
        "keypoints_2d_px": (3.0, 8.0), "body_root_relative_mm": (5.0, 15.0), "hand_root_relative_mm": (5.0, 15.0),
        "translation_mm": (15.0, 40.0), "mesh_root_relative_mm": (5.0, 15.0),
    }
    def summarize(measurements: dict[str, list[Tensor]]) -> dict[str, AccuracyBand]:
        bands: dict[str, AccuracyBand] = {}
        for name, batches in measurements.items():
            values: Tensor = torch.cat(batches)
            median, p95 = torch.quantile(values, torch.tensor([0.5, 0.95])).tolist()
            median_limit, p95_limit = limits[name]
            bands[name] = AccuracyBand(median, p95, float(values.max()), median_limit, p95_limit, median <= median_limit and p95 <= p95_limit)
        return bands

    bands = summarize(errors)
    source_bands = {source: summarize(values) for source, values in source_errors.items()}
    report = BodyAccuracyReport(
        [row for row, _ in inputs], people, bands, acceptance_passes(source_bands, config.stress_sources), per_frame,
        summarize(repeated_errors), source_bands, config.stress_sources, summarize(stress_errors),
        "development" if config.development_frames is not None else "acceptance",
        body_run_identity(effective_fast),
    )
    config.output.parent.mkdir(parents=True, exist_ok=True)
    config.output.write_text(to_json(report) + "\n")
    print(to_json(report.bands), flush=True)
    if config.evidence_dir is not None:
        config.evidence_dir.mkdir(parents=True, exist_ok=True)
        for name, (_, row, image, results) in worst.items():
            save_comparison(config.evidence_dir / f"fast-worst-{name}.rrd", row, image, results)
    return report
