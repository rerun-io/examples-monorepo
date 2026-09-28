"""Run the full pipeline (DetNet-F, KeyNet-F, the LM fit, detection-by-tracking) on catalog segments and score it.

Per segment and hand mode it writes, under ``<output-root>/<name>/<hand mode>/``:

- ``<segment>.npz`` + ``<segment>.json``: the ``SegmentTrack`` (``handtrack.results``);
- ``<segment>.metrics.json``: MKPE, MKA, MKA GT, tracking statistics and DetNet P/R with tracking (``eval.segment``).

With ``detnet_alone`` (the default) the same decode also runs DetNet on every camera of every frame and writes the
DetNet-alone record and its §5.4 scores under ``<output-root>/<name>/detnet/``.

Known hand: the recording's profile model and its ϕ. Unknown hand (§5.1): track the first ``calibration_frames`` frames
with the generic model, calibrate ϕ on their stereo observations (``fit.scale.calibrate_scale``), then track the whole
segment again with the generic model × ϕ.
"""

import hashlib
import math
import os
import tempfile
import time
from contextlib import suppress
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
import rerun as rr
import torch
from beartype.roar import BeartypeException
from rerun.catalog import DatasetEntry
from serde import SerdeError, serde
from serde.json import from_json, to_json
from simplecv.umetrack_temp.generic_hand_model_torch import HandModelTorch

from handtrack.data.catalog import CATALOG_URL, UMETRACK, SegmentInfo, list_segments, select_split
from handtrack.eval.segment import (
    DetectionScore,
    DetNetAloneMetrics,
    HandScore,
    PositionScore,
    SegmentMetrics,
    TrackingScore,
    score_detnet_alone,
    score_track,
)
from handtrack.fit.observations import HandObservation
from handtrack.fit.scale import ScaleCalibration, calibrate_scale, scaled_hand_model
from handtrack.hand.pose import HandPose, generic_hand_model
from handtrack.models.detnet import DetNetF
from handtrack.models.keynet import KeyNetF
from handtrack.oracle import GroundTruthViews, KeyNetOnTruthBoxes, OracleDetector, OracleKeypoints
from handtrack.pipeline import SegmentData, TrackerRun, detnet_alone_track, load_weights, read_segment, run_tracker, segment_track
from handtrack.results import DetectorSource, HandMode, KeypointSource, SegmentTrack, TrackMetadata, save_track
from handtrack.tracker import Detector, DetNetDetector, KeyNetEstimator, KeypointEstimator, Tracker, TrackerConfig

Domain: TypeAlias = Literal["real", "synthetic"]
Interaction: TypeAlias = Literal["any", "separate_hand", "hand_hand"]
DETNET_DIR: str = "detnet"


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Which segments, which networks, which hand modes, and where the outputs go."""

    name: str = "dev"
    """Run name: outputs go to ``<output-root>/<name>/``."""
    segments: tuple[str, ...] = ()
    """Segment ids of ``dataforge-umetrack``; empty selects the test split filtered by ``domain`` and ``interaction``."""
    domain: Domain = "real"
    interaction: Interaction = "any"
    max_segments: int | None = None
    """Keep the first N selected segments (sorted by id)."""
    shard: int = 0
    """Process every ``shards``-th selected segment starting at this one (parallel runs split a split this way)."""
    shards: int = 1
    max_frames: int | None = None
    """Track only the first N frames of each segment."""
    hand_modes: tuple[HandMode, ...] = ("known",)
    detector: DetectorSource = "detnet"
    """``oracle``: ground-truth circles instead of DetNet (separates tracker and fit bugs from network quality)."""
    keypoints: KeypointSource = "keynet"
    """``oracle``: projected ground-truth keypoints plus noise instead of KeyNet; ``keynet_gt_boxes``: KeyNet on ground-truth crops (diagnostic)."""
    checkpoints: Path = Path("/home/pablo/handtrack-data/checkpoints/current")
    """Holds ``detnet.weights.pt`` and ``keynet.weights.pt`` (model-only state_dicts) with ``.sha256`` sidecars."""
    random_weights: bool = False
    """Use randomly initialised networks instead of the checkpoints (plumbing tests only; the numbers mean nothing)."""
    oracle_noise_px: float = 1.5
    """Gaussian noise on oracle keypoints, net-frame pixels."""
    oracle_noise_d_mm: float = 5.0
    """Gaussian noise on oracle d_rel, millimetres."""
    detnet_alone: bool = True
    """Also run DetNet alone on every frame and camera (needs ``detector = detnet``)."""
    calibration_frames: int = 100
    """Unknown hand: frames tracked with the generic model to calibrate ϕ (§5.1)."""
    output_root: Path = Path("/home/pablo/handtrack-data/runs")
    device: str = "cuda"
    catalog_url: str = CATALOG_URL
    seed: int = 0
    """Oracle noise seed."""
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    """Tracker thresholds and fit settings (defaults: the paper's values plus our recorded choices)."""
    cpu_threads: int = 1
    """torch CPU threads: the fit's matrices are tiny, and one thread was the fastest measured (5.2 s vs 6.4 s at 32 for 60 frames)."""


@dataclass(frozen=True, slots=True)
class Networks:
    """The two networks (None when replaced by the oracle) and what they are."""

    detnet: DetNetF | None
    keynet: KeyNetF | None
    detnet_sha256: str
    keynet_sha256: str


def load_networks(config: RunConfig, device: torch.device) -> Networks:
    """Load the current checkpoints once per run (or build random nets); fp32, eval mode."""
    torch.manual_seed(config.seed)
    detnet: DetNetF | None = None
    keynet: KeyNetF | None = None
    detnet_sha256: str = "oracle"
    keynet_sha256: str = "oracle"
    if config.detector == "detnet":
        detnet = DetNetF()
        detnet_sha256 = "random" if config.random_weights else load_weights(detnet, config.checkpoints / "detnet.weights.pt")
        detnet = detnet.to(device).eval()
    if config.keypoints != "oracle":
        keynet = KeyNetF()
        keynet_sha256 = "random" if config.random_weights else load_weights(keynet, config.checkpoints / "keynet.weights.pt")
        keynet = keynet.to(device).eval()
    return Networks(detnet, keynet, detnet_sha256, keynet_sha256)


def select_segments(config: RunConfig, entry: DatasetEntry) -> tuple[SegmentInfo, ...]:
    """The configured segments: the given ids, or the UmeTrack test split filtered by domain and interaction."""
    listed: tuple[SegmentInfo, ...] = list_segments(entry, UMETRACK)
    if config.segments:
        by_id: dict[str, SegmentInfo] = {info.segment_id: info for info in listed}
        missing: list[str] = [segment for segment in config.segments if segment not in by_id]
        if missing:
            raise ValueError(f"segments not found in {UMETRACK}: {missing}")
        chosen: tuple[SegmentInfo, ...] = tuple(by_id[segment] for segment in config.segments)
    else:
        chosen = tuple(
            info for info in select_split(listed, "test") if info.domain == config.domain and config.interaction in ("any", info.interaction)
        )
    chosen = chosen if config.max_segments is None else chosen[: config.max_segments]
    return chosen[config.shard :: config.shards]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics_path(directory: Path, segment: str) -> Path:
    return directory / f"{segment}.metrics.json"


def format_number(value: float | None, digits: int = 1) -> str:
    """A score for a log line or a table cell; an unscored one is a dash."""
    return "–" if value is None else f"{value:.{digits}f}"


def _keypoint_estimator(config: RunConfig, networks: Networks, truth: GroundTruthViews, phi: float) -> KeypointEstimator:
    if networks.keynet is None:
        return OracleKeypoints(truth, phi, config.oracle_noise_px, config.oracle_noise_d_mm, config.seed)
    keynet: KeyNetEstimator = KeyNetEstimator(networks.keynet)
    return KeyNetOnTruthBoxes(truth, keynet) if config.keypoints == "keynet_gt_boxes" else keynet


PHI_RANGE: tuple[float, float] = (0.75, 1.35)
"""Our plausibility bounds on a calibrated ϕ: the 53 UmeTrack subjects' profiles span 0.862-1.187 of the generic hand."""


@dataclass(frozen=True, slots=True)
class HandModelChoice:
    """The model a tracking pass uses and where its ϕ came from."""

    model: HandModelTorch
    phi: float
    calibration_blocks: int
    """Stereo (hand, frame) observations the scale was solved on; 0 for the profile or a fallback."""
    calibration_note: str
    """``profile``, ``calibrated``, ``clamped from <ϕ>`` or ``generic fallback: <reason>``."""


def calibrate_unknown_hand(
    config: RunConfig, data: SegmentData, detector: Detector, networks: Networks, truth: GroundTruthViews, device: torch.device
) -> HandModelChoice:
    """§5.1 offline protocol: track the first frames with the generic model (ϕ = 1), then calibrate ϕ on their stereo observations.

    Without any stereo observation the generic model is used as it is (ϕ = 1); a ϕ outside ``PHI_RANGE`` is clamped.
    """
    generic: HandModelTorch = generic_hand_model()
    frames: int = min(config.calibration_frames, data.frames)
    tracker: Tracker = Tracker(data.rig, data.letterboxes, generic, 1.0, detector, _keypoint_estimator(config, networks, truth, 1.0), config.tracker)
    run: TrackerRun = run_tracker(data, tracker, frames, device)
    hands: list[HandObservation] = []
    initial: list[HandPose | None] = []
    for frame in run.frames:
        for observation, pose in zip(frame.observations, frame.poses, strict=True):
            if observation is not None and pose is not None:
                hands.append(observation)
                initial.append(pose)
    try:
        calibration: ScaleCalibration = calibrate_scale(generic, hands, initial)
    except BeartypeException:
        raise
    except ValueError as error:  # no hand seen in stereo in the calibration frames
        return HandModelChoice(generic, 1.0, 0, f"generic fallback: {error}")
    if not math.isfinite(calibration.phi):
        return HandModelChoice(generic, 1.0, 0, f"generic fallback: calibrated phi is {calibration.phi}")
    phi: float = min(max(calibration.phi, PHI_RANGE[0]), PHI_RANGE[1])
    note: str = "calibrated" if phi == calibration.phi else f"clamped from {calibration.phi:.4f}"
    return HandModelChoice(scaled_hand_model(generic, phi), phi, calibration.blocks, note)


def _write_detnet_alone(root: Path, data: SegmentData, info: SegmentInfo, networks: Networks, run: TrackerRun, frames: int, identity: str) -> None:
    """The DetNet-alone record and its §5.4 scores, when ``run`` carries a DetNet-alone pass."""
    if run.detnet_alone is None:
        return
    meta: TrackMetadata = TrackMetadata(
        segment=info.segment_id,
        detnet_sha256=networks.detnet_sha256,
        keynet_sha256="none",
        hand_mode="known",
        hand_scale=1.0,
        timings_s={"decode": run.timings_s["decode"], "detnet_alone": run.timings_s["detnet_alone"]},
        dataset=info.dataset,
        kind="detnet_alone",
        run_identity_sha256=identity,
    )
    track: SegmentTrack = detnet_alone_track(data, run.detnet_alone, meta)
    npz: Path = save_track(track, root / DETNET_DIR)
    scores: tuple[list[DetectionScore], list[DetectionScore]] = score_detnet_alone(track, data.labels, data.letterboxes)
    metrics: DetNetAloneMetrics = DetNetAloneMetrics(
        segment=info.segment_id,
        domain=info.domain,
        interaction=info.interaction,
        frames=frames,
        detnet_sha256=networks.detnet_sha256,
        per_camera=scores[0],
        per_camera_crop=scores[1],
        track_sha256=file_sha256(npz),
        run_identity_sha256=identity,
    )
    metrics_path(root / DETNET_DIR, info.segment_id).write_text(to_json(metrics))


def run_segment(config: RunConfig, entry: DatasetEntry, info: SegmentInfo, networks: Networks, device: torch.device) -> list[SegmentMetrics]:
    """Track one segment in every configured hand mode, write the records and scores, return the scores."""
    identity: str = ensure_run_identity(config, networks)
    start: float = time.perf_counter()
    data: SegmentData = read_segment(entry, info)
    read_s: float = time.perf_counter() - start
    frames: int = data.frames if config.max_frames is None else min(config.max_frames, data.frames)
    truth: GroundTruthViews = GroundTruthViews.from_labels(data.labels)
    detnet: DetNetDetector | None = None if networks.detnet is None else DetNetDetector(networks.detnet)
    detector: Detector = detnet if detnet is not None else OracleDetector(truth)
    root: Path = config.output_root / config.name
    scores: list[SegmentMetrics] = []
    if not config.hand_modes and detnet is not None and config.detnet_alone:
        _write_detnet_alone(root, data, info, networks, run_tracker(data, None, frames, device, detnet), frames, identity)
    for index, mode in enumerate(config.hand_modes):
        mode_start: float = time.perf_counter()
        calibration_s: float = 0.0
        if mode == "known":
            choice: HandModelChoice = HandModelChoice(data.timeline.hand_model, data.timeline.hand_scale, 0, "profile")
        else:
            choice = calibrate_unknown_hand(config, data, detector, networks, truth, device)
            calibration_s = time.perf_counter() - mode_start
        phi: float = choice.phi
        tracker: Tracker = Tracker(
            data.rig, data.letterboxes, choice.model, phi, detector, _keypoint_estimator(config, networks, truth, phi), config.tracker
        )
        alone: DetNetDetector | None = detnet if config.detnet_alone and index == 0 else None
        run: TrackerRun = run_tracker(data, tracker, frames, device, alone)
        timings: dict[str, float] = {"read": read_s, "calibration": calibration_s, **run.timings_s}
        meta: TrackMetadata = TrackMetadata(
            segment=info.segment_id,
            detnet_sha256=networks.detnet_sha256,
            keynet_sha256=networks.keynet_sha256,
            hand_mode=mode,
            hand_scale=phi,
            timings_s=timings,
            dataset=info.dataset,
            detector=config.detector,
            keypoints=config.keypoints,
            run_identity_sha256=identity,
        )
        track: SegmentTrack = segment_track(data, run, meta)
        directory: Path = root / mode
        npz: Path = save_track(track, directory)
        scored: tuple[PositionScore, list[HandScore], list[DetectionScore], list[DetectionScore]] = score_track(track, data.labels, data.letterboxes)
        metrics: SegmentMetrics = SegmentMetrics(
            segment=info.segment_id,
            domain=info.domain,
            interaction=info.interaction,
            hand_mode=mode,
            hand_scale=phi,
            frames=frames,
            detector=config.detector,
            keypoints=config.keypoints,
            detnet_sha256=networks.detnet_sha256,
            keynet_sha256=networks.keynet_sha256,
            position=scored[0],
            hands=scored[1],
            detnet_with_tracking=scored[2],
            detnet_with_tracking_crop=scored[3],
            keynet_views=int(np.isfinite(track.presence).sum()),
            detnet_runs=int((track.detnet_camera >= 0).sum()),
            track_sha256=file_sha256(npz),
            run_identity_sha256=identity,
            timings_s=timings,
            calibration_blocks=choice.calibration_blocks,
            calibration_note=choice.calibration_note,
        )
        metrics_path(directory, info.segment_id).write_text(to_json(metrics))
        scores.append(metrics)
        _write_detnet_alone(root, data, info, networks, run, frames, identity)
    return scores


def summary_lines(metrics: SegmentMetrics) -> list[str]:
    """Five lines for a progress log."""
    lines: list[str] = [
        f"{metrics.segment} [{metrics.hand_mode}, phi {metrics.hand_scale:.3f} ({metrics.calibration_note}), {metrics.frames} frames, {metrics.detector}/{metrics.keypoints}]",
        f"  MKPE {format_number(metrics.position.mkpe_mm)} mm, MKA {format_number(metrics.position.mka_mm, 2)} (GT {format_number(metrics.position.mka_gt_mm, 2)}) mm/frame², scored keypoints {metrics.position.keypoints}",
    ]
    for hand in metrics.hands:
        tracking: TrackingScore = hand.tracking
        lines.append(
            f"  {hand.side}: MKPE {format_number(hand.position.mkpe_mm)} mm, tracked {tracking.visible_tracked_frames}/{tracking.visible_frames} visible"
            f" ({format_number(tracking.visible_tracked_fraction, 3)}), acquire {tracking.acquire_frames}, drops {tracking.drop_frames}, tracked w/o hand {tracking.tracked_without_hand}"
        )
    lines.append(
        "  DetNet P/R with tracking per camera: "
        + ", ".join(f"cam{s.camera} {format_number(s.precision, 2)}/{format_number(s.recall, 2)}" for s in metrics.detnet_with_tracking)
        + " (in the x1.2 crop: "
        + ", ".join(f"{format_number(s.precision, 2)}/{format_number(s.recall, 2)}" for s in metrics.detnet_with_tracking_crop)
        + ")"
        + f"; {metrics.keynet_views} KeyNet crops, {metrics.detnet_runs} DetNet runs, {metrics.timings_s['total']:.1f} s"
    )
    return lines


def main(config: RunConfig) -> None:
    ensure_run_identity(config)
    torch.set_num_threads(config.cpu_threads)
    device: torch.device = torch.device(config.device)
    entry: DatasetEntry = rr.catalog.CatalogClient(config.catalog_url).get_dataset(UMETRACK)
    networks: Networks = load_networks(config, device)
    root: Path = config.output_root / config.name
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(to_json(RunRecord.from_config(config, networks)))
    for info in select_segments(config, entry):
        for metrics in run_segment(config, entry, info, networks, device):
            print("\n".join(summary_lines(metrics)), flush=True)


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class RunRecord:
    """Run settings, also persisted immutably as ``<run>/identity.json``."""

    name: str
    segments: list[str]
    domain: str
    interaction: str
    max_segments: int | None
    max_frames: int | None
    hand_modes: list[str]
    detector: str
    keypoints: str
    checkpoints: str
    detnet_sha256: str
    keynet_sha256: str
    oracle_noise_px: float
    oracle_noise_d_mm: float
    detnet_alone: bool
    calibration_frames: int
    extrapolate: bool
    presence_threshold: float
    min_keypoint_confidence: float
    tracker: TrackerConfig
    """All tracker thresholds and nested fit settings used for this run."""

    seed: int = 0
    """Network initialization and oracle noise seed."""
    split: str = "test"
    """Selection split, or explicit segment selection."""
    catalog_url: str = CATALOG_URL
    """Source catalog."""
    device: str = "cuda"
    """Compute device."""

    @staticmethod
    def from_config(config: RunConfig, networks: Networks) -> "RunRecord":
        return RunRecord(
            name=config.name,
            segments=list(config.segments),
            domain=config.domain,
            interaction=config.interaction,
            max_segments=config.max_segments,
            max_frames=config.max_frames,
            hand_modes=list(config.hand_modes),
            detector=config.detector,
            keypoints=config.keypoints,
            checkpoints=str(config.checkpoints),
            detnet_sha256=networks.detnet_sha256,
            keynet_sha256=networks.keynet_sha256,
            oracle_noise_px=config.oracle_noise_px,
            oracle_noise_d_mm=config.oracle_noise_d_mm,
            detnet_alone=config.detnet_alone,
            calibration_frames=config.calibration_frames,
            extrapolate=config.tracker.extrapolate,
            presence_threshold=config.tracker.presence_threshold,
            min_keypoint_confidence=config.tracker.min_keypoint_confidence,
            tracker=config.tracker,
            seed=config.seed,
            split="explicit" if config.segments else "test",
            catalog_url=config.catalog_url,
            device=config.device,
        )


def ensure_run_identity(config: RunConfig, networks: Networks | None = None) -> str:
    """Publish an immutable identity; concurrent shards verify the first writer's complete record."""
    if networks is None:
        detnet_sha256: str = "oracle"
        keynet_sha256: str = "oracle"
        if config.detector == "detnet":
            detnet_sha256 = "random" if config.random_weights else file_sha256(config.checkpoints / "detnet.weights.pt")
        if config.keypoints != "oracle":
            keynet_sha256 = "random" if config.random_weights else file_sha256(config.checkpoints / "keynet.weights.pt")
        networks = Networks(None, None, detnet_sha256, keynet_sha256)
    record: RunRecord = RunRecord.from_config(config, networks)
    payload: bytes = to_json(record).encode()
    path: Path = config.output_root / config.name / "identity.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        created: tuple[int, str] = tempfile.mkstemp(prefix=".identity.", dir=path.parent)
        temporary: Path = Path(created[1])
        try:
            with os.fdopen(created[0], "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            with suppress(FileExistsError):
                os.link(temporary, path)
            directory: int = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            temporary.unlink(missing_ok=True)
    try:
        existing: RunRecord = from_json(RunRecord, path.read_text())
    except (SerdeError, ValueError) as error:
        raise ValueError(f"Invalid run identity {path}: {error}") from error
    differing: list[str] = [item.name for item in fields(record) if getattr(existing, item.name) != getattr(record, item.name)]
    if differing:
        raise ValueError(f"Run identity differs at {path}: {', '.join(differing)}; use a new run directory")
    return hashlib.sha256(to_json(existing).encode()).hexdigest()
