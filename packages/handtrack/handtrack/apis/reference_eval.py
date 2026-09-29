"""Restartable UmeTrack reference ladder; calibration uses validation only.

Every shard attempts publication after its last segment. The last finisher writes
summary.json and summary.md only when all selected segment records verify. The
aggregate action can repeat publication. No catalog writes or viewer processes.
"""
import fcntl
import hashlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from rerun.catalog import CatalogClient, DatasetEntry
from serde import SerdeError, serde
from serde.json import from_json, to_json

from handtrack.data.catalog import CATALOG_URL, UMETRACK, SegmentInfo, list_segments, read_statics, read_timeline_table, select_split
from handtrack.eval.segment import combine_positions
from handtrack.models.detnet import DetNetF
from handtrack.pipeline import SegmentData, load_weights, read_segment
from handtrack.reference.catalog import ReferenceLabels, reference_labels
from handtrack.reference.results import (
    Calibration,
    CameraStatistics,
    Mode,
    PositionStatistics,
    ReferenceFrames,
    ReferenceMetrics,
    Summary,
    calibration_summary,
    circle_statistics,
    load_frames,
    position_statistics,
    save_frames,
)
from handtrack.reference.run import LadderResult, run_ladder
from handtrack.reference.state import TrackEnd
from handtrack.reference.upstream import PoseStage, UmeTrack, load_umetrack
from handtrack.tracker import DetNetDetector
from handtrack.train.checkpoint import atomic_write


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class Config:
    action: Literal["evaluate", "calibrate", "aggregate"] = "evaluate"
    """Calibration only reads validation labels; evaluation decodes shared NVDEC frames."""
    umetrack_root: Path = Path("/home/pablo/handtrack-data/umetrack_baseline/UmeTrack")
    """External upstream checkout; loaded lazily by one loader."""
    umetrack_shim: Path = Path("/home/pablo/handtrack-data/umetrack_baseline/shim")
    """A6's existing pytorch3d SO3 shim; no dependency installation or upstream edits."""
    detnet_weights: Path = Path("/home/pablo/handtrack-data/checkpoints/current/detnet.weights.pt")
    """Checkpoint verified by pipeline.load_weights against its .sha256 sidecar."""
    output: Path = Path("/tmp/fleet-artifacts/handtrack/notes/r1-reference/run")
    """Run output directory; use a separate directory for each configuration."""
    calibration: Path | None = None
    """Frozen validation Calibration JSON; required for every circle mode."""
    modes: tuple[Mode, ...] = ("gt_pose", "gt_circle", "detnet", "track")
    """Crop-source ladder levels."""
    track_end: tuple[TrackEnd, ...] = ("geometry", "detnet")
    """Report both tracking end policies by default."""
    detnet_miss_frames: int = 3
    """Consecutive misses in every tracked view before DetNet ends a track."""
    hand_mode: Literal["known", "unknown"] = "known"
    """Unknown-hand calibration is not implemented; rejected explicitly."""
    segments: tuple[str, ...] = ()
    """Empty selects the full testing split (validation for calibrate)."""
    domain: Literal["real", "synthetic", "both"] = "real"
    """Keep domains separate in aggregate tables."""
    shard: int = 0
    """Zero-based shard index."""
    shards: int = 1
    """Number of segment shards."""
    max_frames: int | None = None
    """Optional timeline prefix for smoke tests; full clips are needed for A6 comparison."""
    calibration_stride: int = 1
    """Validation label sampling stride; recorded in the calibration artifact."""
    cpu_threads: int = 1
    """Torch CPU worker threads."""
    catalog_url: str = CATALOG_URL
    """Read-only source catalog."""

    def __post_init__(self) -> None:
        if self.hand_mode != "known":
            raise ValueError("Unknown hand is a deferred stretch goal; use known")
        if self.shards < 1 or not 0 <= self.shard < self.shards or self.detnet_miss_frames < 1 or self.calibration_stride < 1 or self.cpu_threads < 1:
            raise ValueError("Invalid shard, thread, stride, or miss count")
        if self.max_frames is not None and self.max_frames < 1:
            raise ValueError("max_frames must be positive")
        if not self.modes or len(set(self.modes)) != len(self.modes) or not self.track_end or len(set(self.track_end)) != len(self.track_end):
            raise ValueError("Modes and track_end must be nonempty without duplicates")
        if self.action == "calibrate" and (self.shards != 1 or self.max_frames is not None):
            raise ValueError("Calibration uses one validation pass; use calibration_stride for sampling")


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class Identity:
    config: Config
    """Settings shared by shards (shard=0, action=evaluate)."""
    detnet_sha256: str
    """Verified checkpoint payload digest."""
    upstream_weights_sha256: str
    """Pretrained pose network digest."""
    upstream_source_sha256: str
    """Digest of upstream Python source files."""
    implementation_sha256: str
    """Digest of reference implementation source files."""
    calibration_json: str
    """Exact typed calibration record, or empty for L0."""
    selected: list[str]
    """Complete ordered segment selection before sharding."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class SegmentResult:
    identity: str
    """Immutable run identity sha256."""
    segment: str
    """Catalog segment id."""
    metrics: list[ReferenceMetrics]
    """One result per configured mode and policy; order defines the NPZ mode axis."""
    frames_npz: str
    """Basename of the fixed-key frame stream beside this record."""
    frames_sha256: str
    """SHA256 of the exact NPZ payload."""


def source_digest(root: Path, paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def selected_segments(config: Config, entry: DatasetEntry) -> tuple[SegmentInfo, ...]:
    candidates: tuple[SegmentInfo, ...] = select_split(list_segments(entry, UMETRACK), "val" if config.action == "calibrate" else "test")
    candidates = tuple(info for info in candidates if config.domain in ("both", info.domain))
    if config.segments:
        missing: set[str] = set(config.segments) - {info.segment_id for info in candidates}
        if missing:
            raise ValueError(f"Segments outside the selected split/domain: {sorted(missing)}")
        candidates = tuple(info for info in candidates if info.segment_id in config.segments)
    if not candidates:
        raise ValueError("No selected segments")
    return candidates


def ladder_keys(config: Config) -> list[tuple[Mode, TrackEnd]]:
    return [(mode, policy) for mode in config.modes for policy in (config.track_end if mode == "track" else ("geometry",))]


def verified_result(path: Path, identity: str, segment: str, keys: list[tuple[Mode, TrackEnd]]) -> SegmentResult | None:
    if not path.exists():
        return None
    try:
        result: SegmentResult = from_json(SegmentResult, path.read_text())
    except (SerdeError, ValueError):
        return None
    expected: list[tuple[Mode, str]] = [(mode, policy if mode == "track" else "none") for mode, policy in keys]
    if result.identity != identity or result.segment != segment or [(metric.mode, metric.track_end) for metric in result.metrics] != expected:
        return None
    if any(metric.identity != identity or metric.segment != segment for metric in result.metrics):
        return None
    if result.frames_npz != f"{segment}.npz":
        return None
    try:
        stream: ReferenceFrames = load_frames(path.parent / result.frames_npz, result.frames_sha256)
    except ValueError:
        return None
    if stream.error_mm.shape[1] != len(result.metrics) or any(item.frames != len(stream.video_time_ns) for item in result.metrics):
        return None
    return result


def publish_summary(config: Config, selected: tuple[SegmentInfo, ...], identity: str) -> bool:
    records: list[ReferenceMetrics] = []
    streams: dict[str, ReferenceFrames] = {}
    for info in selected:
        result: SegmentResult | None = verified_result(config.output / f"{info.segment_id}.json", identity, info.segment_id, ladder_keys(config))
        if result is None:
            print(f"Summary pending: {info.segment_id}", flush=True)
            return False
        records.extend(result.metrics)
        streams[info.segment_id] = load_frames(config.output / result.frames_npz, result.frames_sha256)
    groups: list[ReferenceMetrics] = []
    for mode_index, (mode, policy) in enumerate(ladder_keys(config)):
        end: str = policy if mode == "track" else "none"
        for domain in sorted({info.domain for info in selected}):
            for interaction in ("all", "hand_hand", "separate_hand"):
                ids: set[str] = {info.segment_id for info in selected if info.domain == domain and interaction in ("all", info.interaction)}
                scores: list[ReferenceMetrics] = [item for item in records if item.mode == mode and item.track_end == end and item.segment in ids]
                if not scores:
                    continue
                present: int = sum(item.gt_present for item in scores)
                posed: int = sum(item.posed for item in scores)
                pooled: list[ReferenceFrames] = [streams[item.segment] for item in scores]
                robust: PositionStatistics = position_statistics(np.concatenate([item.error_mm[:, mode_index] for item in pooled]),
                    np.concatenate([item.posed[:, mode_index] for item in pooled]), np.concatenate([item.gt_valid for item in pooled]))
                cameras: list[CameraStatistics] = circle_statistics(np.concatenate([item.detnet_circle for item in pooled]),
                    np.concatenate([item.detnet_presence for item in pooled]), np.concatenate([item.gt_circle for item in pooled]),
                    np.concatenate([item.visible_landmarks for item in pooled]),
                    np.concatenate([item.selected[:, mode_index] if mode == "detnet" else np.zeros_like(item.selected[:, mode_index]) for item in pooled])) if scores[0].cameras else []
                samples: int = sum(item.error_pairs for item in cameras)
                groups.append(ReferenceMetrics(f"{domain}/{interaction}", mode, end, identity, sum(item.frames for item in scores),
                    combine_positions([item.position for item in scores]), present, posed, sum(item.false_poses for item in scores),
                    posed / present if present else None, samples,
                    sum((item.centre_mean_px or 0.0) * item.error_pairs for item in cameras) / samples if samples else None,
                    sum((item.radius_mean_px or 0.0) * item.error_pairs for item in cameras) / samples if samples else None,
                    robust, cameras))
    summary: Summary = Summary(identity, len(selected), groups)
    atomic_write(config.output / "summary.json", to_json(summary).encode())
    lines: list[str] = [f"# UmeTrack reference — {len(selected)} segments", f"\nIdentity: `{identity}`\n",
        "Position errors are per hand-frame, mean over 21 landmarks. Quantiles pool frame samples (linear interpolation). "
        "<20 and <50 divide by GT-present hand-frames, including misses. Wild is error >200 mm / all posed hand-frames; "
        "false poses lack GT error and cannot enter its numerator. Coverage is GT-present posed / GT-present. Rates are fractions.", "",
        "| group | mode | end | mean mm | median mm | P90 mm | <20 | <50 | wild | coverage | false poses | GT present | all posed | scored | MKA mm/frame² |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for item in groups:
        assert item.robust is not None
        values: list[str] = ["—" if value is None else f"{value:.4f}" for value in
            (item.position.mkpe_mm, item.robust.median_mm, item.robust.p90_mm, item.robust.below_20, item.robust.below_50,
             item.robust.wild, item.coverage)]
        lines.append(f"| {item.segment} | {item.mode} | {item.track_end} | " + " | ".join(values)
            + f" | {item.false_poses} | {item.gt_present} | {item.robust.posed_denominator} | {item.robust.scored} | {item.position.mka_mm} |")
    lines.extend(["", "DetNet circle errors use finite circle pairs with >=19 GT landmarks in front and inside the upstream image bounds, "
        "regardless of detection probability. Detection is presence >0.5. Empty means zero visible landmarks; unavailable GT is excluded. "
        "Camera indices follow the input rig order. L2 selections count hand-camera pairs before pose fitting; wrong = selected - visible selected.", "",
        "| group | mode | end | camera | centre median px | centre mean px | centre P90 px | radius median px | radius mean px | finite pairs | detected / visible pairs | detection rate | false / empty pairs | false rate | L2 selected | visible selected | wrong view |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---:|---:|---:|---:|"])
    for item in groups:
        for camera in item.cameras:
            values = ["—" if value is None else f"{value:.4f}" for value in
                (camera.centre_median_px, camera.centre_mean_px, camera.centre_p90_px, camera.radius_median_px,
                 camera.radius_mean_px, camera.detection_rate, camera.false_detection_rate)]
            lines.append(f"| {item.segment} | {item.mode} | {item.track_end} | {camera.camera} | " + " | ".join(values[:5])
                + f" | {camera.error_pairs} | {camera.detections}/{camera.visible_pairs} | {values[5]} | {camera.false_detections}/{camera.empty_pairs} | {values[6]}"
                + f" | {camera.selected} | {camera.selected_visible} | {camera.wrong_view} |")
    atomic_write(config.output / "summary.md", ("\n".join(lines) + "\n").encode())
    return True


def calibrate(config: Config, entry: DatasetEntry, selected: tuple[SegmentInfo, ...], api: UmeTrack, source: str) -> None:
    ratios: list[float] = []
    angles: list[float] = []
    for info in selected:
        labels: ReferenceLabels = reference_labels(read_statics(entry, info), read_timeline_table(entry, info))
        stage: PoseStage = PoseStage(api, labels, None)
        for row in range(0, len(labels.times), config.calibration_stride):
            if not labels.tracked[row]:
                continue
            stage.set_frame(row)
            poses = stage.ground_truth(row)
            circles, visible = stage.gt_circles(poses)
            values = stage.crop_diagnostics(poses, circles, visible)
            ratios.extend(values[0])
            angles.extend(values[1])
        print(f"Calibrated {info.segment_id}: {len(ratios)} pairs", flush=True)
    result: Calibration = calibration_summary([info.segment_id for info in selected], source, ratios, angles, config.calibration_stride)
    path: Path = config.output / "calibration.json"
    if path.exists():
        raise FileExistsError(f"Calibration already frozen at {path}; use a fresh output directory")
    atomic_write(path, to_json(result).encode())
    print(to_json(result), flush=True)


def main(config: Config) -> None:
    torch.set_num_threads(config.cpu_threads)
    torch.manual_seed(0)
    source: str = source_digest(config.umetrack_root, list((config.umetrack_root / "lib").rglob("*.py")))
    source = hashlib.sha256((source + source_digest(config.umetrack_shim, list(config.umetrack_shim.rglob("*.py")))).encode()).hexdigest()
    api: UmeTrack = load_umetrack(config.umetrack_root, config.umetrack_shim)
    client: CatalogClient = CatalogClient(config.catalog_url)
    entry: DatasetEntry = client.get_dataset(UMETRACK)
    selected: tuple[SegmentInfo, ...] = selected_segments(config, entry)
    if config.action == "calibrate":
        calibrate(config, entry, selected, api, source)
        return
    calibration: Calibration | None = from_json(Calibration, config.calibration.read_text()) if config.calibration is not None else None
    if any(mode != "gt_pose" for mode in config.modes) and calibration is None:
        raise ValueError("Circle modes require --calibration from a validation-only calibration run")
    if calibration is not None and calibration.source_sha256 != source:
        raise ValueError("Calibration used different upstream source")
    model: DetNetF = DetNetF()
    detnet_digest: str = load_weights(model, config.detnet_weights)
    weights: Path = config.umetrack_root / "pretrained_models/pretrained_weights.torch"
    package: Path = Path(__file__).resolve().parents[1]
    record: Identity = Identity(replace(config, shard=0, action="evaluate"), detnet_digest, hashlib.sha256(weights.read_bytes()).hexdigest(),
        source, source_digest(package, [*package.rglob("*.py")]), "" if calibration is None else to_json(calibration), [info.segment_id for info in selected])
    payload: bytes = to_json(record).encode()
    identity: str = hashlib.sha256(payload).hexdigest()
    config.output.mkdir(parents=True, exist_ok=True)
    with (config.output / ".identity.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        identity_path: Path = config.output / "identity.json"
        if identity_path.exists() and identity_path.read_bytes() != payload:
            raise ValueError("Run identity differs; use a fresh output directory")
        atomic_write(identity_path, payload)
    if config.action == "aggregate":
        if not publish_summary(config, selected, identity):
            raise RuntimeError("Cannot aggregate an incomplete run")
        return
    device: torch.device = torch.device("cuda")
    detector: DetNetDetector | None = DetNetDetector(model.to(device).eval()) if any(mode != "gt_pose" for mode in config.modes) else None
    for info in selected[config.shard::config.shards]:
        path: Path = config.output / f"{info.segment_id}.json"
        if verified_result(path, identity, info.segment_id, ladder_keys(config)) is not None:
            print(f"Verified, skipping {info.segment_id}", flush=True)
            continue
        data: SegmentData = read_segment(entry, info)
        labels: ReferenceLabels = reference_labels(read_statics(entry, info), read_timeline_table(entry, info))
        stages: dict[tuple[Mode, TrackEnd], PoseStage] = {key: PoseStage(api, labels, weights) for key in ladder_keys(config)}
        result: LadderResult = run_ladder(data, labels, stages, detector,
            min(data.frames, config.max_frames or data.frames), 1.0 if calibration is None else calibration.median,
            config.detnet_miss_frames, identity, device)
        npz: Path = path.with_suffix(".npz")
        digest: str = save_frames(result.streams, npz)
        atomic_write(path, to_json(SegmentResult(identity, info.segment_id, result.metrics, npz.name, digest)).encode())
        print(f"Completed {info.segment_id}", flush=True)
        del stages
    publish_summary(config, selected, identity)
