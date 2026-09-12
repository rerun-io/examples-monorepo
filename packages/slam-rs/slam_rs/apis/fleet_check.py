"""Score catalog segments against ground truth and measured lane baselines."""

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

import orjson
from serde import coerce, serde
from serde.json import to_json

from slam_rs import _core
from slam_rs.catalog_feed import CatalogSegment, resolve_catalog_segments
from slam_rs.machine import Machine, this_machine, this_peak_rss_mb
from slam_rs.reference import (
    GATE_RATIO,
    SMOKE_SEGMENTS,
    Baseline,
    Measurement,
    ReferenceManifest,
    ReferenceSegment,
    Tier,
    gate_failures,
    load_manifest,
)
from slam_rs.tracking import SegmentRun, run_segment
from slam_rs.trajectory import AteResult, ScoringResult, extent_m, nonfinite_position_text, score_trajectory

Lane: TypeAlias = Literal["cpu", "gpu"]


@dataclass(slots=True, frozen=True)
class ClipResult:
    """One measured clip and all inputs to its verdict."""

    segment_id: str
    """Catalog segment id."""
    measurement: Measurement
    """Measured inputs to the gate."""
    wall_s: float
    """Replay duration."""
    peak_rss_mb: float
    """Peak process memory."""
    config_sha256: str
    """Resolved configuration digest."""
    unscored: str | None
    """Scoring refusal, if any."""
    baseline: Baseline | None
    """Matching lane/profile reference."""

    @property
    def baseline_gt_rmse_cm(self) -> float | None:
        """Matching baseline error, when present."""
        return None if self.baseline is None else self.baseline.gt_rmse_cm

    @property
    def gt_allowed_cm(self) -> float | None:
        """Ten percent above the matching baseline."""
        return None if self.baseline is None else GATE_RATIO * self.baseline.gt_rmse_cm

    @property
    def speed_gated(self) -> bool:
        """Only measurements from the same host, lane, and profile gate speed."""
        return self.baseline is not None and self.measurement.hostname.split(".")[0] == self.baseline.host

    @property
    def failures(self) -> tuple[str, ...]:
        """All failed clauses, including any scoring refusal."""
        failures: list[str] = gate_failures(self.measurement, self.baseline)
        if self.unscored:
            failures.append(f"scoring: {self.unscored}")
        return tuple(failures)

    @property
    def verdict(self) -> str:
        """Verdict with explicit missing-baseline status."""
        return "fail: " + "; ".join(self.failures) if self.failures else "pass" if self.baseline else "pass; no baseline"

    def row(self, machine: Machine) -> str:
        """One printable measurement row."""
        speed: str = "speed gated" if self.speed_gated else "speed not gated on this host"
        return f"| {machine.hostname} | {self.segment_id} | {self.measurement.framesets}/{self.measurement.tracked}/{self.measurement.lost} | GT {self.measurement.gt_rmse_cm:.3f} cm | tracker {self.measurement.median_tracker_ms:.3f} ms; {speed} | {self.verdict} |"


def measure(
    manifest: ReferenceManifest,
    segment: ReferenceSegment,
    gpu: bool = False,
    profile: Literal["reference", "fast"] = "fast",
    catalog: str | None = None,
    source: CatalogSegment | None = None,
) -> ClipResult:
    """Replay a catalog segment and associate estimates with ground truth."""
    if source is None:
        source = resolve_catalog_segments((CatalogSegment(catalog or manifest.catalog_url, segment.dataset_name, segment.segment_id),), require_ground_truth=True)[0]
    if (source.dataset_name, source.segment_id) != (segment.dataset_name, segment.segment_id):
        raise ValueError(f"source {source.dataset_name}/{source.segment_id} does not match segment {segment.dataset_name}/{segment.segment_id}")
    if not source.has_ground_truth:
        raise ValueError(f"{segment.segment_id}: ground-truth layer absent")
    run: SegmentRun = run_segment(manifest, segment, gpu=gpu, profile=profile, source=source)
    scoring: ScoringResult = score_trajectory(run.estimate, run.ground_truth)
    against_gt: AteResult | None = scoring.result
    lane: Lane = this_lane(gpu)
    hostname: str = this_machine().hostname
    baseline: Baseline | None = segment.baseline_for(lane, profile, hostname)
    return ClipResult(
        segment_id=segment.segment_id,
        measurement=Measurement(
            framesets=run.framesets,
            tracked=len(run.estimate),
            lost=run.lost,
            associated=against_gt.n_associated if against_gt else 0,
            gt_rmse_cm=100.0 * against_gt.rmse_m if against_gt else math.nan,
            extent_m=extent_m(run.estimate),
            truth_extent_m=extent_m(run.ground_truth),
            poses_finite=nonfinite_position_text(run.estimate) is None,
            median_tracker_ms=run.median_tracker_ms,
            hostname=hostname,
            lane=lane,
            profile=profile,
        ),
        wall_s=run.wall_s,
        peak_rss_mb=this_peak_rss_mb(),
        config_sha256=run.config_sha256,
        unscored=scoring.unscored,
        baseline=baseline,
    )


@serde(type_check=coerce, deny_unknown_fields=True)
@dataclass(slots=True, frozen=True)
class FleetClipReport:
    """The twelve public fleet columns, in their original order."""

    segment_id: str
    """Catalog segment id."""
    framesets: int
    """Framesets fed."""
    tracked: int
    """Estimated poses."""
    lost: int
    """Framesets left waiting for IMU."""
    gt_rmse_cm: float | None
    """Ground-truth error, or null when unscored."""
    wall_s: float
    """Replay duration."""
    peak_rss_mb: float
    """Peak resident memory."""
    gt_allowed_cm: float | None
    """Error allowance from the matched baseline."""
    baseline_gt_rmse_cm: float | None
    """Matched baseline error."""
    median_tracker_ms: float | None
    """Median tracker cost, or null if unmeasured."""
    speed_gated: bool
    """Whether this host matches the baseline."""
    verdict: str
    """Gate verdict and any refusals."""


@serde(type_check=coerce, deny_unknown_fields=True)
@dataclass(slots=True, frozen=True)
class FleetReport:
    """Machine, run identity and measured clip reports."""

    machine: Machine
    """Measuring host."""
    lane: Lane
    """Execution lane."""
    profile: Literal["reference", "fast"]
    """Configuration overlay."""
    core_sha256: str
    """Extension digest."""
    config_sha256: dict[str, str]
    """Resolved configuration digests by dataset."""
    clips: list[FleetClipReport]
    """Completed clips in run order."""


def this_lane(gpu: bool) -> Lane:
    """Refuse unsupported GPU requests before reading data."""
    if gpu and _core.gpu_backend is None:
        raise ValueError("--gpu needs a core built with a GPU cargo feature")
    return "gpu" if gpu else "cpu"


@dataclass(slots=True)
class Config:
    """Score selected catalog segments."""

    profile: Literal["reference", "fast"] = "fast"
    """Configuration overlay."""
    catalog: str | None = None
    """Catalog URL; defaults to the manifest."""
    segments: tuple[str, ...] = SMOKE_SEGMENTS
    """Segment ids to run."""
    tier: Tier | None = None
    """Select a manifest tier instead of explicit ids."""
    output_json: Path = Path("fleet_check.json")
    """Measurement output."""
    gpu: bool = False
    """Use the GPU frontend."""


def main(config: Config) -> None:
    """Validate sources, measure each segment, and persist results after each run."""
    lane: Lane = this_lane(config.gpu)
    manifest: ReferenceManifest = load_manifest()
    segments: tuple[ReferenceSegment, ...] = (
        manifest.in_tier(config.tier) if config.tier else tuple(manifest.by_id(identifier) for identifier in config.segments)
    )
    if not segments:
        raise ValueError("--segments named no clip")
    sources: tuple[CatalogSegment, ...] = resolve_catalog_segments(
        tuple(CatalogSegment(config.catalog or manifest.catalog_url, segment.dataset_name, segment.segment_id) for segment in segments),
        require_ground_truth=True,
    )
    machine: Machine = this_machine()
    core_sha256: str = hashlib.sha256(Path(_core.__file__).read_bytes()).hexdigest()
    results: list[ClipResult] = []
    config_digests: dict[str, str] = {}
    config.output_json.parent.mkdir(parents=True, exist_ok=True)
    for segment, source in zip(segments, sources, strict=True):
        result: ClipResult = measure(manifest, segment, config.gpu, config.profile, source=source)
        if segment.dataset_name in config_digests and config_digests[segment.dataset_name] != result.config_sha256:
            raise RuntimeError(f"{segment.dataset_name}: configuration changed during replay")
        config_digests[segment.dataset_name] = result.config_sha256
        results.append(result)
        print(result.row(machine), flush=True)
        report: FleetReport = FleetReport(
            machine=machine,
            lane=lane,
            profile=config.profile,
            core_sha256=core_sha256,
            config_sha256=config_digests,
            clips=[
                FleetClipReport(
                    segment_id=row.segment_id,
                    framesets=row.measurement.framesets,
                    tracked=row.measurement.tracked,
                    lost=row.measurement.lost,
                    gt_rmse_cm=row.measurement.gt_rmse_cm if math.isfinite(row.measurement.gt_rmse_cm) else None,
                    wall_s=row.wall_s,
                    peak_rss_mb=row.peak_rss_mb,
                    gt_allowed_cm=row.gt_allowed_cm,
                    baseline_gt_rmse_cm=row.baseline_gt_rmse_cm,
                    median_tracker_ms=row.measurement.median_tracker_ms if math.isfinite(row.measurement.median_tracker_ms) else None,
                    speed_gated=row.speed_gated,
                    verdict=row.verdict,
                )
                for row in results
            ],
        )
        config.output_json.write_text(to_json(report, option=orjson.OPT_INDENT_2) + "\n")
    if any(row.failures for row in results):
        raise SystemExit("\n".join(row.verdict for row in results if row.failures))
