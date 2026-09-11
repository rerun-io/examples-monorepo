"""Score catalog segments against ground truth and measured lane baselines."""

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, TypeAlias

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
        return self.baseline is not None and self.measurement.hostname == self.baseline.host

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
    if not source.has_ground_truth:
        raise ValueError(f"{segment.segment_id}: ground-truth layer absent")
    run: SegmentRun = run_segment(manifest, segment, gpu=gpu, profile=profile, source=source)
    scoring: ScoringResult = score_trajectory(run.estimate, run.ground_truth)
    against_gt: AteResult | None = scoring.result
    lane: Lane = this_lane(gpu)
    baseline: Baseline | None = segment.baseline_for(lane, profile)
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
            hostname=this_machine().hostname,
            lane=lane,
            profile=profile,
        ),
        wall_s=run.wall_s,
        peak_rss_mb=this_peak_rss_mb(),
        config_sha256=run.config_sha256,
        unscored=scoring.unscored,
        baseline=baseline,
    )


def clip_json(clip: ClipResult) -> dict[str, object]:
    """The fleet JSON measurement contract, in its original field order."""
    return {
        "segment_id": clip.segment_id,
        "framesets": clip.measurement.framesets,
        "tracked": clip.measurement.tracked,
        "lost": clip.measurement.lost,
        "gt_rmse_cm": clip.measurement.gt_rmse_cm,
        "wall_s": clip.wall_s,
        "peak_rss_mb": clip.peak_rss_mb,
        "gt_allowed_cm": clip.gt_allowed_cm,
        "baseline_gt_rmse_cm": clip.baseline_gt_rmse_cm,
        "median_tracker_ms": clip.measurement.median_tracker_ms,
        "speed_gated": clip.speed_gated,
        "verdict": clip.verdict,
    }

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
        payload: dict[str, object] = {
            "machine": asdict(machine),
            "lane": lane,
            "profile": config.profile,
            "core_sha256": core_sha256,
            "config_sha256": config_digests,
            "clips": [clip_json(row) for row in results],
        }
        config.output_json.write_text(json.dumps(payload, indent=2) + "\n")
    if any(row.failures for row in results):
        raise SystemExit("\n".join(row.verdict for row in results if row.failures))
