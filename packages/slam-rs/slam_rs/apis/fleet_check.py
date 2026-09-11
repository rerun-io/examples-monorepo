"""Score catalog segments against ground truth and measured lane baselines."""

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, TypeAlias

import pyarrow as pa
from rerun.catalog import CatalogClient, DatasetEntry

from slam_rs import _core
from slam_rs.machine import Machine, this_machine, this_peak_rss_mb
from slam_rs.reference import SMOKE_SEGMENTS, Baseline, ReferenceManifest, ReferenceSegment, Tier, gate_failures, load_manifest
from slam_rs.tracking import SegmentRun, run_segment
from slam_rs.trajectory import AteResult, ate, extent_m, nonfinite_position_text

Lane: TypeAlias = Literal["cpu", "gpu"]


@dataclass(slots=True, frozen=True)
class ClipResult:
    """One measured clip and all inputs to its verdict."""

    segment_id: str
    """Catalog segment id."""
    framesets: int
    """Framesets fed."""
    tracked: int
    """Estimated poses."""
    lost: int
    """Framesets left waiting for IMU."""
    gt_rmse_cm: float
    """Ground-truth ATE."""
    gt_associated: int
    """Estimate poses associated with ground truth."""
    wall_s: float
    """Replay duration."""
    median_tracker_ms: float
    """Median accepted tracker call duration."""
    peak_rss_mb: float
    """Peak process memory."""
    extent_m: float
    """Estimated bounding-box diagonal."""
    truth_extent_m: float
    """Ground-truth bounding-box diagonal."""
    poses_finite: bool
    """Whether estimated positions are finite."""
    unscored: str | None
    """Scoring refusal, if any."""
    config_sha256: str
    """Resolved configuration digest."""
    baseline: Baseline | None
    """Matching lane/profile reference."""
    hostname: str
    """Measuring host."""
    lane: Lane
    """Execution lane."""
    profile: Literal["reference", "fast"]
    """Configuration overlay."""

    @property
    def baseline_gt_rmse_cm(self) -> float | None:
        """Matching baseline error, when present."""
        return None if self.baseline is None else self.baseline.gt_rmse_cm

    @property
    def gt_allowed_cm(self) -> float | None:
        """Ten percent above the matching baseline."""
        return None if self.baseline is None else 1.10 * self.baseline.gt_rmse_cm

    @property
    def speed_gated(self) -> bool:
        """Only measurements from the same host, lane, and profile gate speed."""
        return (
            self.baseline is not None
            and self.hostname == self.baseline.host
            and self.lane == self.baseline.lane
            and self.profile == self.baseline.profile
        )

    @property
    def failures(self) -> tuple[str, ...]:
        """All failed clauses, including any scoring refusal."""
        failures: list[str] = gate_failures(
            framesets=self.framesets,
            tracked=self.tracked,
            lost=self.lost,
            associated=self.gt_associated,
            gt_rmse_cm=self.gt_rmse_cm,
            extent_m=self.extent_m,
            truth_extent_m=self.truth_extent_m,
            poses_finite=self.poses_finite,
            baseline=self.baseline,
            median_tracker_ms=self.median_tracker_ms,
            hostname=self.hostname,
            lane=self.lane,
            profile=self.profile,
        )
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
        return f"| {machine.hostname} | {self.segment_id} | {self.framesets}/{self.tracked}/{self.lost} | GT {self.gt_rmse_cm:.3f} cm | tracker {self.median_tracker_ms:.3f} ms; {speed} | {self.verdict} |"


def check_scoring_inputs(manifest: ReferenceManifest, segment: ReferenceSegment, catalog: str | None = None) -> None:
    """Require the catalog segment and its ground-truth layer before replay."""
    dataset: DatasetEntry = CatalogClient(catalog or manifest.catalog_url).get_dataset(segment.dataset_name)
    if segment.segment_id not in dataset.segment_ids():
        raise ValueError(f"{segment.segment_id}: absent from catalog")
    layers: pa.Table = dataset.manifest().to_arrow_table().select(["rerun_segment_id", "rerun_layer_name"])
    if not any(row["rerun_segment_id"] == segment.segment_id and row["rerun_layer_name"] == "gt" for row in layers.to_pylist()):
        raise ValueError(f"{segment.segment_id}: ground-truth layer absent")


def measure(
    manifest: ReferenceManifest,
    segment: ReferenceSegment,
    gpu: bool = False,
    profile: Literal["reference", "fast"] = "fast",
    catalog: str | None = None,
) -> ClipResult:
    """Replay a catalog segment and associate estimates with ground truth."""
    check_scoring_inputs(manifest, segment, catalog)
    run: SegmentRun = run_segment(manifest, segment, gpu=gpu, profile=profile, catalog=catalog)
    against_gt: AteResult | None = None
    unscored: str | None = nonfinite_position_text(run.estimate)
    if unscored is None:
        try:
            against_gt = ate(run.estimate, run.ground_truth)
        except ValueError as error:
            unscored = str(error)
    lane: Lane = this_lane(gpu)
    baseline: Baseline | None = next((row for row in segment.baseline if row.profile == profile and row.lane == lane), None)
    return ClipResult(
        segment_id=segment.segment_id,
        framesets=run.framesets,
        tracked=len(run.estimate),
        lost=run.lost,
        gt_rmse_cm=100.0 * against_gt.rmse_m if against_gt else math.nan,
        gt_associated=against_gt.n_associated if against_gt else 0,
        wall_s=run.wall_s,
        median_tracker_ms=run.median_tracker_ms,
        peak_rss_mb=this_peak_rss_mb(),
        extent_m=extent_m(run.estimate),
        truth_extent_m=extent_m(run.ground_truth),
        poses_finite=nonfinite_position_text(run.estimate) is None,
        unscored=unscored,
        config_sha256=run.config_sha256,
        baseline=baseline,
        hostname=this_machine().hostname,
        lane=lane,
        profile=profile,
    )


CLIP_JSON_KEYS: tuple[str, ...] = (
    "segment_id",
    "framesets",
    "tracked",
    "lost",
    "gt_rmse_cm",
    "wall_s",
    "peak_rss_mb",
    "gt_allowed_cm",
    "baseline_gt_rmse_cm",
    "median_tracker_ms",
    "speed_gated",
    "verdict",
)


def clip_json(clip: ClipResult) -> dict[str, object]:
    """The fleet JSON measurement contract."""
    return {key: getattr(clip, key) for key in CLIP_JSON_KEYS}


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
    for segment in segments:
        check_scoring_inputs(manifest, segment, config.catalog)
    machine: Machine = this_machine()
    core_sha256: str = hashlib.sha256(Path(_core.__file__).read_bytes()).hexdigest()
    results: list[ClipResult] = []
    config_digests: dict[str, str] = {}
    config.output_json.parent.mkdir(parents=True, exist_ok=True)
    for segment in segments:
        result: ClipResult = measure(manifest, segment, config.gpu, config.profile, config.catalog)
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
