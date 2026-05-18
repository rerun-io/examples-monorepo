import copy
from dataclasses import dataclass, field
from pathlib import Path

import tyro
from jaxtyping import Int
from monopriors.apis.multiview_calibration import MultiViewCalibratorConfig
from numpy import ndarray
from simplecv.data.exoego.hocap import HocapConfig
from simplecv.rerun_log_utils import RerunTyroConfig

from mv_api.api.full_exoego_app import FullExoEgoRunResult, run_full_exoego_app
from mv_api.api.full_exoego_pipeline import (
    AnnotatedMVAPIDatasetUnion,
    RRDPipelineConfig,
    run_full_exoego_pipeline,
)
from mv_api.api.rerun_artifact_compare import (
    RRDComparisonResult,
    RRDStatsQueryResult,
    compare_rrd_files,
    query_rrd_stats,
)
from mv_api.multiview_pose_estimator import MultiviewBodyTrackerConfig


@dataclass
class FullExoEgoOracleComparisonConfig:
    """Configuration for comparing the legacy full pipeline against the node-based app."""

    legacy_rrd_path: Path = Path("artifacts/node-validation/legacy-full-exoego.rrd")
    """Path where the legacy oracle RRD should be saved."""
    node_rrd_path: Path = Path("artifacts/node-validation/node-full-exoego.rrd")
    """Path where the node-based app RRD should be saved."""
    application_id: str = "mv_api_oracle_comparison"
    """Rerun application id used for both outputs."""
    recording_id: str = "mv_api_oracle_comparison"
    """Fixed Rerun recording id used for both outputs."""
    dataset: AnnotatedMVAPIDatasetUnion = field(default_factory=HocapConfig)
    """Dataset factory used by both the legacy and node-based runs."""
    calib_config: MultiViewCalibratorConfig = field(
        default_factory=lambda: MultiViewCalibratorConfig(refine_depth_maps=False, segment_people=False)
    )
    """Calibration config used by both runs."""
    tracker_config: MultiviewBodyTrackerConfig = field(default_factory=MultiviewBodyTrackerConfig)
    """Tracker config used by both runs."""
    calib_ts_nano: int | None = None
    """Optional nanosecond timestamp used to select calibration frames."""
    max_frames: int | None = 1
    """Maximum number of frames to process."""


@dataclass(slots=True)
class FullExoEgoOracleComparisonResult:
    """Comparison output for legacy-vs-node full exo/ego validation."""

    legacy_rrd_path: Path
    """Legacy oracle RRD path."""
    node_rrd_path: Path
    """Node-based app RRD path."""
    node_result: FullExoEgoRunResult
    """Run summary returned by the node-based app."""
    exact_comparison: RRDComparisonResult
    """Exact RRD comparison result."""
    legacy_stats: RRDStatsQueryResult
    """Parsed Rerun stats for the legacy oracle RRD."""
    node_stats: RRDStatsQueryResult
    """Parsed Rerun stats for the node-based app RRD."""
    world_entity_counts_match: bool
    """Whether all ``/world`` entity chunk counts match between artifacts."""


def _world_entity_counts(stats: RRDStatsQueryResult) -> dict[str, int]:
    world_counts: dict[str, int] = {
        entity_path: count
        for entity_path, count in stats.entity_chunk_counts.items()
        if entity_path == "/world" or entity_path.startswith("/world/")
    }
    return world_counts


def _build_pipeline_config(
    *,
    config: FullExoEgoOracleComparisonConfig,
    rrd_path: Path,
) -> RRDPipelineConfig:
    rr_config: RerunTyroConfig = RerunTyroConfig(
        application_id=config.application_id,
        recording_id=config.recording_id,
        save=rrd_path,
        headless=True,
    )
    pipeline_config: RRDPipelineConfig = RRDPipelineConfig(
        rr_config=rr_config,
        dataset=copy.deepcopy(config.dataset),
        calib_config=copy.deepcopy(config.calib_config),
        tracker_config=copy.deepcopy(config.tracker_config),
        calib_ts_nano=config.calib_ts_nano,
        max_frames=config.max_frames,
    )
    return pipeline_config


def run_oracle_comparison(config: FullExoEgoOracleComparisonConfig) -> FullExoEgoOracleComparisonResult:
    """Run legacy and node-based full exo/ego pipelines and compare their RRD artifacts."""
    config.legacy_rrd_path.parent.mkdir(parents=True, exist_ok=True)
    config.node_rrd_path.parent.mkdir(parents=True, exist_ok=True)

    legacy_config: RRDPipelineConfig = _build_pipeline_config(config=config, rrd_path=config.legacy_rrd_path)
    run_full_exoego_pipeline(config=legacy_config)

    node_config: RRDPipelineConfig = _build_pipeline_config(config=config, rrd_path=config.node_rrd_path)
    node_result: FullExoEgoRunResult = run_full_exoego_app(config=node_config)

    legacy_stats: RRDStatsQueryResult = query_rrd_stats(rrd_path=config.legacy_rrd_path)
    node_stats: RRDStatsQueryResult = query_rrd_stats(rrd_path=config.node_rrd_path)
    exact_comparison: RRDComparisonResult = compare_rrd_files(
        expected_rrd_path=config.legacy_rrd_path,
        actual_rrd_path=config.node_rrd_path,
    )
    world_entity_counts_match: bool = _world_entity_counts(legacy_stats) == _world_entity_counts(node_stats)

    return FullExoEgoOracleComparisonResult(
        legacy_rrd_path=config.legacy_rrd_path,
        node_rrd_path=config.node_rrd_path,
        node_result=node_result,
        exact_comparison=exact_comparison,
        legacy_stats=legacy_stats,
        node_stats=node_stats,
        world_entity_counts_match=world_entity_counts_match,
    )


def main(config: FullExoEgoOracleComparisonConfig) -> None:
    """CLI entrypoint for legacy-vs-node Rerun artifact comparison."""
    result: FullExoEgoOracleComparisonResult = run_oracle_comparison(config=config)
    processed_timestamps: Int[ndarray, "n_frames"] = result.node_result.model_backed.processed_timestamps
    print(f"legacy_rrd={result.legacy_rrd_path}")
    print(f"node_rrd={result.node_rrd_path}")
    print(f"node_processed_timestamps={processed_timestamps.shape[0]}")
    print(f"exact_rrd_match={result.exact_comparison.exact_match}")
    print(f"world_entity_counts_match={result.world_entity_counts_match}")
    print(f"legacy_num_rows={result.legacy_stats.overview.get('num_rows')}")
    print(f"node_num_rows={result.node_stats.overview.get('num_rows')}")
    if not result.exact_comparison.exact_match and not result.world_entity_counts_match:
        raise SystemExit(1)


if __name__ == "__main__":
    main(tyro.cli(FullExoEgoOracleComparisonConfig))
