"""Write the MetricTrail catalog layer of finished runs on VSLAM-LAB sequences, and optionally register them.

Implements ``/tmp/fleet-artifacts/metrictrail/p1/catalog/SPEC.md``. One run per segment (seed 1234 by default); the
other seeds' reports only add per-seed properties. Files go to
``<results_root>/catalog/<source>/<run_tag>/<segment>.rrd`` and are never overwritten.

    pixi run -e metrictrail python tools/catalog_layer.py --runs-root /mnt/nas/ssd-8tb/metrictrail/results/upstream/tum/seed1234 \
        --report /tmp/fleet-artifacts/metrictrail/p0/baseline/scores/tum-seed1234/report.json --benchmark tum \
        --source metrictrail_upstream --producer "upstream ViPE @ ead2e00, unmodified" --env <env> \
        [--catalog-url rerun+http://dgx-spark:51235 --replace]
"""

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from metrictrail.apis.vslamlab_eval import Report, RunScore
from metrictrail.benchmarks import BENCHMARKS, BenchmarkName, BenchmarkPreset
from metrictrail.catalog.build import RunProvenance, SeedScores, build_layer, find_score, read_report
from metrictrail.catalog.layer import LayerInputs, LayerProperties, LayerSource, segment_id, write_layer
from metrictrail.catalog.register import dataset_name, open_dataset, register_layers
from metrictrail.eval.protocols import PaperProtocol
from metrictrail.frames import FrameSource, open_frames
from metrictrail.ground_truth import GtContext, load_gt_context, protocol_for
from metrictrail.outputs import RunOutputs, read_run_outputs

VIPE_COMMIT: str = "ead2e008bc055310121736a49ff306bba50d10dc"
"""ViPE commit of ``recipes/vipe-src`` and of the upstream reference runs."""


@dataclass
class Config:
    """Write (and register) MetricTrail catalog layers."""

    runs_root: Path
    """MetricTrail output directory of one seed (``pose/<case>.npz`` for every sequence)."""
    report: Path
    """Evaluation ``report.json`` of exactly that output directory (the numbers of record)."""
    benchmark: BenchmarkName
    """Preset the runs used."""
    source: LayerSource
    """Layer name / run source; never label a port run ``metrictrail_upstream`` or the reverse."""
    producer: str
    """Which tree ran, e.g. ``upstream ViPE @ ead2e00, unmodified (...)`` or ``examples-monorepo metrictrail @ <sha>``."""
    env: str = field(default_factory=lambda: sys.prefix)
    """Pixi environment of the runs."""
    repo_commit: str | None = None
    """Monorepo commit of the code that ran (our port only)."""
    seed: int = 1234
    """``pipeline.slam.seed`` of ``runs_root``."""
    seed_reports: list[Path] = field(default_factory=list)
    """Reports of every seed (this one included), for the per-seed properties."""
    seeds: list[int] = field(default_factory=list)
    """Seeds of ``seed_reports``, same order."""
    sequences: list[str] = field(default_factory=list)
    """VSLAM-LAB sequences; empty: every sequence of the report."""
    frame_root: Path | None = None
    """Parent of the frame directories; None: the protocol's MetricTrail data root."""
    benchmark_root: Path = Path("/mnt/nas/ssd-8tb/datasets/VSLAM-LAB-Benchmark")
    """VSLAM-LAB benchmark root."""
    results_root: Path = Path("/mnt/nas/ssd-8tb/metrictrail/results")
    """Layer files go to ``<results_root>/catalog/<source>/<run_tag>/``."""
    run_tag: str = field(default_factory=lambda: time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    """New directory per build."""
    catalog_url: str | None = None
    """Register on this catalog after writing; None: write only."""
    replace: bool = False
    """Register with REPLACE (a newer run of a layer already registered); default ERROR."""


def main(config: Config) -> list[Path]:
    """Write one layer per sequence; with ``catalog_url``, register them in one call."""
    if len(config.seed_reports) != len(config.seeds):
        raise ValueError("--seed-reports and --seeds must have the same length")
    preset: BenchmarkPreset = BENCHMARKS[config.benchmark]
    protocol: PaperProtocol = protocol_for(config.benchmark)
    report: Report = read_report(config.report)
    seed_reports: list[Report] = [read_report(path) for path in config.seed_reports]
    sequences: list[str] = config.sequences or sorted({score.sequence for score in report.runs})
    frame_root: Path = config.frame_root if config.frame_root is not None else Path(protocol.frame_root)
    out_dir: Path = config.results_root / "catalog" / config.source / config.run_tag
    provenance = RunProvenance(producer=config.producer, vipe_commit=VIPE_COMMIT, repo_commit=config.repo_commit, env=config.env, seed=config.seed)
    paths: list[Path] = []
    segments: list[str] = []
    for sequence in sequences:
        name: str = protocol.upstream_cases.get(sequence, sequence)
        score: RunScore = find_score(report, sequence, name, config.report)
        per_seed: SeedScores | None = None
        if seed_reports:
            values: list[float | None] = [
                find_score(other, sequence, name, path).paper_ate_m for other, path in zip(seed_reports, config.seed_reports, strict=True)
            ]
            if all(value is not None for value in values):
                per_seed = SeedScores(seeds=list(config.seeds), paper_ate_m=[float(value) for value in values if value is not None])
        outputs: RunOutputs = read_run_outputs(config.runs_root, name)
        frames: FrameSource = open_frames(frame_root / name)
        sequence_dir: Path = config.benchmark_root / protocol.benchmark_folder / sequence
        context: GtContext = load_gt_context(sequence_dir, protocol, frames.names)
        built: tuple[LayerInputs, LayerProperties] = build_layer(
            outputs,
            frames,
            context,
            score=score,
            report=report,
            provenance=provenance,
            raster_policy=preset.raster_policy,
            tracking_stride=preset.tracking_stride,
            seed_scores=per_seed,
        )
        segment: str = segment_id(protocol.benchmark_folder, sequence)
        path: Path = write_layer(out_dir / f"{segment}.rrd", segment, config.source, built[0], built[1])
        out_dir.chmod(0o775)
        print(f"wrote {path}: {built[1].num_tracked} poses, {built[1].num_keyframes} keyframes, paper ATE {built[1].paper_ate_m:.6f} m")
        paths.append(path)
        segments.append(segment)
    if config.catalog_url is not None:
        name_of_dataset: str = dataset_name(protocol.dataset)
        register_layers(open_dataset(config.catalog_url, name_of_dataset), paths, segments, config.source, replace=config.replace)
        print(f"registered {len(paths)} {config.source} layers on {config.catalog_url} / {name_of_dataset}; take the catalog snapshot now")
    return paths
