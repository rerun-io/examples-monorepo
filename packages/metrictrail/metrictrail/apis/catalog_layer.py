"""Write the MetricTrail catalog layer of finished runs on VSLAM-LAB sequences, and optionally register them.

Implements ``/tmp/fleet-artifacts/metrictrail/p1/catalog/SPEC.md``. One run per segment (seed 1234 by default); the
other seeds' reports only add per-seed properties. Files go to
``<results_root>/catalog/<source>/<run_tag>/<segment>.rrd`` and are never overwritten.

    pixi run -e metrictrail python tools/catalog_layer.py --runs-root /mnt/nas/ssd-8tb/metrictrail/results/upstream/tum/seed1234 \
        --report /tmp/fleet-artifacts/metrictrail/p0/baseline/scores/tum-seed1234/report.json --benchmark tum \
        --source metrictrail_upstream --producer "upstream ViPE @ ead2e00, unmodified" --env <env> \
        [--catalog-url rerun+http://dgx-spark:51235]

Provenance is the run's, not the writer's: for our port the monorepo commit comes from each run's own
``<runs_root>/runs/<case>/logs/commit.txt`` (``git rev-parse HEAD`` then ``git status --short`` of the worktree when
the run started), and ``{commit}`` in ``--producer`` is replaced by its short form. A run without those logs needs
``--repo-commit``; a ``--repo-commit`` that disagrees with the logs, or a producer naming another commit, is refused.
Runs whose commit was later rebased away pass ``--code-commit`` (the branch commit that carries their code): it becomes
``repo_commit`` and the logs' commit is kept as ``run_commit``.

The TensorRT shipped path (``--source metrictrail``) also takes ``--capture-rrd`` (each run's live capture, ``{case}``
filled in) for the surface normals, and ``--depth-source`` names its depth.
"""

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from metrictrail.apis.vslamlab_eval import Report, RunScore
from metrictrail.benchmarks import BENCHMARKS, BenchmarkName, BenchmarkPreset
from metrictrail.catalog.build import RunProvenance, SeedScores, build_layer, find_score, read_report
from metrictrail.catalog.layer import DEPTH_SOURCE, LayerInputs, LayerProperties, LayerSource, segment_id, write_layer
from metrictrail.catalog.register import dataset_name, open_dataset, register_layer_run
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
    """Which tree ran, e.g. ``upstream ViPE @ ead2e00, unmodified (...)`` or ``examples-monorepo metrictrail @ {commit} (...)``;
    ``{commit}`` becomes the short monorepo commit of each run."""
    env: str = field(default_factory=lambda: sys.prefix)
    """Pixi environment of the runs."""
    repo_commit: str | None = None
    """Monorepo commit of the code that ran (our port only), for runs without ``runs/<case>/logs/commit.txt``; must
    agree with those logs where they exist."""
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
    run_logs_root: Path | None = None
    """Parent of each run's ``<case>/logs/commit.txt``; None: ``<runs_root>/runs``."""
    code_commit: str | None = None
    """Branch commit that carries the runs' code when their own commit (``logs/commit.txt``) was rebased away; it
    becomes ``repo_commit`` and the run's own commit ``run_commit``. Needs the run logs."""
    depth_source: str = DEPTH_SOURCE
    """``depth_source`` property: what the depth images are."""
    capture_rrd: str | None = None
    """Each run's live-capture ``.rrd`` with ``{case}`` for the case name (surface normals); None: no normals."""
    catalog_url: str | None = None
    """Register on this catalog after writing (``ERROR`` where the segment has no such layer yet, ``REPLACE`` where it
    has an older run); None: write only."""


@dataclass(frozen=True, slots=True)
class RunCommit:
    """The monorepo state a run started from (its ``logs/commit.txt``)."""

    sha: str
    """``git rev-parse HEAD``."""
    changes: list[str] | None
    """``git status --short`` lines; None when the worktree was clean."""


def read_run_commit(path: Path) -> RunCommit:
    """Parse a run's ``logs/commit.txt``: a 40-hex commit line, then ``git status --short`` lines.

    Raises:
        ValueError: If the first line is not a full commit hash.
    """
    lines: list[str] = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    sha: str = lines[0].strip() if lines else ""
    if len(sha) != 40 or any(char not in "0123456789abcdef" for char in sha):
        raise ValueError(f"{path}: first line {sha!r} is not a full commit hash")
    return RunCommit(sha=sha, changes=[line.rstrip() for line in lines[1:]] or None)


def run_provenance(config: Config, name: str) -> RunProvenance:
    """Producer and commits of the run of case ``name`` (the run's own logs; see the module docstring).

    Raises:
        ValueError: If a port run has no known commit, ``--repo-commit`` disagrees with the run's logs, or the
            producer names another commit.
    """
    log: Path = (config.run_logs_root if config.run_logs_root is not None else config.runs_root / "runs") / name / "logs" / "commit.txt"
    commit: RunCommit | None = read_run_commit(log) if log.is_file() else None
    if commit is not None and config.repo_commit is not None and not commit.sha.startswith(config.repo_commit):
        raise ValueError(f"--repo-commit {config.repo_commit} but {log} says the run used {commit.sha}")
    if commit is None and config.repo_commit is not None:
        commit = RunCommit(sha=config.repo_commit, changes=None)
    if config.code_commit is not None and not log.is_file():
        raise ValueError(f"--code-commit needs the run's own {log}")
    if config.source != "metrictrail_upstream" and commit is None:
        raise ValueError(f"{log} is missing; pass --repo-commit with the commit the port run used")
    producer: str = config.producer.replace("{commit}", commit.sha[:9]) if commit is not None else config.producer
    if commit is not None and config.source != "metrictrail_upstream" and commit.sha[:9] not in producer:
        raise ValueError(f"producer {producer!r} does not name the run's commit {commit.sha[:9]}; write it as {{commit}}")
    return RunProvenance(
        producer=producer,
        vipe_commit=VIPE_COMMIT,
        repo_commit=config.code_commit if config.code_commit is not None else commit.sha if commit is not None else None,
        repo_changes=commit.changes if commit is not None else None,
        env=config.env,
        seed=config.seed,
        run_commit=commit.sha if config.code_commit is not None and commit is not None else None,
    )


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
    paths: list[Path] = []
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
            provenance=run_provenance(config, name),
            raster_policy=preset.raster_policy,
            tracking_stride=preset.tracking_stride,
            seed_scores=per_seed,
            depth_source=config.depth_source,
            capture_rrd=Path(config.capture_rrd.replace("{case}", name)) if config.capture_rrd is not None else None,
        )
        segment: str = segment_id(protocol.benchmark_folder, sequence)
        path: Path = write_layer(out_dir / f"{segment}.rrd", segment, config.source, built[0], built[1])
        out_dir.chmod(0o775)
        print(f"wrote {path}: {built[1].num_tracked} poses, {built[1].num_keyframes} keyframes, paper ATE {built[1].paper_ate_m:.6f} m")
        paths.append(path)
    if config.catalog_url is not None:
        name_of_dataset: str = dataset_name(protocol.dataset)
        register_layer_run(open_dataset(config.catalog_url, name_of_dataset), paths, config.source)
        print(f"registered {len(paths)} {config.source} layers on {config.catalog_url} / {name_of_dataset}; take the catalog snapshot now")
    return paths
