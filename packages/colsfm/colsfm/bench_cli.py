"""Benchmark two cuSFM-layout runs and render the comparison in Rerun.

```
pixi run -e colsfm python -m colsfm.bench_cli \
    --dataset galileo \
    --run-a data/cusfm_runs/galileo_blobref/cusfm \
    --run-b data/cusfm_runs/galileo_colsfm/cusfm \
    --input-dir data/r2b_galileo \
    --save data/bench/galileo_compare.rrd \
    --report data/bench/galileo_compare.md
```

Run A is the reference (the NVIDIA blob) and lands at `/world/rig_00`; run B is
the candidate (the `colsfm` pipeline) and lands at `/world/rig_01`. The
acceptance bounds of `docs/open-pipeline-plan.md` are checked against B on every
dataset: they are all relative to run A, so they mean the same thing wherever a
reference run exists, and `check_acceptance` drops the rows a dataset cannot
support — RoboCap ships no `ground_truth.txt`, so it gets no ATE row.

The command's **exit status is the acceptance result**: 0 when every measurable
bound is met, 1 when any is not. A bound that could not be measured is printed
and counted separately but does not fail the run; see `acceptance_exit_code`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import tyro

from colsfm.bench_report import render_markdown_report, write_markdown_report
from colsfm.benchmark import (
    AcceptanceBounds,
    AcceptanceCheck,
    Comparison,
    RunArtifacts,
    compare_runs,
    read_ground_truth,
    read_run,
    rig_track_from_frames_meta,
    write_json_report,
)
from colsfm.frames_meta import FRAMES_META_NAME, FramesMeta, read_frames_meta
from colsfm.rerun_log import BenchmarkScene, TrajectorySources, save_comparison


@dataclass(frozen=True)
class BenchConfig:
    """Compare two cuSFM-layout runs of the same dataset."""

    run_a: Path
    """Reference run's `.../cusfm` directory — the NVIDIA blob; becomes `rig_00`."""
    run_b: Path
    """Candidate run's `.../cusfm` directory — the colsfm pipeline; becomes `rig_01`."""
    input_dir: Path
    """Dataset input directory holding `frames_meta.json` and maybe `ground_truth.txt`."""
    dataset: str = "galileo"
    """Label for these runs' dataset; it names the report and the recording, nothing else."""
    name_a: str = "blob"
    """Short label for run A in the report and in the recording's metadata."""
    name_b: str = "colsfm"
    """Short label for run B."""
    save: Path | None = None
    """Write the Rerun recording here; skipped when omitted."""
    report: Path | None = None
    """Write the markdown report here; skipped when omitted."""
    json_report: Path | None = None
    """Write the pyserde JSON dump here; defaults to `report` with a `.json` suffix."""
    bounds: AcceptanceBounds = field(default_factory=AcceptanceBounds)
    """Acceptance bounds checked against run B; the plan's values unless overridden."""


def run(config: BenchConfig) -> Comparison:
    """Read both runs, compute every metric, and write the requested artifacts.

    Args:
        config: Parsed command line.

    Returns:
        The comparison, so callers can assert on it.
    """
    print(f"[bench] A {config.name_a}: {config.run_a}")
    print(f"[bench] B {config.name_b}: {config.run_b}")
    run_a: RunArtifacts = read_run(config.run_a, config.name_a)
    run_b: RunArtifacts = read_run(config.run_b, config.name_b)
    comparison: Comparison = compare_runs(run_a, run_b, config.input_dir, config.dataset, config.bounds)
    print(render_markdown_report(comparison))

    if config.report is not None:
        print(f"[bench] report -> {write_markdown_report(config.report, comparison)}")
    json_path: Path | None = config.json_report or (config.report.with_suffix(".json") if config.report else None)
    if json_path is not None:
        print(f"[bench] json   -> {write_json_report(json_path, comparison)}")

    if config.save is not None:
        input_meta: FramesMeta = read_frames_meta(config.input_dir / FRAMES_META_NAME)
        sources: TrajectorySources = TrajectorySources(
            input_track=rig_track_from_frames_meta(input_meta),
            ground_truth=read_ground_truth(config.input_dir),
        )
        scene: BenchmarkScene = BenchmarkScene(comparison=comparison, run_a=run_a, run_b=run_b, sources=sources)
        print(f"[bench] rerun  -> {save_comparison(config.save, scene)}")

    if comparison.acceptance:
        # A check whose bound could not be measured is neither met nor failed.
        measurable: list[AcceptanceCheck] = [check for check in comparison.acceptance if check.passed is not None]
        failed: list[str] = [check.name for check in measurable if not check.passed]
        print(
            f"[bench] acceptance: {len(measurable) - len(failed)}/{len(comparison.acceptance)} bounds met, "
            f"{len(comparison.acceptance) - len(measurable)} not measurable"
        )
        if failed:
            print(f"[bench] acceptance FAILED: {', '.join(failed)}")
    return comparison


def acceptance_exit_code(checks: Sequence[AcceptanceCheck]) -> int:
    """The process status the acceptance checks imply.

    The policy, in one place because it is the contract a CI job reads:

    * every measurable bound met — including the case of no bounds at all — is 0;
    * any measurable bound failed is 1, however many others passed;
    * a bound that could not be measured (`passed is None`) is reported in the
      report and the console tally but **does not** fail the run. A missing
      ground truth or a run that reported no timings is a gap in the evidence,
      not evidence of a regression.

    Args:
        checks: The comparison's acceptance checks.

    Returns:
        0 or 1, for `SystemExit`.
    """
    return 1 if any(check.passed is False for check in checks) else 0


def main() -> None:
    """Entry point for `python -m colsfm.bench_cli`.

    Raises:
        SystemExit: Always, carrying `acceptance_exit_code`'s status.
    """
    comparison: Comparison = run(tyro.cli(BenchConfig))
    raise SystemExit(acceptance_exit_code(comparison.acceptance))


if __name__ == "__main__":
    main()
