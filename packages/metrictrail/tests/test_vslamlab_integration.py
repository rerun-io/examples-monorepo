"""Integration: the evaluation CLI on a real upstream MetricTrail TUM run (seed 1234, paper raster), through VSLAM-LAB's own evaluation."""

from pathlib import Path

import pytest

from metrictrail.apis.vslamlab_eval import Config, Report, main

RUN_DIR: Path = Path("/mnt/nas/ssd-8tb/metrictrail/results/upstream/tum/seed1234")
FRAME_ROOT: Path = Path("/mnt/nas/ssd-8tb/metrictrail/data/TUM/rgb")
VSLAMLAB_DIR: Path = Path("/tmp/metrictrail-scratch/vslamlab/VSLAM-LAB")
BENCHMARK: Path = Path("/mnt/nas/ssd-8tb/datasets/VSLAM-LAB-Benchmark")

pytestmark = pytest.mark.integration


def test_real_desk_run_scores_the_same_under_both_protocols(tmp_path: Path) -> None:
    for asset in (RUN_DIR / "pose", FRAME_ROOT / "desk", VSLAMLAB_DIR / ".pixi" / "envs" / "vslamlab", BENCHMARK / "RGBDTUM"):
        if not asset.exists():
            pytest.skip(f"missing asset {asset}")
    config = Config(
        runs=[RUN_DIR],
        dataset="rgbdtum",
        out_dir=tmp_path / "out",
        sequences=["rgbd_dataset_freiburg1_desk"],
        naming="upstream",
        frame_root=FRAME_ROOT,
        benchmark_root=BENCHMARK,
        vslamlab_dir=VSLAMLAB_DIR,
    )
    report: Report = main(config)
    run = report.runs[0]
    # Every one of the 613 desk frames has a GT pose within 20 ms, so the two association rules pick the same pairs.
    assert run.paper_num_pairs == run.vslamlab_num_evaluated_frames == 613
    assert run.vslamlab_ate_m == pytest.approx(run.paper_ate_m, rel=1e-6)
    assert run.vslamlab_replica_ate_m == pytest.approx(run.vslamlab_ate_m, rel=1e-6)
    assert (tmp_path / "out" / "report.json").is_file()
    # evo's settings went to the private HOME, not ~/.evo.
    assert (tmp_path / "out" / "evo_home" / ".evo" / "settings.json").is_file()
