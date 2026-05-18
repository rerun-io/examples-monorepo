import subprocess
from pathlib import Path

import numpy as np
import pytest
import rerun as rr
from jaxtyping import Float32
from numpy import ndarray
from simplecv.rerun_log_utils import RerunTyroConfig

from mv_api.api.rerun_artifact_compare import compare_rrd_files, query_rrd_stats


def test_rerun_artifact_compare_queries_stats_and_exact_match(tmp_path: Path) -> None:
    rrd_path: Path = tmp_path / "sample.rrd"
    RerunTyroConfig(
        application_id="mv_api_rrd_compare_test",
        recording_id="fixed_compare_recording",
        save=rrd_path,
        headless=True,
    )
    points: Float32[ndarray, "2 3"] = np.array(
        [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]],
        dtype=np.float32,
    )
    rr.log("/world/points", rr.Points3D(points))
    recording: rr.RecordingStream | None = rr.get_global_data_recording()
    assert recording is not None
    recording.flush(timeout_sec=30.0)

    stats = query_rrd_stats(rrd_path=rrd_path)
    comparison = compare_rrd_files(expected_rrd_path=rrd_path, actual_rrd_path=rrd_path)

    assert stats.returncode == 0
    assert stats.overview["num_entity_paths"] >= 1
    assert stats.entity_chunk_counts["/world/points"] == 1
    assert comparison.exact_match
    assert comparison.returncode == 0


def test_query_rrd_stats_rejects_unparsed_successful_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        del args, kwargs
        return subprocess.CompletedProcess(args=["rerun"], returncode=0, stdout="Summary\n-------\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(ValueError, match="Unable to parse"):
        query_rrd_stats(rrd_path=tmp_path / "sample.rrd")
