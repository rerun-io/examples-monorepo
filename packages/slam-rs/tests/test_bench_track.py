"""The bench harness's schedule, its best-of-medians rule and the selection it refuses.

The first two are the protocol rather than the plumbing: the interleave is what
cancels host drift, and taking the best of the per-round medians is what makes a
row reproduce. None of the three needs a GPU, a dump or a core, so they are
checked here in milliseconds.
"""

import hashlib
import os
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from fixture_types import never

from slam_rs.apis import bench_track
from slam_rs.apis.bench_track import Config, Framesets, Lane, LaneRound, best_median_ms, interleave, main
from slam_rs.catalog_feed import ImuCalib
from slam_rs.reference import SMOKE_SEGMENTS, ReferenceManifest, profiled_config_text


def test_the_schedule_runs_every_lane_once_per_round() -> None:
    """Round-major, lanes in the given order: drift hits both lanes in a round."""
    schedule: list[tuple[int, Lane]] = interleave(("cpu", "gpu"), 3)
    assert schedule == [
        (1, "cpu"),
        (1, "gpu"),
        (2, "cpu"),
        (2, "gpu"),
        (3, "cpu"),
        (3, "gpu"),
    ]


def test_every_round_holds_the_whole_lane_set_in_the_same_order() -> None:
    """The property drift-cancelling needs, over three lanes and four rounds.

    Rounds come out in order and each one is the full lane list, so no lane can
    be measured only early or only late in a sweep.
    """
    lanes: tuple[Lane, ...] = ("cpu", "gpu", "cpu")
    schedule: list[tuple[int, Lane]] = interleave(lanes, 4)
    assert len(schedule) == 12
    rounds: list[int] = [entry[0] for entry in schedule]
    assert rounds == sorted(rounds)
    for round_index in range(1, 5):
        assert [entry[1] for entry in schedule if entry[0] == round_index] == list(lanes)


def _round(lane: Lane, index: int, samples: list[float]) -> LaneRound:
    return LaneRound(
        lane=lane,
        round_index=index,
        track_ms=np.asarray(samples, dtype=np.float64),
        wall_s=1.0,
        cpu_pct=np.asarray([99.8], dtype=np.float64),
    )


def test_the_reported_figure_is_the_lowest_round_median() -> None:
    """Not the mean of the rounds and not the minimum call: the best median."""
    rounds: list[LaneRound] = [
        _round("gpu", 1, [6.0, 6.2, 6.4]),
        _round("gpu", 2, [5.8, 6.0, 6.2]),
        _round("gpu", 3, [1.0, 9.0, 9.4]),
    ]
    assert best_median_ms(rounds) == pytest.approx(6.0)


def test_a_lane_with_no_rounds_is_refused() -> None:
    """An empty lane has no figure, and saying so beats reporting zero."""
    with pytest.raises(ValueError, match="no rounds"):
        best_median_ms([])


def test_an_empty_lane_selection_is_refused_before_any_file_or_affinity_work(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """``--lanes`` with nothing after it measured nothing and ended in a traceback.

    The empty tuple survived every hop: the process was pinned, the dump and the
    config were read, a zero-lane header was printed, and only the summary's
    ``config.lanes[0]`` — the lane every ratio is read against — raised
    ``IndexError: tuple index out of range``. :func:`slam_rs.apis.run` converts
    ``ValueError`` and ``FileNotFoundError`` only, so a mistyped selection
    reached the operator as a traceback after the input work (S25 review).
    """

    monkeypatch.setattr(bench_track, "load_framesets", never("a dump was read for a run with no lane to measure"))
    monkeypatch.setattr(os, "sched_setaffinity", never("a core was pinned for a run with no lane to measure"))
    with pytest.raises(ValueError, match="--lanes named no backend.*cpu, gpu"):
        main(Config(dump=tmp_path / "clip.npz", config=tmp_path / "msdmi_config.json", lanes=(), pin_core=3))


@pytest.mark.parametrize("profile", ["reference", "fast"])
def test_benchmark_records_identify_the_profile_and_one_resolved_config(
    profile: Literal["reference", "fast"],
    manifest: ReferenceManifest,
    imu: ImuCalib,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """RESULT and BEST retain profile identity; CONFIG records the digest once."""
    config_path: Path = manifest.package_root / manifest.dataset(manifest.by_id(SMOKE_SEGMENTS[1]).dataset_name).vio_config
    framesets: Framesets = Framesets(
        images=np.zeros((1, 1, 1, 1), dtype=np.uint8),
        t_ns=np.zeros(1, dtype=np.int64),
        imu_offsets=np.zeros(2, dtype=np.int64),
        imu_t=np.zeros(0, dtype=np.int64),
        imu_gyro=np.zeros((0, 3)),
        imu_accel=np.zeros((0, 3)),
        safe_radius=472.0,
        cameras=(),
        imu=imu,
    )
    monkeypatch.setattr(bench_track, "load_framesets", lambda *_args: framesets)
    monkeypatch.setattr(bench_track, "run_lane", lambda lane, index, *_args: _round(lane, index, [1.0, 2.0]))
    main(Config(dump=tmp_path / "clip.npz", config=config_path, profile=profile, lanes=("cpu",), rounds=2))
    printed: str = capsys.readouterr().out
    records: list[str] = [line for line in printed.splitlines() if line.startswith(("RESULT ", "BEST "))]
    assert len(records) == 3
    assert all(f"profile={profile}" in record for record in records)
    digest: str = hashlib.sha256(profiled_config_text(config_path, profile).encode("utf-8")).hexdigest()
    assert printed.count(f"config_sha256={digest}") == 1
