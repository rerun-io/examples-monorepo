"""What a RoboCap fleet row says, and what it reads its own cost against.

The tool this covers runs on machines with no NAS, no repository and — on the cap
— no pixi and no viewer, so what is under test is the part that needs none of
that: the budget arithmetic a row reports, how the row reads, and the two facts
about the machine it carries beside the numbers. The replay itself is
:func:`slam_rs.tracking.run_robocap`, which the probe does **not** drive: the
probe owns its own loop, because of the Rerun rung, and what the two share is
the estimator's two files and the clock rule
(:func:`slam_rs.tracking.robocap_estimator_files`,
:func:`slam_rs.tracking.robocap_cpp_trajectory`). Putting both on one loop
behind an observer is a deferred follow-up. The ``slow`` test below is what runs
this one on the real rig.
"""

import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from fixture_types import never
from jaxtyping import Float64, Int64
from numpy import ndarray

from slam_rs.apis import robocap_fleet
from slam_rs.apis.robocap_fleet import BUDGET_15FPS_MS, BUDGET_30FPS_MS, Config, RobocapRow, main, measure
from slam_rs.machine import Machine, this_machine
from slam_rs.reference import ReferenceManifest, RobocapSession, profiled_config_text
from slam_rs.tracking import SegmentRun
from slam_rs.trajectory import ASSOCIATION_TOLERANCE_NS, Trajectory, empty_trajectory, shift_clock

CAP: Machine = Machine(hostname="robocap_f403b0", arch="aarch64", libc="2.41", cores=8)
"""The RK3588 cap, which is the machine every budget in this module is for."""
ROW: RobocapRow = RobocapRow(
    machine=CAP,
    segment_id="robocap-s15",
    framesets=1588,
    tracked=1588,
    lost=0,
    cpp_rmse_cm=7.91,
    cpp_max_cm=16.99,
    cpp_median_cm=6.98,
    wall_s=158.8,
    ms_per_frameset=100.0,
    cpp_wall_s=88.9,
    realtime_factor_15fps=BUDGET_15FPS_MS / 100.0,
    realtime_factor_30fps=BUDGET_30FPS_MS / 100.0,
    peak_rss_mb=512.0,
    temp_c_before=40.7,
    temp_c_after=51.8,
    cross_platform_ate_cm=0.004,
    unscored=None,
)
"""Session 15 at a round 100 ms per frameset, so the budget arithmetic is readable."""


def test_a_row_reads_its_cost_against_the_caps_two_input_budgets() -> None:
    """66.7 ms at 15 fps and 33.3 ms at 30 fps, and 100 ms per frameset misses both.

    The factor is the budget over the cost, so it reads the way a realtime
    factor should: at or above 1.0 the machine keeps up, and 0.67 means it needs
    half again as long as the sensor gives it.
    """
    assert round(BUDGET_15FPS_MS, 2) == 66.67
    assert round(BUDGET_30FPS_MS, 2) == 33.33
    assert ROW.realtime_factor_15fps == pytest.approx(0.667, abs=0.001)
    assert ROW.realtime_factor_30fps == pytest.approx(0.333, abs=0.001)


def test_the_row_carries_the_machine_the_temperature_and_both_budgets() -> None:
    """One markdown row: the machine, the agreement, the cost, the two factors, the heat."""
    cells: list[str] = [cell.strip() for cell in ROW.row().strip().strip("|").split("|")]
    assert cells[:4] == ["robocap_f403b0", "aarch64", "8", "robocap-s15"]
    assert cells[4] == "1588/1588/0"
    assert cells[5] == "7.91"
    assert cells[6] == "0.004"
    assert cells[7:10] == ["158.8", "100.0", "88.9"]
    assert cells[10:12] == ["0.67x", "0.33x"]
    assert cells[12] == "512"
    assert cells[13] == "40.7 → 51.8"


def test_a_machine_that_measures_neither_a_cpp_wall_nor_a_temperature_says_so() -> None:
    """Every optional column is a dash, not a zero: a missing measurement is not a measurement of zero."""
    bare: RobocapRow = replace(ROW, cpp_wall_s=None, temp_c_before=None, temp_c_after=None, cross_platform_ate_cm=None)
    cells: list[str] = [cell.strip() for cell in bare.row().strip().strip("|").split("|")]
    assert cells[6] == "—"
    assert cells[9] == "—"
    assert cells[13] == "—"


def test_the_session_is_named_the_way_a_fleet_row_names_it(manifest: ReferenceManifest) -> None:
    """``s00000015`` in the manifest is ``robocap-s15`` in a row, for both sessions."""
    assert [session.fleet_id for session in manifest.robocap.sessions] == ["robocap-s15", "robocap-s21"]


def test_both_outputs_survive_a_directory_that_is_not_there_yet(manifest: ReferenceManifest, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A run that measured 52.9 s of video must not lose it to a missing ``out/``."""
    monkeypatch.setattr(robocap_fleet, "measure", lambda *_args: (ROW, empty_trajectory()))
    output: Path = tmp_path / "out" / "robocap_fleet.json"
    main(Config(output_json=output))
    assert json.loads(output.read_text())["segment_id"] == "robocap-s15"
    assert output.with_suffix(".csv").is_file()


def test_a_scoring_input_that_is_not_here_is_refused_before_the_replay(
    manifest: ReferenceManifest, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """52.9 s of video must not be spent to reach a missing ``slam`` layer or a mistyped ``--reference-csv``.

    Both were opened only after the whole session had replayed: the C++ layer to
    score against, and the other machine's trajectory for the cross-platform
    figure. A relocated corpus can be partial and a path is typed by hand, so
    both are opened before the estimator is fed anything.
    """

    monkeypatch.setattr(robocap_fleet, "run_robocap", never("the session replayed before its scoring inputs were opened"))
    session: RobocapSession = manifest.robocap.session("s00000015")
    absent: RobocapSession = replace(session, slam_url=f"file://{tmp_path / 'slam.rrd'}")
    with pytest.raises(FileNotFoundError, match="slam.rrd is not a file on this machine"):
        measure(manifest, absent, Config(), this_machine())

    # The layer is here; the other machine's trajectory is a typo.
    (tmp_path / "slam.rrd").write_bytes(b"")
    with pytest.raises(FileNotFoundError, match="typo.csv is not a file on this machine"):
        measure(manifest, absent, Config(reference_csv=tmp_path / "typo.csv"), this_machine())


def test_an_estimate_on_another_clock_is_a_row_and_not_a_traceback(
    manifest: ReferenceManifest, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The cost this row exists to report must survive an estimate that scores against nothing.

    :func:`~slam_rs.trajectory.ate` needs an association and not a pose count,
    so an estimate on the wrong clock — the ``slam`` layer's ``video_time``
    against the device clock every basalt CSV beside it uses — used to traceback
    after the whole 52.9 s clip had replayed, losing the wall, the budget and the
    temperatures the RoboCap lane exists to measure. The agreement reads as NaN,
    the reason is on the row, and both outputs are written before the run exits
    non-zero.
    """
    poses: int = 30
    reference: Trajectory = Trajectory(
        t_ns=np.arange(poses, dtype=np.int64) * 33_000_000, position_m=np.zeros((poses, 3)), quaternion_wxyz=np.zeros((poses, 4))
    )
    monkeypatch.setattr(robocap_fleet, "robocap_cpp_trajectory", lambda *_args: reference)
    monkeypatch.setattr(
        robocap_fleet,
        "run_robocap",
        lambda *_args, **_kwargs: SegmentRun(estimate=shift_clock(reference, 4_800_000_000_000), framesets=poses, lost=0, wall_s=3.0),
    )
    (tmp_path / "slam.rrd").write_bytes(b"")
    session: RobocapSession = replace(manifest.robocap.session("s00000015"), slam_url=f"file://{tmp_path / 'slam.rrd'}")

    row: RobocapRow
    estimate: Trajectory
    row, estimate = measure(manifest, session, Config(), CAP)
    assert row.unscored is not None
    assert f"no pose associated within {ASSOCIATION_TOLERANCE_NS} ns" in row.unscored
    assert math.isnan(row.cpp_rmse_cm)
    assert math.isnan(row.cpp_max_cm)
    assert math.isnan(row.cpp_median_cm)
    # The cost is measured, not scored, so it is a number on exactly this row.
    assert row.ms_per_frameset == pytest.approx(100.0)
    assert len(estimate) == poses

    monkeypatch.setattr(robocap_fleet, "measure", lambda *_args: (row, estimate))
    output: Path = tmp_path / "robocap_fleet.json"
    with pytest.raises(SystemExit, match="no pose associated within"):
        main(Config(output_json=output))
    written: dict = json.loads(output.read_text())
    assert math.isnan(written["cpp_rmse_cm"])
    assert "no pose associated within" in written["unscored"]
    assert output.with_suffix(".csv").is_file()


def test_a_non_finite_estimate_is_a_row_and_not_an_alignment_traceback(
    manifest: ReferenceManifest, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The cost of 52.9 s of video must survive a diverged estimator too.

    An estimate whose positions carry a NaN associates on the ``slam`` layer's
    own clock, so a row was built by aligning it — and
    :func:`~slam_rs.trajectory.rigid_alignment` hands the covariance to
    ``np.linalg.svd``, which raises ``LinAlgError: SVD did not converge``. That
    is not the :class:`ValueError` the association case is caught as, so the
    whole replay was lost to a traceback: no row, no trajectory and no JSON on
    exactly the machine whose numbers this lane exists to collect. Finiteness is
    tested before the alignment, the clause names it, the agreement reads as
    NaN, the cost beside it is still a number, and both outputs are written
    before the run exits non-zero.
    """
    poses: int = 30
    t_ns: Int64[ndarray, " n"] = np.arange(poses, dtype=np.int64) * 33_000_000
    reference: Trajectory = Trajectory(
        t_ns=t_ns, position_m=np.arange(3 * poses, dtype=np.float64).reshape(poses, 3), quaternion_wxyz=np.zeros((poses, 4))
    )
    positions: Float64[ndarray, "n 3"] = np.arange(3 * poses, dtype=np.float64).reshape(poses, 3).copy()
    positions[7, 2] = np.inf
    monkeypatch.setattr(robocap_fleet, "robocap_cpp_trajectory", lambda *_args: reference)
    monkeypatch.setattr(
        robocap_fleet,
        "run_robocap",
        lambda *_args, **_kwargs: SegmentRun(
            estimate=Trajectory(t_ns=t_ns, position_m=positions, quaternion_wxyz=np.zeros((poses, 4))), framesets=poses, lost=0, wall_s=3.0
        ),
    )
    monkeypatch.setattr(robocap_fleet, "ate", never("an estimate with a non-finite position was handed to the alignment"))
    (tmp_path / "slam.rrd").write_bytes(b"")
    session: RobocapSession = replace(manifest.robocap.session("s00000015"), slam_url=f"file://{tmp_path / 'slam.rrd'}")

    row: RobocapRow
    estimate: Trajectory
    row, estimate = measure(manifest, session, Config(), CAP)
    assert row.unscored is not None
    assert "1 of 30 estimated positions is not finite" in row.unscored
    assert math.isnan(row.cpp_rmse_cm)
    assert math.isnan(row.cpp_max_cm)
    assert math.isnan(row.cpp_median_cm)
    assert row.cross_platform_ate_cm is None
    # Measured, not scored, so it is a number on exactly this row.
    assert row.ms_per_frameset == pytest.approx(100.0)
    assert len(estimate) == poses

    monkeypatch.setattr(robocap_fleet, "measure", lambda *_args: (row, estimate))
    output: Path = tmp_path / "robocap_fleet.json"
    with pytest.raises(SystemExit, match="is not finite"):
        main(Config(output_json=output))
    written: dict = json.loads(output.read_text())
    assert math.isnan(written["cpp_rmse_cm"])
    assert "is not finite" in written["unscored"]
    assert output.with_suffix(".csv").is_file()


@pytest.mark.slow
def test_the_real_session_replays_on_this_machine_and_agrees_with_the_cpp(manifest: ReferenceManifest) -> None:
    """The first second of session 15 off the NAS, scored against the C++ layer beside it.

    A second is thirty framesets, which is enough to prove the lane end to end —
    the rig opens, the four cameras decode, the estimator tracks, the C++ layer
    is read and associated — without paying the whole 52.9 s clip in the test
    suite. The accuracy figure the fleet reports comes from the whole clip.
    """
    session: RobocapSession = manifest.robocap.session("s00000015")
    if not session.base_path.is_file():
        pytest.skip(f"{session.base_path} is not mounted on this host")
    row: RobocapRow
    estimate: Trajectory
    row, estimate = measure(manifest, session, Config(seconds=1.0, window_s=5.0), this_machine())
    assert row.segment_id == "robocap-s15"
    assert row.lost == 0
    assert row.tracked == row.framesets
    assert row.cpp_rmse_cm < 20.0
    assert len(estimate) == row.tracked
    assert row.ms_per_frameset > 0.0
    assert row.cross_platform_ate_cm is None
    # The manifest's own measured C++ wall, not a None the report has to fill in.
    assert row.cpp_wall_s == 88.91


def test_the_only_session_with_a_measured_cpp_wall_is_the_one_that_has_one(manifest: ReferenceManifest) -> None:
    """A column the tool can never fill teaches the next reader that the measurement does not exist.

    The 88.91 s was measured once, on cap A, and had been typed into the report
    by hand while ``measure`` wrote ``None`` unconditionally. Session 21 has no
    C++ wall and still says so.
    """
    assert manifest.robocap.session("s00000015").expected_cpp_wall_s == 88.91
    assert manifest.robocap.session("s00000021").expected_cpp_wall_s is None


@pytest.mark.parametrize("profile", ["reference", "fast"])
def test_result_carries_the_profile_and_resolved_config(
    profile: Literal["reference", "fast"],
    manifest: ReferenceManifest,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The exported session identifies the config used under either profile."""
    monkeypatch.setattr(robocap_fleet, "measure", lambda *_args: (ROW, empty_trajectory()))
    output: Path = tmp_path / "robocap.json"
    main(Config(profile=profile, output_json=output))
    written: dict = json.loads(output.read_text())
    resolved: str = profiled_config_text(manifest.package_root / manifest.robocap.vio_config, profile, manifest.package_root / "configs/profiles")
    assert written["profile"] == profile
    assert written["config_sha256"] == hashlib.sha256(resolved.encode("utf-8")).hexdigest()
