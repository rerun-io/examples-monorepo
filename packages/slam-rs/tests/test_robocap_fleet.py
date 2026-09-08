"""What a RoboCap fleet row says, and what it reads its own cost against.

The tool this covers runs on machines with no NAS, no repository and — on the cap
— no pixi and no viewer, so what is under test is the part that needs none of
that: the budget arithmetic a row reports, how the row reads, and the two facts
about the machine it carries beside the numbers. The replay itself is the same
:func:`slam_rs.tracking.run_robocap` the probe drives and the ``slow`` test below
is what runs it on the real rig.
"""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from slam_rs.apis import robocap_fleet
from slam_rs.apis.robocap_fleet import BUDGET_15FPS_MS, BUDGET_30FPS_MS, Config, RobocapRow, main, measure
from slam_rs.machine import Machine, this_machine
from slam_rs.reference import MANIFEST_PATH, ReferenceManifest, RobocapSession
from slam_rs.trajectory import Trajectory, empty_trajectory

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


def test_both_outputs_survive_a_directory_that_is_not_there_yet(
    manifest: ReferenceManifest, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A run that measured 52.9 s of video must not lose it to a missing ``out/``."""
    monkeypatch.setattr(robocap_fleet, "measure", lambda *_args: (ROW, empty_trajectory()))
    output: Path = tmp_path / "out" / "robocap_fleet.json"
    main(Config(manifest=MANIFEST_PATH, output_json=output))
    assert json.loads(output.read_text())["segment_id"] == "robocap-s15"
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

