"""What a RoboCap fleet row says, and what it reads its own cost against.

The tool this covers runs on machines with no NAS, no repository and — on the cap
— no pixi and no viewer, so what is under test is the part that needs none of
that: the budget arithmetic a row reports, how the row reads, and the two facts
about the machine it carries beside the numbers. The replay itself is the same
:func:`slam_rs.tracking.run_robocap` the probe drives and the ``slow`` test below
is what runs it on the real rig.
"""

from dataclasses import replace
from pathlib import Path

import pytest

from slam_rs.apis.fleet_check import Machine
from slam_rs.apis.robocap_fleet import FRAMESET_BUDGET_MS, RobocapRow, measure, this_temperature_c
from slam_rs.reference import ReferenceManifest, RobocapSession, load_manifest
from slam_rs.trajectory import Trajectory

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
    realtime_factor_15fps=FRAMESET_BUDGET_MS[15] / 100.0,
    realtime_factor_30fps=FRAMESET_BUDGET_MS[30] / 100.0,
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
    assert FRAMESET_BUDGET_MS[15] == pytest.approx(66.67, abs=0.01)
    assert FRAMESET_BUDGET_MS[30] == pytest.approx(33.33, abs=0.01)
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


def test_the_warmest_zone_wins_and_an_unreadable_one_is_skipped(tmp_path: Path) -> None:
    """The cap publishes seven zones that disagree by a degree, and any of them may vanish.

    A temperature is context for a wall, so a zone that cannot be parsed must
    cost the reading nothing — losing a 158 s measurement to a sysfs file is the
    wrong trade.
    """
    zones: Path = tmp_path / "sys" / "class" / "thermal"
    for index, millidegrees in enumerate(["40700", "39800", "51800\n"]):
        (zones / f"thermal_zone{index}").mkdir(parents=True)
        (zones / f"thermal_zone{index}" / "temp").write_text(millidegrees)
    (zones / "thermal_zone3").mkdir()
    (zones / "thermal_zone3" / "temp").write_text("not a number")
    assert this_temperature_c(tmp_path) == pytest.approx(51.8)


def test_a_machine_with_no_thermal_zones_reports_none(tmp_path: Path) -> None:
    """macOS publishes no zones, and the column has to stay empty rather than read 0 °C."""
    assert this_temperature_c(tmp_path) is None


def test_the_session_is_named_the_way_a_fleet_row_names_it() -> None:
    """``s00000015`` in the manifest is ``robocap-s15`` in a row, for both sessions."""
    manifest: ReferenceManifest = load_manifest()
    assert [session.fleet_id for session in manifest.robocap.sessions] == ["robocap-s15", "robocap-s21"]


@pytest.mark.slow
def test_the_real_session_replays_on_this_machine_and_agrees_with_the_cpp() -> None:
    """The first second of session 15 off the NAS, scored against the C++ layer beside it.

    A second is thirty framesets, which is enough to prove the lane end to end —
    the rig opens, the four cameras decode, the estimator tracks, the C++ layer
    is read and associated — without paying the whole 52.9 s clip in the test
    suite. The accuracy figure the fleet reports comes from the whole clip.
    """
    manifest: ReferenceManifest = load_manifest()
    session: RobocapSession = manifest.robocap.session("s00000015")
    row: RobocapRow
    estimate: Trajectory
    row, estimate = measure(manifest, session, seconds=1.0, window_s=5.0, reference_csv=None)
    assert row.segment_id == "robocap-s15"
    assert row.lost == 0
    assert row.tracked == row.framesets
    assert row.cpp_rmse_cm < 20.0
    assert len(estimate) == row.tracked
    assert row.ms_per_frameset > 0.0
    assert row.cross_platform_ate_cm is None
