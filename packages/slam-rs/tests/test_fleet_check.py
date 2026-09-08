"""What a fleet row says, and when it says the machine failed (D60).

The tool that produces these rows runs on machines with no NAS, no repository
and — on the pack target — no pixi, so what is under test here is the part that
needs none of that: the verdict a clip's numbers earn, and how the row reads.
"""

import os
import platform
import resource
from types import SimpleNamespace

import pytest

from slam_rs.apis.fleet_check import ClipResult, Machine, this_libc, this_peak_rss_mb

MACHINE: Machine = Machine(hostname="pablo-rpi", arch="aarch64", libc="2.36", cores=4)
"""A four-core Pi, which is the smallest machine that runs a full install."""
PASSING: ClipResult = ClipResult(
    segment_id="msd-index__MIO_others__MIO10_short_2_panorama",
    framesets=412,
    tracked=412,
    lost=0,
    cpp_rmse_cm=0.31,
    gt_rmse_cm=1.50,
    cpp_gt_band_cm=(1.427751, 1.427823),
    wall_s=30.0,
    cpp_wall_s=7.674,
    peak_rss_mb=512.0,
)
"""The smoke clip as this host measures it, on a machine four times slower."""


def test_a_clip_inside_the_bands_passes_wherever_it_ran() -> None:
    """The three D60 clauses, and the speed ratio that is not one of them.

    The Pi is four times slower than the x86-64 host the C++ wall was measured
    on. That is a fact about the machine, not a failure of the port, so the ratio
    is reported and the verdict does not read it — the gate's speed clause is
    against a wall measured on one host and means nothing on another.
    """
    assert PASSING.failures == ()
    assert PASSING.verdict == "pass"
    assert PASSING.cpp_wall_ratio == 30.0 / 7.674


def test_every_clause_a_machine_can_miss_is_named_in_its_own_row() -> None:
    """A lost frameset, a path error and a ground-truth error outside the band."""
    from dataclasses import replace

    lost: ClipResult = replace(PASSING, tracked=410, lost=2)
    assert lost.verdict.startswith("fail")
    assert "2 of 412" in lost.failures[0]

    adrift: ClipResult = replace(PASSING, cpp_rmse_cm=2.5, gt_rmse_cm=9.0)
    assert len(adrift.failures) == 2
    assert "2.50 cm from the C++ trajectory" in adrift.failures[0]
    assert "9.00 cm from ground truth" in adrift.failures[1]
    # 1.2 x the worse of the C++'s own two precisions on this clip.
    assert adrift.gt_allowed_cm == 1.2 * 1.427823


def test_the_row_carries_the_machine_beside_the_numbers() -> None:
    """One markdown row: the machine, the clip, both errors, the verdict, the cost."""
    cells: list[str] = [cell.strip() for cell in PASSING.row(MACHINE).strip().strip("|").split("|")]
    assert cells[:4] == ["pablo-rpi", "aarch64", "2.36", "4"]
    assert cells[4].endswith("MIO10_short_2_panorama")
    assert cells[5] == "412/412/0"
    assert cells[6] == "0.31"
    assert cells[7] == "1.50 (allowed 1.71)"
    assert cells[8] == "pass"
    assert cells[9] == "30.00"
    assert cells[10] == "3.91x"
    assert cells[11] == "512"


def test_a_machine_without_glibc_still_names_its_c_library(monkeypatch: pytest.MonkeyPatch) -> None:
    """macOS has no ``CS_GNU_LIBC_VERSION`` and raises on the name, so the row says ``libSystem``.

    The row's third cell is what limits where a compiled core can be carried,
    which on Linux is the glibc version and on macOS is the system release the
    extension binds ``libSystem`` from. Asking for the glibc name on a Mac is a
    ``ValueError``, not a ``None``, so the name has to be looked up before it is
    asked for.
    """
    monkeypatch.setattr(os, "confstr_names", {})
    monkeypatch.setattr(platform, "mac_ver", lambda: ("26.5.1", ("", "", ""), "arm64"))
    assert this_libc() == "libSystem, macOS 26.5.1"


def test_the_peak_resident_set_is_megabytes_on_both_kinds_of_machine(monkeypatch: pytest.MonkeyPatch) -> None:
    """``ru_maxrss`` counts kilobytes on Linux and bytes on macOS, and the pack target is judged on the number.

    The clip that fits in 2 GB is the whole question on the constrained device,
    so a row reading 382,544 MB on a Mac is not a cosmetic slip: it is the one
    figure that decides whether the pack can go anywhere.
    """
    half_a_gigabyte: int = 512 * 1024 * 1024
    monkeypatch.setattr(resource, "getrusage", lambda _who: SimpleNamespace(ru_maxrss=half_a_gigabyte // 1024))
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    assert this_peak_rss_mb() == 512.0
    monkeypatch.setattr(resource, "getrusage", lambda _who: SimpleNamespace(ru_maxrss=half_a_gigabyte))
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    assert this_peak_rss_mb() == 512.0
