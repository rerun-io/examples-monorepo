"""What a fleet row says about the machine it was measured on.

Both fleet tools report the same five facts and neither owns them: the C library
that decides where a compiled core can be carried, the peak resident set a 2 GB
device is judged on, and — on a fanless board — how hot the die got. Nothing
here needs a corpus, a viewer or a NAS, which is the point: these run on the
constrained target too.
"""

import os
import platform
import resource
from pathlib import Path
from types import SimpleNamespace

import pytest

from slam_rs.machine import this_libc, this_peak_rss_mb, this_temperature_c


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
