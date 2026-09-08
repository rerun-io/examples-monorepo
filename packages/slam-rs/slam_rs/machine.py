"""What the machine a row was measured on is, in the terms a fleet table reads it by.

A number off one host means nothing beside a number off another unless the two
hosts are named, so every fleet row carries its machine: the hostname, the
architecture, the C library that limits where a compiled core can travel, the
cores the decode may use, the peak resident set a 2 GB device is judged on and —
on a fanless board that throttles — how hot the die got. Both fleet tools report
these and neither owns them, which is why they live here rather than inside the
one that happened to need them first.
"""

import os
import platform
import resource
from dataclasses import dataclass
from pathlib import Path

THERMAL_ZONES: str = "sys/class/thermal/thermal_zone*/temp"
"""Where Linux publishes die temperatures, relative to the root; the cap has seven zones and a Mac has none."""


@dataclass(slots=True, frozen=True)
class Machine:
    """The host a row was measured on."""

    hostname: str
    """What the machine calls itself."""
    arch: str
    """``platform.machine()``: the port is built for ``x86_64`` and ``aarch64``."""
    libc: str
    """The C library the compiled core is linked against, which is what limits where a pack can go."""
    cores: int
    """Cores visible to the process; the estimator runs single-threaded, the decode does not."""


def this_libc() -> str:
    """The C library this machine offers, in the terms that decide where a compiled core can travel.

    ``CS_GNU_LIBC_VERSION`` is a glibc name and macOS raises ``ValueError`` on
    it rather than returning ``None``, so it has to be looked up before it is
    asked for. Where there is no glibc the answer is the macOS release, because
    that is what the extension binds ``libSystem`` from.
    """
    if "CS_GNU_LIBC_VERSION" in os.confstr_names:
        version: str | None = os.confstr("CS_GNU_LIBC_VERSION")
        if version is not None:
            return version.removeprefix("glibc ")
    macos: str = platform.mac_ver()[0]
    return f"libSystem, macOS {macos}" if macos else "unknown"


def this_peak_rss_mb() -> float:
    """The largest resident set this process has held, in megabytes.

    ``ru_maxrss`` counts kilobytes on Linux and bytes on macOS. The unit cannot
    be guessed, because on the constrained target this number is the whole
    question.
    """
    peak: int = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024.0 * 1024.0) if platform.system() == "Darwin" else peak / 1024.0


def this_machine() -> Machine:
    """What this host is, as a row names it."""
    return Machine(hostname=platform.node(), arch=platform.machine(), libc=this_libc(), cores=os.cpu_count() or 0)


def this_temperature_c(root: Path = Path("/")) -> float | None:
    """Warmest thermal zone this machine publishes, in degrees, or None where it publishes none.

    The number matters on the cap and only there: it is a fanless board in a
    plastic shell that throttles, and a wall measured on a hot die is not the
    wall a cold one gives. Every zone is read and the largest wins, because
    which zone is the SoC differs per board — on the cap zones 0-3 and 6 track
    together and 4-5 sit a degree lower. A zone that cannot be read is skipped
    rather than raised on: a temperature is context for a wall, and no wall
    should be lost to a sysfs file that went away between the glob and the read.

    Args:
        root: Filesystem root to read the zones under; the parameter exists so a
            test can hand over a directory instead of the machine it runs on.
    """
    readings: list[float] = []
    for zone in root.glob(THERMAL_ZONES):
        try:
            readings.append(int(zone.read_text().strip()) / 1000.0)
        except (OSError, ValueError):
            continue
    return max(readings) if readings else None
