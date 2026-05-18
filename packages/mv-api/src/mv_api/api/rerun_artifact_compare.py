import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class RRDStatsQueryResult:
    """Parsed output from ``rerun rrd stats`` for one artifact."""

    rrd_path: Path
    """RRD file that was queried."""
    returncode: int
    """Process return code from the Rerun CLI."""
    stdout: str
    """Raw standard output from the Rerun CLI."""
    stderr: str
    """Raw standard error from the Rerun CLI."""
    overview: dict[str, float]
    """Numeric overview values parsed from the stats output."""
    entity_chunk_counts: dict[str, int]
    """Number of chunks per entity path parsed from the stats output."""


@dataclass(slots=True)
class RRDComparisonResult:
    """Result from comparing two Rerun artifacts with ``rerun rrd compare``."""

    expected_rrd_path: Path
    """Oracle RRD file path."""
    actual_rrd_path: Path
    """Candidate RRD file path."""
    returncode: int
    """Process return code from the Rerun CLI."""
    stdout: str
    """Raw standard output from the Rerun CLI."""
    stderr: str
    """Raw standard error from the Rerun CLI."""
    exact_match: bool
    """Whether the Rerun CLI reported the two artifacts as matching."""


def _parse_numeric_prefix(raw_value: str) -> float:
    normalized: str = raw_value.replace("\u2009", "").strip()
    first_token: str = normalized.split()[0]
    cleaned: str = first_token.rstrip("%")
    return float(cleaned)


def _parse_rrd_stats_stdout(stdout: str) -> tuple[dict[str, float], dict[str, int]]:
    overview: dict[str, float] = {}
    entity_chunk_counts: dict[str, int] = {}
    in_overview: bool = False
    in_entity_counts: bool = False

    for raw_line in stdout.splitlines():
        line: str = raw_line.strip()
        if line == "Overview":
            in_overview = True
            in_entity_counts = False
            continue
        if line == "Num chunks per entity":
            in_overview = False
            in_entity_counts = True
            continue
        if not line or set(line) == {"-"}:
            continue

        if in_overview and " = " in line:
            key, raw_value = line.split(" = ", maxsplit=1)
            overview[key] = _parse_numeric_prefix(raw_value)
            continue

        if in_entity_counts and line.startswith("/") and ": " in line:
            entity_path, raw_count = line.rsplit(": ", maxsplit=1)
            entity_chunk_counts[entity_path] = int(raw_count.replace("\u2009", ""))

    return overview, entity_chunk_counts


def query_rrd_stats(*, rrd_path: Path) -> RRDStatsQueryResult:
    """Query an RRD file with the Rerun CLI and parse numeric stats."""
    stats_result: subprocess.CompletedProcess[str] = subprocess.run(
        ["rerun", "rrd", "stats", str(rrd_path)],
        check=False,
        text=True,
        capture_output=True,
    )
    overview: dict[str, float]
    entity_chunk_counts: dict[str, int]
    overview, entity_chunk_counts = _parse_rrd_stats_stdout(stats_result.stdout)
    if stats_result.returncode == 0 and not overview and not entity_chunk_counts:
        msg: str = "Unable to parse `rerun rrd stats` output; CLI format may have changed."
        raise ValueError(msg)
    return RRDStatsQueryResult(
        rrd_path=rrd_path,
        returncode=stats_result.returncode,
        stdout=stats_result.stdout,
        stderr=stats_result.stderr,
        overview=overview,
        entity_chunk_counts=entity_chunk_counts,
    )


def compare_rrd_files(*, expected_rrd_path: Path, actual_rrd_path: Path) -> RRDComparisonResult:
    """Compare two RRD files with the Rerun CLI."""
    compare_result: subprocess.CompletedProcess[str] = subprocess.run(
        ["rerun", "rrd", "compare", str(expected_rrd_path), str(actual_rrd_path)],
        check=False,
        text=True,
        capture_output=True,
    )
    return RRDComparisonResult(
        expected_rrd_path=expected_rrd_path,
        actual_rrd_path=actual_rrd_path,
        returncode=compare_result.returncode,
        stdout=compare_result.stdout,
        stderr=compare_result.stderr,
        exact_match=compare_result.returncode == 0,
    )
