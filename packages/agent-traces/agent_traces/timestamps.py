"""Exact ISO wall timestamps shared by transcript parsers."""

import re
from datetime import datetime, timedelta

TIMESTAMP_PATTERN: re.Pattern[str] = re.compile(
    r"([0-9]{4})-([0-9]{2})-([0-9]{2})T([0-9]{2}):([0-9]{2}):([0-9]{2})"
    r"(?:[.,]([0-9]{1,9}))?(Z|[+-][0-9]{2}:[0-9]{2})"
)


def parse_timestamp_ns(text: str) -> int:
    """Parse a zoned ISO timestamp with one to nine optional fractional digits."""
    matched: re.Match[str] | None = TIMESTAMP_PATTERN.fullmatch(text)
    if matched is None:
        raise ValueError(f"invalid timestamp: {text!r}")
    stamp: datetime = datetime(
        int(matched.group(1)),
        int(matched.group(2)),
        int(matched.group(3)),
        int(matched.group(4)),
        int(matched.group(5)),
        int(matched.group(6)),
    )
    zone: str = matched.group(8)
    offset_seconds: int = 0
    if zone != "Z":
        hours: int = int(zone[1:3])
        minutes: int = int(zone[4:6])
        if hours > 23 or minutes > 59:
            raise ValueError(f"invalid timestamp offset: {zone!r}")
        offset_seconds = (hours * 3600 + minutes * 60) * (1 if zone[0] == "+" else -1)
    delta: timedelta = stamp - datetime(1970, 1, 1)
    fraction_ns: int = int((matched.group(7) or "").ljust(9, "0"))
    return (delta.days * 86400 + delta.seconds - offset_seconds) * 1_000_000_000 + fraction_ns
