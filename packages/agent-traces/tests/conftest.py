"""Small synthetic Claude session fixtures."""

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import orjson
import pytest


@dataclass(slots=True)
class SessionBuilder:
    """Write deterministic JSONL records without real transcript data."""

    path: Path
    """Main transcript path."""
    index: int = 0
    """Next timestamp step in seconds."""
    records: list[dict[str, object]] = field(default_factory=list)
    """Synthetic records written by this builder."""

    def add(self, kind: str, *, path: Path | None = None, **fields: object) -> None:
        """Append a record at the next whole-second timestamp."""
        target: Path = path or self.path
        target.parent.mkdir(parents=True, exist_ok=True)
        stamp: datetime = datetime(2026, 9, 18, 20, tzinfo=UTC) + timedelta(seconds=self.index)
        record: dict[str, object] = {"type": kind, "timestamp": stamp.isoformat(), **fields}
        with target.open("ab") as stream:
            stream.write(orjson.dumps(record) + b"\n")
        self.records.append(record)
        self.index += 1


@pytest.fixture
def session_builder(tmp_path: Path) -> SessionBuilder:
    """Create a Claude home with a deterministic session id."""
    return SessionBuilder(tmp_path / ".claude-alt/projects/project/session-123.jsonl")


@pytest.fixture
def png_bytes() -> bytes:
    """Generate one opaque red PNG pixel with standard-library encoders."""
    import struct
    import zlib

    def chunk(kind: bytes, data: bytes) -> bytes:
        """Encode one PNG chunk with its checksum."""
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"\x00\xff\x00\x00"))
        + chunk(b"IEND", b"")
    )
