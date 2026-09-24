"""Small synthetic Claude session fixtures."""

from dataclasses import dataclass
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

    def add(self, kind: str, *, path: Path | None = None, **fields: object) -> None:
        """Append a record at the next whole-second timestamp."""
        target: Path = path or self.path
        target.parent.mkdir(parents=True, exist_ok=True)
        stamp: datetime = datetime(2026, 9, 18, 20, tzinfo=UTC) + timedelta(seconds=self.index)
        record: dict[str, object] = {"type": kind, "timestamp": stamp.isoformat(), "uuid": f"record-{self.index}", **fields}
        with target.open("ab") as stream:
            stream.write(orjson.dumps(record) + b"\n")
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


@dataclass(slots=True)
class RolloutBuilder:
    """Synthetic Codex envelopes, with no private rollout data."""

    path: Path
    """Rollout destination."""
    index: int = 0
    """Next timestamp step."""

    def add(self, kind: str, **payload: object) -> None:
        """Append a payload in a deterministic envelope."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        stamp: datetime = datetime(2026, 9, 18, 20, tzinfo=UTC) + timedelta(seconds=self.index)
        with self.path.open("ab") as stream:
            stream.write(orjson.dumps({"timestamp": stamp.isoformat(), "type": kind, "payload": payload}) + b"\n")
        self.index += 1

    def meta(self, thread_id: str = "thread", version: str = "0.153.4", **fields: object) -> None:
        """Write the required metadata."""
        self.add("session_meta", id=thread_id, cli_version=version, model_provider="openai", **fields)

    def item(self, item_type: str, turn_id: str = "turn", **fields: object) -> None:
        """Complete one item with known execution timing."""
        self.add(
            "event_msg",
            type="item_completed",
            turn_id=turn_id,
            started_at_ms=1789761601000,
            completed_at_ms=1789761601250,
            item={"type": item_type, **fields},
        )


@pytest.fixture
def rollout_builder(tmp_path: Path) -> RolloutBuilder:
    """Create a synthetic Codex home."""
    return RolloutBuilder(tmp_path / ".codex-alt/sessions/2026/09/18/rollout-thread.jsonl")
