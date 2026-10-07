"""Small synthetic Claude session fixtures."""

import base64
import socket
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import orjson
import pyarrow as pa
import pytest
import rerun_cli
from rerun.chunk import RrdReader

from agent_traces import claude, codex
from agent_traces.events import Session


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
    return SessionBuilder(tmp_path / ".claude/projects/project/session-123.jsonl")


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


@pytest.fixture
def png_base64(png_bytes: bytes) -> str:
    """Encoded synthetic PNG for source metadata."""
    return base64.b64encode(png_bytes).decode()


@pytest.fixture
def png_block(png_base64: str) -> dict[str, object]:
    """Claude content block for the synthetic PNG."""
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": png_base64}}


@pytest.fixture
def rerun_binary() -> Path:
    """Use the active environment's viewer and catalog binary."""
    binary: Path = Path(rerun_cli.__file__).parent / "rerun"
    if not binary.is_file():
        pytest.skip("Rerun binary missing from the active environment")
    return binary


@pytest.fixture
def free_port() -> int:
    """Choose a free loopback port for a disposable Rerun process."""
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        return reservation.getsockname()[1]


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
            completed_at_ms=1789761601250,
            item={"type": item_type, **fields},
        )


@pytest.fixture
def rollout_builder(tmp_path: Path) -> RolloutBuilder:
    """Create a synthetic Codex home."""
    return RolloutBuilder(tmp_path / ".codex/sessions/2026/09/18/rollout-thread.jsonl")


def read_entities(path: Path) -> dict[str, pa.Table]:
    """Read saved entity columns through the Rerun file reader."""
    batches: dict[str, list[pa.RecordBatch]] = {}
    for chunk in RrdReader(path).stream().to_chunks():
        batch: pa.RecordBatch = chunk.to_record_batch()
        # Scalar style is static; this helper returns samples and recording properties.
        if "SeriesLines:names" in batch.schema.names and "wall" not in batch.schema.names:
            continue
        batches.setdefault(str(chunk.entity_path), []).append(batch)
    return {
        entity: pa.concat_tables([pa.Table.from_batches([batch]) for batch in parts], promote_options="default") for entity, parts in batches.items()
    }


def parse_session(path: Path) -> Session:
    """Parse the inventoried Claude source through the public provider boundary."""
    return claude.session_source(path).parse()


def parse_rollout(path: Path) -> Session:
    """Parse the inventoried Codex source through the public provider boundary."""
    return codex.session_source(path).parse()


def metadata_values(table: pa.Table, key: str) -> list[list[object]]:
    """Read one sparse metadata key from saved JSON without changing the recording schema."""
    values: list[dict[str, object]] = [orjson.loads(row[0]) for row in table["metadata_json"].to_pylist()]
    return [[row[key]] if key in row else [] for row in values]


def agent_rows(table: pa.Table, agent_id: str) -> pa.Table:
    """Select saved rows by their typed agent identity; an empty ID denotes the main agent."""
    return table.take([index for index, value in enumerate(table["agent_id"].to_pylist()) if value == [agent_id]])
