"""Provider dispatch for multi-provider conversion commands."""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from serde import from_dict, serde

from agent_traces import claude, codex
from agent_traces.sources import Discovery, SessionSource, iter_jsonl


@runtime_checkable
class Provider(Protocol):
    """Provider operations used by both commands."""

    def discover(self, home: Path) -> Discovery: ...
    def session_source(self, path: Path) -> SessionSource: ...


@serde
@dataclass(frozen=True, slots=True)
class _FirstRecord:
    """Minimal provider discriminator, independent of either record schema."""

    type: str = ""
    """First record tag."""
    id: str = ""
    """Legacy Codex identity."""
    timestamp: str | None = None
    """Legacy Codex timestamp."""


def provider_for(path: Path) -> Provider:
    """Detect a home by layout or a transcript by its first record."""
    path = path.expanduser()
    if not path.is_file():
        for provider in (claude, codex):
            if provider.is_home(path):
                return provider
        raise ValueError(f"{path}: neither Claude nor Codex home layout found")
    def decode(raw: dict[str, object], index: int) -> _FirstRecord:
        """Read the discriminator without interpreting provider fields."""
        return from_dict(_FirstRecord, raw)

    for first in iter_jsonl(path, decode, skipped=None):
        return codex if first.type == "session_meta" or (not first.type and first.id and first.timestamp) else claude
    raise ValueError(f"{path}: no decodable records")


def discover(home: Path) -> Discovery:
    """Choose a provider by home layout and discover its sessions once."""
    for provider in (claude, codex):
        if provider.is_home(home):
            return provider.discover(home)
    raise ValueError(f"{home}: neither Claude nor Codex home layout found")


def session_source(path: Path) -> SessionSource:
    """Resolve a single transcript using its detected provider."""
    return provider_for(path).session_source(path)
