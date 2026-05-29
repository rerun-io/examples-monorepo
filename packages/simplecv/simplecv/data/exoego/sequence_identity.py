from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def _normalize_part(value: object) -> tuple[str, ...]:
    raw: str = str(value).strip().replace("\\", "/")
    parts: tuple[str, ...] = tuple(part for part in raw.split("/") if part)
    if any(part in {".", ".."} for part in parts):
        raise ValueError(f"Invalid sequence identity part: {value!r}")
    if any("__" in part for part in parts):
        raise ValueError(f"Sequence identity parts cannot contain '__': {value!r}")
    return parts


@dataclass(frozen=True)
class SequenceIdentity:
    """Stable catalog identity for one exo/ego sequence."""

    dataset: str
    parts: tuple[str, ...]

    def __post_init__(self) -> None:
        dataset_parts: tuple[str, ...] = _normalize_part(self.dataset)
        if len(dataset_parts) != 1:
            raise ValueError(f"Dataset identity must be one path part, got {self.dataset!r}")

        normalized_parts: tuple[str, ...] = tuple(part for value in self.parts for part in _normalize_part(value))
        if not normalized_parts:
            raise ValueError("Sequence identity needs at least one part")

        object.__setattr__(self, "dataset", dataset_parts[0])
        object.__setattr__(self, "parts", normalized_parts)

    @property
    def sequence_key(self) -> str:
        """Human-readable identity within the catalog Dataset."""
        return "/".join(self.parts)

    @property
    def recording_id(self) -> str:
        """Stable Rerun segment ID. Includes dataset to avoid cross-Dataset collisions."""
        return "__".join((self.dataset, *self.parts))

    def rrd_path(self, root: Path) -> Path:
        """Return ``<root>/<dataset>/<parts...>.rrd``."""
        sequence_dir: Path = root / self.dataset
        for part in self.parts[:-1]:
            sequence_dir /= part
        return sequence_dir / f"{self.parts[-1]}.rrd"
