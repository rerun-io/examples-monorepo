"""Stable sequence identity: one value derives key, recording_id, and filename.

Borrowed from simplecv's exoego ``sequence_identity.py``; dataforge keeps the
"id = filename = key = segment" rule, so the recording_id doubles as the rrd
filename stem (see ``dataforge.paths.rrd_path``).
"""

from __future__ import annotations

from dataclasses import dataclass


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
    """Stable identity for one dataset sequence (= one recording, one base rrd)."""

    dataset: str
    """Dataset name, a single path part (e.g. ``robocap``)."""
    parts: tuple[str, ...]
    """Sequence parts within the dataset (e.g. ``(device, session, segment)``)."""

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
        """Human-readable identity within the dataset."""
        return "/".join(self.parts)

    @property
    def recording_id(self) -> str:
        """Stable Rerun recording/segment ID; includes dataset to avoid cross-dataset collisions."""
        return "__".join((self.dataset, *self.parts))
