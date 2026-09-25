"""``dataforge-convert``: write the layer rrds of one or every sequence."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from socket import gethostname

from beartype.roar import BeartypeException

from dataforge import paths
from dataforge.datasets import AnnotatedDatasetUnion, RobocapConfig
from dataforge.datasets.base import DataforgeDataset, DataforgeDatasetConfig
from dataforge.identity import SequenceIdentity
from dataforge.timing import ConvertRecord, append_record, capture_span, sequence_timer
from dataforge.writing import CONVERT_SCHEMA_VERSION


@dataclass
class Config:
    """Convert discovered sequences into their base (and derived) layer recordings."""

    dataset: AnnotatedDatasetUnion = field(default_factory=RobocapConfig)
    """Dataset to convert; the raw-tree location lives on the dataset config."""
    sequence: str | None = None
    """Convert only this sequence (its ``sequence_key``, ``recording_id``, or any part)."""
    force: bool = False
    """Rewrite recordings that already exist instead of skipping them."""


def select(discovered: list[tuple[SequenceIdentity, object]], sequence: str | None) -> list[tuple[SequenceIdentity, object]]:
    """Filter discovered ``(identity, source)`` pairs by key, recording id, or a single part."""
    if sequence is None:
        return discovered
    matches: list[tuple[SequenceIdentity, object]] = [
        (identity, source)
        for identity, source in discovered
        if sequence in (identity.sequence_key, identity.recording_id) or sequence in identity.parts
    ]
    if not matches:
        raise ValueError(f"No sequence matches {sequence!r} among {len(discovered)} discovered sequences")
    return matches


def converter_version() -> str:
    """Identify the conversion schema and the checkout containing this source."""
    try:
        commit: str = subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "--short=12", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.SubprocessError, OSError):
        return CONVERT_SCHEMA_VERSION
    return f"{CONVERT_SCHEMA_VERSION}+{commit}"


def file_stamps(targets: dict[str, Path]) -> dict[str, tuple[int, int]]:
    """``(mtime_ns, size)`` of each existing layer file; a changed stamp means this run wrote it."""
    stamps: dict[str, tuple[int, int]] = {}
    for layer, path in targets.items():
        if path.is_file():
            stat: os.stat_result = path.stat()
            stamps[layer] = (stat.st_mtime_ns, stat.st_size)
    return stamps


def main(config: Config) -> None:
    """Convert every selected sequence serially, surviving individual failures.

    One bad sequence must not throw away a batch that is hours in (a lesson from
    robocap's malformed ``s10``), so failures are reported and the run continues;
    the process still exits non-zero so a caller can tell a partial run apart
    from a clean one.
    """
    dataset_config: DataforgeDatasetConfig = config.dataset
    dataset: DataforgeDataset = dataset_config.setup()
    selected: list[tuple[SequenceIdentity, object]] = select(dataset.discover(), config.sequence)
    print(f"converting {len(selected)} sequence(s)")
    failed: list[str] = []
    source_version: str = converter_version()
    root: Path = paths.output_root()
    for identity, source in selected:
        targets: dict[str, Path] = {layer: paths.rrd_path(root, layer=layer, identity=identity) for layer in dataset.layers}
        before: dict[str, tuple[int, int]] = file_stamps(targets)
        failure: str | None = None
        capture_s: float | None = None
        with sequence_timer() as timer:
            try:
                target: Path = dataset.convert(identity, source, force=config.force)
                if not target.is_file():
                    raise RuntimeError(f"convert produced no recording for {identity.sequence_key}")
                if any(before.get(layer) != stamp for layer, stamp in file_stamps(targets).items()):
                    capture_s = capture_span(target)
            except BeartypeException:
                raise
            except Exception as error:
                failure = f"{type(error).__name__}: {error}"
                print(f"FAILED {identity.sequence_key}: {failure}")
                failed.append(identity.sequence_key)
        elapsed: float = timer.total_s
        written: dict[str, int] = {layer: stamp[1] for layer, stamp in file_stamps(targets).items() if before.get(layer) != stamp}
        append_record(
            root / "timing/convert.jsonl",
            ConvertRecord(
                dataset=dataset_config.name,
                recording_id=identity.recording_id,
                converter_version=source_version,
                started_at=timer.started_at,
                stage_s=timer.stage_s,
                total_s=elapsed,
                capture_s=capture_s,
                layer_bytes=written,
                host=gethostname(),
                skipped=not written and failure is None,
                error=failure,
            ),
        )
    if failed:
        raise SystemExit(f"{len(failed)} of {len(selected)} sequence(s) failed: {', '.join(failed)}")
