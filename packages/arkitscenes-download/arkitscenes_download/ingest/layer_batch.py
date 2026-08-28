"""Skip-if-exists batch execution for per-segment layer generators."""

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from beartype.roar import BeartypeException


def segment_ids_from_selection(video_id: str | None, video_ids_file: Path | None) -> list[str]:
    """Resolve the mutually exclusive ``--video-id`` / ``--video-ids-file`` selection."""
    if (video_id is None) == (video_ids_file is None):
        raise SystemExit("provide exactly one of --video-id or --video-ids-file")
    if video_id is not None:
        return [video_id]
    assert video_ids_file is not None
    return [line.strip() for line in video_ids_file.read_text().splitlines() if line.strip()]


@dataclass(slots=True)
class LayerBatchSummary:
    """Per-segment outcomes of one batch run."""

    done: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)


def run_layer_batch(
    video_ids: list[str],
    output_path_for: Callable[[str], Path],
    generate: Callable[[str, Path], str],
    register: Callable[[str, Path], None] | None,
    *,
    force: bool,
    label: str,
) -> LayerBatchSummary:
    """Generate and register one layer per segment, resuming from files already on disk.

    An existing output file skips generation but is still registered: registration
    is idempotent (REPLACE) and cheap, and re-applying it is what makes a file left
    behind by a failed or disabled registration eventually reach the catalog.

    Args:
        video_ids: Segments to process, in order.
        output_path_for: The layer file a segment produces.
        generate: Writes one segment's layer to the given path; returns a short
            description for the progress line (for example ``"651 frames"``).
        register: Registers one written layer, or None to only write files.
        force: Regenerate segments whose output already exists.
        label: Batch name used in the summary line.

    Returns:
        Which segments completed, were skipped (existing file, registration
        re-applied), or failed. A failure never stops the batch; the caller
        decides the exit status.
    """
    summary: LayerBatchSummary = LayerBatchSummary()
    for video_id in video_ids:
        output_path: Path = output_path_for(video_id)
        existing: bool = output_path.is_file() and not force
        start: float = time.perf_counter()
        try:
            description: str = "exists" if existing else generate(video_id, output_path)
            if register is not None:
                register(video_id, output_path)
        except BeartypeException:
            raise
        except Exception as error:
            summary.failed.append(video_id)
            print(f"FAIL {video_id}: {type(error).__name__}: {error}", flush=True)
            continue
        if existing:
            summary.skipped.append(video_id)
            print(f"SKIP {video_id}: {output_path} exists — registered, not regenerated (use --force)", flush=True)
            continue
        summary.done.append(video_id)
        print(
            f"DONE {video_id}: {description} in {time.perf_counter() - start:.1f}s "
            f"({len(summary.done)} done, {len(summary.skipped)} skipped, {len(summary.failed)} failed)",
            flush=True,
        )
    print(f"{label}: {len(summary.done)} done, {len(summary.skipped)} skipped, {len(summary.failed)} failed of {len(video_ids)}")
    if summary.failed:
        print("failed segments: " + ", ".join(summary.failed))
    return summary
