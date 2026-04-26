"""Write the ``view_coordinates`` layer — world-frame axis convention.

What this layer contributes to the composed segment:

- static ``rr.ViewCoordinates`` at ``/world`` matching the dataset's
  published convention (see ``slam_evals.data.datasets``).

Why a dedicated layer (instead of folding into ``calibration.rrd``):

- Convention assignment is its own iteration loop — each new dataset
  added to the spec registry only needs a new ``view_coordinates.rrd``;
  ``calibration.rrd``, ``groundtruth.rrd``, ``rgb_*.rrd``, etc. stay
  byte-identical. This matches the layer model's design intent of one
  orthogonal concern per file.

- Easy to selectively re-emit: ``tools/ingest.py --layers view_coordinates
  --only <slug>`` rewrites only this tiny file.

The layer is omitted entirely for datasets whose ``DatasetSpec`` doesn't
set ``world_view_coordinates`` — the catalog falls back to the viewer's
default world frame for those segments (current behavior).
"""

from __future__ import annotations

from pathlib import Path

import rerun as rr

from slam_evals.data.types import Sequence


def write_view_coordinates_layer(
    sequence: Sequence,
    *,
    out_path: Path,
    application_id: str = "slam-evals",
) -> Path:
    """Write ``view_coordinates.rrd`` for ``sequence``. Returns the output path.

    Caller must verify ``sequence.dataset_spec.world_view_coordinates is not
    None`` first; we ``raise`` otherwise rather than silently emitting an
    empty file.
    """
    spec = sequence.dataset_spec
    if spec is None or spec.world_view_coordinates is None:
        raise ValueError(
            f"sequence {sequence.slug!r} has no world_view_coordinates "
            f"in its DatasetSpec — caller should have gated on this"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    rec = rr.RecordingStream(
        application_id=application_id,
        recording_id=sequence.recording_id,
        send_properties=True,
    )
    with rec:
        rr.log("/world", spec.world_view_coordinates, static=True, recording=rec)

    rec.save(str(out_path))
    return out_path
