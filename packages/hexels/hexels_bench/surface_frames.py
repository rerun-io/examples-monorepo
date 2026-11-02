"""Per-view surface frames (RGB, median and expected depth, alpha) from a `gsplat render --surface`
recording or from upstream's CUDA renderer, in camera order.

Both sources yield the same `SurfaceFrame`, so fusion and parity read depth only through this module.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import rerun.chunk as rrc
from jaxtyping import Float32, UInt8
from numpy import ndarray

from hexels_bench.cameras import ContractCamera

FRAME_PARTS: tuple[str, ...] = ("rgb", "median_depth", "expected_depth", "alpha")
"""Entities under `world/camera/<surface-name>/` that fusion reads (the CLI also logs `normal`)."""


@dataclass(frozen=True, slots=True)
class SurfaceFrame:
    """One rendered view."""

    name: str
    """Camera name (image stem)."""
    rgb: UInt8[ndarray, "h w 3"]
    """Rendered color."""
    median_depth: Float32[ndarray, "h w"]
    """Camera z of the last hit while transmittance > 0.5; 0 without a hit."""
    expected_depth: Float32[ndarray, "h w"]
    """Alpha-weighted camera z divided by alpha; 0 without a hit."""
    alpha: Float32[ndarray, "h w"]
    """1 - final transmittance."""


def _decode(part: str, payload: ndarray, camera: ContractCamera) -> ndarray:
    if part == "rgb":
        return payload.reshape(camera.height, camera.width, 3)
    if part == "normal":
        return payload.view(np.float32).reshape(camera.height, camera.width, 3)
    return payload.view(np.float32).reshape(camera.height, camera.width)


def rrd_views(
    rrd: Path, surface_name: str, parts: tuple[str, ...], cameras: list[ContractCamera], keep: set[int] | None = None
) -> Iterator[tuple[int, dict[str, ndarray]]]:
    """`(view, {part: image})` of a surface recording in `view` order, reading its chunks lazily (a T&T
    recording is tens of GB): a view is yielded as soon as it and all earlier views are complete, then
    dropped. Views outside `keep` are skipped without decoding."""
    entities: dict[str, str] = {f"/world/camera/{surface_name}/{part}": part for part in parts}
    wanted: list[int] = sorted(keep) if keep is not None else list(range(len(cameras)))
    pending: dict[int, dict[str, ndarray]] = {}
    position: int = 0
    for chunk in rrc.RrdReader(rrd).store().stream().filter(content=list(entities)):
        part: str = entities[chunk.entity_path]
        batch: pa.RecordBatch = chunk.to_record_batch()
        buffer_column: str = next(name for name in batch.schema.names if name.endswith(":buffer"))
        instances: pa.Array = batch.column(buffer_column).flatten()
        for row, view in enumerate(batch.column("view").to_numpy().tolist()):
            if keep is None or view in keep:
                payload: ndarray = instances[row].values.to_numpy(zero_copy_only=False)
                pending.setdefault(view, {})[part] = _decode(part, payload, cameras[view])
        while position < len(wanted) and len(pending.get(wanted[position], {})) == len(parts):
            yield wanted[position], pending.pop(wanted[position])
            position += 1
    if position != len(wanted) or pending:
        raise ValueError(f"{rrd}: {position} complete views of {len(wanted)} wanted ({len(pending)} incomplete)")


def rrd_frames(rrd: Path, surface_name: str, cameras: list[ContractCamera]) -> Iterator[SurfaceFrame]:
    """Every frame of a `gsplat render --surface` recording, in camera order."""
    for view, parts in rrd_views(rrd, surface_name, FRAME_PARTS, cameras):
        yield SurfaceFrame(
            name=cameras[view].name,
            rgb=parts["rgb"],
            median_depth=parts["median_depth"],
            expected_depth=parts["expected_depth"],
            alpha=parts["alpha"],
        )
