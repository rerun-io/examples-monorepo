"""Native PLY import and optional compute-viewer selection."""

import warnings
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

import numpy as np
import pyarrow as pa
import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import Float32
from numpy import ndarray
from rerun.chunk import ChunkStore, RrdReader

from gsplat_rust_renderer.nerfbaselines import DEFAULT_SCENE

# Canonical DC basis; consumers cast to their output precision. Keep f64 here
# so the seeded synthetic initialization remains byte-identical.
SH_C0: float = 0.28209479177387814
SPLATS_ENTITY: str = "/world/splats"
SPLATS_VISUALIZER: str = "ComputeGaussianSplats3D"


def compute_visualizer(render_mode: Literal["default", "mip"] = "default") -> rrb.Visualizer:
    """Explicit custom-viewer choice; stock viewers cannot execute this visualizer."""
    warnings.warn("ComputeGaussianSplats3D requires the custom viewer; stock Rerun will show no splats for this explicit override.", stacklevel=2)
    descriptor: rr.ComponentDescriptor = rr.ComponentDescriptor(
        f"{SPLATS_VISUALIZER}:render_mode", archetype=SPLATS_VISUALIZER, component_type="rerun.components.Text"
    )
    return rrb.Visualizer(SPLATS_VISUALIZER, overrides=[rr.components.TextBatch([render_mode]).described(descriptor)])


@dataclass(frozen=True)
class SplatSource:
    """Shared source and rendering choices for the logging tools."""

    compute: bool = False
    """Explicitly select the custom compute visualizer; normally each viewer selects its renderer."""
    render_mode: Literal["default", "mip"] | None = None
    """Per-entity compute rendering mode, stored in the blueprint."""
    scene: str = DEFAULT_SCENE
    """nerfbaselines scene whose pretrained PLY is loaded when --ply-path is omitted."""
    ply_path: Path | None = None
    """Explicit Gaussian splat .ply path; defaults to the pretrained --scene PLY."""


def splat_overrides(compute: bool, render_mode: Literal["default", "mip"] | None) -> dict[str, rrb.Visualizer]:
    """Validate mode selection and produce the shared splat override."""
    if render_mode is not None and not compute:
        raise ValueError("--render-mode requires --compute")
    return {SPLATS_ENTITY: compute_visualizer(render_mode or "default")} if compute else {}


def log_ply(path: Path, *, entity: str = SPLATS_ENTITY, recording: rr.RecordingStream | None = None) -> Float32[ndarray, "2 3"]:
    """Import static native splats and return their 2nd/98th percentile camera bounds.

    Rerun owns PLY decoding. Its chunk reader preserves the native components while
    keeping the logger's entity path and camera framing independent of file names.
    """
    with TemporaryDirectory(prefix="gsplat-ply-") as directory:
        rrd: Path = Path(directory) / "native.rrd"
        source: rr.RecordingStream = rr.RecordingStream("gsplat-ply-import", send_properties=False)
        source.save(rrd)
        try:
            # Allocate the clear before imported row IDs, in the private recording.
            rr.log("/splats.ply", rr.Clear(recursive=True), static=True, recording=source)
            rr.log_file_from_contents("splats.ply", path.read_bytes(), static=True, recording=source)
            source.flush(timeout_sec=30.0)
        finally:
            source.disconnect()
        store: ChunkStore = RrdReader(rrd).stream().filter(content="/splats.ply").map(lambda chunk: chunk.with_entity_path(entity)).collect()
        batches: list[pa.RecordBatch] = [chunk.to_record_batch() for chunk in store.stream().to_chunks() if "GaussianSplats3D:centers" in chunk.to_record_batch().schema.names]
        if not batches:
            raise ValueError(f"{path}: no Gaussian splat centers")
        centers: Float32[ndarray, "n 3"] = np.concatenate(
            [
                batch.column("GaussianSplats3D:centers").flatten().flatten().to_numpy().reshape(-1, 3)
                for batch in batches
            ]
        )
        if len(centers) == 0 or not np.isfinite(centers).all():
            raise ValueError(f"{path}: empty or nonfinite Gaussian splat centers")
        bounds: Float32[ndarray, "2 3"] = np.percentile(centers, [2.0, 98.0], axis=0).astype(np.float32)
        rr.send_chunks(store, recording=recording)
        return bounds
