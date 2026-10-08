"""Native PLY import and optional compute-viewer selection."""

import warnings
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import Float32
from numpy import ndarray
from rerun.chunk import ChunkStore, RrdReader

# Canonical DC basis; consumers cast to their output precision. Keep f64 here
# so the seeded synthetic initialization remains byte-identical.
SH_C0: float = 0.28209479177387814
SPLATS_ENTITY: str = "/world/splats"
SPLATS_VISUALIZER: str = "ComputeGaussianSplats3D"


def compute_visualizer(render_mode: Literal["default", "mip"] = "default") -> rrb.Visualizer:
    """Explicit custom-viewer choice; stock viewers cannot execute this visualizer."""
    warnings.warn("ComputeGaussianSplats3D requires the custom viewer; stock Rerun will show no splats for this explicit override.", stacklevel=2)
    descriptor: rr.ComponentDescriptor = rr.ComponentDescriptor("ComputeGaussianSplats3D:render_mode", component_type="rerun.components.Text")
    return rrb.Visualizer(SPLATS_VISUALIZER, overrides=[rr.components.TextBatch([render_mode]).described(descriptor)])


def log_ply(path: Path, *, recording: rr.RecordingStream | None = None) -> Float32[ndarray, "2 3"]:
    """Import static native splats and return their 2nd/98th percentile camera bounds.

    Rerun owns PLY decoding. Its chunk reader preserves the native components while
    keeping the logger's entity path and camera framing independent of file names.
    """
    # Clear before the importer allocates row IDs: a newer static clear hides older chunks.
    rr.log(SPLATS_ENTITY, rr.Clear(recursive=True), static=True, recording=recording)
    with TemporaryDirectory(prefix="gsplat-ply-") as directory:
        rrd: Path = Path(directory) / "native.rrd"
        source: rr.RecordingStream = rr.RecordingStream("gsplat-ply-import", send_properties=False)
        source.save(rrd)
        try:
            rr.log_file_from_contents("splats.ply", path.read_bytes(), static=True, recording=source)
            source.flush(timeout_sec=30.0)
        finally:
            source.disconnect()
        store: ChunkStore = RrdReader(rrd).stream().filter(content="/splats.ply").map(lambda chunk: chunk.with_entity_path(SPLATS_ENTITY)).collect()
        centers: Float32[ndarray, "n 3"] = np.concatenate(
            [
                chunk.to_record_batch().column("GaussianSplats3D:centers").flatten().flatten().to_numpy().reshape(-1, 3)
                for chunk in store.stream().to_chunks()
            ]
        )
        rr.send_chunks(store, recording=recording)
        return np.percentile(centers, [2.0, 98.0], axis=0).astype(np.float32)
