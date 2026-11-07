"""The TSDF mesh the app grows while DPVO tracks, logged block by block under ``dpvo/mesh``.

:class:`metrictrail.capture.CaptureSession` feeds it each frame on which DPVO's keyframe graph grew, at DPVO's current
estimate of that frame's pose. Later bundle adjustment does not move what was integrated, so the run's final mesh
(:mod:`metrictrail.surface`, final poses) replaces it (:meth:`LiveSurface.retire`).
"""

from dataclasses import dataclass, field

import rerun as rr
from jaxtyping import Float32, UInt8
from numpy import ndarray
from tsdf_rs import TsdfVolume

from metrictrail import entities
from metrictrail.surface import DEFAULT_TSDF, MeshArrays, TsdfSettings, integrate_frame, new_volume


@dataclass(slots=True)
class LiveSurface:
    """A TSDF volume of one recording's tracking pass and the cadence that logs its changed blocks."""

    rec: rr.RecordingStream
    """Recording the mesh blocks are logged into (the app's streamed recording)."""
    settings: TsdfSettings = DEFAULT_TSDF
    """Voxel size, truncation and depth range of the volume."""
    log_every: int = 8
    """Integrated frames between two logs of the changed blocks."""
    volume: TsdfVolume = field(init=False)
    """The volume of the current pass."""
    integrated: int = field(init=False, default=0)
    """Frames integrated into :attr:`volume`."""

    def __post_init__(self) -> None:
        self.volume = new_volume(self.settings)

    def reset(self) -> None:
        """Clear the logged mesh and start an empty volume: a new pass (FP32 recovery) re-tracks from the start."""
        self.rec.log(entities.DPVO_MESH, rr.Clear(recursive=True), static=True)
        self.volume = new_volume(self.settings)
        self.integrated = 0

    def integrate(
        self,
        depth_m: Float32[ndarray, "h w"],
        rgb: UInt8[ndarray, "h w 3"],
        intrinsics_px: Float32[ndarray, "4"],
        world_T_cam: Float32[ndarray, "4 4"],
    ) -> None:
        """Integrate one frame at ``world_T_cam``; log the changed blocks every :attr:`log_every` integrated frames."""
        if not integrate_frame(self.volume, self.settings, depth_m, intrinsics_px, world_T_cam, rgb):
            return
        self.integrated += 1
        if self.integrated % self.log_every == 0:
            self.flush()

    def flush(self) -> None:
        """Log the blocks changed since the last flush (a block whose surface vanished is cleared)."""
        blocks: list[tuple[tuple[int, int, int], MeshArrays]] = self.volume.extract_dirty_block_meshes()
        for (x, y, z), (vertices_xyz, triangles, colors_rgb, normals_xyz) in blocks:
            path: str = f"{entities.DPVO_MESH}/{x}_{y}_{z}"
            if triangles.shape[0] == 0:
                self.rec.log(path, rr.Clear(recursive=False), static=True)
                continue
            self.rec.log(
                path,
                rr.Mesh3D(vertex_positions=vertices_xyz, triangle_indices=triangles, vertex_normals=normals_xyz, vertex_colors=colors_rgb),
                static=True,
            )

    def retire(self) -> None:
        """Clear the logged mesh once the run's final mesh supersedes it."""
        self.rec.log(entities.DPVO_MESH, rr.Clear(recursive=True), static=True)
