"""The TSDF mesh the app grows while DPVO tracks, logged under ``dpvo/mesh`` as it grows.

:class:`metrictrail.capture.CaptureSession` feeds it each frame on which DPVO's keyframe graph grew, at DPVO's current
estimate of that frame's pose. The changed super-blocks are logged on the timeline (scrubbing shows the growth); at the
pass's last integrated frame the blocks are cleared and the whole surface is logged as one mesh
(:class:`simplecv.rerun_live_mesh.LiveMeshLog`). Later bundle adjustment does not move what was integrated; the run's
final mesh (:mod:`metrictrail.surface`, final poses) is logged separately, in the world tab.
"""

from dataclasses import dataclass, field

import rerun as rr
from jaxtyping import Float32, UInt8
from numpy import ndarray
from simplecv.rerun_live_mesh import LiveMeshLog
from tsdf_rs import TsdfVolume

from metrictrail import entities
from metrictrail.surface import DEFAULT_TSDF, TsdfSettings, integrate_frame, new_volume


@dataclass(slots=True)
class LiveSurface:
    """A TSDF volume of one recording's tracking pass and the cadence that logs its changed blocks."""

    rec: rr.RecordingStream
    """Recording the mesh is logged into (the app's streamed recording)."""
    settings: TsdfSettings = DEFAULT_TSDF
    """Voxel size, truncation and depth range of the volume."""
    log_every: int = 8
    """Integrated frames between two logs of the changed blocks."""
    volume: TsdfVolume = field(init=False)
    """The volume of the current pass."""
    integrated: int = field(init=False, default=0)
    """Frames integrated into :attr:`volume`."""
    last_frame: int | None = field(init=False, default=None)
    """Tracked-frame index of the last integrated frame: :meth:`finish` logs the whole surface there."""
    mesh_log: LiveMeshLog = field(init=False)
    """The ``dpvo/mesh`` entities."""

    def __post_init__(self) -> None:
        self.volume = new_volume(self.settings)
        self.mesh_log = LiveMeshLog(entities.DPVO_MESH, self.rec)

    def reset(self) -> None:
        """Clear the logged blocks from this time point on and start an empty volume: a new pass (FP32 recovery)
        re-tracks from the start. Its blocks overlap the earlier pass's on the frames both tracked."""
        if self.integrated > 0:
            self.mesh_log.restart()
        self.volume = new_volume(self.settings)
        self.integrated = 0
        self.last_frame = None

    def integrate(
        self,
        depth_m: Float32[ndarray, "h w"],
        rgb: UInt8[ndarray, "h w 3"],
        intrinsics_px: Float32[ndarray, "4"],
        world_T_cam: Float32[ndarray, "4 4"],
        frame: int,
    ) -> None:
        """Integrate tracked frame ``frame`` at ``world_T_cam``; log the changed blocks every :attr:`log_every`
        integrated frames, at the time point the caller set."""
        if not integrate_frame(self.volume, self.settings, depth_m, intrinsics_px, world_T_cam, rgb):
            return
        self.integrated += 1
        self.last_frame = frame
        if self.integrated % self.log_every == 0:
            self.mesh_log.log_changed(self.volume.extract_dirty_block_meshes())

    def finish(self) -> None:
        """The pass's end: at its last integrated frame, clear the blocks and log the whole surface as one mesh."""
        if self.last_frame is None:
            return
        self.rec.set_time(entities.FRAME_TIMELINE, sequence=self.last_frame)
        self.mesh_log.finish(self.volume.extract_mesh())
