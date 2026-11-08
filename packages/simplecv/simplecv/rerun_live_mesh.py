"""A TSDF surface that grows on the timeline: changed blocks while the run goes on, one mesh at its end.

Every package that grows a TSDF mesh logs it this way (mast3r-slam, MetricTrail, mvs). While the run goes on, each
changed TSDF block is merged into its super-block (``group`` blocks per edge) and that super-block is logged again as
its own ``Mesh3D`` at the current time point: scrubbing back shows the reconstruction grow, and an update costs only
what changed. At the run's last time point the block subtree is cleared and the whole surface is logged once as a
single mesh, so the end of the recording draws one mesh entity instead of hundreds.

Meshes are ``tsdf_rs`` tuples (``TsdfVolume.extract_mesh`` / ``extract_dirty_block_meshes``); this module does not
import ``tsdf_rs``. The caller sets the time point before each call, as for any ``rr.log``.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import TypeAlias

import numpy as np
import rerun as rr
from jaxtyping import Float32, UInt8, UInt32
from numpy import ndarray

MeshArrays: TypeAlias = tuple[
    Float32[ndarray, "v 3"],
    UInt32[ndarray, "t 3"],
    UInt8[ndarray, "v 3"] | None,
    Float32[ndarray, "v 3"],
]
"""``(vertices, triangles, colors, normals)`` as ``tsdf_rs`` returns them; colors are None without colour fusion."""
BlockKey: TypeAlias = tuple[int, int, int]
"""A TSDF block's integer coordinates."""

BLOCKS: str = "blocks"
"""Child of the root that holds one entity per super-block while the run goes on."""
FULL: str = "full"
"""Child of the root that holds the single mesh of the run's end."""


def merge_meshes(meshes: Sequence[MeshArrays]) -> MeshArrays:
    """One mesh from several: vertices stacked, triangle indices offset; colours only if every part has them."""
    offsets: list[int] = np.cumsum([0] + [mesh[0].shape[0] for mesh in meshes[:-1]]).tolist()
    vertices: Float32[ndarray, "v 3"] = np.concatenate([mesh[0] for mesh in meshes]).astype(np.float32, copy=False)
    triangles: UInt32[ndarray, "t 3"] = np.concatenate([mesh[1] + np.uint32(offset) for mesh, offset in zip(meshes, offsets, strict=True)]).astype(
        np.uint32, copy=False
    )
    colors: UInt8[ndarray, "v 3"] | None = None
    if all(mesh[2] is not None for mesh in meshes):
        colors = np.concatenate([mesh[2] for mesh in meshes if mesh[2] is not None])
    normals: Float32[ndarray, "v 3"] = np.concatenate([mesh[3] for mesh in meshes]).astype(np.float32, copy=False)
    return vertices, triangles, colors, normals


def front_faced_mesh(mesh: MeshArrays) -> rr.Mesh3D:
    """A Mesh3D that renders only front faces (the side the cameras saw), so walls seen from outside a room do not
    hide it."""
    vertices, triangles, colors, normals = mesh
    return rr.Mesh3D(
        vertex_positions=vertices,
        triangle_indices=triangles,
        vertex_colors=colors,
        vertex_normals=normals,
        face_rendering=rr.components.MeshFaceRendering.Front,
    )


@dataclass
class LiveMeshLog:
    """Logs a growing TSDF surface under ``root``: temporal super-blocks while running, one mesh at the end.

    Entities: ``root/blocks/<x>_<y>_<z>`` (super-block coordinates) while the run goes on, ``root/full`` from the
    run's last time point. Nothing is static, so every time point shows the surface as it was then.
    """

    root: str
    """Entity path of the surface, e.g. ``world/mesh``."""
    recording: rr.RecordingStream | None = None
    """Recording to log into; None: the active one."""
    group: int = 4
    """Super-block edge, in TSDF blocks: ``group**3`` blocks share one entity, which cuts the viewer's per-entity cost
    while keeping an update local. The viewer pays per entity on every frame: on MetricTrail's pool clip (4 cm voxels)
    group 2 left 1,320 entities and 28 ms per frame in playback, group 4 322 entities and 10 ms."""
    _groups: dict[BlockKey, dict[BlockKey, MeshArrays]] = field(default_factory=dict, init=False, repr=False)
    """Latest non-empty mesh of every block, by super-block, to rebuild a super-block when one of its blocks changes."""

    def log_changed(self, changed: Iterable[tuple[BlockKey, MeshArrays]]) -> None:
        """Log again every super-block with a changed block (an empty mesh means the block's surface vanished)."""
        dirty: set[BlockKey] = set()
        for key, mesh in changed:
            group_key: BlockKey = (key[0] // self.group, key[1] // self.group, key[2] // self.group)
            blocks: dict[BlockKey, MeshArrays] = self._groups.setdefault(group_key, {})
            if mesh[1].shape[0] == 0:
                blocks.pop(key, None)
            else:
                blocks[key] = mesh
            dirty.add(group_key)
        for group_key in sorted(dirty):
            path: str = f"{self.root}/{BLOCKS}/{group_key[0]}_{group_key[1]}_{group_key[2]}"
            parts: list[MeshArrays] = [mesh for _, mesh in sorted(self._groups[group_key].items())]
            if parts:
                rr.log(path, front_faced_mesh(merge_meshes(parts)), recording=self.recording)
            else:
                rr.log(path, rr.Clear(recursive=False), recording=self.recording)

    def restart(self) -> None:
        """Clear every block from this time point on (a re-fusion that logs all of its blocks again follows)."""
        rr.log(f"{self.root}/{BLOCKS}", rr.Clear(recursive=True), recording=self.recording)
        self._groups = {}

    def finish(self, mesh: MeshArrays) -> None:
        """The run's end: clear the blocks and log ``mesh``, the whole surface, as one entity at this time point."""
        self.restart()
        if mesh[1].shape[0] > 0:
            rr.log(f"{self.root}/{FULL}", front_faced_mesh(mesh), recording=self.recording)
