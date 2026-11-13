"""Sparse colour TSDF fusion and marching cubes (Rust: packages/tsdf-rs/crates/tsdf).

Arrays are C-contiguous and row-major; a Fortran-ordered or strided array raises ValueError. World units are those of
the voxel size (metres in MetricTrail). Depth is camera Z, not ray length. The GIL is released while the volume
integrates or extracts.
"""

from typing import TypeAlias

import numpy as np
from jaxtyping import Float32, UInt8, UInt32

_Mesh: TypeAlias = tuple[
    Float32[np.ndarray, "v 3"],
    UInt32[np.ndarray, "t 3"],
    UInt8[np.ndarray, "v 3"] | None,
    Float32[np.ndarray, "v 3"],
]
"""``(vertices, triangles, colors, normals)``: world positions; vertex indices, counter-clockwise seen from outside;
RGB of the nearest voxel (None without colour fusion); unit TSDF-gradient normals, pointing toward the cameras."""

class TsdfVolume:
    """Sparse, unbounded TSDF in 16x16x16-voxel blocks; voxel ``(x, y, z)`` samples ``((x, y, z) + 0.5) * voxel_size``."""

    def __init__(self, voxel_size: float, sdf_trunc: float, fuse_color: bool) -> None:
        """``voxel_size`` and ``sdf_trunc`` (truncation distance) are finite and positive; ``fuse_color`` averages RGB."""
        ...
    def integrate(
        self,
        depth_hw: Float32[np.ndarray, "h w"],
        K_33: Float32[np.ndarray, "3 3"],
        cam_T_world_44: Float32[np.ndarray, "4 4"],
        color_hw3: UInt8[np.ndarray, "h w 3"] | None = None,
        *,
        min_depth: float = 0.0,
        max_depth: float = ...,
    ) -> None:
        """Fuse one depth frame seen by a pinhole camera ``K_33`` at the rigid world-to-camera ``cam_T_world_44``.

        Pixels with depth outside ``[min_depth, max_depth]`` (default: every positive finite depth), non-finite or
        non-positive are skipped.
        """
        ...
    def extract_mesh(self) -> _Mesh:
        """The mesh of every observed zero crossing; deterministic, independent of the thread count."""
        ...
    def extract_dirty_block_meshes(self) -> list[tuple[tuple[int, int, int], _Mesh]]:
        """Per-block meshes of the blocks changed since the last call, by sorted block key.

        A block whose mesh is byte-identical to the one this method last returned for it is left out.

        The union of the latest mesh per key equals :meth:`extract_mesh` triangle for triangle; a vertex on a block
        seam has the same position, colour and normal in every block. A block whose surface vanished comes once with
        an empty mesh; blocks that never had surface are omitted.
        """
        ...
