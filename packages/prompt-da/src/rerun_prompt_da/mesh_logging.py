"""Fused-TSDF-mesh logging shared by the catalog and demo tools.

Kept dependency-light (rerun, numpy, open3d only) because it is imported from
both the catalog lane and the demo env, which do not share heavier deps such
as ``arkitscenes_download``.
"""

import numpy as np
import open3d as o3d
import rerun as rr


def log_fused_mesh(
    recording: rr.RecordingStream | None,
    entity_path: str,
    mesh: o3d.geometry.TriangleMesh,
    *,
    static: bool = True,
) -> None:
    """Log a fused TSDF mesh, statically by default.

    Same see-through treatment as the ARKit mesh: cull back-facing walls so the
    3D view looks into the room from outside. Pass ``static=False`` to log the
    mesh at the current time instead (incremental reconstruction views), and
    ``recording=None`` to use the global recording stream.
    """
    rr.log(
        entity_path,
        rr.Mesh3D(
            vertex_positions=np.asarray(mesh.vertices),
            triangle_indices=np.asarray(mesh.triangles),
            vertex_normals=np.asarray(mesh.vertex_normals),
            vertex_colors=np.asarray(mesh.vertex_colors),
            face_rendering=rr.components.MeshFaceRendering.Front,
        ),
        static=static,
        recording=recording,
    )
