"""Tests for shared fused-mesh Rerun logging."""

import numpy as np
import open3d as o3d
import pytest
import rerun as rr

from simplecv.ops.tsdf_depth_fuser import log_fused_mesh


def test_log_fused_mesh_uses_entity_first_and_keyword_recording(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the common entity/data arguments positional and logging controls keyword-only."""
    mesh: o3d.geometry.TriangleMesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.zeros((3, 3), dtype=np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(np.array([[0, 1, 2]], dtype=np.int32))
    mesh.vertex_normals = o3d.utility.Vector3dVector(np.zeros((3, 3), dtype=np.float64))
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.ones((3, 3), dtype=np.float64))
    recording: rr.RecordingStream = rr.RecordingStream(application_id="test-mesh-logging", recording_id="mesh")
    captured: dict[str, object] = {}

    def fake_log(entity_path: str, archetype: rr.Mesh3D, *, static: bool, recording: rr.RecordingStream | None) -> None:
        captured.update(entity_path=entity_path, archetype=archetype, static=static, recording=recording)

    monkeypatch.setattr(rr, "log", fake_log)

    log_fused_mesh("world/mesh", mesh, recording=recording, static=False)

    assert captured["entity_path"] == "world/mesh"
    assert isinstance(captured["archetype"], rr.Mesh3D)
    assert captured["static"] is False
    assert captured["recording"] is recording
