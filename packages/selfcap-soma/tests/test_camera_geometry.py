import ast
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("cv2", reason="requires OpenCV projection backend")
from selfcap_soma.camera_geometry import Camera, project, triangulate_frames


def test_reused_geometry_matches_source_ast():
    root = Path(__file__).resolve().parents[3]
    original = ast.parse((root / "packages/selfcap-curate/selfcap_curate/keypoints.py").read_text())
    reused = ast.parse((root / "packages/selfcap-soma/selfcap_soma/camera_geometry.py").read_text())
    names = {"Camera", "EgoCamera", "TriangulationBatch", "camera_transforms", "project", "project_ego", "triangulate_frames"}
    original_nodes = {
        n.name: ast.dump(n, include_attributes=False) for n in original.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names
    }
    reused_nodes = {
        n.name: ast.dump(n, include_attributes=False) for n in reused.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names
    }
    assert reused_nodes == original_nodes


def test_two_view_triangulation_recovers_point_without_third_view(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    intrinsics = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]])
    right_pose = np.eye(4)
    right_pose[0, 3] = 1.0
    cameras = [Camera(intrinsics, np.eye(4), (0.0, 0.0), "a"), Camera(intrinsics, right_pose, (0.0, 0.0), "b")]
    point = np.array([[0.0, 0.0, 5.0]])
    pixels = np.array([[[[50.0, 50.0]], [[30.0, 50.0]]]])
    result = triangulate_frames(cameras, pixels, np.ones((1, 2, 1)))
    np.testing.assert_allclose(result.xyz[0], point, atol=1e-5)
    np.testing.assert_allclose(project(cameras[1], point)[0], [30.0, 50.0, 5.0])
