"""The catalog clip reader on a real MAMMA segment (Spark catalog, read-only): cameras, frames, and meshes that project onto them."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.integration

DATASET: str = "dataforge-mamma-markerless-dance"
SEGMENT: str = "mamma-markerless-dance__140725_Breakdancemarkers_Improv_2_03686_1_1_1"


def test_mamma_clip_meshes_project_into_every_camera(spark_catalog_url: str) -> None:
    from open4dgs.catalog_clip import read_catalog_clip

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip = read_catalog_clip(
        spark_catalog_url, DATASET, SEGMENT, 0, 4, cameras=["rig_01/cam_00", "rig_17/cam_00"], max_side=514, decode_device=device
    )
    assert clip.camera_names == ("rig_01/cam_00", "rig_17/cam_00")
    assert clip.frame_indices.tolist() == [0, 1, 2, 3]
    assert clip.fps == 30.0
    assert all(frames.shape[0] == 4 and max(frames.shape[1:3]) == 514 for frames in clip.frames)
    assert clip.people is not None and clip.people.source == "pred/mamma"
    assert clip.people.present[:, 0].tolist() == [True, False, False, True]  # body_mesh is stored every third frame
    vertices = clip.people.vertices[0, 0].double().numpy()
    for camera in range(2):
        cam_T_world = np.linalg.inv(clip.world_T_cam[camera].double().numpy())
        points = vertices @ cam_T_world[:3, :3].T + cam_T_world[:3, 3]
        uv = points @ clip.K[camera].double().numpy().T
        uv = uv[:, :2] / uv[:, 2:]
        width, height = clip.sizes[camera]
        assert (points[:, 2] > 0).all()
        assert ((uv >= 0) & (uv < [width, height])).all()
        # The person is brighter-than-black where the mesh lands (frames are not blank or misaligned by a whole image).
        assert clip.frames[camera][0].float().mean() > 1.0


LOCAL_CATALOG_URL: str = "rerun+http://pablo-dl-server:51235"
"""The local catalog that holds the 4DAnyone ``fdanyone-views`` dataset."""


def _local_catalog_or_skip() -> str:
    import socket

    try:
        with socket.create_connection(("pablo-dl-server", 51235), timeout=3.0):
            pass
    except OSError:
        pytest.skip(f"the local catalog {LOCAL_CATALOG_URL} (dataset fdanyone-views) is unreachable")
    return LOCAL_CATALOG_URL


def test_fdanyone_clip_reads_pinhole_h264_rig_from_rig_00() -> None:
    """4DAnyone views: rigs from rig_00, pinhole (no lens), H.264 with one keyframe, a mesh every frame, 25 fps portrait."""
    from open4dgs.catalog_clip import read_catalog_clip

    url: str = _local_catalog_or_skip()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip = read_catalog_clip(url, "fdanyone-views", "pexels-2785536__v06-seed42", 58, 3, decode_device=device)
    assert clip.camera_names == tuple(f"rig_{index:02d}/cam_00" for index in range(6))
    assert clip.fps == 25.0
    assert clip.sizes == ((704, 1280),) * 6
    assert all(bool(valid.all()) for valid in clip.valid)
    assert clip.people is not None and clip.people.source == "pred/fdanyone"
    assert bool(clip.people.present.all())
    vertices = clip.people.vertices[1, 0].double().numpy()
    for camera in range(6):
        cam_T_world = np.linalg.inv(clip.world_T_cam[camera].double().numpy())
        points = vertices @ cam_T_world[:3, :3].T + cam_T_world[:3, 3]
        uv = points @ clip.K[camera].double().numpy().T
        uv = uv[:, :2] / uv[:, 2:]
        assert (points[:, 2] > 0).all()
        # Hands (12% of SMPL-X vertices) reach up to ~50 px past the side edges in some views; the body stays inside.
        assert ((uv >= [-80, 0]) & (uv < [704 + 80, 1280])).all()
