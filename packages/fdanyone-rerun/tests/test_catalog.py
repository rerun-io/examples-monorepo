"""Unit tests for the run -> catalog segment mapping (no GPU, no catalog server)."""

import math

import numpy as np
import pytest
import rerun as rr
import rerun.chunk as rrc


def _look_from(yaw_deg: float, pitch_deg: float, distance: float = 2.5) -> np.ndarray:
    """world_T_cam of an OpenCV camera on a ring around the origin, looking at it (world Y up)."""
    yaw, pitch = math.radians(yaw_deg), math.radians(pitch_deg)
    forward = np.array([math.sin(yaw) * math.cos(pitch), -math.sin(pitch), -math.cos(yaw) * math.cos(pitch)])
    right = np.cross(forward, [0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    pose = np.eye(4)
    pose[:3, :3] = np.stack([right, down, forward], axis=1)
    pose[:3, 3] = -forward * distance
    return pose


def test_segment_id_names_clip_views_and_seed() -> None:
    from fdanyone_rerun.catalog import segment_id

    assert segment_id("pexels-2785536", 6, 42) == "pexels-2785536__v06-seed42"
    assert segment_id("pexels-2785536", 18, 7) == "pexels-2785536__v18-seed7"
    with pytest.raises(ValueError):
        segment_id("Pexels 2785536", 6, 42)


@pytest.mark.parametrize(
    ("run_entity", "is_pinhole", "expected"),
    [
        ("/world/cameras/00/video", False, "/world/rig_00/cam_00/pinhole/video"),
        ("/world/cameras/17", True, "/world/rig_17/cam_00/pinhole"),
        ("/world/cameras/17", False, "/world/rig_17/cam_00"),
        ("/world/source", True, "/world/source/pinhole"),
        ("/world/source", False, "/world/source"),
        ("/world/source/video", False, "/world/source/pinhole/video"),
        ("/world/body/mesh", False, "/world/pred/fdanyone/people/person_00/body/mesh"),
        ("/world/body/keypoints", False, "/world/pred/fdanyone/people/person_00/mhr70_xyz"),
        ("/previews/03", False, None),
        ("/log", False, None),
    ],
)
def test_catalog_entity_maps_the_run_layout_onto_rigs(run_entity: str, is_pinhole: bool, expected: str | None) -> None:
    from fdanyone_rerun.catalog import catalog_entity

    assert catalog_entity(run_entity, is_pinhole=is_pinhole) == expected


def test_catalog_entities_are_what_the_rig_reader_parses() -> None:
    from simplecv.catalog_rig_layout import CatalogComponent, parse_rig_layout

    from fdanyone_rerun.catalog import catalog_entity

    components: list[CatalogComponent] = []
    for view in range(18):
        node = catalog_entity(f"/world/cameras/{view:02d}", is_pinhole=False)
        pinhole = catalog_entity(f"/world/cameras/{view:02d}", is_pinhole=True)
        video = catalog_entity(f"/world/cameras/{view:02d}/video", is_pinhole=False)
        assert node is not None and pinhole is not None and video is not None
        components += [
            CatalogComponent.parse(node, "Transform3D:mat3x3", True),
            CatalogComponent.parse(pinhole, "Pinhole:image_from_camera", True),
            CatalogComponent.parse(video, "VideoStream:sample", False),
        ]
    layout = parse_rig_layout(components)
    assert [camera.name for camera in layout.cameras] == [f"rig_{view:02d}/cam_00" for view in range(18)]
    assert all(camera.has_calibration and not camera.rig_is_moving for camera in layout.cameras)


@pytest.mark.parametrize(("yaw", "pitch"), [(0.0, 15.0), (60.0, 15.0), (300.0, -10.0), (180.0, 45.0)])
def test_view_pose_recovers_ring_yaw_and_pitch(yaw: float, pitch: float) -> None:
    from fdanyone_rerun.catalog import view_pose

    pose = view_pose(_look_from(yaw, pitch))
    assert pose.yaw_deg == pytest.approx(yaw, abs=1e-3)
    assert pose.pitch_deg == pytest.approx(pitch, abs=1e-3)


def test_cam_T_world_inverts_the_pose() -> None:
    from fdanyone_rerun.catalog import cam_T_world

    pose = _look_from(37.0, 20.0)
    np.testing.assert_allclose(cam_T_world(pose) @ pose, np.eye(4), atol=1e-12)
    # The origin (the subject) sits in front of the camera, on its optical axis.
    np.testing.assert_allclose(cam_T_world(pose) @ [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 2.5, 1.0], atol=1e-12)


def test_retime_moves_data_onto_frame_index_and_video_time() -> None:
    from fdanyone_rerun.catalog import FRAME_INDEX, VIDEO_TIME, retime

    frames = np.arange(5)
    chunk = rrc.Chunk.from_columns(
        "/world/body/keypoints",
        indexes=[rr.TimeColumn("time", duration=frames / 25.0), rr.TimeColumn("frame", sequence=frames)],
        columns=rr.Points3D.columns(positions=np.arange(15, dtype=np.float32).reshape(5, 3)),
    )
    moved = retime(chunk, "/world/pred/fdanyone/people/person_00/mhr70_xyz", 25.0)
    batch = moved.to_record_batch()
    assert moved.entity_path == "/world/pred/fdanyone/people/person_00/mhr70_xyz"
    assert sorted(moved.timeline_names) == [FRAME_INDEX, VIDEO_TIME]
    assert batch.column(FRAME_INDEX).to_pylist() == list(range(5))
    assert batch.column(VIDEO_TIME).cast("int64").to_pylist() == [0, 40_000_000, 80_000_000, 120_000_000, 160_000_000]
    assert batch.column("Points3D:positions").to_pylist() == chunk.to_record_batch().column("Points3D:positions").to_pylist()


def test_camera_chunks_store_cam_T_world_as_the_calibration_reader_expects() -> None:
    from simplecv.catalog_calibration import camera_models_from_static_table

    from fdanyone_rerun.catalog import RunCamera, camera_chunks

    pose = _look_from(120.0, 15.0)
    intrinsics = np.array([[1664.0, 0.0, 352.0], [0.0, 1664.0, 640.0], [0.0, 0.0, 1.0]])
    camera = RunCamera(view=4, world_T_cam=pose, intrinsics=intrinsics, width=704, height=1280)
    pinhole = rrc.Chunk.from_columns(
        "/world/rig_04/cam_00/pinhole", indexes=[], columns=rr.Pinhole.columns(image_from_camera=[intrinsics], resolution=[[704.0, 1280.0]])
    )
    table = rrc.ChunkStore.from_chunks([*camera_chunks(camera, codec="h264"), pinhole]).reader(index=None).to_arrow_table()
    parsed = camera_models_from_static_table(table, ["/world/rig_04/cam_00"], "test")["/world/rig_04/cam_00"].pinhole
    assert parsed.extrinsics.cam_R_world is not None and parsed.extrinsics.cam_t_world is not None
    assert parsed.intrinsics.k_matrix is not None
    np.testing.assert_allclose(parsed.extrinsics.cam_R_world, pose[:3, :3].T, atol=1e-6)
    np.testing.assert_allclose(parsed.extrinsics.cam_t_world, -pose[:3, :3].T @ pose[:3, 3], atol=1e-6)
    np.testing.assert_allclose(parsed.intrinsics.k_matrix, intrinsics)
    assert table.column("/world/rig_04/cam_00:yaw_deg").to_pylist()[0] == [pytest.approx(120.0, abs=1e-3)]
