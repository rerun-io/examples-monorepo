"""HOT3D public reader contracts with tiny read-only source trees."""

import json
from pathlib import Path

import pytest

from dataforge.datasets.hot3d import Hot3dAriaConfig, Hot3dQuest3Config
from dataforge.datasets.hot3d_source import Device


def test_discovery_checks_sizes_and_sidecars_without_markers(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    groups = {"ground_truth": ["metadata.json", "camera_models.json"], "hand_data": ["hands.jsonl"]}
    sequences = {}
    for name, size in (("P0001_complete", 4), ("P0001_partial", 2), ("P0001_absent", 0)):
        source = tmp_path / "aria" / name
        source.mkdir(parents=True)
        if size:
            (source / "recording.vrs").write_bytes(b"v" * size)
        (source / "metadata.json").write_text(json.dumps({"have_hand_object_pose_gt": True, "participant_id": "P0001", "object_uids": []}))
        for filename in ("camera_models.json", "hands.jsonl", "timecode_devicetime_mapping.csv"):
            (source / filename).touch()
        sequences[name] = {"main_vrs": {"file_size_bytes": 4}}
    (tmp_path / "Hot3DAria_download_urls-20260924.json").write_text(
        json.dumps({"sequences": sequences, "sequence_config": {"data_groups": groups, "release": "v4.0.0"}})
    )
    dataset = Hot3dAriaConfig(root=tmp_path).setup()
    dataset.download()
    found = dataset.discover()
    assert [identity.recording_id for identity, _ in found] == ["hot3d-aria__P0001_complete"]
    assert "skip P0001_partial" in capsys.readouterr().out
    (tmp_path / "aria/P0001_complete/hands.jsonl").unlink()
    assert dataset.discover() == []


def test_no_gt_needs_only_base_inputs(tmp_path: Path) -> None:
    source = tmp_path / "quest3/test_sequence"
    source.mkdir(parents=True)
    (source / "recording.vrs").write_bytes(b"v")
    (source / "camera_models.json").write_text("[]")
    (source / "metadata.json").write_text(json.dumps({"have_hand_object_pose_gt": False, "participant_id": "test", "object_uids": []}))
    (tmp_path / "Hot3DQuest_download_urls-20260924.json").write_text(
        json.dumps(
            {
                "sequences": {source.name: {"main_vrs": {"file_size_bytes": 1}}},
                "sequence_config": {"data_groups": {"ground_truth": ["missing.csv"], "hand_data": ["missing.jsonl"]}, "release": "v4.0.0"},
            }
        )
    )
    dataset = Hot3dQuest3Config(root=tmp_path).setup()
    identity, _ = dataset.discover()[0]
    assert list(dataset.targets(identity)) == ["base"]


def test_exact_mapping_keeps_three_rows_and_mano_uses_own_stamp(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from dataforge.datasets.hot3d_source import LabelClock, read_hands, read_mano, read_masks

    (tmp_path / "timecode_devicetime_mapping.csv").write_text("timecode_ns,devicetime_ns\n100,1000\n110,1010\n122,1022\n")
    clock = LabelClock.read(tmp_path)
    (tmp_path / "umetrack_hand_pose_trajectory.jsonl").write_text(
        "\n".join(json.dumps({"timestamp_ns": stamp, "hand_poses": {}}) for stamp in [100, 110, 122])
    )
    assert list(read_hands(tmp_path, clock)) == [1000, 1010, 1022]
    (tmp_path / "mano_hand_pose_trajectory.jsonl").write_text(
        "\n".join(json.dumps({"timestamp_ns": stamp, "hand_poses": {}}) for stamp in [0, 110, 122])
    )
    assert list(read_mano(tmp_path, clock)) == [1010, 1022]
    (tmp_path / "masks").mkdir()
    (tmp_path / "masks/mask_qa_pass.csv").write_text("timestamp[ns],stream_id,mask\n0,1201-1,True\n110,1201-1,False\n110,1201-2,True\n")
    flags = read_masks(tmp_path, clock)["qa_pass"]
    assert flags["1201-1"][1].tolist() == [False]
    assert flags["1201-2"][1].tolist() == [True]
    assert "dropped 1 MANO" in capsys.readouterr().out
    (tmp_path / "umetrack_hand_pose_trajectory.jsonl").write_text(json.dumps({"timestamp_ns": 111, "hand_poses": {}}))
    with pytest.raises(ValueError, match="matching mapping.timecode_ns"):
        read_hands(tmp_path, clock)


def test_missing_poses_never_hold_previous_row(tmp_path: Path) -> None:
    import numpy as np

    from dataforge.datasets.hot3d_source import LabelClock, pose_array, read_poses

    path = tmp_path / "headset_trajectory.csv"
    path.write_text(
        "object_uid,timestamp[ns],t_wo_x[m],t_wo_y[m],t_wo_z[m],q_wo_w,q_wo_x,q_wo_y,q_wo_z\nhead,10,1,2,3,1,0,0,0\nhead,30,4,5,6,1,0,0,0\n"
    )
    poses = pose_array(read_poses(path, LabelClock({10: 10, 20: 20, 30: 30}))["head"], np.array([10, 20, 30], dtype=np.int64))
    assert np.isnan(poses[1]).all()
    np.testing.assert_array_equal(poses[2, :3, 3], [4, 5, 6])


@pytest.mark.parametrize("dual_focal", [False, True])
def test_rotation_matches_projectaria_calibration(dual_focal: bool) -> None:
    import numpy as np
    from projectaria_tools.core.calibration import CameraCalibration, CameraModelType, rotate_camera_calib_cw90deg
    from projectaria_tools.core.sophus import SE3

    from dataforge import aria
    from dataforge.datasets.hot3d_vrs import CameraModel, CameraTransform

    params = [500.0, 600.0, 480.0, *([0.001] * 6), 0.01, 0.02, 0.003, 0.004, 0.005, 0.006]
    transform = CameraTransform(np.array([1.0, 0.0, 0.0, 0.0]), np.array([0.1, 0.2, 0.3]))
    model = CameraModel("left", "1201-1", 1280, 1024, "CameraModelType.FISHEYE624", [params[0], *params] if dual_focal else params, transform, 1.5)
    native = CameraCalibration("left", CameraModelType.FISHEYE624, np.asarray(params), SE3.from_matrix(transform.matrix()), 1280, 1024, None, 1.5, "")
    rotated = rotate_camera_calib_cw90deg(native)
    expected = aria.fisheye62_from_aria(rotated, rig_T_cam=np.asarray(rotated.get_transform_device_camera().to_matrix()), name="left")
    actual = model.camera(rotate_cw90=True)
    assert (actual.intrinsics.width, actual.intrinsics.height) == (1024, 1280)
    assert actual.intrinsics.k_matrix is not None and expected.intrinsics.k_matrix is not None
    assert actual.extrinsics.world_R_cam is not None and expected.extrinsics.world_R_cam is not None
    assert actual.extrinsics.world_t_cam is not None and expected.extrinsics.world_t_cam is not None
    np.testing.assert_allclose(actual.intrinsics.k_matrix, expected.intrinsics.k_matrix)
    np.testing.assert_allclose(actual.extrinsics.world_R_cam, expected.extrinsics.world_R_cam)
    np.testing.assert_allclose(actual.extrinsics.world_t_cam, expected.extrinsics.world_t_cam)
    assert actual.distortion == expected.distortion
    full_model = model.calibration(rotate_cw90=True)
    np.testing.assert_allclose(full_model.get_projection_params(), rotated.get_projection_params())
    projected = full_model.project(np.array([0.1, 0.2, 1.0]))
    expected_projected = rotated.project(np.array([0.1, 0.2, 1.0]))
    assert projected is not None and expected_projected is not None
    np.testing.assert_allclose(projected, expected_projected)
    if dual_focal:
        from dataclasses import replace

        unequal = replace(model, params=[500.0, 510.0, *params[1:]])
        with pytest.raises(ValueError, match=r"fx=500.0, fy=510.0"):
            unequal.camera(rotate_cw90=True)


def test_fk_missing_hand_is_nan_and_right_hand_is_mirrored() -> None:
    import numpy as np
    from simplecv.umetrack_temp.generic_hand_model_numpy import HandModelNumpy

    from dataforge.datasets.hot3d_hands import evaluate_hands
    from dataforge.datasets.hot3d_source import UmeFrame, UmePose, Wrist

    # A rigid synthetic model: every landmark/vertex belongs to the root frame.
    rest = np.tile(np.array([[10.0, 20.0, 30.0]], dtype=np.float32), (21, 1))
    axes = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (22, 1))
    weights = np.zeros((21, 17), dtype=np.float32)
    weights[:, 0] = 1.0
    model = HandModelNumpy(
        joint_rotation_axes=axes,
        joint_rest_positions=np.zeros((22, 3), dtype=np.float32),
        joint_frame_index=np.zeros(22, dtype=np.int64),
        joint_parent=np.full(22, -1, dtype=np.int64),
        joint_first_child=np.full(22, -1, dtype=np.int64),
        joint_next_sibling=np.full(22, -1, dtype=np.int64),
        landmark_rest_positions=rest,
        landmark_rest_bone_weights=np.ones((21, 1), dtype=np.float32),
        landmark_rest_bone_indices=np.zeros((21, 1), dtype=np.int64),
        hand_scale=np.array(1.0, dtype=np.float32),
        mesh_vertices=rest.copy(),
        mesh_triangles=np.array([[0, 1, 2]], dtype=np.int64),
        dense_bone_weights=weights,
        joint_limits=np.zeros((22, 2), dtype=np.float32),
    )
    wrist = Wrist(np.array([1.0, 2.0, 3.0], dtype=np.float32), np.array([1.0, 0.0, 0.0, 0.0]))
    pose = UmePose(wrist, np.zeros(22, dtype=np.float32), 0.75)
    result = evaluate_hands(model, {10: UmeFrame(10, {"0": pose, "1": pose}), 20: UmeFrame(20, {})}, np.array([10, 20], dtype=np.int64))
    np.testing.assert_allclose(result.positions[0, 91], [1.01, 2.02, 3.03], atol=1e-6)
    np.testing.assert_allclose(result.positions[0, 112], [0.99, 2.02, 3.03], atol=1e-6)
    assert result.confidence[0, 91] == 0.75
    assert np.isnan(result.positions[1]).all()
    assert (result.scores[:, 1] == 0.0).all()
    assert np.isnan(result.angles[:, 1]).all()


@pytest.mark.parametrize("reader_name", ["read_hands", "read_mano", "read_masks", "read_poses"])
@pytest.mark.parametrize("stamps", [(20, 20), (20, 10)])
def test_readers_reject_non_increasing_device_stamps(tmp_path: Path, reader_name: str, stamps: tuple[int, int]) -> None:
    from dataforge.datasets import hot3d_source

    clock = hot3d_source.LabelClock({10: 10, 20: 20})
    if reader_name in ("read_hands", "read_mano"):
        name = "umetrack" if reader_name == "read_hands" else "mano"
        (tmp_path / f"{name}_hand_pose_trajectory.jsonl").write_text(
            "\n".join(json.dumps({"timestamp_ns": stamp, "hand_poses": {}}) for stamp in stamps)
        )
        source = tmp_path
    elif reader_name == "read_masks":
        (tmp_path / "masks").mkdir()
        (tmp_path / "masks/mask_qa_pass.csv").write_text("timestamp[ns],stream_id,mask\n" + "".join(f"{stamp},1201-1,True\n" for stamp in stamps))
        source = tmp_path
    else:
        source = tmp_path / "headset_trajectory.csv"
        source.write_text(
            "object_uid,timestamp[ns],t_wo_x[m],t_wo_y[m],t_wo_z[m],q_wo_w,q_wo_x,q_wo_y,q_wo_z\n"
            + "".join(f"head,{stamp},0,0,0,1,0,0,0\n" for stamp in stamps)
        )
    with pytest.raises(ValueError, match="increasing|duplicate"):
        getattr(hot3d_source, reader_name)(source, clock)


def test_quest_off_grid_label_raises(tmp_path: Path) -> None:
    import numpy as np

    from dataforge.datasets.hot3d_source import Hot3dSource, Metadata, read_labels

    (tmp_path / "headset_trajectory.csv").write_text(
        "object_uid,timestamp[ns],t_wo_x[m],t_wo_y[m],t_wo_z[m],q_wo_w,q_wo_x,q_wo_y,q_wo_z\nhead,11,0,0,0,1,0,0,0\n"
    )
    source = Hot3dSource(tmp_path, Metadata(True, "test", []))
    with pytest.raises(ValueError, match="timestamp_ns=11.*label census"):
        read_labels(source, "quest3", np.array([10, 20], dtype=np.int64), None)


def test_nearest_framesets_clamp_and_tie_goes_left() -> None:
    import numpy as np

    from dataforge.datasets.hot3d_vrs import nearest_framesets

    actual = nearest_framesets(np.array([10, 20, 30], dtype=np.int64), np.array([0, 10, 14, 15, 16, 25, 30, 40], dtype=np.int64))
    np.testing.assert_array_equal(actual, [0, 0, 0, 0, 1, 1, 2, 2])


def test_discovery_skips_truncated_metadata_and_manifest_absent_sequence(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    sequences = {}
    for name in ("complete", "truncated", "unlisted"):
        source = tmp_path / "quest3" / name
        source.mkdir(parents=True)
        (source / "recording.vrs").write_bytes(b"v")
        (source / "camera_models.json").write_text("[]")
        (source / "metadata.json").write_text(
            '{"have_hand_object_pose_gt":'
            if name == "truncated"
            else json.dumps({"have_hand_object_pose_gt": False, "participant_id": "test", "object_uids": []})
        )
        if name != "unlisted":
            sequences[name] = {"main_vrs": {"file_size_bytes": 1}}
    manifest = tmp_path / "Hot3DQuest_download_urls-20260924.json"
    manifest.write_text(json.dumps({"sequences": sequences, "sequence_config": {"data_groups": {}, "release": "v4.0.0"}}))
    dataset = Hot3dQuest3Config(root=tmp_path).setup()
    found = dataset.discover()
    assert [source.path.name for _, source in found] == ["complete"]
    output = capsys.readouterr().out
    assert f"{tmp_path}/quest3/truncated/metadata.json" in output
    assert "skip unlisted: absent from URL list" in output
    # Targets use the discovered metadata even if the source later becomes unreadable.
    (found[0][1].path / "metadata.json").write_text("{")
    assert list(dataset.targets(found[0][0])) == ["base"]
    manifest.write_text("{")
    with pytest.raises((ValueError, json.JSONDecodeError)):
        dataset.discover()


@pytest.mark.parametrize("device", ["aria", "quest3"])
def test_no_gt_census_uses_primary_camera_without_reading_labels(tmp_path: Path, device: Device) -> None:
    import numpy as np

    from dataforge.datasets.hot3d_source import Hot3dSource, Metadata, read_labels

    source = Hot3dSource(tmp_path, Metadata(False, "test", []))
    labels = read_labels(source, device, np.array([10, 20, 30], dtype=np.int64), 25)
    np.testing.assert_array_equal(labels.times_ns, [10, 20])
    assert not labels.hands and not labels.mano and not labels.masks and not labels.headset and not labels.objects


def test_aria_census_keeps_missing_rows_and_preview_validates_full_clock(tmp_path: Path) -> None:
    import numpy as np

    from dataforge.datasets.hot3d_source import Hot3dSource, Metadata, read_labels

    (tmp_path / "timecode_devicetime_mapping.csv").write_text("timecode_ns,devicetime_ns\n100,9\n200,19\n300,29\n")
    header = "object_uid,timestamp[ns],t_wo_x[m],t_wo_y[m],t_wo_z[m],q_wo_w,q_wo_x,q_wo_y,q_wo_z\n"
    (tmp_path / "headset_trajectory.csv").write_text(header + "head,100,0,0,0,1,0,0,0\nhead,300,0,0,0,1,0,0,0\n")
    (tmp_path / "dynamic_objects.csv").write_text(header)
    for name in ("umetrack", "mano"):
        (tmp_path / f"{name}_hand_pose_trajectory.jsonl").write_text(
            "\n".join(json.dumps({"timestamp_ns": stamp, "hand_poses": {}}) for stamp in (100, 300))
        )
    source = Hot3dSource(tmp_path, Metadata(True, "test", []))
    labels = read_labels(source, "aria", np.array([10, 20, 30], dtype=np.int64), 20)
    np.testing.assert_array_equal(labels.times_ns, [9, 19])
    assert list(labels.hands) == [9]
    assert list(labels.mano) == [9]
    # The full mapping validates later source rows even for a short preview.
    (tmp_path / "mano_hand_pose_trajectory.jsonl").write_text(json.dumps({"timestamp_ns": 301, "hand_poses": {}}))
    with pytest.raises(ValueError, match="timestamp_ns=301"):
        read_labels(source, "aria", np.array([10, 20, 30], dtype=np.int64), 20)


def test_full_lens_projection_matches_reference_and_rotated_pixels() -> None:
    import numpy as np

    from dataforge.datasets.hot3d_layers import project_keypoints
    from dataforge.datasets.hot3d_vrs import CameraModel, CameraTransform

    params = np.array([300.0, 320.0, 240.0, 0.02, -0.004, 0.001, -0.0002, 0.00003, -0.000004, 0.003, -0.005, 0.007, -0.002, 0.004, -0.001])
    model = CameraModel(
        "test",
        "1201-1",
        640,
        480,
        "CameraModelType.FISHEYE624",
        params.tolist(),
        CameraTransform(np.array([1.0, 0.0, 0.0, 0.0]), np.array([0.25, -0.5, 0.125])),
        0.5,
    )
    # Independent FISHEYE624 equation: radial angular polynomial, then
    # tangential and thin-prism offsets computed from the radial coordinates.
    point = np.array([0.25, 0.125, 1.0])
    radius = np.linalg.norm(point[:2])
    theta = np.arctan2(radius, point[2])
    theta_d = theta * (1.0 + sum(params[3 + i] * theta ** (2 * i + 2) for i in range(6)))
    x, y = point[:2] * theta_d / radius
    r2 = x * x + y * y
    p0, p1 = params[9:11]
    s0, s1, s2, s3 = params[11:15]
    expected = (
        params[0]
        * np.array(
            [
                x + p0 * (r2 + 2 * x * x) + 2 * p1 * x * y + s0 * r2 + s1 * r2**2,
                y + p1 * (r2 + 2 * y * y) + 2 * p0 * x * y + s2 * r2 + s3 * r2**2,
            ]
        )
        + params[1:3]
    )
    world_T_device = np.tile(np.eye(4), (2, 1, 1))
    world_T_device[0, :3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    world_T_device[0, :3, 3] = [1, 2, 3]
    world_T_device[1] = np.nan
    positions = np.full((2, 133, 3), np.nan, dtype=np.float32)
    world_T_cam = world_T_device[0] @ model.device_T_camera.matrix()
    for slot, camera_point in [(91, point), (92, [0, 0, -1]), (93, [0.75, 0, 1])]:
        positions[0, slot] = world_T_cam[:3, :3] @ camera_point + world_T_cam[:3, 3]
    positions[1] = positions[0]
    scores = np.full((2, 133), 0.75, dtype=np.float32)
    native = project_keypoints(model.calibration(rotate_cw90=False), world_T_device, positions, scores)
    np.testing.assert_allclose(native.positions[0, 91], expected, atol=1e-6, rtol=0.0)
    rotated = project_keypoints(model.calibration(rotate_cw90=True), world_T_device, positions, scores)
    np.testing.assert_allclose(rotated.positions[0, 91], [479 - expected[1], expected[0]], atol=1e-6, rtol=0.0)
    assert rotated.confidence[0, 91] == 0.75
    assert np.isnan(rotated.positions[0, [0, 92, 93]]).all()
    assert (rotated.confidence[0, [0, 92, 93]] == 0.0).all()
    assert np.isnan(rotated.positions[1]).all()
    assert (rotated.confidence[1] == 0.0).all()


@pytest.mark.parametrize("config_type,count", [(Hot3dAriaConfig, 3), (Hot3dQuest3Config, 2)])
def test_fisheye_panes_show_only_video_and_projected_keypoints(config_type: type[Hot3dAriaConfig] | type[Hot3dQuest3Config], count: int) -> None:
    import rerun.blueprint as rrb
    from test_blueprints import children

    from dataforge import schema

    dataset = config_type().setup()
    assert "projections" in dataset.layers
    world, column = children(dataset.default_blueprint().root_container, rrb.Horizontal)
    assert isinstance(world, rrb.Spatial3DView)
    assert world.contents == ["/world/**"]
    panes = children(column, rrb.Vertical)
    assert len(panes) == count
    for cam, pane in enumerate(panes):
        assert isinstance(pane, rrb.Spatial2DView)
        assert pane.contents == [schema.video_path(0, cam), schema.coco133_uv_projected_path(0, cam)]
        assert isinstance(pane.contents, list)
        assert "/world" not in pane.contents and "/world/**" not in pane.contents


@pytest.mark.parametrize("valid_radius,width", [(10.0, 640), (None, 330)])
def test_projection_rejects_valid_radius_and_image_bounds(valid_radius: float | None, width: int) -> None:
    import numpy as np
    from projectaria_tools.core.calibration import CameraCalibration, CameraModelType
    from projectaria_tools.core.sophus import SE3

    from dataforge.datasets.hot3d_layers import project_keypoints

    camera = CameraCalibration(
        "test", CameraModelType.FISHEYE624, np.array([300.0, 320.0, 240.0, *([0.0] * 12)]), SE3(), width, 480, valid_radius, 1.5, ""
    )
    positions = np.full((1, 133, 3), np.nan, dtype=np.float32)
    positions[0, 91] = [0.5, 0.0, 1.0]
    projected = project_keypoints(camera, np.eye(4)[None], positions, np.ones((1, 133), dtype=np.float32))
    assert np.isnan(projected.positions).all()
    assert (projected.confidence == 0.0).all()
