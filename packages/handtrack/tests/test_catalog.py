import numpy as np
import pyarrow as pa
import pytest
import torch
from scipy.spatial.transform import Rotation

from handtrack.data.catalog import (
    HOT3D_QUEST3,
    HOT3D_QUEST3_LAYOUT,
    PROFILE_COLUMN,
    SHOW3D,
    SHOW3D_HELDOUT_SUBJECTS,
    SHOW3D_LAYOUT,
    UMETRACK,
    UMETRACK_LAYOUT,
    UMETRACK_VALIDATION_USERS,
    CatalogDataError,
    DatasetLayout,
    SegmentInfo,
    bool_rows,
    camera_angles,
    column_major_3x3,
    hand_timeline,
    layout_for,
    list_rows,
    point_rows,
    read_rig,
    rotation_from_quaternion_xyzw,
    segment_infos,
    select_split,
    timeline_columns,
)
from handtrack.hand.pose import GENERIC_HAND_MODEL


def test_list_rows_reads_first_value_and_nan_for_missing() -> None:
    fixed: pa.ChunkedArray = pa.chunked_array(
        [pa.array([[[1.0, 2.0, 3.0]], None], type=pa.list_(pa.list_(pa.float32(), 3))), pa.array([[], [[4.0, None, 6.0]]], type=pa.list_(pa.list_(pa.float32(), 3)))]
    )
    rows: np.ndarray = list_rows(fixed, 3)
    np.testing.assert_array_equal(rows[0], [1.0, 2.0, 3.0])
    assert np.isnan(rows[1]).all() and np.isnan(rows[2]).all()
    np.testing.assert_array_equal(rows[3, [0, 2]], [4.0, 6.0])
    assert np.isnan(rows[3, 1])
    scalars: np.ndarray = list_rows(pa.chunked_array([pa.array([[0.5], None, [1.0]], type=pa.list_(pa.float64()))]), 1)
    np.testing.assert_array_equal(scalars[[0, 2], 0], [0.5, 1.0])
    assert np.isnan(scalars[1, 0])
    # A sliced chunk keeps its offsets into the unsliced child.
    sliced: np.ndarray = list_rows(pa.chunked_array([pa.array([[1.0], [2.0], [3.0]], type=pa.list_(pa.float64())).slice(1)]), 1)
    np.testing.assert_array_equal(sliced[:, 0], [2.0, 3.0])


def test_point_rows() -> None:
    points = pa.list_(pa.list_(pa.float32(), 2))
    column: pa.ChunkedArray = pa.chunked_array([pa.array([[[1.0, 2.0], [3.0, 4.0]], None, [[5.0, 6.0]]], type=points)])
    rows: np.ndarray = point_rows(column, 2, 2)
    np.testing.assert_array_equal(rows[0], [[1.0, 2.0], [3.0, 4.0]])
    assert np.isnan(rows[1:]).all()


def test_bool_rows() -> None:
    value, present = bool_rows(pa.chunked_array([pa.array([[True], None, [False], []], type=pa.list_(pa.bool_()))]))
    assert value.tolist() == [True, False, False, False]
    assert present.tolist() == [True, False, True, False]


def test_quaternion_and_column_major() -> None:
    rotation: Rotation = Rotation.from_euler("xyz", [0.3, -0.7, 1.1])
    np.testing.assert_allclose(rotation_from_quaternion_xyzw(rotation.as_quat().astype(np.float32)), rotation.as_matrix(), atol=1e-6)
    matrix: np.ndarray = np.arange(9, dtype=np.float32).reshape(3, 3)
    np.testing.assert_array_equal(column_major_3x3(matrix.T.reshape(9).copy()), matrix)


def _segment_table() -> pa.Table:
    ume = [
        ("umetrack__real__hand_hand__training__user_00__recording_00", "real", "hand_hand", "training", "user_00"),
        ("umetrack__synthetic__hand_hand__training__user_00__recording_00", "synthetic", "hand_hand", "training", "user_00"),
        (f"umetrack__real__separate_hand__training__{UMETRACK_VALIDATION_USERS[0]}__recording_01", "real", "separate_hand", "training", UMETRACK_VALIDATION_USERS[0]),
        ("umetrack__real__hand_hand__testing__user_05__recording_00", "real", "hand_hand", "testing", "user_05"),
    ]
    return pa.table(
        {
            "rerun_segment_id": [row[0] for row in ume],
            "rerun_layer_names": [["base", "hand_pose"]] * len(ume),
            "property:episode:domain": [[row[1]] for row in ume],
            "property:episode:interaction": [[row[2]] for row in ume],
            "property:episode:split": [[row[3]] for row in ume],
            "property:episode:user": [[row[4]] for row in ume],
            "property:capture:num_frames": [[451]] * len(ume),
            "property:capture:fps": [[30]] * len(ume),
        }
    )


def test_umetrack_splits() -> None:
    infos: tuple[SegmentInfo, ...] = segment_infos(UMETRACK, _segment_table())
    assert [info.domain for info in infos] == ["real", "real", "real", "synthetic"]
    train: tuple[SegmentInfo, ...] = select_split(infos, "train")
    assert {info.subject for info in train} == {"user_00"} and len(train) == 2
    assert [info.subject for info in select_split(infos, "val")] == [UMETRACK_VALIDATION_USERS[0]]
    assert [info.split for info in select_split(infos, "test")] == ["testing"]


def test_show3d_needs_hand_pose_and_holds_out_subjects() -> None:
    table: pa.Table = pa.table(
        {
            "rerun_segment_id": ["show3d__A__x_1", "show3d__B__y_2", f"show3d__{SHOW3D_HELDOUT_SUBJECTS[0]}__z_3", "show3d__C__t_4"],
            "rerun_layer_names": [["base", "hand_pose"], ["base"], ["base", "hand_pose"], ["base", "hand_pose"]],
            "property:episode:split": [["train"], ["test"], ["train"], ["test"]],
            "property:episode:subject_id": [["A"], ["B"], [SHOW3D_HELDOUT_SUBJECTS[0]], ["C"]],
            "property:episode:action": [["x"], ["y"], ["z"], ["t"]],
            "property:capture:num_frames": [[1785], [100], [900], [50]],
        }
    )
    infos: tuple[SegmentInfo, ...] = segment_infos(SHOW3D, table)
    assert [info.segment_id for info in infos] == ["show3d__A__x_1", "show3d__C__t_4", f"show3d__{SHOW3D_HELDOUT_SUBJECTS[0]}__z_3"]
    assert all(info.fps == 60 and info.domain == "show3d" for info in infos)
    assert [info.subject for info in select_split(infos, "train")] == ["A"]
    assert [info.subject for info in select_split(infos, "val")] == [SHOW3D_HELDOUT_SUBJECTS[0]]
    with pytest.raises(ValueError, match="no hand labels"):
        select_split(infos, "test")


def _statics(fisheye: bool, relation: int = 2, layout: DatasetLayout | None = None, resolution: tuple[float, float] | None = None) -> pa.Table:
    layout = layout or (UMETRACK_LAYOUT if fisheye else SHOW3D_LAYOUT)
    size: list[float] = list(resolution or ((636.0, 480.0) if fisheye else (1024.0, 1280.0)))
    columns: dict[str, list] = {"rerun_segment_id": ["s"]}
    for index, camera in enumerate(layout.cameras):
        rotation: np.ndarray = Rotation.from_euler("y", 0.2 * index).as_matrix()
        columns[f"{camera}:Transform3D:relation"] = [[relation]]
        columns[f"{camera}:Transform3D:mat3x3"] = [[rotation.T.reshape(9).tolist()]]
        columns[f"{camera}:Transform3D:translation"] = [[[0.01 * index, 0.0, 0.0]]]
        intrinsics: np.ndarray = np.array([[240.0, 0.0, 318.0], [0.0, 241.0, 239.0], [0.0, 0.0, 1.0]])
        columns[f"{camera}/pinhole:Pinhole:image_from_camera"] = [[intrinsics.T.reshape(9).tolist()]]
        columns[f"{camera}/pinhole:Pinhole:resolution"] = [[size]]
        if fisheye:
            columns[f"{camera}/pinhole:simplecv.components.DistortionCoefficients"] = [[[0.1, 0.01, 0.0, 0.0, 0.0, 0.0, 0.001, -0.001]]]
            columns[f"{camera}/pinhole:simplecv.components.DistortionModel"] = [["kannala_brandt"]]
    columns[PROFILE_COLUMN] = [[GENERIC_HAND_MODEL.read_text()]]
    return pa.table(columns)


def test_read_rig_from_statics() -> None:
    info: SegmentInfo = SegmentInfo(UMETRACK, "s", "real", "hand_hand", "training", "user_00", 451, 30)
    rig, letterboxes = read_rig(_statics(fisheye=True), info)
    assert rig.names == UMETRACK_LAYOUT.cameras and rig.fisheye62 is not None and tuple(rig.fisheye62.shape) == (4, 8)
    np.testing.assert_allclose(rig.cam_from_rig[1, :3, :3].numpy(), Rotation.from_euler("y", 0.2).as_matrix(), atol=1e-6)
    torch.testing.assert_close(rig.principal[0], torch.tensor([318.0, 239.0]))
    assert all(letterbox.pad_x == 2.0 for letterbox in letterboxes)
    show3d_rig, show3d_letterboxes = read_rig(_statics(fisheye=False), SegmentInfo(SHOW3D, "s", "show3d", "x", "train", "A", 10, 60))
    assert show3d_rig.fisheye62 is None and all(letterbox.quarter_turn_cw for letterbox in show3d_letterboxes)
    with pytest.raises(ValueError, match="ChildFromParent"):
        read_rig(_statics(fisheye=True, relation=1), info)


@pytest.mark.parametrize("missing_confidence", [False, True])
def test_hand_timeline_umetrack_masks(missing_confidence: bool) -> None:
    info: SegmentInfo = SegmentInfo(UMETRACK, "s", "real", "separate_hand", "training", "user_00", 3, 30)
    frames: int = 3
    quaternion: list[float] = [0.0, 0.0, 0.0, 1.0]
    columns: dict[str, pa.Array] = {"video_time": pa.array([0, 33_000_000, 66_000_000], type=pa.duration("ns"))}
    four = pa.list_(pa.list_(pa.float32(), 4))
    three = pa.list_(pa.list_(pa.float32(), 3))
    names: list[str] = timeline_columns(UMETRACK_LAYOUT)
    columns[names[0]] = pa.array([[quaternion], [quaternion], [[np.nan] * 4]], type=four)
    columns[names[1]] = pa.array([[[0.0, 0.0, 0.0]]] * frames, type=three)
    columns[names[2]] = pa.array([[False], [True], [False]], type=pa.list_(pa.bool_()))
    for side, present in (("left", True), ("right", False)):
        base: str = f"/world/gt/hands/{side}"
        columns[f"{base}/confidence:Scalars:scalars"] = pa.array([[1.0 if present else 0.0]] * frames, type=pa.list_(pa.float64()))
        columns[f"{base}/joint_angles:joint_angles"] = pa.array([[[0.1] * 22] if present else []] * frames, type=pa.list_(pa.list_(pa.float32(), 22)))
        columns[f"{base}/wrist:Transform3D:quaternion"] = pa.array([[quaternion] if present else None] * frames, type=four)
        columns[f"{base}/wrist:Transform3D:translation"] = pa.array([[[0.0, 0.0, 0.3]] if present else None] * frames, type=three)
    if missing_confidence:
        columns["/world/gt/hands/left/confidence:Scalars:scalars"] = pa.array([None] * frames, type=pa.list_(pa.float64()))
    timeline = hand_timeline(pa.table(columns), _statics(fisheye=True), info)
    assert timeline.headset_valid.tolist() == [True, False, False]
    assert torch.isnan(timeline.world_from_rig[1:]).all()
    assert timeline.has_pose.tolist() == [[True, False]] * frames
    assert timeline.confidence[:, 1].eq(0).all()
    assert torch.isnan(timeline.poses[1].translation).all()
    assert timeline.hand_scale == pytest.approx(1.0)
    if missing_confidence:
        from handtrack.data.segment_labels import segment_labels
        rig, letterboxes = read_rig(_statics(fisheye=True), info)
        assert torch.isnan(timeline.confidence[:, 0]).all()
        labels = segment_labels(timeline, rig, letterboxes, np.arange(frames, dtype=np.int64), False)
        assert not labels.image_valid.any()


@pytest.mark.parametrize('dataset', [UMETRACK, SHOW3D])
@pytest.mark.parametrize('property_name', ['episode:split', 'episode:user'])
@pytest.mark.parametrize('value', ['missing', None, [], [None], [''], ['   ']])
def test_partition_properties_are_required(dataset: str, property_name: str, value: object) -> None:
    table = _segment_table()
    if dataset == SHOW3D:
        table = table.rename_columns(['property:episode:subject_id' if name == 'property:episode:user' else name for name in table.column_names])
        table = table.set_column(table.column_names.index('property:episode:split'), 'property:episode:split', pa.array([['train']] * table.num_rows))
        property_name = property_name.replace('episode:user', 'episode:subject_id')
    name = f'property:{property_name}'
    table = table.drop([name])
    if value != 'missing':
        table = table.append_column(name, pa.array([value] * table.num_rows, type=pa.list_(pa.string())))
    with pytest.raises(ValueError, match=f'{dataset} .*{property_name}'):
        segment_infos(dataset, table)


def _hot3d_table() -> pa.Table:
    return pa.table(
        {
            "rerun_segment_id": ["hot3d-quest3__P0003_cccc", "hot3d-quest3__P0003_bbbb", "hot3d-quest3__P0002_aaaa"],
            "rerun_layer_names": [["base", "hand_pose", "hand_mesh"], ["base"], ["base", "hand_pose"]],
            "property:episode:participant_id": [["P0003"], ["P0003"], ["P0002"]],
            "property:episode:has_gt": [[True], [False], [True]],
            "property:capture:num_frames": [[395], [1200], [3981]],
        }
    )


def test_hot3d_keeps_labelled_scenes_as_an_unseen_test_set() -> None:
    infos: tuple[SegmentInfo, ...] = segment_infos(HOT3D_QUEST3, _hot3d_table())
    assert [(info.segment_id, info.subject, info.num_frames) for info in infos] == [("hot3d-quest3__P0002_aaaa", "P0002", 3981), ("hot3d-quest3__P0003_cccc", "P0003", 395)]
    assert all((info.domain, info.split, info.fps, info.interaction) == ("hot3d", "train", 30, "") for info in infos)
    assert select_split(infos, "test") == infos
    for split in ("train", "val"):
        with pytest.raises(CatalogDataError, match="unseen test set"):
            select_split(infos, split)


@pytest.mark.parametrize("value", ["missing", [None], ["  "]])
def test_hot3d_participant_is_required(value: object) -> None:
    table: pa.Table = _hot3d_table().drop(["property:episode:participant_id"])
    if value != "missing":
        table = table.append_column("property:episode:participant_id", pa.array([value] * table.num_rows, type=pa.list_(pa.string())))
    with pytest.raises(ValueError, match=f"{HOT3D_QUEST3} .*episode:participant_id"):
        segment_infos(HOT3D_QUEST3, table)


def test_hot3d_layout_rig_and_letterbox() -> None:
    layout: DatasetLayout = layout_for(HOT3D_QUEST3)
    assert layout is HOT3D_QUEST3_LAYOUT and layout.rig_index == 0 and layout.cameras == ("/world/rig_00/cam_00", "/world/rig_00/cam_01")
    assert (layout.pool_stride, layout.tracker_step) == (6, 1)  # 30 fps: the 5 fps pool and the 30 Hz tracker step
    ids: list[int] = [lay.camera_offset + index for lay in (UMETRACK_LAYOUT, SHOW3D_LAYOUT, HOT3D_QUEST3_LAYOUT) for index in range(len(lay.cameras))]
    assert ids == list(range(8))
    info: SegmentInfo = SegmentInfo(HOT3D_QUEST3, "s", "hot3d", "", "train", "P0002", 10, 30)
    rig, letterboxes = read_rig(_statics(fisheye=True, layout=layout, resolution=(1024.0, 1280.0)), info)
    assert rig.names == layout.cameras and rig.fisheye62 is not None and tuple(rig.fisheye62.shape) == (2, 8)
    assert all(letterbox.quarter_turn_cw and letterbox.scale == 0.46875 for letterbox in letterboxes)


def test_camera_angles_come_from_umetrack_statics_only() -> None:
    umetrack: pa.Table = _statics(fisheye=True)
    for index, camera in enumerate(UMETRACK_LAYOUT.cameras):
        umetrack = umetrack.append_column(f"{camera}/pinhole:source_camera_angle_deg", pa.array([[90.0 * index]], type=pa.list_(pa.float64())))
    assert camera_angles(umetrack, SegmentInfo(UMETRACK, "s", "real", "hand_hand", "training", "user_00", 3, 30)) == (0.0, 90.0, 180.0, 270.0)
    hot3d: pa.Table = _statics(fisheye=True, layout=HOT3D_QUEST3_LAYOUT, resolution=(1024.0, 1280.0))
    assert camera_angles(hot3d, SegmentInfo(HOT3D_QUEST3, "s", "hot3d", "", "train", "P0002", 3, 30)) == (0.0, 0.0)


def test_hand_timeline_hot3d_reads_a_quaternion_rig_without_untracked_and_its_qa_flags() -> None:
    info: SegmentInfo = SegmentInfo(HOT3D_QUEST3, "s", "hot3d", "", "train", "P0002", 3, 30)
    frames: int = 3
    quaternion: list[float] = [0.0, 0.0, 0.0, 1.0]
    four = pa.list_(pa.list_(pa.float32(), 4))
    three = pa.list_(pa.list_(pa.float32(), 3))
    scalar = pa.list_(pa.float64())
    names: list[str] = timeline_columns(HOT3D_QUEST3_LAYOUT)
    assert not any(name.endswith(":untracked") for name in names)
    columns: dict[str, pa.Array] = {"video_time": pa.array([0, 33_333_333, 66_666_666], type=pa.duration("ns"))}
    columns["/world/rig_00:Transform3D:quaternion"] = pa.array([[quaternion], [quaternion], [[np.nan] * 4]], type=four)
    columns["/world/rig_00:Transform3D:translation"] = pa.array([[[0.0, 0.0, 0.0]]] * frames, type=three)
    for side, present in (("left", True), ("right", False)):
        base: str = f"/world/gt/hands/{side}"
        columns[f"{base}/confidence:Scalars:scalars"] = pa.array([[1.0 if present else 0.0]] * frames, type=scalar)
        columns[f"{base}/joint_angles:joint_angles"] = pa.array([[[0.1] * 22] if present else []] * frames, type=pa.list_(pa.list_(pa.float32(), 22)))
        columns[f"{base}/wrist:Transform3D:quaternion"] = pa.array([[quaternion] if present else None] * frames, type=four)
        columns[f"{base}/wrist:Transform3D:translation"] = pa.array([[[0.0, 0.0, 0.3]] if present else None] * frames, type=three)
    columns["/world/gt/quality/rig_00/cam_00/qa_pass:Scalars:scalars"] = pa.array([[1.0], [1.0], [0.0]], type=scalar)
    columns["/world/gt/quality/rig_00/cam_01/qa_pass:Scalars:scalars"] = pa.array([[1.0], None, [1.0]], type=scalar)
    table: pa.Table = pa.table(columns)
    assert sorted(table.column_names[1:]) == sorted(names)
    statics: pa.Table = _statics(fisheye=True, layout=HOT3D_QUEST3_LAYOUT, resolution=(1024.0, 1280.0))
    timeline = hand_timeline(table, statics, info)
    assert timeline.headset_valid.tolist() == [True, True, False]
    assert timeline.has_pose.tolist() == [[True, False]] * frames
    # A missing flag fails: the image's quality is unknown.
    assert timeline.camera_valid is not None and timeline.camera_valid.tolist() == [[True, True], [True, False], [False, True]]
    with pytest.raises(CatalogDataError, match="qa_pass"):
        hand_timeline(table.drop(["/world/gt/quality/rig_00/cam_01/qa_pass:Scalars:scalars"]), statics, info)
