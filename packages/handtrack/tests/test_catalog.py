import numpy as np
import pyarrow as pa
import pytest
import torch
from scipy.spatial.transform import Rotation

from handtrack.data.catalog import (
    PROFILE_COLUMN,
    SHOW3D,
    SHOW3D_HELDOUT_SUBJECTS,
    SHOW3D_LAYOUT,
    UMETRACK,
    UMETRACK_LAYOUT,
    UMETRACK_VALIDATION_USERS,
    SegmentInfo,
    bool_rows,
    column_major_3x3,
    hand_timeline,
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


def _statics(fisheye: bool, relation: int = 2) -> pa.Table:
    layout = UMETRACK_LAYOUT if fisheye else SHOW3D_LAYOUT
    columns: dict[str, list] = {"rerun_segment_id": ["s"]}
    for index, camera in enumerate(layout.cameras):
        rotation: np.ndarray = Rotation.from_euler("y", 0.2 * index).as_matrix()
        columns[f"{camera}:Transform3D:relation"] = [[relation]]
        columns[f"{camera}:Transform3D:mat3x3"] = [[rotation.T.reshape(9).tolist()]]
        columns[f"{camera}:Transform3D:translation"] = [[[0.01 * index, 0.0, 0.0]]]
        intrinsics: np.ndarray = np.array([[240.0, 0.0, 318.0], [0.0, 241.0, 239.0], [0.0, 0.0, 1.0]])
        columns[f"{camera}/pinhole:Pinhole:image_from_camera"] = [[intrinsics.T.reshape(9).tolist()]]
        columns[f"{camera}/pinhole:Pinhole:resolution"] = [[[636.0, 480.0] if fisheye else [1024.0, 1280.0]]]
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


def test_hand_timeline_umetrack_masks() -> None:
    info: SegmentInfo = SegmentInfo(UMETRACK, "s", "real", "separate_hand", "training", "user_00", 3, 30)
    frames: int = 3
    quaternion: list[float] = [0.0, 0.0, 0.0, 1.0]
    columns: dict[str, pa.Array] = {"video_time": pa.array([0, 33_000_000, 66_000_000], type=pa.duration("ns"))}
    four = pa.list_(pa.list_(pa.float32(), 4))
    three = pa.list_(pa.list_(pa.float32(), 3))
    names: list[str] = timeline_columns(UMETRACK_LAYOUT, show3d=False)
    columns[names[0]] = pa.array([[quaternion], [quaternion], [[np.nan] * 4]], type=four)
    columns[names[1]] = pa.array([[[0.0, 0.0, 0.0]]] * frames, type=three)
    columns[names[2]] = pa.array([[False], [True], [False]], type=pa.list_(pa.bool_()))
    for side, present in (("left", True), ("right", False)):
        base: str = f"/world/gt/hands/{side}"
        columns[f"{base}/confidence:Scalars:scalars"] = pa.array([[1.0 if present else 0.0]] * frames, type=pa.list_(pa.float64()))
        columns[f"{base}/joint_angles:joint_angles"] = pa.array([[[0.1] * 22] if present else []] * frames, type=pa.list_(pa.list_(pa.float32(), 22)))
        columns[f"{base}/wrist:Transform3D:quaternion"] = pa.array([[quaternion] if present else None] * frames, type=four)
        columns[f"{base}/wrist:Transform3D:translation"] = pa.array([[[0.0, 0.0, 0.3]] if present else None] * frames, type=three)
    timeline = hand_timeline(pa.table(columns), _statics(fisheye=True), info)
    assert timeline.headset_valid.tolist() == [True, False, False]
    assert torch.isnan(timeline.world_from_rig[1:]).all()
    assert timeline.has_pose.tolist() == [[True, False]] * frames
    assert timeline.confidence[:, 1].eq(0).all()
    assert torch.isnan(timeline.poses[1].translation).all()
    assert timeline.hand_scale == pytest.approx(1.0)
