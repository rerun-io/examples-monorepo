import csv
import json
from pathlib import Path

import av
import numpy as np
import pytest
import rerun as rr
from jaxtyping import UInt8
from numpy import ndarray

from simplecv.apis.view_exoego import VisualizeConfig, visualize_exo_ego
from simplecv.configs.exoego_dataset_configs import dataset_defaults
from simplecv.data.exoego import epfl_smart_kitchen as epfl_module
from simplecv.data.exoego.epfl_smart_kitchen import (
    EpflSmartKitchenConfig,
    EpflSmartKitchenSequence,
    _parse_numeric_cell,
    load_hololens_world_to_camera_poses,
    video_path_for_camera,
)
from simplecv.rerun_log_utils import RerunTyroConfig
from simplecv.rrd_query_utils import RRDQuerySession, first_valid_value

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
MONOREPO_ROOT: Path = PROJECT_ROOT.parents[1]


def _have_mano_pkls() -> bool:
    mano_root: Path = PROJECT_ROOT / "simplecv" / "data"
    return (mano_root / "MANO_RIGHT.pkl").exists() and (mano_root / "MANO_LEFT.pkl").exists()


def _frame_pixels(frame_idx: int) -> UInt8[ndarray, "h w 3"]:
    pixels: UInt8[ndarray, "h w 3"] = np.empty((32, 32, 3), dtype=np.uint8)
    pixels[..., 0] = np.uint8(frame_idx * 30)
    pixels[..., 1] = np.uint8(40 + frame_idx * 20)
    pixels[..., 2] = np.uint8(80 + frame_idx * 10)
    return pixels


def _write_synthetic_mp4(path: Path, num_frames: int = 3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    container: av.container.OutputContainer = av.open(str(path), mode="w")
    stream: av.video.stream.VideoStream = container.add_stream("h264", rate=30, options={"preset": "ultrafast", "crf": "0"})
    stream.width = 32
    stream.height = 32
    stream.pix_fmt = "yuv420p"
    stream.max_b_frames = 0
    for frame_idx in range(num_frames):
        frame: av.VideoFrame = av.VideoFrame.from_ndarray(_frame_pixels(frame_idx), format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def _write_minimal_public_release(root: Path) -> None:
    session_dir: Path = root / "Public_release_videos" / "train" / "YH2002" / "2023_12_04_10_15_23"
    videos_dir: Path = session_dir / "videos"
    meta_dir: Path = session_dir / "meta_data"
    meta_dir.mkdir(parents=True, exist_ok=True)
    _write_synthetic_mp4(videos_dir / "hololens.mp4")
    _write_synthetic_mp4(videos_dir / "output0.mp4")

    camera_matrix: dict[str, dict[str, list[list[float]] | list[float] | int]] = {
        "hololens": {
            "K": [[100.0, 0.0, 16.0], [0.0, 100.0, 16.0], [0.0, 0.0, 1.0]],
            "dist": [],
            "width": 32,
            "height": 32,
        },
        "output0": {
            "world2cam": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            "K": [[100.0, 0.0, 16.0], [0.0, 100.0, 16.0], [0.0, 0.0, 1.0]],
            "dist": [],
            "width": 32,
            "height": 32,
        },
    }
    (meta_dir / "camera_matrix.json").write_text(json.dumps(camera_matrix))
    world2holo: list[list[float]] = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    with (meta_dir / "holo_data_wpose.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["world2holo"])
        writer.writeheader()
        for _ in range(3):
            writer.writerow({"world2holo": json.dumps(world2holo)})
    (meta_dir / "timestamps.txt").write_text("1000\n34333\n67666\n")

    pose_dir: Path = root / "Public_release_pose" / "train" / "YH2002" / "2023_12_04_10_15_23" / "pose_3d"
    pose_dir.mkdir(parents=True, exist_ok=True)
    body_kpts: list[list[float]] = [[float(idx), float(idx + 1), float(idx + 2)] for idx in range(17)]
    hand_kpts: list[list[float]] = [[float(idx), float(idx + 1), float(idx + 2)] for idx in range(42)]
    body_conf: list[float] = [1.0] * 17
    hand_conf: list[float] = [1.0] * 42
    left_shape: list[float] = [float(idx) * 0.01 for idx in range(10)]
    right_shape: list[float] = [value + 1e-5 for value in left_shape]
    left_pose: list[float] = [0.01] * 48
    right_pose: list[float] = [0.02] * 48
    with (pose_dir / "pose3d_smpl.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["kp3ds", "kp3ds_conf", "l2_dist"])
        writer.writeheader()
        for _ in range(3):
            writer.writerow(
                {
                    "kp3ds": json.dumps(body_kpts),
                    "kp3ds_conf": json.dumps(body_conf),
                    "l2_dist": "0.01",
                }
            )
    with (pose_dir / "pose3d_mano.csv").open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "kp3ds",
                "kp3ds_conf",
                "l2_dist_left",
                "l2_dist_right",
                "left_poses",
                "right_poses",
                "left_Rh",
                "right_Rh",
                "left_Th",
                "right_Th",
                "left_shapes",
                "right_shapes",
            ],
        )
        writer.writeheader()
        for _ in range(3):
            writer.writerow(
                {
                    "kp3ds": json.dumps(hand_kpts),
                    "kp3ds_conf": json.dumps(hand_conf),
                    "l2_dist_left": "0.01",
                    "l2_dist_right": "0.01",
                    "left_poses": json.dumps(left_pose),
                    "right_poses": json.dumps(right_pose),
                    "left_Rh": json.dumps([0.0, 0.0, 0.0]),
                    "right_Rh": json.dumps([0.0, 0.0, 0.0]),
                    "left_Th": json.dumps([0.1, 0.2, 0.3]),
                    "right_Th": json.dumps([0.4, 0.5, 0.6]),
                    "left_shapes": json.dumps(left_shape),
                    "right_shapes": json.dumps(right_shape),
                }
            )


def test_epfl_smart_kitchen_sequence_identity_includes_split_participant_and_session() -> None:
    cfg = EpflSmartKitchenConfig(
        root_directory=Path("data/epfl-smart-kitchen/sample"),
        split="train",
        participant_id="YH2002",
        session_name="2023_12_04_10_15_23",
    )

    identity = EpflSmartKitchenSequence.sequence_identity_for_config(cfg)

    assert identity.dataset == "epfl-smart-kitchen"
    assert identity.sequence_key == "train/YH2002/2023_12_04_10_15_23"
    assert identity.recording_id == "epfl-smart-kitchen__train__YH2002__2023_12_04_10_15_23"
    assert identity.rrd_path(Path("/tmp/catalog")) == Path("/tmp/catalog/epfl-smart-kitchen/train/YH2002/2023_12_04_10_15_23.rrd")


def test_epfl_smart_kitchen_is_registered_for_generic_exoego_viewer() -> None:
    assert isinstance(dataset_defaults["epfl-smart-kitchen"], EpflSmartKitchenConfig)


def test_epfl_smart_kitchen_default_root_uses_av1_mirror() -> None:
    cfg = EpflSmartKitchenConfig()

    assert cfg.root_directory == Path("/mnt/8tb/data/epfl-smart-kitchen-av1")


def test_epfl_smart_kitchen_ignores_generated_hololens_sidecar_when_present(tmp_path: Path) -> None:
    _write_minimal_public_release(tmp_path)
    cfg = EpflSmartKitchenConfig(
        root_directory=tmp_path,
        split="train",
        participant_id="YH2002",
        session_name="2023_12_04_10_15_23",
    )
    sidecar_path: Path = tmp_path / "Public_release_videos" / "train" / "YH2002" / "2023_12_04_10_15_23" / "videos" / "hololens.rerun_h264.mp4"
    sidecar_path.write_bytes(b"sidecar")

    assert video_path_for_camera(cfg, "hololens").name == "hololens.mp4"
    assert video_path_for_camera(cfg, "output0").name == "output0.mp4"


def test_epfl_smart_kitchen_uses_compact_image_planes(tmp_path: Path) -> None:
    _write_minimal_public_release(tmp_path)
    cfg = EpflSmartKitchenConfig(
        root_directory=tmp_path,
        split="train",
        participant_id="YH2002",
        session_name="2023_12_04_10_15_23",
        exo_camera_names=("output0",),
        load_labels=False,
    )
    sequence = EpflSmartKitchenSequence(cfg)

    assert sequence.ego_sequence is not None
    assert sequence.exo_sequence is not None
    assert sequence.ego_sequence.image_plane_distance == 0.1
    assert sequence.exo_sequence.image_plane_distance == 0.25


def test_epfl_smart_kitchen_sample_download_task_matches_hocap_style() -> None:
    pixi_text: str = (MONOREPO_ROOT / "pixi.toml").read_text()

    assert "[feature.simplecv.tasks._download-simplecv-epfl-smart-kitchen-sample]" in pixi_text
    assert "pablovela5620/epfl-smart-kitchen-sample" in pixi_text
    assert "data/epfl-smart-kitchen/sample.zip" in pixi_text
    assert "data/epfl-smart-kitchen/sample" in pixi_text


def test_epfl_smart_kitchen_counts_train_and_test_sessions(tmp_path: Path) -> None:
    (tmp_path / "Public_release_pose" / "train" / "YH2002" / "2023_12_04_10_15_23" / "pose_3d").mkdir(parents=True)
    (tmp_path / "Public_release_videos" / "train" / "YH2002" / "2023_12_04_10_15_23").mkdir(parents=True)
    (tmp_path / "Public_release_pose" / "test" / "YH2003" / "2023_12_05_11_16_24" / "pose_3d").mkdir(parents=True)
    (tmp_path / "Public_release_videos" / "test" / "YH2003" / "2023_12_05_11_16_24").mkdir(parents=True)
    (tmp_path / "Public_release_pose" / "test" / "YH2004" / "pose_only_session" / "pose_3d").mkdir(parents=True)
    cfg = EpflSmartKitchenConfig(root_directory=tmp_path)

    assert EpflSmartKitchenSequence.num_sequences_for_config(cfg) == 2


def test_epfl_smart_kitchen_numeric_parser_preserves_large_values_and_non_finites() -> None:
    parsed = _parse_numeric_cell("[100001.0, NaN, -Infinity, inf, +inf]")

    assert float(parsed[0]) == pytest.approx(100001.0)
    assert np.isnan(parsed[1])
    assert np.isneginf(parsed[2])
    assert np.isposinf(parsed[3])
    assert np.isposinf(parsed[4])


def test_epfl_smart_kitchen_pyserde_hand_row_decodes_csv_string_arrays_and_aliases() -> None:
    from simplecv.data.exoego.epfl_smart_kitchen import _parse_hand_pose_rows

    rows = _parse_hand_pose_rows(
        [
            {
                "kp3ds": json.dumps([[float(idx), float(idx + 1), float(idx + 2)] for idx in range(42)]),
                "kp3ds_conf": json.dumps([1.0] * 42),
                "l2_dist_left": "",
                "l2_dist_right": "0.01",
                "left_poses": json.dumps([0.01] * 48),
                "right_poses": json.dumps([0.02] * 45),
                "left_RH": json.dumps([0.0, 0.1, 0.2]),
                "right_RH": json.dumps([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
                "left_TH": json.dumps([0.1, 0.2, 0.3]),
                "right_TH": json.dumps([0.4, 0.5, 0.6]),
                "left_shapes": json.dumps([float(idx) * 0.01 for idx in range(10)]),
                "right_shapes": json.dumps([float(idx) * 0.02 for idx in range(10)]),
            }
        ],
        path=Path("pose3d_mano.csv"),
    )
    row = rows[0]

    assert row.kp3ds.dtype == np.float32
    assert row.kp3ds.shape == (42, 3)
    assert row.kp3ds_conf is not None
    assert row.kp3ds_conf.shape == (42,)
    assert row.l2_dist_left is None
    assert row.l2_dist_right == pytest.approx(0.01)
    assert row.left_Rh.shape == (3,)
    assert row.right_Rh.shape == (3, 3)
    np.testing.assert_allclose(row.left_Th, np.array([0.1, 0.2, 0.3], dtype=np.float32))
    np.testing.assert_allclose(row.right_Th, np.array([0.4, 0.5, 0.6], dtype=np.float32))


def test_epfl_smart_kitchen_pyserde_hand_row_decodes_canonical_csv_strings() -> None:
    from serde import from_dict

    from simplecv.data.exoego.epfl_smart_kitchen import EpflHandPoseRow

    row = from_dict(
        EpflHandPoseRow,
        {
            "kp3ds": json.dumps([[float(idx), float(idx + 1), float(idx + 2)] for idx in range(42)]),
            "kp3ds_conf": json.dumps([1.0] * 42),
            "l2_dist_left": "",
            "l2_dist_right": "0.01",
            "left_poses": json.dumps([0.01] * 48),
            "right_poses": json.dumps([0.02] * 45),
            "left_Rh": json.dumps([0.0, 0.1, 0.2]),
            "right_Rh": json.dumps([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            "left_Th": json.dumps([0.1, 0.2, 0.3]),
            "right_Th": json.dumps([0.4, 0.5, 0.6]),
            "left_shapes": json.dumps([float(idx) * 0.01 for idx in range(10)]),
            "right_shapes": json.dumps([float(idx) * 0.02 for idx in range(10)]),
        },
    )

    assert row.kp3ds.dtype == np.float32
    assert row.kp3ds.shape == (42, 3)
    assert row.kp3ds_conf is not None
    assert row.kp3ds_conf.shape == (42,)
    assert row.l2_dist_left is None
    assert row.l2_dist_right == pytest.approx(0.01)
    assert row.left_Rh.shape == (3,)
    assert row.right_Rh.shape == (3, 3)
    np.testing.assert_allclose(row.left_Th, np.array([0.1, 0.2, 0.3], dtype=np.float32))
    np.testing.assert_allclose(row.right_Th, np.array([0.4, 0.5, 0.6], dtype=np.float32))


def test_epfl_smart_kitchen_sequence_loads_no_label_ego_and_exo_streams(tmp_path: Path) -> None:
    _write_minimal_public_release(tmp_path)
    cfg = EpflSmartKitchenConfig(
        root_directory=tmp_path,
        split="train",
        participant_id="YH2002",
        session_name="2023_12_04_10_15_23",
        exo_camera_names=("output0",),
        load_labels=False,
    )

    sequence = EpflSmartKitchenSequence(cfg)
    sample = sequence[0]

    assert sequence.canonical_stream_name in {"ego/hololens", "exo/output0"}
    assert sample.labels is None
    assert sample.ego_cam_params_list is not None
    assert sample.exo_cam_params_list is not None
    assert sample.exo_cam_params_list[0] is not None
    assert sample.ego_cam_params_list[0].name == "hololens"
    assert sample.exo_cam_params_list[0].name == "output0"
    np.testing.assert_array_equal(
        sequence.stream_timestamps_ns["ego/hololens"],
        np.array([0, 33333000, 66666000], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        sequence.stream_timestamps_ns["exo/output0"],
        np.array([0, 33333000, 66666000], dtype=np.int64),
    )


def test_epfl_smart_kitchen_hololens_video_dimensions_are_loaded_once(tmp_path: Path, monkeypatch) -> None:
    _write_minimal_public_release(tmp_path)
    camera_matrix_path: Path = tmp_path / "Public_release_videos" / "train" / "YH2002" / "2023_12_04_10_15_23" / "meta_data" / "camera_matrix.json"
    camera_matrix: dict[str, dict] = json.loads(camera_matrix_path.read_text())
    del camera_matrix["hololens"]["width"]
    del camera_matrix["hololens"]["height"]
    camera_matrix_path.write_text(json.dumps(camera_matrix))
    calls: list[Path] = []

    def fake_video_size(video_path: Path) -> tuple[int, int]:
        calls.append(video_path)
        return 32, 32

    monkeypatch.setattr(epfl_module, "_video_size", fake_video_size)
    from simplecv.data.ego.epfl_smart_kitchen_ego import EpflSmartKitchenEgoSequence

    cfg = EpflSmartKitchenConfig(
        root_directory=tmp_path,
        split="train",
        participant_id="YH2002",
        session_name="2023_12_04_10_15_23",
    )

    ego_sequence = EpflSmartKitchenEgoSequence(cfg)

    assert len(ego_sequence.ego_cam_dict["hololens"]) == 3
    assert calls == [video_path_for_camera(cfg, "hololens")]


def test_epfl_smart_kitchen_hololens_pose_loader_carries_previous_pose_for_empty_tracking_rows(
    tmp_path: Path,
) -> None:
    _write_minimal_public_release(tmp_path)
    cfg = EpflSmartKitchenConfig(
        root_directory=tmp_path,
        split="train",
        participant_id="YH2002",
        session_name="2023_12_04_10_15_23",
    )
    holo_path: Path = tmp_path / "Public_release_videos" / "train" / "YH2002" / "2023_12_04_10_15_23" / "meta_data" / "holo_data_wpose.csv"
    first_pose: list[list[float]] = [
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 2.0],
        [0.0, 0.0, 1.0, 3.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    last_pose: list[list[float]] = [
        [1.0, 0.0, 0.0, 4.0],
        [0.0, 1.0, 0.0, 5.0],
        [0.0, 0.0, 1.0, 6.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    with holo_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["world2holo"])
        writer.writeheader()
        writer.writerow({"world2holo": json.dumps(first_pose)})
        writer.writerow({"world2holo": "[]"})
        writer.writerow({"world2holo": json.dumps(last_pose)})

    with pytest.warns(UserWarning, match="missing 1 HoloLens world2holo"):
        holo_poses = load_hololens_world_to_camera_poses(cfg)

    poses = holo_poses.cam_T_world_list
    assert len(poses) == 3
    np.testing.assert_allclose(poses[0], np.array(first_pose, dtype=np.float32))
    # The dropout frame still holds a carried-forward placeholder (so the list stays
    # frame-aligned), but valid_mask flags it so the viewer never renders that stale pose.
    np.testing.assert_allclose(poses[1], np.array(first_pose, dtype=np.float32))
    np.testing.assert_allclose(poses[2], np.array(last_pose, dtype=np.float32))
    np.testing.assert_array_equal(holo_poses.valid_mask, np.array([True, False, True]))


def test_epfl_smart_kitchen_hololens_pose_loader_warns_when_leading_rows_use_first_valid_pose(
    tmp_path: Path,
) -> None:
    _write_minimal_public_release(tmp_path)
    cfg = EpflSmartKitchenConfig(
        root_directory=tmp_path,
        split="train",
        participant_id="YH2002",
        session_name="2023_12_04_10_15_23",
    )
    holo_path: Path = tmp_path / "Public_release_videos" / "train" / "YH2002" / "2023_12_04_10_15_23" / "meta_data" / "holo_data_wpose.csv"
    first_valid_pose: list[list[float]] = [
        [1.0, 0.0, 0.0, 7.0],
        [0.0, 1.0, 0.0, 8.0],
        [0.0, 0.0, 1.0, 9.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    with holo_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["world2holo"])
        writer.writeheader()
        writer.writerow({"world2holo": "[]"})
        writer.writerow({"world2holo": json.dumps(first_valid_pose)})

    with pytest.warns(UserWarning, match="first valid pose for leading gaps"):
        holo_poses = load_hololens_world_to_camera_poses(cfg)

    poses = holo_poses.cam_T_world_list
    assert len(poses) == 2
    np.testing.assert_allclose(poses[0], np.array(first_valid_pose, dtype=np.float32))
    np.testing.assert_allclose(poses[1], np.array(first_valid_pose, dtype=np.float32))
    np.testing.assert_array_equal(holo_poses.valid_mask, np.array([False, True]))


def test_epfl_smart_kitchen_sequence_loads_coco133_labels_and_mano_stack(tmp_path: Path) -> None:
    if not _have_mano_pkls():
        pytest.skip("MANO model files are not available")

    _write_minimal_public_release(tmp_path)
    cfg = EpflSmartKitchenConfig(
        root_directory=tmp_path,
        split="train",
        participant_id="YH2002",
        session_name="2023_12_04_10_15_23",
        exo_camera_names=("output0",),
        load_labels=True,
    )

    sequence = EpflSmartKitchenSequence(cfg)
    labels = sequence.exoego_labels

    assert labels is not None
    assert labels.xyzc_stack.shape == (3, 133, 4)
    np.testing.assert_array_equal(labels.timestamps_ns, np.array([0, 33333000, 66666000], dtype=np.int64))
    np.testing.assert_allclose(labels.xyzc_stack[0, 0, :3], np.array([0.0, 1.0, 2.0], dtype=np.float32))
    np.testing.assert_allclose(labels.xyzc_stack[0, 91, :3], np.array([0.0, 1.0, 2.0], dtype=np.float32))
    np.testing.assert_allclose(labels.xyzc_stack[0, 112, :3], np.array([21.0, 22.0, 23.0], dtype=np.float32))
    assert float(labels.xyzc_stack[0, 91, 3]) == 1.0
    assert float(labels.xyzc_stack[0, 112, 3]) == 1.0

    assert labels.mano_stack is not None
    assert labels.mano_stack.use_pca is False
    assert labels.mano_stack.so3.shape == (3, 2, 48)
    assert labels.mano_stack.trans.shape == (3, 2, 3)
    # Per-hand betas (index 0 = right, 1 = left); the fixture offsets right by 1e-5.
    assert labels.mano_stack.betas.shape == (2, 10)
    np.testing.assert_allclose(
        labels.mano_stack.betas_for(0),
        np.array([float(idx) * 0.01 + 1e-5 for idx in range(10)], dtype=np.float32),
        rtol=0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        labels.mano_stack.betas_for(1),
        np.array([float(idx) * 0.01 for idx in range(10)], dtype=np.float32),
        rtol=0,
        atol=1e-6,
    )
    # Root rotation is always taken from Rh (zeros in this fixture), overlaid on
    # the first three pose entries; the remaining 45 finger pose values come
    # from the *_poses CSV column.
    np.testing.assert_allclose(labels.mano_stack.so3[0, 0, :3], np.zeros(3, dtype=np.float32))
    np.testing.assert_allclose(labels.mano_stack.so3[0, 0, 3:], np.array([0.02] * 45, dtype=np.float32))
    np.testing.assert_allclose(labels.mano_stack.so3[0, 1, :3], np.zeros(3, dtype=np.float32))
    np.testing.assert_allclose(labels.mano_stack.so3[0, 1, 3:], np.array([0.01] * 45, dtype=np.float32))


def test_epfl_smart_kitchen_rejects_short_mano_pose_vectors_instead_of_padding_pca(
    tmp_path: Path,
) -> None:
    _write_minimal_public_release(tmp_path)
    hand_path: Path = tmp_path / "Public_release_pose" / "train" / "YH2002" / "2023_12_04_10_15_23" / "pose_3d" / "pose3d_mano.csv"
    with hand_path.open(newline="") as file:
        reader = csv.DictReader(file)
        rows: list[dict[str, str]] = [dict(row) for row in reader]
        fieldnames: list[str] = list(reader.fieldnames or [])
    rows[0]["left_poses"] = json.dumps([0.1] * 12)
    with hand_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    cfg = EpflSmartKitchenConfig(
        root_directory=tmp_path,
        split="train",
        participant_id="YH2002",
        session_name="2023_12_04_10_15_23",
        exo_camera_names=("output0",),
        load_labels=True,
    )

    with pytest.raises(ValueError, match="looks like PCA coefficients"):
        EpflSmartKitchenSequence(cfg)


def test_epfl_smart_kitchen_ego_sequence_warns_when_reusing_final_pose(tmp_path: Path) -> None:
    _write_minimal_public_release(tmp_path)
    holo_path: Path = tmp_path / "Public_release_videos" / "train" / "YH2002" / "2023_12_04_10_15_23" / "meta_data" / "holo_data_wpose.csv"
    with holo_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["world2holo"])
        writer.writeheader()
        writer.writerow({"world2holo": json.dumps([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])})
        writer.writerow({"world2holo": json.dumps([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 2.0], [0.0, 0.0, 1.0, 3.0], [0.0, 0.0, 0.0, 1.0]])})
    from simplecv.data.ego.epfl_smart_kitchen_ego import EpflSmartKitchenEgoSequence

    cfg = EpflSmartKitchenConfig(
        root_directory=tmp_path,
        split="train",
        participant_id="YH2002",
        session_name="2023_12_04_10_15_23",
    )
    ego_sequence = EpflSmartKitchenEgoSequence(cfg)

    with pytest.warns(UserWarning, match="reusing the final pose"):
        ego_sequence[2]


def test_epfl_smart_kitchen_ego_cams_use_nan_extrinsics_on_tracking_dropouts(tmp_path: Path) -> None:
    _write_minimal_public_release(tmp_path)
    holo_path: Path = tmp_path / "Public_release_videos" / "train" / "YH2002" / "2023_12_04_10_15_23" / "meta_data" / "holo_data_wpose.csv"
    identity_pose: list[list[float]] = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    with holo_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["world2holo"])
        writer.writeheader()
        writer.writerow({"world2holo": json.dumps(identity_pose)})
        writer.writerow({"world2holo": "[]"})  # HoloLens tracking lost this frame
        writer.writerow({"world2holo": json.dumps(identity_pose)})
    from simplecv.data.ego.epfl_smart_kitchen_ego import EpflSmartKitchenEgoSequence

    cfg = EpflSmartKitchenConfig(
        root_directory=tmp_path,
        split="train",
        participant_id="YH2002",
        session_name="2023_12_04_10_15_23",
    )

    with pytest.warns(UserWarning, match="missing 1 HoloLens world2holo"):
        ego_sequence = EpflSmartKitchenEgoSequence(cfg)

    cams = ego_sequence.ego_cam_dict["hololens"]
    assert len(cams) == 3
    # Live frames keep finite extrinsics; the dropout frame is NaN so Rerun renders no
    # ego frustum and the keypoint projection comes out NaN (no 2D dots) instead of
    # freezing on the carried-forward pose. The video is logged untouched elsewhere.
    assert np.isfinite(cams[0].extrinsics.cam_t_world).all()
    assert np.isnan(cams[1].extrinsics.cam_t_world).all()
    assert np.isnan(cams[1].extrinsics.world_t_cam).all()
    assert np.isnan(cams[1].extrinsics.cam_T_world[:3]).all()
    assert np.isfinite(cams[2].extrinsics.cam_t_world).all()


def test_epfl_smart_kitchen_visualized_rrd_contains_video_labels_and_mano(tmp_path: Path) -> None:
    if not _have_mano_pkls():
        pytest.skip("MANO model files are not available")

    _write_minimal_public_release(tmp_path)
    cfg = EpflSmartKitchenConfig(
        root_directory=tmp_path,
        split="train",
        participant_id="YH2002",
        session_name="2023_12_04_10_15_23",
        exo_camera_names=("output0",),
        load_labels=True,
    )
    sequence = EpflSmartKitchenSequence(cfg)
    rrd_path: Path = tmp_path / "epfl-mini.rrd"
    viz_config = VisualizeConfig(
        rr_config=RerunTyroConfig(
            application_id="test-epfl-visualize",
            recording_id="test-epfl-visualize",
            save=rrd_path,
        ),
        dataset=cfg,
        log_exo=True,
        log_ego=True,
        log_labels=True,
        log_mano=True,
    )
    rec: rr.RecordingStream = viz_config.rr_config.rec_stream

    visualize_exo_ego(sequence, viz_config)
    rec.flush(timeout_sec=60.0)

    query_session = RRDQuerySession(rrd_path)
    try:
        column_names: list[str] = query_session._dataset_view("/**").arrow_schema().names
        assert "/world/gt/mano/right/mesh:Mesh3D:vertex_positions" in column_names
        assert "/world/gt/mano/left/mesh:Mesh3D:vertex_positions" in column_names
        assert "/world/gt/mano/right/mesh:Mesh3D:vertex_normals" not in column_names
        assert "/world/gt/mano/left/mesh:Mesh3D:vertex_normals" not in column_names
        assert "/world/gt/mano/coco133_xyz:Points3D:positions" in column_names
        assert "/world/ego/hololens/pinhole/video:VideoStream:codec" in column_names
        assert "/world/exo/output0/pinhole/video:VideoStream:codec" in column_names

        ego_plane_table = query_session.read_arrow(
            contents="/**",
            selectors=["/world/ego/hololens/pinhole:Pinhole:image_plane_distance"],
            index=None,
        )
        exo_plane_table = query_session.read_arrow(
            contents="/**",
            selectors=["/world/exo/output0/pinhole:Pinhole:image_plane_distance"],
            index=None,
        )
        ego_plane = first_valid_value(ego_plane_table["/world/ego/hololens/pinhole:Pinhole:image_plane_distance"])
        exo_plane = first_valid_value(exo_plane_table["/world/exo/output0/pinhole:Pinhole:image_plane_distance"])
        assert float(ego_plane[0]) == pytest.approx(0.1)
        assert float(exo_plane[0]) == pytest.approx(0.25)
    finally:
        query_session.close()


def test_epfl_smart_kitchen_visualized_rrd_can_log_mano_vertex_normals(tmp_path: Path) -> None:
    if not _have_mano_pkls():
        pytest.skip("MANO model files are not available")

    _write_minimal_public_release(tmp_path)
    cfg = EpflSmartKitchenConfig(
        root_directory=tmp_path,
        split="train",
        participant_id="YH2002",
        session_name="2023_12_04_10_15_23",
        exo_camera_names=("output0",),
        load_labels=True,
    )
    sequence = EpflSmartKitchenSequence(cfg)
    rrd_path: Path = tmp_path / "epfl-mini-with-mano-normals.rrd"
    viz_config = VisualizeConfig(
        rr_config=RerunTyroConfig(
            application_id="test-epfl-visualize-normals",
            recording_id="test-epfl-visualize-normals",
            save=rrd_path,
        ),
        dataset=cfg,
        log_exo=True,
        log_ego=True,
        log_labels=True,
        log_mano=True,
        log_mano_vertex_normals=True,
    )
    rec: rr.RecordingStream = viz_config.rr_config.rec_stream

    visualize_exo_ego(sequence, viz_config)
    rec.flush(timeout_sec=60.0)

    query_session = RRDQuerySession(rrd_path)
    try:
        column_names: list[str] = query_session._dataset_view("/**").arrow_schema().names
        assert "/world/gt/mano/right/mesh:Mesh3D:vertex_positions" in column_names
        assert "/world/gt/mano/left/mesh:Mesh3D:vertex_positions" in column_names
        assert "/world/gt/mano/right/mesh:Mesh3D:vertex_normals" in column_names
        assert "/world/gt/mano/left/mesh:Mesh3D:vertex_normals" in column_names
    finally:
        query_session.close()


def test_epfl_smart_kitchen_empty_hand_l2_cells_do_not_drop_keypoints(tmp_path: Path) -> None:
    if not _have_mano_pkls():
        pytest.skip("MANO model files are not available")

    _write_minimal_public_release(tmp_path)
    hand_path: Path = tmp_path / "Public_release_pose" / "train" / "YH2002" / "2023_12_04_10_15_23" / "pose_3d" / "pose3d_mano.csv"
    with hand_path.open(newline="") as file:
        reader = csv.DictReader(file)
        rows: list[dict[str, str]] = [dict(row) for row in reader]
        fieldnames: list[str] = list(reader.fieldnames or [])
    rows[0]["l2_dist_left"] = ""
    rows[0]["l2_dist_right"] = ""
    with hand_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    cfg = EpflSmartKitchenConfig(
        root_directory=tmp_path,
        split="train",
        participant_id="YH2002",
        session_name="2023_12_04_10_15_23",
        exo_camera_names=("output0",),
        load_labels=True,
    )

    with pytest.warns(UserWarning, match="empty EPFL hand l2 distance values"):
        sequence = EpflSmartKitchenSequence(cfg)
    labels = sequence.exoego_labels

    assert labels is not None
    assert float(labels.xyzc_stack[0, 91, 3]) == 1.0
    assert float(labels.xyzc_stack[0, 112, 3]) == 1.0


def test_epfl_smart_kitchen_empty_body_l2_cells_do_not_drop_keypoints(tmp_path: Path) -> None:
    if not _have_mano_pkls():
        pytest.skip("MANO model files are not available")

    _write_minimal_public_release(tmp_path)
    body_path: Path = tmp_path / "Public_release_pose" / "train" / "YH2002" / "2023_12_04_10_15_23" / "pose_3d" / "pose3d_smpl.csv"
    with body_path.open(newline="") as file:
        reader = csv.DictReader(file)
        rows: list[dict[str, str]] = [dict(row) for row in reader]
        fieldnames: list[str] = list(reader.fieldnames or [])
    rows[0]["l2_dist"] = ""
    with body_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    cfg = EpflSmartKitchenConfig(
        root_directory=tmp_path,
        split="train",
        participant_id="YH2002",
        session_name="2023_12_04_10_15_23",
        exo_camera_names=("output0",),
        load_labels=True,
    )

    with pytest.warns(UserWarning, match="empty EPFL body l2 distance values"):
        sequence = EpflSmartKitchenSequence(cfg)
    labels = sequence.exoego_labels

    assert labels is not None
    assert float(labels.xyzc_stack[0, 0, 3]) == 1.0


def test_epfl_smart_kitchen_label_load_fails_on_timestamp_count_mismatch(tmp_path: Path) -> None:
    _write_minimal_public_release(tmp_path)
    timestamps_path: Path = tmp_path / "Public_release_videos" / "train" / "YH2002" / "2023_12_04_10_15_23" / "meta_data" / "timestamps.txt"
    timestamps_path.write_text("1000\n34333\n")
    cfg = EpflSmartKitchenConfig(
        root_directory=tmp_path,
        split="train",
        participant_id="YH2002",
        session_name="2023_12_04_10_15_23",
        exo_camera_names=("output0",),
        load_labels=True,
    )

    with pytest.raises(ValueError, match="label frame count must match RGB timestamp count"):
        EpflSmartKitchenSequence(cfg)
