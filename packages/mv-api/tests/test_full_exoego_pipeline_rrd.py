import subprocess
from collections.abc import Generator
from pathlib import Path
from typing import cast

import numpy as np
import open3d as o3d
import pytest
import rerun as rr
import torch
from jaxtyping import Float32, Int, UInt8
from monopriors.apis.multiview_calibration import MultiViewCalibrator, MVCalibResults
from numpy import ndarray
from rerun.components.view_coordinates import ViewCoordinates
from simplecv.apis.view_exoego import LogPaths, SceneSetupResult
from simplecv.camera_parameters import Extrinsics, Intrinsics, PinholeParameters
from simplecv.data.ego.base_ego import BaseEgoSequence, CameraParam, EgoData
from simplecv.data.exo.base_exo import BaseExoSequence, ExoData
from simplecv.data.exoego.base_exoego import BaseExoEgoSequence, ExoEgoLabels, ExoEgoSample, RigLayout
from simplecv.data.exoego.hocap import HocapConfig
from simplecv.rerun_log_utils import RerunTyroConfig
from simplecv.video_io import MultiVideoReader, VideoReader
from wilor_nano.hand_keypoints import WilorHandKeypointDetector

from mv_api.api import exoego_nodes, full_exoego_app, full_exoego_pipeline
from mv_api.api.full_exoego_pipeline import CameraSource, RRDPipelineConfig
from mv_api.multiview_pose_estimator import MultiviewBodyTracker, MultiviewBodyTrackerConfig, MVHistory


class FakeExoEgoSequence(BaseExoEgoSequence[HocapConfig]):
    def __init__(self) -> None:
        # Skip the base timeline machinery but provide the side sequences the
        # real build_rig_layout reads (both None -> empty rig layout).
        self.ego_sequence = None
        self.exo_sequence = None

    def build_rig_layout(self, *, world_path: Path = Path("world"), log_exo: bool = True, log_ego: bool = True) -> RigLayout:
        # The fakes don't model the camera lists the real layout walks; an
        # empty layout keeps rig logging a no-op and exercises the pipeline's
        # legacy flat-path fallback the assertions below encode.
        del world_path, log_exo, log_ego
        return RigLayout(rigs=[], exo_cam_paths={}, ego_cam_paths={})

    @property
    def world_coordinate_system(self) -> ViewCoordinates:
        return rr.ViewCoordinates.RIGHT_HAND_Z_UP

    def _build_ego(self) -> BaseEgoSequence[HocapConfig] | None:
        return None

    def _build_exo(self) -> BaseExoSequence[HocapConfig] | None:
        return None

    def __getitem__(
        self,
        idx: int | None = None,
        ts_nano: np.timedelta64 | None = None,
    ) -> ExoEgoSample:
        del idx, ts_nano
        return ExoEgoSample(canonical_index=0, canonical_timestamp_ns=0)

    def load_stream_timestamps_ns(self) -> dict[str, Int[ndarray, "n_frames"]]:
        return {}

    def load_labels(self) -> ExoEgoLabels | None:
        return None

    @classmethod
    def iter_episode_sequences(cls, cfg: HocapConfig) -> Generator["FakeExoEgoSequence", None, None]:
        del cfg
        yield cls()

    @classmethod
    def num_sequences_for_config(cls, cfg: HocapConfig) -> int:
        del cfg
        return 1


class FakeVideoReader(VideoReader):
    def __init__(self) -> None:
        self._fps: float = 30.0
        self._frame_cnt: int = 2
        self._width: int = 32
        self._height: int = 32

    def __getitem__(self, index: int | slice) -> UInt8[ndarray, "h w 3"] | list[UInt8[ndarray, "h w 3"]]:
        if isinstance(index, slice):
            return [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(*index.indices(self._frame_cnt))]
        return np.zeros((32, 32, 3), dtype=np.uint8)

    def get_frame(self, frame_id: int) -> UInt8[ndarray, "h w 3"]:
        del frame_id
        return np.zeros((32, 32, 3), dtype=np.uint8)


class FakeMultiVideoReader(MultiVideoReader):
    def __init__(self) -> None:
        self.video_readers: list[VideoReader] = [FakeVideoReader()]

    def __getitem__(self, idx: int) -> list[UInt8[ndarray, "h w 3"]]:
        frame_list: list[UInt8[ndarray, "h w 3"]] = []
        for reader in self.video_readers:
            frame: UInt8[ndarray, "h w 3"] = cast(UInt8[ndarray, "h w 3"], reader[idx])
            frame_list.append(frame)
        return frame_list


class FakeExoSequence(BaseExoSequence[HocapConfig]):
    def __init__(self) -> None:
        self.exo_video_readers: MultiVideoReader = FakeMultiVideoReader()
        self._video_path_list: list[Path] = []
        self._exo_cam_list: list[PinholeParameters | None] = []

    def __getitem__(self, idx: int) -> ExoData:
        del idx
        raise NotImplementedError

    def load_video_paths(self) -> list[Path]:
        return []

    def load_exo_cams(self) -> list[PinholeParameters | None]:
        return []

    @property
    def image_plane_distance(self) -> float:
        return 0.1


class FakeEgoSequence(BaseEgoSequence[HocapConfig]):
    def __init__(self) -> None:
        self.ego_video_readers: MultiVideoReader = FakeMultiVideoReader()
        self._ego_video_name_list: list[str] = []

    def __getitem__(self, idx: int) -> EgoData:
        del idx
        raise NotImplementedError

    def load_video_paths(self) -> list[Path]:
        return []

    def load_ego_cams(self) -> dict[str, list[CameraParam]]:
        return {}

    def align_cams_and_videos(  # pyrefly: ignore[bad-override]
        self,
        video_path_list: list[Path],
        ego_cam_dict: dict[str, list[CameraParam]],
    ) -> tuple[dict[str, list[CameraParam]], dict[str, Path]]:
        return ego_cam_dict, {}

    @property
    def image_plane_distance(self) -> float:
        return 0.1


class FakeModelBackedSequence(FakeExoEgoSequence):
    def __init__(self) -> None:
        self.exo_sequence: BaseExoSequence[HocapConfig] | None = FakeExoSequence()
        self.ego_sequence: BaseEgoSequence[HocapConfig] | None = FakeEgoSequence()


class FakeDatasetCameraExoSequence(FakeExoSequence):
    def __init__(self) -> None:
        super().__init__()
        self._exo_cam_list = [_fake_pinhole("cam0")]
        self._video_path_list = [Path("cam0.mp4")]

    @property
    def exo_video_names(self) -> list[str]:
        return ["cam0"]


class FakeDatasetCameraEgoSequence(FakeEgoSequence):
    def __init__(self) -> None:
        super().__init__()
        self._dataset_ego_cam_dict: dict[str, list[CameraParam]] = {
            "hololens_kv5h72": [_fake_pinhole("hololens_kv5h72")]
        }
        self._ego_video_name_list = ["hololens_kv5h72"]

    @property
    def ego_cam_dict(self) -> dict[str, list[CameraParam]]:
        return self._dataset_ego_cam_dict


class FakeDatasetCameraModelBackedSequence(FakeExoEgoSequence):
    def __init__(self) -> None:
        self.exo_sequence: BaseExoSequence[HocapConfig] | None = FakeDatasetCameraExoSequence()
        self.ego_sequence: BaseEgoSequence[HocapConfig] | None = FakeDatasetCameraEgoSequence()


def _fake_pinhole(name: str) -> PinholeParameters:
    intrinsics: Intrinsics = Intrinsics.from_focal_principal_point(
        camera_conventions="RDF",
        fl_x=50.0,
        fl_y=50.0,
        cx=16.0,
        cy=16.0,
        height=32,
        width=32,
    )
    extrinsics: Extrinsics = Extrinsics(
        cam_R_world=np.eye(3, dtype=np.float32),
        cam_t_world=np.zeros(3, dtype=np.float32),
    )
    return PinholeParameters(name=name, intrinsics=intrinsics, extrinsics=extrinsics)


class FakeMultiViewCalibrator(MultiViewCalibrator):
    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs

    def __call__(self, *, rgb_list: list[UInt8[ndarray, "H W 3"]]) -> MVCalibResults:
        del rgb_list
        pcd: o3d.geometry.PointCloud = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array([[0.0, 0.0, 2.0], [0.1, 0.0, 2.0]], dtype=np.float32))
        pcd.colors = o3d.utility.Vector3dVector(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32))
        return MVCalibResults(depth_list=[], pinhole_param_list=[_fake_pinhole("cam0"), _fake_pinhole("rgb")], pcd=pcd)


class FakeBodyTracker(MultiviewBodyTracker):
    def __init__(
        self,
        config: MultiviewBodyTrackerConfig,
        filter_body_idxes: Int[ndarray, "idx"] | None = None,
    ) -> None:
        del filter_body_idxes
        self.config: MultiviewBodyTrackerConfig = config
        self.num_keypoints: int = 133

    def __call__(
        self,
        *,
        frames_rgb: UInt8[torch.Tensor, "n_views h w 3"],
        pinhole_list: list[PinholeParameters],
        pred_state: MVHistory,
        pinhole_log_paths: list[Path] | None = None,
        recording: rr.RecordingStream | None = None,
    ) -> MVHistory:
        del frames_rgb, pinhole_list, pinhole_log_paths, recording
        xyzc: Float32[ndarray, "133 4"] = np.full((133, 4), np.nan, dtype=np.float32)
        xyzc[:, 0:3] = np.array([0.0, 0.0, 2.0], dtype=np.float32)
        xyzc[:, 3] = 1.0
        pred_state.xyzc_t1 = pred_state.xyzc_t
        pred_state.xyzc_t = xyzc
        return pred_state


class FakeHandKeypointDetector(WilorHandKeypointDetector):
    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs

    def __call__(self, *args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("hand detector should not run when projected hand boxes are degenerate")


def test_pipeline_writes_rrd_with_expected_scene_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rrd_path: Path = tmp_path / "hocap-smoke.rrd"

    def fake_setup(self: HocapConfig) -> FakeExoEgoSequence:
        return FakeExoEgoSequence()

    def fake_setup_scene(
        exoego_sequence: FakeExoEgoSequence,
        *,
        rig_layout: object,
        parent_log_path: Path,
        timeline: str,
        log_ego: bool,
        log_exo: bool,
        recording: rr.RecordingStream | None = None,
    ) -> SceneSetupResult:
        del exoego_sequence, rig_layout, log_ego, log_exo
        rr.set_time(timeline, sequence=0, recording=recording)
        rr.log(str(parent_log_path / "exo" / "cam0" / "pinhole" / "video"), rr.TextDocument("exo"), recording=recording)
        rr.log(str(parent_log_path / "ego" / "rgb" / "pinhole" / "video"), rr.TextDocument("ego"), recording=recording)
        timestamps: Int[ndarray, "n_frames"] = np.array([0], dtype=np.int64)
        return SceneSetupResult(
            log_paths=LogPaths(
                exo_video_log_paths=[parent_log_path / "exo" / "cam0" / "pinhole" / "video"],
                ego_video_log_paths=[parent_log_path / "ego" / "rgb" / "pinhole" / "video"],
            ),
            shortest_timestamp=timestamps,
        )

    def fake_model_backed_pipeline(
        *,
        config: RRDPipelineConfig,
        exoego_sequence: FakeExoEgoSequence,
        scene_setup_result: SceneSetupResult,
        parent_log_path: Path,
        timeline: str,
        exo_cam_paths: dict[str, Path] | None = None,
        ego_cam_paths: dict[str, Path] | None = None,
        recording: rr.RecordingStream | None = None,
    ) -> None:
        del config, exoego_sequence, scene_setup_result, timeline, exo_cam_paths, ego_cam_paths
        points3d: Float32[ndarray, "1 3"] = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        points2d: Float32[ndarray, "1 2"] = np.array([[12.0, 24.0]], dtype=np.float32)
        rr.log(str(parent_log_path / "gt" / "coco133_xyz"), rr.Points3D(points3d), recording=recording)
        rr.log(
            str(parent_log_path / "ego" / "rgb" / "pinhole" / "pred" / "coco133_uv" / "projected"),
            rr.Points2D(points2d),
            recording=recording,
        )

    monkeypatch.setattr(HocapConfig, "setup", fake_setup)
    monkeypatch.setattr(exoego_nodes, "setup_scene", fake_setup_scene)
    monkeypatch.setattr(full_exoego_pipeline, "run_model_backed_pipeline", fake_model_backed_pipeline)

    rr_config: RerunTyroConfig = RerunTyroConfig(
        application_id="mv_api_test",
        save=rrd_path,
        headless=True,
    )
    config: RRDPipelineConfig = RRDPipelineConfig(
        rr_config=rr_config,
        dataset=HocapConfig(),
        max_frames=1,
    )

    run_result: full_exoego_app.FullExoEgoRunResult = full_exoego_app.run_full_exoego_app(config=config)

    assert rrd_path.exists()
    assert rrd_path.stat().st_size > 0
    assert run_result.rrd_path == rrd_path
    assert run_result.parent_log_path == Path("world")
    assert run_result.timeline == "video_time"
    np.testing.assert_array_equal(run_result.model_backed.processed_timestamps, np.array([0], dtype=np.int64))

    stats_result: subprocess.CompletedProcess[str] = subprocess.run(
        ["rerun", "rrd", "stats", str(rrd_path)],
        check=False,
        text=True,
        capture_output=True,
    )
    assert stats_result.returncode == 0, stats_result.stderr
    assert "/world/exo/cam0/pinhole/video" in stats_result.stdout
    assert "/world/ego/rgb/pinhole/video" in stats_result.stdout
    assert "/world/gt/coco133_xyz" in stats_result.stdout
    assert "/world/ego/rgb/pinhole/pred/coco133_uv/projected" in stats_result.stdout


def test_model_backed_hook_logs_calibrated_predictions_with_fake_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rrd_path: Path = tmp_path / "hocap-fake-models.rrd"

    def fake_setup(self: HocapConfig) -> FakeModelBackedSequence:
        del self
        return FakeModelBackedSequence()

    def fake_setup_scene(
        exoego_sequence: FakeModelBackedSequence,
        *,
        rig_layout: object,
        parent_log_path: Path,
        timeline: str,
        log_ego: bool,
        log_exo: bool,
        recording: rr.RecordingStream | None = None,
    ) -> SceneSetupResult:
        del exoego_sequence, rig_layout, log_ego, log_exo
        rr.set_time(timeline, duration=np.timedelta64(0, "ns"), recording=recording)
        rr.log(str(parent_log_path / "exo" / "cam0" / "pinhole" / "video"), rr.TextDocument("exo"), recording=recording)
        rr.log(
            str(parent_log_path / "ego" / "hololens_kv5h72" / "pinhole" / "video"),
            rr.TextDocument("ego"),
            recording=recording,
        )
        timestamps: Int[ndarray, "n_frames"] = np.array([0], dtype=np.int64)
        return SceneSetupResult(
            log_paths=LogPaths(
                exo_video_log_paths=[parent_log_path / "exo" / "cam0" / "pinhole" / "video"],
                ego_video_log_paths=[parent_log_path / "ego" / "hololens_kv5h72" / "pinhole" / "video"],
            ),
            shortest_timestamp=timestamps,
        )

    monkeypatch.setattr(HocapConfig, "setup", fake_setup)
    monkeypatch.setattr(exoego_nodes, "setup_scene", fake_setup_scene)
    monkeypatch.setattr(full_exoego_pipeline, "MultiViewCalibrator", FakeMultiViewCalibrator)
    monkeypatch.setattr(full_exoego_pipeline, "MultiviewBodyTracker", FakeBodyTracker)
    monkeypatch.setattr(full_exoego_pipeline, "WilorHandKeypointDetector", FakeHandKeypointDetector)
    def fake_estimate_voxel_size(points: Float32[ndarray, "num_points 3"], target_points: int) -> float:
        del points, target_points
        return 0.01

    monkeypatch.setattr(full_exoego_pipeline, "estimate_voxel_size", fake_estimate_voxel_size)

    rr_config: RerunTyroConfig = RerunTyroConfig(
        application_id="mv_api_fake_models_test",
        save=rrd_path,
        headless=True,
    )
    config: RRDPipelineConfig = RRDPipelineConfig(
        rr_config=rr_config,
        dataset=HocapConfig(),
        max_frames=1,
    )

    full_exoego_app.run_full_exoego_app(config=config)

    stats_result: subprocess.CompletedProcess[str] = subprocess.run(
        ["rerun", "rrd", "stats", str(rrd_path)],
        check=False,
        text=True,
        capture_output=True,
    )
    assert stats_result.returncode == 0, stats_result.stderr
    assert "/world/gt/env_pointcloud" in stats_result.stdout
    assert "/world/gt/coco133_xyz" in stats_result.stdout
    assert "/world/exo/cam0/pinhole/gt/coco133_uv" in stats_result.stdout
    assert "/world/ego/hololens_kv5h72/pinhole/pred/coco133_uv/projected" in stats_result.stdout


@pytest.mark.parametrize("camera_source", ["auto", "dataset"])
def test_dataset_camera_sources_skip_estimated_environment_logging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    camera_source: CameraSource,
) -> None:
    rrd_path: Path = tmp_path / f"hocap-{camera_source}-cameras.rrd"

    def fake_setup(self: HocapConfig) -> FakeDatasetCameraModelBackedSequence:
        del self
        return FakeDatasetCameraModelBackedSequence()

    def fake_setup_scene(
        exoego_sequence: FakeDatasetCameraModelBackedSequence,
        *,
        rig_layout: object,
        parent_log_path: Path,
        timeline: str,
        log_ego: bool,
        log_exo: bool,
        recording: rr.RecordingStream | None = None,
    ) -> SceneSetupResult:
        del exoego_sequence, rig_layout, log_ego, log_exo
        rr.set_time(timeline, duration=np.timedelta64(0, "ns"), recording=recording)
        rr.log(str(parent_log_path / "exo" / "cam0" / "pinhole" / "video"), rr.TextDocument("exo"), recording=recording)
        rr.log(
            str(parent_log_path / "ego" / "hololens_kv5h72" / "pinhole" / "video"),
            rr.TextDocument("ego"),
            recording=recording,
        )
        timestamps: Int[ndarray, "n_frames"] = np.array([0], dtype=np.int64)
        return SceneSetupResult(
            log_paths=LogPaths(
                exo_video_log_paths=[parent_log_path / "exo" / "cam0" / "pinhole" / "video"],
                ego_video_log_paths=[parent_log_path / "ego" / "hololens_kv5h72" / "pinhole" / "video"],
            ),
            shortest_timestamp=timestamps,
        )

    class FailingMultiViewCalibrator(FakeMultiViewCalibrator):
        def __call__(self, *, rgb_list: list[UInt8[ndarray, "H W 3"]]) -> MVCalibResults:
            del rgb_list
            raise AssertionError("GT camera mode should not estimate cameras")

    monkeypatch.setattr(HocapConfig, "setup", fake_setup)
    monkeypatch.setattr(exoego_nodes, "setup_scene", fake_setup_scene)
    monkeypatch.setattr(full_exoego_pipeline, "MultiViewCalibrator", FailingMultiViewCalibrator)
    monkeypatch.setattr(full_exoego_pipeline, "MultiviewBodyTracker", FakeBodyTracker)
    monkeypatch.setattr(full_exoego_pipeline, "WilorHandKeypointDetector", FakeHandKeypointDetector)

    rr_config: RerunTyroConfig = RerunTyroConfig(
        application_id="mv_api_dataset_camera_test",
        save=rrd_path,
        headless=True,
    )
    config: RRDPipelineConfig = RRDPipelineConfig(
        rr_config=rr_config,
        dataset=HocapConfig(),
        camera_source=camera_source,
        max_frames=1,
    )

    full_exoego_app.run_full_exoego_app(config=config)

    stats_result: subprocess.CompletedProcess[str] = subprocess.run(
        ["rerun", "rrd", "stats", str(rrd_path)],
        check=False,
        text=True,
        capture_output=True,
    )
    assert stats_result.returncode == 0, stats_result.stderr
    assert "/world/gt/env_pointcloud" not in stats_result.stdout
    assert "/world/gt/env_mesh" not in stats_result.stdout
    assert "/world/gt/coco133_xyz" in stats_result.stdout
    assert "/world/exo/cam0/pinhole/gt/coco133_uv" in stats_result.stdout
    assert "/world/ego/hololens_kv5h72/pinhole/pred/coco133_uv/projected" in stats_result.stdout
