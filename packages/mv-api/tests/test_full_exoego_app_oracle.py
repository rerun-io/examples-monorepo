from collections.abc import Generator
from pathlib import Path

import numpy as np
import pytest
import rerun as rr
from jaxtyping import Float32, Int
from numpy import ndarray
from rerun.components.view_coordinates import ViewCoordinates
from simplecv.apis.view_exoego import LogPaths, SceneSetupResult
from simplecv.data.ego.base_ego import BaseEgoSequence
from simplecv.data.exo.base_exo import BaseExoSequence
from simplecv.data.exoego.base_exoego import BaseExoEgoSequence, ExoEgoLabels, ExoEgoSample
from simplecv.data.exoego.hocap import HocapConfig
from simplecv.rerun_log_utils import RerunTyroConfig

from mv_api.api import full_exoego_app, full_exoego_pipeline
from mv_api.api.full_exoego_pipeline import RRDPipelineConfig
from mv_api.api.rerun_artifact_compare import compare_rrd_files, query_rrd_stats


class FakeExoEgoSequence(BaseExoEgoSequence[HocapConfig]):
    def __init__(self) -> None:
        pass

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


def test_node_app_rrd_matches_legacy_pipeline_oracle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy_rrd_path: Path = tmp_path / "legacy.rrd"
    node_rrd_path: Path = tmp_path / "node-app.rrd"

    def fake_setup(self: HocapConfig) -> FakeExoEgoSequence:
        del self
        return FakeExoEgoSequence()

    def fake_setup_scene(
        exoego_sequence: FakeExoEgoSequence,
        *,
        parent_log_path: Path,
        timeline: str,
        log_ego: bool,
        log_exo: bool,
        recording: rr.RecordingStream | None = None,
    ) -> SceneSetupResult:
        del exoego_sequence, log_ego, log_exo
        rr.set_time(timeline=timeline, duration=np.timedelta64(0, "ns"), recording=recording)
        rr.log(str(parent_log_path / "exo" / "cam0" / "pinhole" / "video"), rr.TextDocument("exo"), recording=recording)
        rr.log(str(parent_log_path / "ego" / "rgb" / "pinhole" / "video"), rr.TextDocument("ego"), recording=recording)
        shortest_timestamp: Int[ndarray, "n_frames"] = np.array([0], dtype=np.int64)
        return SceneSetupResult(
            log_paths=LogPaths(
                exo_video_log_paths=[parent_log_path / "exo" / "cam0" / "pinhole" / "video"],
                ego_video_log_paths=[parent_log_path / "ego" / "rgb" / "pinhole" / "video"],
            ),
            shortest_timestamp=shortest_timestamp,
        )

    def fake_model_backed_pipeline(
        *,
        config: RRDPipelineConfig,
        exoego_sequence: FakeExoEgoSequence,
        scene_setup_result: SceneSetupResult,
        parent_log_path: Path,
        timeline: str,
        recording: rr.RecordingStream | None = None,
    ) -> None:
        del config, exoego_sequence, scene_setup_result, timeline
        points3d: Float32[ndarray, "2 3"] = np.array(
            [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]],
            dtype=np.float32,
        )
        points2d: Float32[ndarray, "2 2"] = np.array(
            [[12.0, 24.0], [20.0, 28.0]],
            dtype=np.float32,
        )
        rr.log(str(parent_log_path / "gt" / "coco133_xyz"), rr.Points3D(points3d), recording=recording)
        rr.log(
            str(parent_log_path / "ego" / "rgb" / "pinhole" / "pred" / "coco133_uv" / "projected"),
            rr.Points2D(points2d),
            recording=recording,
        )

    monkeypatch.setattr(HocapConfig, "setup", fake_setup)
    monkeypatch.setattr(full_exoego_pipeline, "setup_scene", fake_setup_scene)
    monkeypatch.setattr(full_exoego_pipeline, "_run_model_backed_pipeline", fake_model_backed_pipeline)

    legacy_config: RRDPipelineConfig = RRDPipelineConfig(
        rr_config=RerunTyroConfig(
            application_id="mv_api_oracle_test",
            recording_id="fixed_oracle_recording",
            save=legacy_rrd_path,
            headless=True,
        ),
        dataset=HocapConfig(),
        max_frames=1,
    )
    full_exoego_pipeline.run_full_exoego_pipeline(config=legacy_config)

    node_config: RRDPipelineConfig = RRDPipelineConfig(
        rr_config=RerunTyroConfig(
            application_id="mv_api_oracle_test",
            recording_id="fixed_oracle_recording",
            save=node_rrd_path,
            headless=True,
        ),
        dataset=HocapConfig(),
        max_frames=1,
    )
    node_result: full_exoego_app.FullExoEgoRunResult = full_exoego_app.run_full_exoego_app(config=node_config)

    assert node_result.rrd_path == node_rrd_path
    assert node_result.parent_log_path == Path("world")
    assert node_result.timeline == "video_time"

    node_stats = query_rrd_stats(rrd_path=node_rrd_path)
    comparison = compare_rrd_files(expected_rrd_path=legacy_rrd_path, actual_rrd_path=node_rrd_path)

    assert node_stats.returncode == 0, node_stats.stderr
    assert node_stats.entity_chunk_counts["/world/gt/coco133_xyz"] == 1
    assert node_stats.entity_chunk_counts["/world/ego/rgb/pinhole/pred/coco133_uv/projected"] == 1
    assert comparison.exact_match, comparison.stderr
