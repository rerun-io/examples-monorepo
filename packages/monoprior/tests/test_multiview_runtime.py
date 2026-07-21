from pathlib import Path
from threading import Event, Thread

import numpy as np
import pytest
from simplecv.camera_orient_utils import rotation_matrix_between
from simplecv.camera_parameters import Extrinsics, Intrinsics, PinholeParameters

from monopriors.apis.calibrate_synced_videos import CalibConfig
from monopriors.apis.multiview_calibration import (
    MultiViewCalibratorConfig,
    run_multiview_calibration,
)
from monopriors.apis.multiview_geometry import MultiviewGeometryConfig, run_multiview_geometry
from monopriors.gradio_ui.multiview_calibration_ui import _prepare_request as _prepare_calibration_request
from monopriors.gradio_ui.multiview_geometry_ui import _prepare_request
from monopriors.models.multiview.multiview_model import MultiviewPred
from monopriors.models.multiview.multiview_predictor import (
    MultiviewPredictor,
    MultiviewPredictorCache,
    MultiviewPredictorConfig,
)


def _prediction() -> MultiviewPred:
    pinhole = PinholeParameters(
        name="camera_0",
        intrinsics=Intrinsics(
            camera_conventions="RDF",
            fl_x=10.0,
            fl_y=10.0,
            cx=1.0,
            cy=1.0,
            width=2,
            height=2,
        ),
        extrinsics=Extrinsics(
            cam_R_world=np.eye(3, dtype=np.float32),
            cam_t_world=np.zeros(3, dtype=np.float32),
        ),
    )
    return MultiviewPred(
        cam_name="camera_0",
        rgb_image=np.zeros((2, 2, 3), dtype=np.uint8),
        depth_map=np.ones((2, 2), dtype=np.float32),
        confidence_mask=np.ones((2, 2), dtype=np.float32),
        pinhole_param=pinhole,
    )


def test_geometry_has_one_config_source_and_forwards_center_method() -> None:
    class FakePredictor(MultiviewPredictor):
        def __init__(self) -> None:
            self.config = MultiviewPredictorConfig(device="cpu")
            self.center_method = ""
            self.preprocessing_mode = ""

        def __call__(
            self,
            rgb_list: list[np.ndarray],
            *,
            preprocessing_mode: str = "pad",
            center_method: str = "none",
        ) -> list[MultiviewPred]:
            self.center_method = center_method
            self.preprocessing_mode = preprocessing_mode
            return [_prediction()]

    predictor = FakePredictor()
    config = MultiviewGeometryConfig(
        keep_top_percent=100.0,
        preprocessing_mode="crop",
        center_method="focus",
    )

    result = run_multiview_geometry(
        rgb_list=[np.zeros((2, 2, 3), dtype=np.uint8)],
        multiview_predictor=predictor,
        config=config,
    )

    assert predictor.center_method == "focus"
    assert predictor.preprocessing_mode == "crop"
    assert len(result.mv_pred_list) == 1
    assert not hasattr(config, "model_name")
    assert not hasattr(config, "device")


def test_predictor_cache_reuses_runtime_identity_and_closes_before_replacement() -> None:
    events: list[tuple[str, MultiviewPredictorConfig]] = []

    class FakePredictor(MultiviewPredictor):
        def __init__(self, config: MultiviewPredictorConfig) -> None:
            self.config = config
            events.append(("create", config))

        def close(self) -> None:
            events.append(("close", self.config))

    cache = MultiviewPredictorCache(factory=FakePredictor)
    vggt_config = MultiviewPredictorConfig(model_name="vggt", device="cpu")
    g3t_config = MultiviewPredictorConfig(model_name="g3t", device="cpu")

    with cache.acquire(vggt_config) as first:
        pass
    with cache.acquire(vggt_config) as reused:
        assert reused is first
    with cache.acquire(g3t_config):
        pass

    assert events == [("create", vggt_config), ("close", vggt_config), ("create", g3t_config)]


def test_predictor_cache_reuses_configs_with_the_same_runtime_identity() -> None:
    created: list[MultiviewPredictorConfig] = []

    class FakePredictor(MultiviewPredictor):
        def __init__(self, config: MultiviewPredictorConfig) -> None:
            self.config = config
            created.append(config)

        def close(self) -> None:
            pass

    cache = MultiviewPredictorCache(factory=FakePredictor)
    vggt = MultiviewPredictorConfig(model_name="vggt", device="cpu")
    equivalent_vggt = MultiviewPredictorConfig(
        model_name="vggt",
        device="cpu",
        local_files_only=True,
        g3t_compile=True,
    )
    cpu_g3t = MultiviewPredictorConfig(model_name="g3t", device="cpu")
    equivalent_cpu_g3t = MultiviewPredictorConfig(model_name="g3t", device="cpu", g3t_compile=True)

    with cache.acquire(vggt) as first_vggt:
        pass
    with cache.acquire(equivalent_vggt) as second_vggt:
        assert second_vggt is first_vggt
    with cache.acquire(cpu_g3t) as first_g3t:
        pass
    with cache.acquire(equivalent_cpu_g3t) as second_g3t:
        assert second_g3t is first_g3t

    assert created == [vggt, cpu_g3t]


def test_predictor_cache_distinguishes_effective_cuda_compilation() -> None:
    created: list[MultiviewPredictorConfig] = []

    class FakePredictor(MultiviewPredictor):
        def __init__(self, config: MultiviewPredictorConfig) -> None:
            self.config = config
            created.append(config)

        def close(self) -> None:
            pass

    cache = MultiviewPredictorCache(factory=FakePredictor)
    eager = MultiviewPredictorConfig(model_name="g3t", device="cuda")
    compiled = MultiviewPredictorConfig(model_name="g3t", device="cuda", g3t_compile=True)

    with cache.acquire(eager):
        pass
    with cache.acquire(compiled):
        pass

    assert created == [eager, compiled]


def test_predictor_cache_cannot_replace_a_backend_while_a_run_uses_it() -> None:
    class FakePredictor(MultiviewPredictor):
        def __init__(self, config: MultiviewPredictorConfig) -> None:
            self.config = config

        def close(self) -> None:
            pass

    cache = MultiviewPredictorCache(factory=FakePredictor)
    vggt_config = MultiviewPredictorConfig(model_name="vggt", device="cpu")
    g3t_config = MultiviewPredictorConfig(model_name="g3t", device="cpu")
    first_acquired = Event()
    release_first = Event()
    second_acquired = Event()

    def use_first() -> None:
        with cache.acquire(vggt_config):
            first_acquired.set()
            assert release_first.wait(timeout=2.0)

    def use_second() -> None:
        assert first_acquired.wait(timeout=2.0)
        with cache.acquire(g3t_config):
            second_acquired.set()

    first_thread = Thread(target=use_first)
    second_thread = Thread(target=use_second)
    first_thread.start()
    second_thread.start()
    assert first_acquired.wait(timeout=2.0)
    assert not second_acquired.wait(timeout=0.1)
    release_first.set()
    first_thread.join(timeout=2.0)
    second_thread.join(timeout=2.0)
    assert second_acquired.is_set()


def test_gradio_request_captures_backend_and_operation_config_together(monkeypatch: pytest.MonkeyPatch) -> None:
    rgb_list = [np.zeros((2, 2, 3), dtype=np.uint8)]
    monkeypatch.setattr(
        "monopriors.gradio_ui.multiview_geometry_ui._parse_and_load_images",
        lambda _files: rgb_list,
    )

    request = _prepare_request(
        ["unused.jpg"],
        model_name="g3t",
        keep_top_percent=42.0,
        preprocessing_mode="crop",
        verbose=False,
    )

    assert request.rgb_list is rgb_list
    assert request.predictor_config == MultiviewPredictorConfig(
        model_name="g3t",
        device="cuda",
    )
    assert request.geometry_config == MultiviewGeometryConfig(
        keep_top_percent=42.0,
        preprocessing_mode="crop",
        verbose=False,
    )


def test_calibration_request_composes_the_canonical_geometry_config(monkeypatch: pytest.MonkeyPatch) -> None:
    rgb_list = [np.zeros((2, 2, 3), dtype=np.uint8)]
    monkeypatch.setattr(
        "monopriors.gradio_ui.multiview_calibration_ui._parse_and_load_images",
        lambda _files: rgb_list,
    )

    request = _prepare_calibration_request(
        ["unused.jpg"],
        model_name="g3t",
        keep_top_percent=42.0,
        refine_depth_maps=False,
        segment_people=False,
        preprocessing_mode="crop",
    )

    assert request.rgb_list is rgb_list
    assert request.config.predictor_config.model_name == "g3t"
    assert request.config.geometry_config == MultiviewGeometryConfig(
        keep_top_percent=42.0,
        preprocessing_mode="crop",
        verbose=True,
    )


def test_synced_video_calibration_exposes_the_canonical_backend_config() -> None:
    predictor_config = MultiviewPredictorConfig(model_name="g3t", device="cpu")
    geometry_config = MultiviewGeometryConfig(preprocessing_mode="crop")

    config = CalibConfig(
        videos_dir=Path("capture"),
        predictor_config=predictor_config,
        geometry_config=geometry_config,
    )

    assert config.predictor_config is predictor_config
    assert config.geometry_config is geometry_config


@pytest.mark.parametrize(
    ("source", "target"),
    [
        (np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 1.0])),
        (np.array([0.0, 0.0, -1.0]), np.array([0.0, 0.0, 1.0])),
    ],
)
def test_shared_rotation_handles_aligned_and_antiparallel_vectors(
    source: np.ndarray,
    target: np.ndarray,
) -> None:
    rotation = rotation_matrix_between(source, target)
    np.testing.assert_allclose(rotation @ source, target, rtol=0.0, atol=1.0e-7)


def test_calibration_pipeline_uses_the_supplied_predictor() -> None:
    class FakePredictor(MultiviewPredictor):
        def __init__(self) -> None:
            self.config = MultiviewPredictorConfig(device="cpu")
            self.calls = 0

        def __call__(
            self,
            rgb_list: list[np.ndarray],
            *,
            preprocessing_mode: str = "pad",
            center_method: str = "none",
        ) -> list[MultiviewPred]:
            self.calls += 1
            return [_prediction()]

    predictor = FakePredictor()
    rgb_list = [np.zeros((2, 2, 3), dtype=np.uint8)]
    config = MultiViewCalibratorConfig(
        predictor_config=predictor.config,
        geometry_config=MultiviewGeometryConfig(keep_top_percent=100.0),
        refine_depth_maps=False,
        segment_people=False,
    )

    result = run_multiview_calibration(
        rgb_list=rgb_list,
        multiview_predictor=predictor,
        config=config,
        parent_log_path=Path("world"),
    )

    assert predictor.calls == 1
    assert len(result.pinhole_param_list) == 1
    assert np.asarray(result.pcd.points).shape == (4, 3)
