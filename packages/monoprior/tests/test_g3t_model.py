import math
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from simplecv.camera_parameters import Extrinsics, Intrinsics, PinholeParameters
from torch import nn
from vggt.models.vggt import VGGT

from monopriors.apis.multiview_calibration import MultiViewCalibrator, MultiViewCalibratorConfig
from monopriors.apis.multiview_geometry import MultiviewGeometryConfig, run_multiview_geometry
from monopriors.gradio_ui.multiview_geometry_ui import _prepare_request
from monopriors.models.multiview.multiview_predictor import (
    G3T_CHECKPOINT_REVISION,
    G3TBackend,
    MultiviewPredictor,
    MultiviewPredictorCache,
    MultiviewPredictorConfig,
    decode_g3t_camera_heads,
)
from monopriors.models.multiview.vggt_model import (
    MultiviewModelPredictions,
    MultiviewPred,
    generate_multiview_pred,
    preprocess_images,
)
from monopriors.third_party.g3t.layers.attention import Attention
from monopriors.third_party.g3t.layers.rope import PositionGetter, RotaryPositionEmbedding2D
from monopriors.third_party.g3t.models.g3t import G3T


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


def _rgb_image(height: int = 28, width: int = 42) -> np.ndarray:
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[..., 0] = np.arange(width, dtype=np.uint8)[None, :]
    return rgb


class FakeG3T(G3T):
    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.point_head: nn.Module | None = nn.Identity()

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        assert self.point_head is None
        batch: int = 1
        num_cams: int = images.shape[0]
        height: int = images.shape[-2]
        width: int = images.shape[-1]
        local_pose_encoding = torch.zeros((batch, num_cams, 6), dtype=torch.float32)
        local_pose_encoding[..., 3] = 1.0
        local_pose_encoding[..., 4:] = math.pi / 2.0
        relative_pose_encoding = torch.zeros((batch, num_cams, 5), dtype=torch.float32)
        relative_pose_encoding[..., 0] = 1.0
        relative_pose_encoding[..., 4] = 1.0
        return {
            "local_pose_enc": local_pose_encoding,
            "global_pose_enc": relative_pose_encoding,
            "depth": torch.ones((batch, num_cams, height, width, 1), dtype=torch.float32),
            "depth_conf": torch.ones((batch, num_cams, height, width), dtype=torch.float32),
        }


def test_g3t_predictor_pins_checkpoint_skips_point_head_and_preserves_gravity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = FakeG3T()

    def load_g3t(
        _model_class: type[G3T],
        repo_id: str,
        *,
        revision: str,
        local_files_only: bool,
    ) -> FakeG3T:
        assert repo_id == "thatbrguy/g3t"
        assert revision == G3T_CHECKPOINT_REVISION
        assert local_files_only is True
        return model

    def fail_if_auto_orient_is_called(*_args: object, **_kwargs: object) -> None:
        pytest.fail("G3T must use its predicted gravity instead of estimating up from cameras")

    monkeypatch.setattr(G3T, "from_pretrained", classmethod(load_g3t))
    monkeypatch.setattr(
        "monopriors.models.multiview.multiview_predictor.auto_orient_and_center_poses",
        fail_if_auto_orient_is_called,
    )
    predictor = MultiviewPredictor(
        MultiviewPredictorConfig(model_name="g3t", device="cpu", preprocessing_mode="pad", local_files_only=True)
    )

    predictions = predictor([_rgb_image()], center_method="none")

    assert len(predictions) == 1
    assert predictions[0].depth_map.shape == (28, 42)
    np.testing.assert_allclose(
        predictions[0].pinhole_param.extrinsics.world_R_cam @ np.array([0.0, -1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        atol=1e-6,
    )


def test_preprocess_images_parallel_path_preserves_per_image_results() -> None:
    images = [_rgb_image(height=28, width=42), _rgb_image(height=42, width=28)]

    batch = preprocess_images(images, mode="pad")
    individual = [preprocess_images([image], mode="pad") for image in images]

    torch.testing.assert_close(batch.images, torch.stack([result.images[0] for result in individual]), rtol=0.0, atol=0.0)
    assert batch.metadata == [result.metadata[0] for result in individual]


def test_fast_g3t_rgb_materialization_stays_within_five_percent() -> None:
    rgb = _rgb_image(height=48, width=64)
    preprocessed = preprocess_images([rgb], mode="pad")

    def model_predictions() -> MultiviewModelPredictions:
        return MultiviewModelPredictions(
            depth=np.ones((1, 1, 518, 518, 1), dtype=np.float32),
            depth_conf=np.ones((1, 1, 518, 518), dtype=np.float32),
            intrinsic=np.array([[[[259.0, 0.0, 259.0], [0.0, 259.0, 259.0], [0.0, 0.0, 1.0]]]], dtype=np.float32),
            cam_T_world_b34=np.array(
                [[[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]]],
                dtype=np.float32,
            ),
        )

    reference = generate_multiview_pred(
        model_predictions(),
        img_tensors=preprocessed.images,
        rgb_list=[rgb],
        metadata_list=preprocessed.metadata,
        fast_rgb=False,
    )[0]
    candidate = generate_multiview_pred(
        model_predictions(),
        img_tensors=preprocessed.images,
        rgb_list=[rgb],
        metadata_list=preprocessed.metadata,
        fast_rgb=True,
    )[0]

    assert float(np.mean(np.abs(reference.rgb_image.astype(np.int16) - candidate.rgb_image.astype(np.int16)) <= 13)) >= 0.95
    np.testing.assert_array_equal(candidate.depth_map, reference.depth_map)
    np.testing.assert_array_equal(candidate.confidence_mask, reference.confidence_mask)
    np.testing.assert_allclose(candidate.pinhole_param.intrinsics.k_matrix, reference.pinhole_param.intrinsics.k_matrix)


def test_graphable_rope_preserves_original_frequency_lookup() -> None:
    rope = RotaryPositionEmbedding2D()
    tokens = torch.randn(2, 4, 9, 64)
    positions = torch.randint(0, 4, (2, 9, 2))
    feature_dim = tokens.shape[-1] // 2
    cos_comp, sin_comp = rope._compute_frequency_components(
        feature_dim,
        int(positions.max()) + 1,
        tokens.device,
        tokens.dtype,
    )
    vertical, horizontal = tokens.chunk(2, dim=-1)
    expected = torch.cat(
        (
            rope._apply_1d_rope(vertical, positions[..., 0], cos_comp, sin_comp),
            rope._apply_1d_rope(horizontal, positions[..., 1], cos_comp, sin_comp),
        ),
        dim=-1,
    )

    actual = rope(tokens, positions)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_g3t_compilation_can_be_disabled() -> None:
    assert MultiviewPredictorConfig(model_name="g3t").g3t_compile is True
    assert MultiviewPredictorConfig(model_name="g3t", g3t_compile=False).g3t_compile is False


def test_g3t_compile_warms_frame_and_global_position_caches() -> None:
    rope = RotaryPositionEmbedding2D()
    attention = Attention(dim=1024, num_heads=16, rope=rope)
    position_getter = PositionGetter()
    aggregator = SimpleNamespace(
        patch_size=14,
        patch_start_idx=5,
        position_getter=position_getter,
        rope=rope,
        frame_blocks=[SimpleNamespace(attn=attention)],
    )
    backend = object.__new__(G3TBackend)
    backend.model = SimpleNamespace(aggregator=aggregator)
    backend.dtype = torch.bfloat16
    backend._warmed_input_shapes = set()

    backend._warm_position_caches(torch.zeros((2, 3, 28, 42)))

    assert (2, 3) in position_getter.position_cache
    assert (32, 11, torch.device("cpu"), torch.float32) in rope.frequency_cache
    assert (32, 22, torch.device("cpu"), torch.float32) in rope.frequency_cache


def test_g3t_keeps_camera_heads_fp32_but_allows_mixed_precision_depth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    disabled_autocast_depths: list[int] = []

    @contextmanager
    def record_disabled_autocast(_device_type: str, *, enabled: bool) -> Iterator[None]:
        assert enabled is False
        disabled_autocast_depths.append(1)
        try:
            yield
        finally:
            disabled_autocast_depths.pop()

    class FakeAggregator(nn.Module):
        def forward(self, images: torch.Tensor) -> tuple[list[torch.Tensor], int]:
            batch, num_cams = images.shape[:2]
            return [torch.zeros((batch, num_cams, 1, 2))], 0

    class FakeCameraHead(nn.Module):
        def forward(self, _tokens: list[torch.Tensor]) -> list[torch.Tensor]:
            assert disabled_autocast_depths
            return [torch.zeros((1, 1, 6))]

    class FakeDepthHead(nn.Module):
        def forward(
            self,
            _tokens: list[torch.Tensor],
            *,
            images: torch.Tensor,
            patch_start_idx: int,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            assert not disabled_autocast_depths
            assert patch_start_idx == 0
            shape = (*images.shape[:2], *images.shape[-2:], 1)
            depth = torch.ones(shape, dtype=torch.bfloat16)
            return depth, depth.squeeze(-1)

    model = object.__new__(G3T)
    nn.Module.__init__(model)
    model.aggregator = FakeAggregator()
    model.add_module("local_camera_head", FakeCameraHead())
    model.add_module("global_camera_head", FakeCameraHead())
    model.add_module("depth_head", FakeDepthHead())
    model.point_head = None
    monkeypatch.setattr(torch.amp, "autocast", record_disabled_autocast)

    predictions = model(torch.zeros((1, 3, 2, 2)))

    assert predictions["depth"].dtype is torch.float32
    assert predictions["depth_conf"].dtype is torch.float32


def test_g3t_camera_heads_compose_local_and_relative_poses() -> None:
    local_pose_encoding = torch.tensor(
        [[[0.0, 0.0, 0.0, 1.0, math.pi / 2.0, math.pi / 2.0]]], dtype=torch.float32
    )
    relative_pose_encoding = torch.tensor([[[1.0, 2.0, 3.0, 0.0, 1.0]]], dtype=torch.float32)

    cam_T_world_b34, intrinsic_b33 = decode_g3t_camera_heads(
        local_pose_encoding=local_pose_encoding,
        relative_pose_encoding=relative_pose_encoding,
        image_size_hw=(100, 200),
    )

    expected_cam_T_world_b34 = torch.tensor(
        [[[[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 2.0], [0.0, 0.0, 1.0, 3.0]]]], dtype=torch.float32
    )
    expected_intrinsic_b33 = torch.tensor(
        [[[[100.0, 0.0, 100.0], [0.0, 50.0, 50.0], [0.0, 0.0, 1.0]]]], dtype=torch.float32
    )
    torch.testing.assert_close(cam_T_world_b34, expected_cam_T_world_b34)
    torch.testing.assert_close(intrinsic_b33, expected_intrinsic_b33)


def test_vggt_backend_preserves_existing_up_estimation_and_common_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeVGGT(VGGT):
        def __init__(self) -> None:
            nn.Module.__init__(self)

        def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
            num_cams: int = images.shape[0]
            height: int = images.shape[-2]
            width: int = images.shape[-1]
            pose_encoding = torch.zeros((1, num_cams, 9), dtype=torch.float32)
            pose_encoding[..., 3] = 1.0
            pose_encoding[..., 7:] = math.pi / 2.0
            return {
                "pose_enc": pose_encoding,
                "depth": torch.ones((1, num_cams, height, width, 1), dtype=torch.float32),
                "depth_conf": torch.ones((1, num_cams, height, width), dtype=torch.float32),
            }

    model = FakeVGGT()
    orientation_calls: list[tuple[str, str]] = []

    def load_vggt(_model_class: type[VGGT], repo_id: str, *, local_files_only: bool) -> FakeVGGT:
        assert repo_id == "facebook/VGGT-1B"
        assert local_files_only is True
        return model

    def record_orientation(
        poses: np.ndarray,
        *,
        method: str,
        center_method: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        orientation_calls.append((method, center_method))
        return poses[:, :3, :], np.eye(4)

    monkeypatch.setattr(VGGT, "from_pretrained", classmethod(load_vggt))
    monkeypatch.setattr(
        "monopriors.models.multiview.multiview_predictor.auto_orient_and_center_poses",
        record_orientation,
    )
    predictor = MultiviewPredictor(
        MultiviewPredictorConfig(model_name="vggt", device="cpu", preprocessing_mode="pad", local_files_only=True)
    )

    predictions = predictor([_rgb_image()], center_method="focus")

    assert len(predictions) == 1
    assert orientation_calls == [("up", "focus")]


def test_geometry_has_one_config_source_and_forwards_center_method() -> None:
    class FakePredictor(MultiviewPredictor):
        def __init__(self) -> None:
            self.config = MultiviewPredictorConfig(device="cpu")
            self.center_method = ""

        def __call__(self, rgb_list: list[np.ndarray], *, center_method: str = "none") -> list[MultiviewPred]:
            self.center_method = center_method
            return [_prediction()]

    predictor = FakePredictor()
    config = MultiviewGeometryConfig(keep_top_percent=100.0, center_method="focus")

    result = run_multiview_geometry(
        rgb_list=[np.zeros((2, 2, 3), dtype=np.uint8)],
        multiview_predictor=predictor,
        config=config,
    )

    assert predictor.center_method == "focus"
    assert len(result.mv_pred_list) == 1
    assert not hasattr(config, "model_name")
    assert not hasattr(config, "device")


def test_predictor_cache_reuses_exact_config_and_closes_before_replacement() -> None:
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
        preprocessing_mode="crop",
    )
    assert request.geometry_config == MultiviewGeometryConfig(keep_top_percent=42.0, verbose=False)


def test_calibrator_rejects_predictor_from_a_different_config() -> None:
    class FakePredictor(MultiviewPredictor):
        def __init__(self, config: MultiviewPredictorConfig) -> None:
            self.config = config

    vggt_config = MultiviewPredictorConfig(model_name="vggt", device="cpu")
    g3t_config = MultiviewPredictorConfig(model_name="g3t", device="cpu")

    with pytest.raises(ValueError, match="does not match"):
        MultiViewCalibrator(
            parent_log_path=Path("world"),
            config=MultiViewCalibratorConfig(
                predictor_config=g3t_config,
                refine_depth_maps=False,
                segment_people=False,
            ),
            multiview_predictor=FakePredictor(vggt_config),
        )
