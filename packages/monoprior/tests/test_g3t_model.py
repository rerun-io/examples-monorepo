import math
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import open3d as o3d
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from simplecv.camera_parameters import Extrinsics, Intrinsics, PinholeParameters
from torch import nn
from vggt.models.vggt import VGGT

import monopriors.apis.multiview_calibration as multiview_calibration
from monopriors.apis.multiview_calibration import (
    MVCalibResults,
    log_calibration_results,
)
from monopriors.models.multiview.multiview_model import (
    MultiviewModelPredictions,
    MultiviewPred,
    filter_confidences,
    generate_multiview_pred,
    preprocess_images,
    remove_padding_from_prediction,
)
from monopriors.models.multiview.multiview_pointcloud import (
    mv_pred_to_filtered_pointcloud,
    mv_pred_to_pointcloud,
)
from monopriors.models.multiview.multiview_predictor import (
    G3T_CHECKPOINT_REVISION,
    MultiviewPredictor,
    MultiviewPredictorConfig,
    decode_g3t_camera_heads,
)
from monopriors.third_party.g3t.layers.attention import Attention
from monopriors.third_party.g3t.layers.rope import PositionGetter, RotaryPositionEmbedding2D
from monopriors.third_party.g3t.models.aggregator import Aggregator
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


def test_dense_pointcloud_supports_different_view_shapes() -> None:
    first = _prediction()
    second = _prediction()
    second.cam_name = "camera_1"
    second.rgb_image = np.zeros((2, 3, 3), dtype=np.uint8)
    second.depth_map = np.arange(1, 7, dtype=np.float32).reshape(2, 3)
    second.confidence_mask = np.ones((2, 3), dtype=np.float32)
    second.pinhole_param.intrinsics.width = 3

    expected = np.concatenate([mv_pred_to_pointcloud([first]), mv_pred_to_pointcloud([second])])
    actual = mv_pred_to_pointcloud([first, second])

    assert actual.shape == (10, 3)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_filtered_pointcloud_unprojects_only_budgeted_confident_pixels() -> None:
    prediction = _prediction()
    prediction.rgb_image = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
    prediction.depth_map = np.arange(1, 17, dtype=np.float32).reshape(4, 4)
    prediction.confidence_mask = np.ones((4, 4), dtype=np.float32)
    prediction.pinhole_param.intrinsics.width = 4
    prediction.pinhole_param.intrinsics.height = 4
    confidence_masks = [np.full((4, 4), 255, dtype=np.uint8)]

    points, colors = mv_pred_to_filtered_pointcloud(
        [prediction],
        confidence_masks,
        target_points=4,
    )

    selected_pixel_ids = colors[:, 0] // 3
    dense_points = mv_pred_to_pointcloud([prediction])
    assert len(points) == 4
    assert len(np.unique(selected_pixel_ids)) == 4
    np.testing.assert_allclose(points, dense_points[selected_pixel_ids], rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_array_equal(colors, prediction.rgb_image.reshape(-1, 3)[selected_pixel_ids])


def test_filtered_pointcloud_applies_confidence_to_refined_depths() -> None:
    prediction = _prediction()
    prediction.rgb_image = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    confidence_mask = np.array([[255, 0], [0, 255]], dtype=np.uint8)
    refined_depth = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32)

    points, colors = mv_pred_to_filtered_pointcloud(
        [prediction],
        [confidence_mask],
        depth_list=[refined_depth],
        target_points=10,
    )

    dense_refined_points = mv_pred_to_pointcloud([prediction], depth_list=[refined_depth])
    np.testing.assert_allclose(points, dense_refined_points[[0, 3]], rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(colors, prediction.rgb_image.reshape(-1, 3)[[0, 3]])


def test_calibration_logging_does_not_filter_the_bounded_pointcloud_twice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    pointcloud.colors = o3d.utility.Vector3dVector(np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    result = MVCalibResults(depth_list=[], pinhole_param_list=[], pcd=pointcloud)

    monkeypatch.setattr(multiview_calibration.rr, "send_blueprint", lambda **_kwargs: None)
    monkeypatch.setattr(multiview_calibration.rr, "log", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(multiview_calibration.rr, "set_time", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        multiview_calibration,
        "estimate_voxel_size",
        lambda *_args, **_kwargs: pytest.fail("point cloud was filtered a second time"),
        raising=False,
    )

    output = log_calibration_results(
        rgb_list=[np.zeros((2, 2, 3), dtype=np.uint8)],
        output=result,
        parent_log_path=Path("world"),
        timeline="frame",
    )

    np.testing.assert_array_equal(np.asarray(output.pcd.points), np.asarray(pointcloud.points))
    np.testing.assert_array_equal(np.asarray(output.pcd.colors), np.asarray(pointcloud.colors))


@settings(max_examples=75, deadline=None)
@given(
    data=st.data(),
    height=st.integers(min_value=3, max_value=12),
    width=st.integers(min_value=3, max_value=12),
    focal_length=st.floats(min_value=1.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    principal_x=st.floats(min_value=-5.0, max_value=15.0, allow_nan=False, allow_infinity=False),
    principal_y=st.floats(min_value=-5.0, max_value=15.0, allow_nan=False, allow_infinity=False),
    rotation=st.floats(min_value=-math.pi, max_value=math.pi, allow_nan=False, allow_infinity=False),
    translation=st.tuples(
        st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    ),
)
def test_filtered_pointcloud_is_exact_confident_subset_of_dense_unprojection(
    data: st.DataObject,
    height: int,
    width: int,
    focal_length: float,
    principal_x: float,
    principal_y: float,
    rotation: float,
    translation: tuple[float, float, float],
) -> None:
    pixel_count = height * width
    depth_map = np.asarray(
        data.draw(
            st.lists(
                st.floats(min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False),
                min_size=pixel_count,
                max_size=pixel_count,
            ),
            label="depth_map",
        ),
        dtype=np.float32,
    ).reshape(height, width)
    confident_pixel_ids = data.draw(
        st.sets(st.integers(min_value=0, max_value=pixel_count - 1), min_size=4, max_size=pixel_count),
        label="confident_pixel_ids",
    )
    target_points = data.draw(
        st.integers(min_value=1, max_value=max(1, len(confident_pixel_ids) // 4)),
        label="target_points",
    )
    confidence_mask = np.zeros(pixel_count, dtype=np.uint8)
    confidence_mask[list(confident_pixel_ids)] = 255
    confidence_mask = confidence_mask.reshape(height, width)

    pixel_ids = np.arange(pixel_count, dtype=np.uint8)
    rgb_image = np.stack((pixel_ids, 255 - pixel_ids, np.full(pixel_count, 173, dtype=np.uint8)), axis=1).reshape(height, width, 3)
    cos_rotation = math.cos(rotation)
    sin_rotation = math.sin(rotation)
    cam_R_world = np.asarray(
        [[cos_rotation, -sin_rotation, 0.0], [sin_rotation, cos_rotation, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    prediction = MultiviewPred(
        cam_name="camera_0",
        rgb_image=rgb_image,
        depth_map=depth_map,
        confidence_mask=confidence_mask.astype(np.float32),
        pinhole_param=PinholeParameters(
            name="camera_0",
            intrinsics=Intrinsics(
                camera_conventions="RDF",
                fl_x=focal_length,
                fl_y=focal_length,
                cx=principal_x,
                cy=principal_y,
                width=width,
                height=height,
            ),
            extrinsics=Extrinsics(
                cam_R_world=cam_R_world,
                cam_t_world=np.asarray(translation, dtype=np.float32),
            ),
        ),
    )

    points, colors = mv_pred_to_filtered_pointcloud(
        [prediction],
        [confidence_mask],
        target_points=target_points,
    )

    selected_pixel_ids = colors[:, 0].astype(np.int64)
    dense_points = mv_pred_to_pointcloud([prediction])
    assert len(points) == min(target_points, np.count_nonzero(confidence_mask))
    assert len(np.unique(selected_pixel_ids)) == len(selected_pixel_ids)
    assert np.all(confidence_mask.reshape(-1)[selected_pixel_ids] != 0)
    np.testing.assert_array_equal(colors, rgb_image.reshape(-1, 3)[selected_pixel_ids])
    np.testing.assert_allclose(points, dense_points[selected_pixel_ids], rtol=1.0e-5, atol=1.0e-5)


@settings(max_examples=75, deadline=None)
@given(
    data=st.data(),
    height=st.integers(min_value=1, max_value=20),
    width=st.integers(min_value=1, max_value=20),
    keep_top_percent=st.integers(min_value=1, max_value=100),
)
def test_confidence_filter_matches_linear_percentile_mask(
    data: st.DataObject,
    height: int,
    width: int,
    keep_top_percent: int,
) -> None:
    values = data.draw(
        st.lists(
            st.integers(min_value=0, max_value=20),
            min_size=height * width,
            max_size=height * width,
        ),
        label="confidence_values",
    )
    confidence = np.asarray(values, dtype=np.float32).reshape(height, width)
    threshold = np.percentile(confidence, 100.0 - keep_top_percent)
    expected = (((confidence >= threshold) & (confidence > 1.0e-5)) * 255).astype(np.uint8)

    actual = filter_confidences(confidence, keep_top_percent)

    np.testing.assert_array_equal(actual, expected)


def _rgb_image(height: int = 28, width: int = 42) -> np.ndarray:
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[..., 0] = np.arange(width, dtype=np.uint8)[None, :]
    return rgb


class FakeG3T(G3T):
    def __init__(self) -> None:
        nn.Module.__init__(self)

    def to(self, *args: object, **kwargs: object) -> "FakeG3T":
        assert not hasattr(self, "point_head")
        return super().to(*args, **kwargs)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        assert not hasattr(self, "point_head")
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
        MultiviewPredictorConfig(model_name="g3t", device="cpu", local_files_only=True)
    )

    predictions = predictor([_rgb_image()], preprocessing_mode="pad", center_method="none")

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


def test_multiview_materialization_supports_mixed_aspect_ratios() -> None:
    images = [_rgb_image(height=28, width=42), _rgb_image(height=42, width=28)]
    preprocessed = preprocess_images(images, mode="pad")
    predictions = MultiviewModelPredictions(
        depth=np.ones((1, 2, 518, 518, 1), dtype=np.float32),
        depth_conf=np.ones((1, 2, 518, 518), dtype=np.float32),
        intrinsic=np.repeat(
            np.array([[[[259.0, 0.0, 259.0], [0.0, 259.0, 259.0], [0.0, 0.0, 1.0]]]], dtype=np.float32),
            2,
            axis=1,
        ),
        cam_T_world_b34=np.repeat(
            np.array(
                [[[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]]],
                dtype=np.float32,
            ),
            2,
            axis=1,
        ),
    )

    materialized = generate_multiview_pred(
        predictions,
        img_tensors=preprocessed.images,
        rgb_list=images,
        metadata_list=preprocessed.metadata,
    )

    assert [prediction.depth_map.shape for prediction in materialized] == [(28, 42), (42, 28)]


@pytest.mark.parametrize("pixel_value", [0, 127, 255])
def test_common_rgb_materialization_preserves_constant_images(pixel_value: int) -> None:
    rgb = np.full((42, 42, 3), pixel_value, dtype=np.uint8)
    preprocessed = preprocess_images([rgb], mode="pad")
    resized_height, resized_width = preprocessed.images.shape[-2:]
    predictions = MultiviewModelPredictions(
        depth=np.ones((1, 1, resized_height, resized_width, 1), dtype=np.float32),
        depth_conf=np.ones((1, 1, resized_height, resized_width), dtype=np.float32),
        intrinsic=np.array([[[[259.0, 0.0, 259.0], [0.0, 259.0, 259.0], [0.0, 0.0, 1.0]]]], dtype=np.float32),
        cam_T_world_b34=np.array(
            [[[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]]],
            dtype=np.float32,
        ),
    )

    materialized = generate_multiview_pred(
        predictions,
        img_tensors=preprocessed.images,
        rgb_list=[rgb],
        metadata_list=preprocessed.metadata,
    )

    np.testing.assert_array_equal(materialized[0].rgb_image, rgb)


def test_common_rgb_materialization_stays_within_five_percent_of_float_resize() -> None:
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

    candidate = generate_multiview_pred(
        model_predictions(),
        img_tensors=preprocessed.images,
        rgb_list=[rgb],
        metadata_list=preprocessed.metadata,
    )[0]

    processed_rgb = preprocessed.images[0].permute(1, 2, 0).numpy(force=True)
    processed_rgb = remove_padding_from_prediction(processed_rgb, preprocessed.metadata[0])
    resized_rgb = cv2.resize(processed_rgb, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    normalized_rgb = (resized_rgb - resized_rgb.min()) / (resized_rgb.max() - resized_rgb.min())
    reference_rgb = (normalized_rgb * 255).clip(0, 255).astype(np.uint8)

    assert float(np.mean(np.abs(reference_rgb.astype(np.int16) - candidate.rgb_image.astype(np.int16)) <= 13)) >= 0.95


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


def test_g3t_uses_stable_eager_inference_by_default() -> None:
    assert MultiviewPredictorConfig(model_name="g3t").g3t_compile is False
    assert MultiviewPredictorConfig(model_name="g3t", g3t_compile=True).g3t_compile is True


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
    Aggregator.warm_shape_caches(
        aggregator,
        num_cams=2,
        height=28,
        width=42,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

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
        MultiviewPredictorConfig(model_name="vggt", device="cpu", local_files_only=True)
    )

    predictions = predictor([_rgb_image()], preprocessing_mode="pad", center_method="focus")

    assert len(predictions) == 1
    assert orientation_calls == [("up", "focus")]


def test_g3t_checkpoint_loading_only_allows_removed_point_head_weights() -> None:
    model: G3T = G3T.__new__(G3T)
    nn.Module.__init__(model)
    model.register_parameter("kept", nn.Parameter(torch.zeros(1)))

    incompatible = model.load_state_dict(
        {"kept": torch.ones(1), "point_head.obsolete": torch.zeros(1)},
        strict=False,
    )

    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == ["point_head.obsolete"]
    with pytest.raises(RuntimeError, match="missing=.*kept"):
        model.load_state_dict({"point_head.obsolete": torch.zeros(1)}, strict=False)
    with pytest.raises(RuntimeError, match="unexpected=.*other"):
        model.load_state_dict({"kept": torch.ones(1), "other": torch.zeros(1)}, strict=False)
