import numpy as np
import pytest
from jaxtyping import Float32, Int, UInt8
from numpy import ndarray
from simplecv.camera_parameters import Extrinsics, Intrinsics, PinholeParameters
from wilor_nano.hand_keypoints import KeypointResults

from mv_api.api.full_exoego_pipeline import (
    CALIB_TARGET_RESOLUTION,
    _extract_wilor_uv,
    _mask_points_outside_intrinsics,
    _scale_pinhole_to_image_shape,
    _select_pose_camera_params,
    compute_square_bbox,
    frame_index_to_timestamp,
    resize_images_to_common_resolution,
    timestamp_to_frame_index,
)


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
        world_R_cam=np.eye(3, dtype=np.float32),
        world_t_cam=np.zeros(3, dtype=np.float32),
    )
    return PinholeParameters(name=name, intrinsics=intrinsics, extrinsics=extrinsics)


def test_timestamp_helpers_map_to_closest_frame_at_or_before_timestamp() -> None:
    timestamps_ns: Int[ndarray, "num_frames"] = np.array([100, 200, 400], dtype=np.int64)

    assert timestamp_to_frame_index(time_ns=50, frame_timestamps_ns=timestamps_ns) == 0
    assert timestamp_to_frame_index(time_ns=199, frame_timestamps_ns=timestamps_ns) == 0
    assert timestamp_to_frame_index(time_ns=200, frame_timestamps_ns=timestamps_ns) == 1
    assert timestamp_to_frame_index(time_ns=900, frame_timestamps_ns=timestamps_ns) == 2
    assert frame_index_to_timestamp(frame_timestamps_ns=timestamps_ns, frame_index=1) == 200

    with pytest.raises(IndexError):
        frame_index_to_timestamp(frame_timestamps_ns=timestamps_ns, frame_index=3)


def test_resize_images_to_common_resolution_keeps_uniform_shapes() -> None:
    image_a: UInt8[ndarray, "h w 3"] = np.full((4, 2, 3), 10, dtype=np.uint8)
    image_b: UInt8[ndarray, "h w 3"] = np.full((2, 4, 3), 20, dtype=np.uint8)

    resized: list[UInt8[ndarray, "h w 3"]] = resize_images_to_common_resolution(
        images=[image_a, image_b],
        target_size=(2, 2),
    )

    assert [image.shape for image in resized] == [(2, 2, 3), (2, 2, 3)]
    assert resized[0].dtype == np.uint8
    assert resized[1].dtype == np.uint8


def test_scale_pinhole_to_image_shape_maps_resized_calibration_back_to_video_pixels() -> None:
    intrinsics: Intrinsics = Intrinsics.from_focal_principal_point(
        camera_conventions="RDF",
        fl_x=800.0,
        fl_y=720.0,
        cx=640.0,
        cy=360.0,
        height=CALIB_TARGET_RESOLUTION[1],
        width=CALIB_TARGET_RESOLUTION[0],
    )
    extrinsics: Extrinsics = Extrinsics(
        world_R_cam=np.eye(3, dtype=np.float32),
        world_t_cam=np.zeros(3, dtype=np.float32),
    )
    resized_camera: PinholeParameters = PinholeParameters(
        name="exo_640x480",
        extrinsics=extrinsics,
        intrinsics=intrinsics,
    )

    video_camera: PinholeParameters = _scale_pinhole_to_image_shape(
        camera=resized_camera,
        image_shape=(480, 640),
        source_size=CALIB_TARGET_RESOLUTION,
    )

    assert video_camera.name == "exo_640x480"
    assert video_camera.extrinsics is resized_camera.extrinsics
    assert video_camera.intrinsics.width == 640
    assert video_camera.intrinsics.height == 480
    assert video_camera.intrinsics.fl_x == pytest.approx(400.0)
    assert video_camera.intrinsics.fl_y == pytest.approx(480.0)
    assert video_camera.intrinsics.cx == pytest.approx(320.0)
    assert video_camera.intrinsics.cy == pytest.approx(240.0)


def test_select_pose_camera_params_uses_dataset_in_auto_and_allows_estimated_override() -> None:
    estimated_exo: list[PinholeParameters] = [_fake_pinhole("estimated_exo")]
    estimated_ego: list[PinholeParameters] = [_fake_pinhole("estimated_ego")]
    dataset_exo: list[PinholeParameters] = [_fake_pinhole("dataset_exo")]
    dataset_ego: list[list[PinholeParameters]] = [[_fake_pinhole("dataset_ego")]]

    auto_selection = _select_pose_camera_params(
        camera_source="auto",
        estimated_exo_pinhole_param_list=estimated_exo,
        estimated_ego_pinhole_param_list=estimated_ego,
        dataset_exo_pinhole_param_list=dataset_exo,
        dataset_ego_pinhole_param_lists=dataset_ego,
    )

    assert auto_selection.source == "dataset"
    assert auto_selection.exo_pinhole_param_list is dataset_exo
    assert auto_selection.ego_pinhole_param_lists is dataset_ego
    assert not auto_selection.log_estimated_pinholes

    estimated_selection = _select_pose_camera_params(
        camera_source="estimated",
        estimated_exo_pinhole_param_list=estimated_exo,
        estimated_ego_pinhole_param_list=estimated_ego,
        dataset_exo_pinhole_param_list=dataset_exo,
        dataset_ego_pinhole_param_lists=dataset_ego,
    )

    assert estimated_selection.source == "estimated"
    assert estimated_selection.exo_pinhole_param_list is estimated_exo
    assert estimated_selection.ego_pinhole_param_lists == [[estimated_ego[0]]]
    assert estimated_selection.log_estimated_pinholes


def test_select_pose_camera_params_falls_back_to_estimated_and_requires_dataset_when_forced() -> None:
    estimated_exo: list[PinholeParameters] = [_fake_pinhole("estimated_exo")]
    estimated_ego: list[PinholeParameters] = [_fake_pinhole("estimated_ego")]

    auto_selection = _select_pose_camera_params(
        camera_source="auto",
        estimated_exo_pinhole_param_list=estimated_exo,
        estimated_ego_pinhole_param_list=estimated_ego,
        dataset_exo_pinhole_param_list=None,
        dataset_ego_pinhole_param_lists=None,
    )

    assert auto_selection.source == "estimated"
    assert auto_selection.exo_pinhole_param_list is estimated_exo
    assert auto_selection.ego_pinhole_param_lists == [[estimated_ego[0]]]
    assert auto_selection.log_estimated_pinholes

    with pytest.raises(ValueError, match="camera_source='dataset'"):
        _select_pose_camera_params(
            camera_source="dataset",
            estimated_exo_pinhole_param_list=estimated_exo,
            estimated_ego_pinhole_param_list=estimated_ego,
            dataset_exo_pinhole_param_list=None,
            dataset_ego_pinhole_param_lists=None,
        )


def test_mask_points_outside_intrinsics_removes_off_image_keypoints() -> None:
    intrinsics: Intrinsics = Intrinsics.from_focal_principal_point(
        camera_conventions="RDF",
        fl_x=50.0,
        fl_y=50.0,
        cx=50.0,
        cy=40.0,
        height=80,
        width=100,
    )
    uv: UInt8[ndarray, "5 2"] = np.array(
        [
            [0, 0],
            [99, 79],
            [100, 10],
            [10, 80],
            [20, 20],
        ],
        dtype=np.uint8,
    )
    confidences: UInt8[ndarray, "5"] = np.ones(5, dtype=np.uint8)

    masked_uv, masked_confidences = _mask_points_outside_intrinsics(
        uv=uv.astype(np.float32),
        confidences=confidences.astype(np.float32),
        intrinsics=intrinsics,
    )

    np.testing.assert_allclose(masked_uv[0], np.array([0.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(masked_uv[1], np.array([99.0, 79.0], dtype=np.float32))
    np.testing.assert_allclose(masked_uv[4], np.array([20.0, 20.0], dtype=np.float32))
    assert np.isnan(masked_uv[2]).all()
    assert np.isnan(masked_uv[3]).all()
    assert masked_confidences[0] == pytest.approx(1.0)
    assert masked_confidences[1] == pytest.approx(1.0)
    assert masked_confidences[4] == pytest.approx(1.0)
    assert np.isnan(masked_confidences[2])
    assert np.isnan(masked_confidences[3])


def test_compute_square_bbox_expands_and_clips_to_intrinsics() -> None:
    intrinsics: Intrinsics = Intrinsics.from_focal_principal_point(
        camera_conventions="RDF",
        fl_x=50.0,
        fl_y=50.0,
        cx=50.0,
        cy=40.0,
        height=80,
        width=100,
    )
    hand_uv: UInt8[ndarray, "21 2"] = np.zeros((21, 2), dtype=np.uint8)
    hand_uv[0] = np.array([10, 20], dtype=np.uint8)
    hand_uv[1] = np.array([30, 40], dtype=np.uint8)
    hand_uv[2:] = np.array([20, 30], dtype=np.uint8)

    bbox = compute_square_bbox(
        hand_uv=hand_uv.astype(np.float32),
        intrinsics=intrinsics,
        expansion_ratio=0.5,
    )

    assert bbox is not None
    np.testing.assert_allclose(bbox, np.array([5.0, 15.0, 35.0, 45.0], dtype=np.float32))


def test_compute_square_bbox_uses_finite_hand_keypoints() -> None:
    intrinsics: Intrinsics = Intrinsics.from_focal_principal_point(
        camera_conventions="RDF",
        fl_x=50.0,
        fl_y=50.0,
        cx=50.0,
        cy=40.0,
        height=80,
        width=100,
    )
    hand_uv: Float32[ndarray, "21 2"] = np.full((21, 2), np.nan, dtype=np.float32)
    hand_uv[0] = np.array([10.0, 20.0], dtype=np.float32)
    hand_uv[1] = np.array([30.0, 40.0], dtype=np.float32)
    hand_uv[2] = np.array([20.0, 30.0], dtype=np.float32)

    bbox: Float32[ndarray, "4"] | None = compute_square_bbox(
        hand_uv=hand_uv,
        intrinsics=intrinsics,
        expansion_ratio=0.5,
    )

    assert bbox is not None
    np.testing.assert_allclose(bbox, np.array([5.0, 15.0, 35.0, 45.0], dtype=np.float32))


def test_extract_wilor_uv_converts_keypoint_results_to_float32() -> None:
    keypoints_2d: ndarray = np.arange(42, dtype=np.float64).reshape(1, 21, 2)
    scores: ndarray = np.ones((1, 21), dtype=np.float64)
    wilor_pred: KeypointResults = KeypointResults(
        keypoints_2d=keypoints_2d,
        scores=scores,
        global_orient=None,
        hand_pose=None,
        betas=None,
    )

    pred_uv: Float32[ndarray, "1 21 2"] = _extract_wilor_uv(wilor_pred)

    assert pred_uv.dtype == np.float32
    np.testing.assert_allclose(pred_uv, keypoints_2d.astype(np.float32))
