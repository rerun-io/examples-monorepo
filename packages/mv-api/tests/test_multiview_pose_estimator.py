from typing import Any, cast

import numpy as np
import pytest
import torch
from jaxtyping import Float32, Float64, UInt8
from numpy import ndarray
from posekit.predictions import BoxDetections, Keypoints2d
from posekit.skeletons import COCO_133
from simplecv.camera_parameters import Extrinsics, Intrinsics, PinholeParameters
from torch import Tensor

from mv_api import multiview_pose_estimator as estimator_module
from mv_api.multiview_pose_estimator import MultiviewBodyTracker, MultiviewBodyTrackerConfig, MVHistory


def test_compute_hand_bbox_expands_square_and_clips_to_image() -> None:
    hand_uv: Float32[ndarray, "4 2"] = np.array(
        [
            [10.0, 10.0],
            [20.0, 30.0],
            [15.0, 25.0],
            [np.nan, np.nan],
        ],
        dtype=np.float32,
    )

    bbox: Float32[ndarray, "4"] | None = MultiviewBodyTracker._compute_hand_bbox(
        hand_uv=hand_uv,
        image_shape=(40, 25),
        expansion_ratio=0.0,
    )

    expected_bbox: Float32[ndarray, "4"] = np.array([5.0, 10.0, 24.0, 30.0], dtype=np.float32)
    assert bbox is not None
    np.testing.assert_allclose(bbox, expected_bbox)


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


class _FailingDetector:
    def __call__(self, frames_rgb: UInt8[Tensor, "b h w 3"]) -> BoxDetections:
        del frames_rgb
        raise AssertionError("tracking should provide the selected-view bbox without running detection")


class _RecordingPoseModel:
    def __init__(self) -> None:
        self.seen_pixel_values: list[int] = []
        self.skeleton = COCO_133

    def __call__(self, frames_rgb: UInt8[Tensor, "b h w 3"], detections: BoxDetections) -> Keypoints2d:
        assert detections.num_detections > 0
        self.seen_pixel_values.append(int(frames_rgb[0, 0, 0, 0]))
        num_instances: int = detections.num_detections
        return Keypoints2d(
            xy=torch.zeros((num_instances, 133, 2), dtype=torch.float32),
            scores=torch.ones((num_instances, 133), dtype=torch.float32),
            frame_indices=detections.frame_indices,
            skeleton=COCO_133,
        )


def test_multiview_tracker_filters_images_and_pinholes_for_selected_detection_cameras(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_batch_triangulate(
        keypoints_2d: Float32[ndarray, "n_views n_kpts 3"],
        projection_matrices: Float32[ndarray, "n_views 3 4"],
        min_views: int,
    ) -> Float64[ndarray, "n_kpts 4"]:
        assert keypoints_2d.shape[0] == 1
        assert projection_matrices.shape[0] == 1
        assert min_views == 2
        return np.zeros((133, 4), dtype=np.float64)

    monkeypatch.setattr(estimator_module, "batch_triangulate", fake_batch_triangulate)
    tracker: MultiviewBodyTracker = object.__new__(MultiviewBodyTracker)
    tracker.config = MultiviewBodyTrackerConfig(cams_for_detection_idx=[1], perform_tracking=True, device="cpu")
    tracker.num_keypoints = 133
    tracker.filter_body_idxes = np.arange(133, dtype=np.intp)
    tracker.det_model = cast(Any, _FailingDetector())
    pose_model: _RecordingPoseModel = _RecordingPoseModel()
    tracker.pose_model = cast(Any, pose_model)
    tracker.hand_keypoint_engine = None

    xyzc_t: Float32[ndarray, "133 4"] = np.zeros((133, 4), dtype=np.float32)
    xyzc_t[:, 2] = 2.0
    xyzc_t[:, 3] = 1.0
    pred_state: MVHistory = MVHistory(xyzc_t=xyzc_t, xyzc_t1=xyzc_t.copy())
    frames_rgb: UInt8[Tensor, "2 32 32 3"] = torch.stack(
        [
            torch.full((32, 32, 3), 10, dtype=torch.uint8),
            torch.full((32, 32, 3), 20, dtype=torch.uint8),
        ]
    )
    pinhole_list: list[PinholeParameters] = [_fake_pinhole("cam0"), _fake_pinhole("cam1")]

    output_state: MVHistory = tracker(
        frames_rgb=frames_rgb,
        pinhole_list=pinhole_list,
        pred_state=pred_state,
    )

    assert pose_model.seen_pixel_values == [20]
    assert output_state.uvc_t is not None
    assert output_state.uvc_t.shape == (1, 133, 3)
