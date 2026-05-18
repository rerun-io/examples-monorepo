import numpy as np
import pytest
from jaxtyping import Float32
from numpy import ndarray

from sapiens2_pose.api.pose_artifact import (
    PoseArtifactComparisonConfig,
    PosePredictionArtifact,
    compare_pose_prediction_artifacts,
    load_pose_prediction_artifact,
    save_pose_prediction_artifact,
)


def test_pose_prediction_artifacts_compare_with_relative_tolerance() -> None:
    baseline: PosePredictionArtifact = PosePredictionArtifact(
        bboxes=np.asarray([[1.0, 2.0, 20.0, 24.0]], dtype=np.float32),
        keypoints=np.asarray([[[10.0, 20.0], [40.0, 60.0]]], dtype=np.float32),
        scores=np.asarray([[0.9, 0.8]], dtype=np.float32),
    )
    candidate: PosePredictionArtifact = PosePredictionArtifact(
        bboxes=np.asarray([[1.01, 2.01, 20.01, 24.01]], dtype=np.float32),
        keypoints=np.asarray([[[10.2, 19.8], [39.5, 60.5]]], dtype=np.float32),
        scores=np.asarray([[0.88, 0.81]], dtype=np.float32),
    )

    metrics = compare_pose_prediction_artifacts(
        baseline,
        candidate,
        PoseArtifactComparisonConfig(max_relative_difference=0.05),
    )

    assert metrics.max_relative_difference <= 0.05
    assert metrics.compared_values == 10


def test_pose_prediction_artifacts_fail_when_relative_tolerance_is_exceeded() -> None:
    baseline_keypoints: Float32[ndarray, "1 1 2"] = np.asarray([[[10.0, 20.0]]], dtype=np.float32)
    candidate_keypoints: Float32[ndarray, "1 1 2"] = np.asarray([[[11.0, 20.0]]], dtype=np.float32)
    baseline: PosePredictionArtifact = PosePredictionArtifact(
        bboxes=np.asarray([[1.0, 2.0, 20.0, 24.0]], dtype=np.float32),
        keypoints=baseline_keypoints,
        scores=np.asarray([[0.9]], dtype=np.float32),
    )
    candidate: PosePredictionArtifact = PosePredictionArtifact(
        bboxes=np.asarray([[1.0, 2.0, 20.0, 24.0]], dtype=np.float32),
        keypoints=candidate_keypoints,
        scores=np.asarray([[0.9]], dtype=np.float32),
    )

    with pytest.raises(AssertionError, match="max_relative_difference"):
        compare_pose_prediction_artifacts(
            baseline,
            candidate,
            PoseArtifactComparisonConfig(max_relative_difference=0.05),
        )


def test_pose_prediction_artifacts_can_compare_visible_keypoints_by_bbox_fraction() -> None:
    baseline: PosePredictionArtifact = PosePredictionArtifact(
        bboxes=np.asarray([[0.0, 0.0, 100.0, 100.0]], dtype=np.float32),
        keypoints=np.asarray([[[50.0, 50.0], [0.0, 0.0]]], dtype=np.float32),
        scores=np.asarray([[0.9, 0.01]], dtype=np.float32),
    )
    candidate: PosePredictionArtifact = PosePredictionArtifact(
        bboxes=np.asarray([[0.0, 0.0, 100.0, 100.0]], dtype=np.float32),
        keypoints=np.asarray([[[53.0, 54.0], [500.0, 500.0]]], dtype=np.float32),
        scores=np.asarray([[0.91, 0.01]], dtype=np.float32),
    )

    metrics = compare_pose_prediction_artifacts(
        baseline,
        candidate,
        PoseArtifactComparisonConfig(
            max_relative_difference=0.05,
            keypoint_score_threshold=0.1,
            max_keypoint_bbox_fraction=0.05,
        ),
    )

    assert metrics.compared_keypoints == 1
    assert metrics.max_keypoint_bbox_fraction is not None
    assert metrics.max_keypoint_bbox_fraction <= 0.05


def test_pose_prediction_artifact_round_trips_npz(tmp_path) -> None:
    artifact_path = tmp_path / "pose_prediction.npz"
    artifact: PosePredictionArtifact = PosePredictionArtifact(
        bboxes=np.asarray([[1.0, 2.0, 20.0, 24.0]], dtype=np.float32),
        keypoints=np.asarray([[[10.0, 20.0], [40.0, 60.0]]], dtype=np.float32),
        scores=np.asarray([[0.9, 0.8]], dtype=np.float32),
    )

    save_pose_prediction_artifact(artifact, artifact_path)
    loaded: PosePredictionArtifact = load_pose_prediction_artifact(artifact_path)

    assert artifact_path.exists()
    assert np.array_equal(loaded.bboxes, artifact.bboxes)
    assert np.array_equal(loaded.keypoints, artifact.keypoints)
    assert np.array_equal(loaded.scores, artifact.scores)
