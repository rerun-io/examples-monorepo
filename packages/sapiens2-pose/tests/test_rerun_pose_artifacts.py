from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rerun as rr
from jaxtyping import Float32
from numpy import ndarray

from sapiens2_pose.api.rerun_pose_artifact import (
    RerunPoseArtifactComparisonConfig,
    compare_rerun_pose_artifacts,
    load_rerun_pose_artifact,
)


def _write_pose_rrd(
    rrd_path: Path,
    *,
    bbox_xyxy: Float32[ndarray, "4"],
    keypoints: Float32[ndarray, "k 2"],
) -> None:
    """Write a minimal pose RRD with the same entity/component layout as the demos."""
    recording: rr.RecordingStream = rr.RecordingStream(application_id="test_pose_rrd")
    rrd_path.parent.mkdir(parents=True, exist_ok=True)
    with recording:
        rr.set_time("iteration", sequence=0)
        rr.log(
            "image/person_0/bbox",
            rr.Boxes2D(
                array=np.asarray(bbox_xyxy, dtype=np.float32).reshape(1, 4),
                array_format=rr.Box2DFormat.XYXY,
                class_ids=0,
                labels="person_0",
            ),
        )
        rr.log("image/person_0/keypoints", rr.Points2D(positions=np.asarray(keypoints, dtype=np.float32)))
    recording.save(str(rrd_path))


def test_load_rerun_pose_artifact_queries_boxes_and_keypoints(tmp_path: Path) -> None:
    rrd_path: Path = tmp_path / "pose.rrd"
    bbox: Float32[ndarray, "4"] = np.asarray([10.0, 20.0, 110.0, 220.0], dtype=np.float32)
    keypoints: Float32[ndarray, "2 2"] = np.asarray([[30.0, 40.0], [np.nan, np.nan]], dtype=np.float32)

    _write_pose_rrd(rrd_path, bbox_xyxy=bbox, keypoints=keypoints)
    artifact = load_rerun_pose_artifact(rrd_path)

    assert np.allclose(artifact.bboxes, bbox.reshape(1, 4))
    assert np.allclose(artifact.keypoints[0, 0], keypoints[0])
    assert np.isnan(artifact.keypoints[0, 1]).all()


def test_compare_rerun_pose_artifacts_uses_bbox_and_visible_keypoint_tolerances(tmp_path: Path) -> None:
    baseline_rrd_path: Path = tmp_path / "baseline.rrd"
    candidate_rrd_path: Path = tmp_path / "candidate.rrd"
    baseline_bbox: Float32[ndarray, "4"] = np.asarray([0.0, 0.0, 100.0, 100.0], dtype=np.float32)
    candidate_bbox: Float32[ndarray, "4"] = np.asarray([0.0, 0.0, 101.0, 100.0], dtype=np.float32)
    baseline_keypoints: Float32[ndarray, "2 2"] = np.asarray([[50.0, 50.0], [np.nan, np.nan]], dtype=np.float32)
    candidate_keypoints: Float32[ndarray, "2 2"] = np.asarray([[53.0, 54.0], [np.nan, np.nan]], dtype=np.float32)
    config: RerunPoseArtifactComparisonConfig = RerunPoseArtifactComparisonConfig(
        max_relative_difference=0.05,
        max_keypoint_bbox_fraction=0.05,
    )

    _write_pose_rrd(baseline_rrd_path, bbox_xyxy=baseline_bbox, keypoints=baseline_keypoints)
    _write_pose_rrd(candidate_rrd_path, bbox_xyxy=candidate_bbox, keypoints=candidate_keypoints)

    metrics = compare_rerun_pose_artifacts(
        load_rerun_pose_artifact(baseline_rrd_path),
        load_rerun_pose_artifact(candidate_rrd_path),
        config,
    )

    assert metrics.compared_bboxes == 4
    assert metrics.compared_keypoints == 1
    assert metrics.max_keypoint_bbox_fraction <= 0.05


def test_compare_rerun_pose_artifacts_fails_when_visibility_changes(tmp_path: Path) -> None:
    baseline_rrd_path: Path = tmp_path / "baseline.rrd"
    candidate_rrd_path: Path = tmp_path / "candidate.rrd"
    bbox: Float32[ndarray, "4"] = np.asarray([0.0, 0.0, 100.0, 100.0], dtype=np.float32)

    _write_pose_rrd(
        baseline_rrd_path,
        bbox_xyxy=bbox,
        keypoints=np.asarray([[50.0, 50.0], [np.nan, np.nan]], dtype=np.float32),
    )
    _write_pose_rrd(
        candidate_rrd_path,
        bbox_xyxy=bbox,
        keypoints=np.asarray([[50.0, 50.0], [60.0, 60.0]], dtype=np.float32),
    )

    with pytest.raises(AssertionError, match="visibility masks differ"):
        compare_rerun_pose_artifacts(
            load_rerun_pose_artifact(baseline_rrd_path),
            load_rerun_pose_artifact(candidate_rrd_path),
            RerunPoseArtifactComparisonConfig(),
        )
