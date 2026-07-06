import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rerun.experimental as rrx
from jaxtyping import Float32, UInt8
from numpy import ndarray
from sapiens2_pose.api.pose_artifact import PoseArtifactComparisonConfig

from sapiens_coco133_pose.api.artifact import (
    POINTS_AVERAGE_CONFIDENCE_COMPONENT,
    POINTS_CONFIDENCE_COMPONENT,
    Coco133PoseFrame,
    compare_video_pose_rrds,
    load_rerun_video_pose_artifact,
    write_video_pose_rrd,
)


def test_write_video_pose_rrd_round_trips_bbox_and_coco133_keypoints(tmp_path: Path) -> None:
    image_rgb: UInt8[ndarray, "h w 3"] = np.zeros((16, 20, 3), dtype=np.uint8)
    bboxes: Float32[ndarray, "n 4"] = np.asarray([[1.0, 2.0, 10.0, 14.0]], dtype=np.float32)
    keypoints: Float32[ndarray, "n 133 2"] = np.full((1, 133, 2), np.nan, dtype=np.float32)
    scores: Float32[ndarray, "n 133"] = np.zeros((1, 133), dtype=np.float32)
    keypoints[0, 0] = np.asarray([3.0, 4.0], dtype=np.float32)
    keypoints[0, 1] = np.asarray([5.0, 6.0], dtype=np.float32)
    keypoints[0, 2] = np.asarray([7.0, 8.0], dtype=np.float32)
    scores[0, 0] = 0.9
    scores[0, 1] = 0.8
    scores[0, 2] = 0.25
    frame: Coco133PoseFrame = Coco133PoseFrame(
        frame_index=0,
        timestamp_ns=123,
        image_rgb=image_rgb,
        bboxes=bboxes,
        keypoints=keypoints,
        scores=scores,
    )
    rrd_path: Path = tmp_path / "golden.rrd"

    write_video_pose_rrd([frame], rrd_path=rrd_path, keypoint_threshold=0.5)
    artifact = load_rerun_video_pose_artifact(rrd_path)

    assert artifact.frame_indices.tolist() == [0]
    assert artifact.person_frame_indices.tolist() == [0]
    np.testing.assert_allclose(artifact.bboxes, bboxes)
    assert artifact.keypoints.shape == (1, 133, 2)
    np.testing.assert_allclose(artifact.keypoints[0, :2], keypoints[0, :2])
    assert np.isnan(artifact.keypoints[0, 2, :]).all()
    assert np.isnan(artifact.keypoints[0, 3:, :]).all()
    np.testing.assert_allclose(artifact.scores[0, :3], scores[0, :3])

    reader = rrx.RrdReader(str(rrd_path))
    recordings: list[Any] = list(reader.recordings())
    keypoint_batches: list[dict[str, list[Any]]] = [
        chunk.to_record_batch().to_pydict()
        for chunk in reader.stream(store=recordings[0]).to_chunks()
        if str(chunk.entity_path) == "/video/person_0/keypoints"
    ]
    assert keypoint_batches
    assert POINTS_CONFIDENCE_COMPONENT in keypoint_batches[0]
    assert POINTS_AVERAGE_CONFIDENCE_COMPONENT in keypoint_batches[0]


def test_video_pose_rrd_uses_one_stable_video_entity_for_frame_overlays(tmp_path: Path) -> None:
    image_rgb: UInt8[ndarray, "h w 3"] = np.zeros((16, 20, 3), dtype=np.uint8)
    bboxes: Float32[ndarray, "n 4"] = np.asarray([[1.0, 2.0, 10.0, 14.0]], dtype=np.float32)
    keypoints: Float32[ndarray, "n 133 2"] = np.full((1, 133, 2), np.nan, dtype=np.float32)
    scores: Float32[ndarray, "n 133"] = np.zeros((1, 133), dtype=np.float32)
    keypoints[0, 0] = np.asarray([3.0, 4.0], dtype=np.float32)
    scores[0, 0] = 0.9
    frames: list[Coco133PoseFrame] = [
        Coco133PoseFrame(frame_index=0, timestamp_ns=0, image_rgb=image_rgb, bboxes=bboxes, keypoints=keypoints, scores=scores),
        Coco133PoseFrame(frame_index=1, timestamp_ns=33_000_000, image_rgb=image_rgb, bboxes=bboxes + 1.0, keypoints=keypoints + 1.0, scores=scores),
    ]
    rrd_path: Path = tmp_path / "stable_video.rrd"

    write_video_pose_rrd(frames, rrd_path=rrd_path, keypoint_threshold=0.5)

    reader = rrx.RrdReader(str(rrd_path))
    recordings: list[Any] = list(reader.recordings())
    entity_paths: set[str] = {str(chunk.entity_path) for chunk in reader.stream(store=recordings[0]).to_chunks()}
    assert "/video/image" in entity_paths
    assert "/video/person_0/bbox" in entity_paths
    assert "/video/person_0/keypoints" in entity_paths
    assert all("/video/frame_" not in entity_path for entity_path in entity_paths)

    artifact = load_rerun_video_pose_artifact(rrd_path)
    assert artifact.frame_indices.tolist() == [0, 1]
    assert artifact.person_frame_indices.tolist() == [0, 1]
    np.testing.assert_allclose(artifact.bboxes, np.stack([bboxes[0], bboxes[0] + 1.0], axis=0))


def test_rrd_candidate_matches_baseline_when_paths_are_provided() -> None:
    baseline_rrd = os.environ.get("SAPIENS_COCO133_BASELINE_RRD")
    candidate_rrd = os.environ.get("SAPIENS_COCO133_CANDIDATE_RRD")
    if baseline_rrd is None or candidate_rrd is None:
        pytest.skip("Set SAPIENS_COCO133_BASELINE_RRD and SAPIENS_COCO133_CANDIDATE_RRD to run RRD parity.")

    metrics = compare_video_pose_rrds(Path(baseline_rrd), Path(candidate_rrd), PoseArtifactComparisonConfig(0.0, keypoint_score_threshold=0.3, max_keypoint_bbox_fraction=0.0))

    assert metrics.max_absolute_difference == 0.0
    assert metrics.max_keypoint_bbox_fraction == 0.0
