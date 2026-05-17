from pathlib import Path

import numpy as np
from jaxtyping import Float32, UInt8
from numpy import ndarray
from PIL import Image

from sapiens2_pose.api.image_pose import ImagePoseConfig, run_image_pose
from sapiens2_pose.api.pose_artifact import PosePredictionArtifact, load_pose_prediction_artifact


def test_run_image_pose_writes_rrd_and_numeric_artifact_with_mocked_inference(tmp_path: Path) -> None:
    image_path: Path = tmp_path / "input.png"
    rrd_path: Path = tmp_path / "baseline.rrd"
    artifact_path: Path = tmp_path / "baseline.npz"
    image_rgb: UInt8[ndarray, "h w 3"] = np.zeros((24, 32, 3), dtype=np.uint8)
    Image.fromarray(image_rgb).save(image_path)

    def fake_detect_persons(
        image: UInt8[ndarray, "h w 3"],
        **_kwargs: object,
    ) -> Float32[ndarray, "n 4"]:
        assert image.shape == (24, 32, 3)
        return np.asarray([[1.0, 2.0, 20.0, 22.0]], dtype=np.float32)

    def fake_estimate_pose(
        image: UInt8[ndarray, "h w 3"],
        bboxes: Float32[ndarray, "n 4"],
        **_kwargs: object,
    ) -> PosePredictionArtifact:
        assert image.shape == (24, 32, 3)
        assert bboxes.shape == (1, 4)
        keypoints: Float32[ndarray, "n k 2"] = np.asarray([[[5.0, 6.0], [10.0, 12.0]]], dtype=np.float32)
        scores: Float32[ndarray, "n k"] = np.asarray([[0.9, 0.8]], dtype=np.float32)
        return PosePredictionArtifact(bboxes=bboxes, keypoints=keypoints, scores=scores)

    summary = run_image_pose(
        ImagePoseConfig(image_path=image_path, rrd_path=rrd_path, artifact_path=artifact_path),
        detect_persons_fn=fake_detect_persons,
        estimate_pose_fn=fake_estimate_pose,
    )

    artifact: PosePredictionArtifact = load_pose_prediction_artifact(artifact_path)
    assert summary.person_count == 1
    assert summary.model_size == "0.4B"
    assert rrd_path.exists()
    assert rrd_path.stat().st_size > 0
    assert np.array_equal(artifact.bboxes, np.asarray([[1.0, 2.0, 20.0, 22.0]], dtype=np.float32))
    assert np.array_equal(artifact.keypoints, np.asarray([[[5.0, 6.0], [10.0, 12.0]]], dtype=np.float32))
    assert np.array_equal(artifact.scores, np.asarray([[0.9, 0.8]], dtype=np.float32))
