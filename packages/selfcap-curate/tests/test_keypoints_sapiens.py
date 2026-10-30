"""Sapiens model selection and isolated output layer contracts."""

import numpy as np
import rerun as rr
from rerun.chunk import RrdReader

from selfcap_curate.keypoints import Camera, Keypoints3D, SelectedView, write_keypoints
from selfcap_curate.keypoints_gpu import keypoint_models


def test_sapiens_selects_batch_four_and_same_detector(monkeypatch):
    from posekit.models.sapiens import SapiensPose2d, SapiensPoseConfig
    from posekit.models.yolox import YoloxDetector, YoloxDetectorConfig

    configs = []
    detector = object.__new__(YoloxDetector)
    pose = object.__new__(SapiensPose2d)

    def setup_detector(config):
        configs.append(config)
        return detector

    def setup_pose(config):
        configs.append(config)
        return pose

    monkeypatch.setattr(YoloxDetectorConfig, "setup", setup_detector)
    monkeypatch.setattr(SapiensPoseConfig, "setup", setup_pose)
    keypoint_models.cache_clear()
    assert keypoint_models("sapiens2") == (detector, pose)
    assert configs[0].score_thr == 0.3
    assert configs[0].backend.max_batch_size == 8
    assert configs[1].model_size == "0.4B"
    assert configs[1].backend.max_batch_size == configs[1].backend.opt_batch_size == 4
    keypoint_models.cache_clear()


def test_sapiens_layer_does_not_copy_calibration_or_shared_entities(tmp_path):
    base = tmp_path / "calib.rrd"
    with rr.RecordingStream("test", recording_id="selfcap__synthetic", send_properties=False) as recording:
        recording.save(base)
        recording.log("/world/rig_02/cam_00", rr.Transform3D(translation=[0, 0, 0]), static=True)
        recording.send_property("calib", rr.AnyValues(method="untouched"))
    original = base.read_bytes()
    camera = Camera(np.eye(3), np.eye(4), (0.0, 0.0), "/world/rig_02/cam_00")
    result = Keypoints3D(
        np.array([0.0, 0.1]), np.ones((2, 133, 3), np.float32), np.full((2, 133), 2, np.uint8),
        np.ones((2, 133), np.float32), {"phone": SelectedView(np.ones((2, 133, 2), np.float32), np.ones((2, 133)))},
        {"phone": camera}, 10.0,
    )
    output = tmp_path / "sapiens.rrd"
    write_keypoints(base, output, result, model="sapiens2")
    assert base.read_bytes() == original
    paths = {str(chunk.entity_path) for chunk in RrdReader(output).stream()}
    assert "/world/gt/sapiens2/coco133_xyz" in paths
    assert "/world/rig_02/cam_00/pinhole/sapiens2/coco133_uv" in paths
    assert "/world/rig_02/cam_00/pinhole/sapiens2/coco133_uv_projected" in paths
    assert "/__properties/keypoints_sapiens2" in paths
    assert all("sapiens2" in path for path in paths)
