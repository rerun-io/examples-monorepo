"""What the keypoints writer shows: colours on each confidence scale, link colour, and the method it records."""

from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
import rerun as rr
from jaxtyping import Float32, Float64, UInt8
from numpy import ndarray
from rerun.chunk import RrdReader

from selfcap_curate.keypoints import HEADSET_COVERED, Camera, EgoCamera, Keypoints3D, PoseModel, SelectedView, project, write_keypoints


def layer(tmp_path: Path, model: PoseModel = "rtmw") -> Path:
    """Write two frames: joints 3, 4, 6 carry the same scores in 3D and in 2D, joint 7 a 2D score just under the hide threshold."""
    calib: Path = tmp_path / "calib.rrd"
    with rr.RecordingStream("test", recording_id="selfcap__synthetic", send_properties=False) as recording:
        recording.save(calib)
        recording.send_property("calib", rr.AnyValues(method="unchanged"))
    K: Float64[ndarray, "3 3"] = np.array([[900.0, 0.0, 960.0], [0.0, 900.0, 540.0], [0.0, 0.0, 1.0]])
    camera: Camera = Camera(K, np.eye(4), (0.0, 0.0), "/world/rig_00/cam_00")
    times: Float64[ndarray, "2"] = np.array([0.0, 0.04])
    xyz: Float32[ndarray, "2 133 3"] = np.tile(np.linspace([-0.3, -0.3, 3.5], [0.3, 0.3, 4.5], 133), (2, 1, 1)).astype(np.float32)
    confidence: Float32[ndarray, "2 133"] = np.full((2, 133), 0.9, np.float32)
    confidence[:, [3, 4, 6]] = [0.2, 0.6, 1.0]
    scores: Float64[ndarray, "2 133"] = np.full((2, 133), 0.9)
    scores[:, [3, 4, 6, 7]] = [0.7, 0.85, 1.0, 0.699]
    uv: Float32[ndarray, "2 133 2"] = np.tile(project(camera, xyz[0].astype(np.float64))[:, :2], (2, 1, 1)).astype(np.float32)
    result: Keypoints3D = Keypoints3D(
        times, xyz, np.full((2, 133), 4, np.uint8), confidence, {"phone": SelectedView(uv, scores)}, {"phone": camera}, 25.0
    )
    ego: EgoCamera = EgoCamera(4, 0, K, np.zeros(14), (1920, 1080), np.eye(4), times, np.stack([np.eye(4)] * 2))
    output: Path = tmp_path / "layer.rrd"
    write_keypoints(calib, output, result, ego=(ego,), model=model)
    return output


def first_row(path: Path, entity: str, component: str) -> list:
    """First row of a component on one entity, static or temporal."""
    for chunk in RrdReader(path).stream():
        batch: pa.RecordBatch = chunk.to_record_batch()
        if str(chunk.entity_path) == entity and component in batch.schema.names:
            return batch.column(component).to_pylist()[0]
    raise AssertionError(f"{entity} has no {component}")


def colours(path: Path, entity: str, kind: str, joints: list[int]) -> UInt8[ndarray, "k 3"]:
    packed: list[int] = first_row(path, entity, f"{kind}:colors")
    return np.array([list(int(packed[joint]).to_bytes(4, "big")[:3]) for joint in joints], dtype=np.uint8)


@pytest.mark.parametrize(
    ("entity", "kind", "joints"),
    [
        ("/world/pred/rtmw_triangulated/coco133_xyz", "Points3D", [3, 4, 6]),
        ("/world/skeleton", "Points3D", [3, 4, 6]),
        ("/world/rig_00/cam_00/pinhole/coco133_uv_projected", "Points2D", [3, 4, 6]),
        ("/world/rig_04/cam_00/pinhole/coco133_uv_projected", "Points2D", [3, 4, 6]),
        ("/world/rig_04/cam_00/pinhole/skeleton", "Points2D", [3, 4, 6]),
    ],
)
def test_3d_confidence_runs_from_red_at_the_floor_to_green_at_one(tmp_path: Path, entity: str, kind: str, joints: list[int]) -> None:
    """Every entity coloured by the Kineo score: 0.2 red, 0.6 halfway (yellow), 1.0 green."""
    np.testing.assert_allclose(colours(layer(tmp_path), entity, kind, joints), [[255, 0, 0], [255, 255, 0], [0, 255, 0]], atol=1)


@pytest.mark.parametrize(("model", "parent"), [("rtmw", "pred/rtmw"), ("sapiens2", "sapiens2")])
def test_2d_detections_store_every_score_and_show_only_the_confident_ones(tmp_path: Path, model: PoseModel, parent: str) -> None:
    """Detector scores are another scale: 0.7 red, 0.85 yellow, 1.0 green. The data keeps 0.699 and the headset joints; the display copy does not."""
    path: Path = layer(tmp_path, model)
    entity: str = f"/world/rig_00/cam_00/pinhole/{parent}/coco133_uv"
    np.testing.assert_allclose(colours(path, entity, "Points2D", [3, 4, 6]), [[255, 0, 0], [255, 255, 0], [0, 255, 0]], atol=1)
    positions: Float64[ndarray, "133 2"] = np.array(first_row(path, entity, "Points2D:positions"), dtype=np.float64)
    assert np.isfinite(positions).all()
    scores: Float64[ndarray, "133"] = np.array(first_row(path, entity, "simplecv.KeypointConfidence2D:confidences"), dtype=np.float64)
    np.testing.assert_allclose(scores[[0, 7]], [0.9, 0.699], rtol=1e-6)
    ids: list[int] = first_row(path, f"{entity}_min70", "Points2D:keypoint_ids")
    assert ids == [joint for joint in range(133) if joint != 7 and joint not in HEADSET_COVERED]
    np.testing.assert_array_equal(np.array(first_row(path, f"{entity}_min70", "Points2D:positions"), dtype=np.float64), positions[ids])


@pytest.mark.parametrize("model", ["rtmw", "sapiens2"])
def test_the_properties_describe_the_current_method_and_the_masked_joints(tmp_path: Path, model: PoseModel) -> None:
    properties: str = "/__properties/calib" if model == "rtmw" else "/__properties/keypoints_sapiens2"
    path: Path = layer(tmp_path, model)
    method: str = first_row(path, properties, "kp3d_method")[0]
    assert method == (
        "confidence-weighted DLT: best view-pair seed, 6 rounds of robust reweighting over every view (Cauchy at 8 px, no weight beyond 24 px); "
        "confidence: Kineo pairwise reprojection score (arXiv 2510.24464)"
    )
    assert "RANSAC" not in method
    masked: str = first_row(path, properties, "kp3d_missing_under_headset")[0]
    assert "body 0-2 and face 23-90" in masked
    assert "both models" not in masked


@pytest.mark.parametrize(("model", "entity"), [("rtmw", "/world/rig_00/cam_00/pinhole/pred/rtmw"), ("sapiens2", "/world/rig_00/cam_00/pinhole/sapiens2")])
def test_the_2d_model_links_are_the_blue_of_the_display_skeleton(tmp_path: Path, model: PoseModel, entity: str) -> None:
    path: Path = layer(tmp_path, model)
    blue_rgba: int = 0x0000FFFF

    def class_colour(context_entity: str) -> int:
        return first_row(path, context_entity, "AnnotationContext:context")[0][0]["class_description"]["info"]["color"]

    assert class_colour(entity) == blue_rgba
    assert class_colour("/world/skeleton") == blue_rgba
