from dataclasses import fields
from inspect import signature
from pathlib import Path

import numpy as np
import pytest
import rerun as rr
from dataforge.triangulation import HEADSET_COVERED
from rerun.server import Server

from selfcap_soma import batch, catalog, evaluation_io
from selfcap_soma.catalog import RigPaths, exo_columns, exo_view_track


def test_sapiens_selection_uses_registered_confidence_and_ignores_legacy_predictions() -> None:
    names = [
        "/world/pred/sapiens2/coco133_xyz:Points3D:positions",
        "/world/gt/sapiens2/coco133_xyz:Points3D:positions",
        "/world/gt/sapiens2/coco133_xyz:simplecv.KeypointConfidence3D:confidences",
    ]
    position, confidence = exo_columns(names, "sapiens2")
    assert position == "/world/gt/sapiens2/coco133_xyz:Points3D:positions"
    assert confidence.endswith("KeypointConfidence3D:confidences")
    with pytest.raises(ValueError, match="unique"):
        exo_columns([], "sapiens2")


def test_rtmw_is_the_default_exo_source() -> None:
    """A run that names no source reads the layer the keypoints stage writes by default."""
    defaults = {
        "batch": next(f.default for f in fields(batch.Config) if f.name == "exo_source"),
        "read_window": signature(catalog.read_window).parameters["source"].default,
        "read_views": signature(evaluation_io.read_views).parameters["source"].default,
    }
    assert defaults == dict.fromkeys(defaults, "rtmw")


def test_view_track_prefers_the_keypoints_layer_over_align_detections() -> None:
    """A selfcap dataset holds both: align writes pinhole/pred/sapiens2, keypoints writes pinhole/sapiens2."""
    camera = "/world/rig_00/cam_00"
    tracks = [f"{camera}/pinhole/pred/sapiens2/coco133_uv", f"{camera}/pinhole/sapiens2/coco133_uv", f"{camera}/pinhole/pred/rtmw/coco133_uv"]
    names = [
        *(f"{track}:Points2D:positions" for track in tracks),
        *(f"{track}:simplecv.KeypointConfidence2D:confidences" for track in tracks),
        f"{camera}/pinhole/pred/sapiens2/coco133_uv_min70:Points2D:positions",
        f"{camera}/pinhole/sapiens2/coco133_uv_min70:Points2D:positions",
        f"{camera}/pinhole/sapiens2/coco133_uv_min70:simplecv.KeypointConfidence2D:confidences",
        f"{camera}/pinhole/pred/rtmw/coco133_uv_min70:Points2D:positions",
        f"{camera}/pinhole/pred/rtmw/coco133_uv_min70:simplecv.KeypointConfidence2D:confidences",
        f"{camera}/pinhole/sapiens2/coco133_uv_projected:Points2D:positions",
    ]
    assert exo_view_track(names, camera, "sapiens2") == f"{camera}/pinhole/sapiens2/coco133_uv"
    assert exo_view_track(names, camera, "rtmw") == f"{camera}/pinhole/pred/rtmw/coco133_uv"
    # A dataset that only went through align still has its detections.
    align_only = [name for name in names if not name.startswith(f"{camera}/pinhole/sapiens2/")]
    assert exo_view_track(align_only, camera, "sapiens2") == f"{camera}/pinhole/pred/sapiens2/coco133_uv"
    assert exo_view_track(names, "/world/rig_01/cam_00", "sapiens2") is None


def test_read_views_thresholds_the_uncensored_entity_itself(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The stored entity keeps 0.69 and the headset joints; a view holds only what fitting and scoring may use."""
    camera = "/world/rig_00/cam_00"
    xy = np.ones((1, 133, 2), dtype=np.float32)
    scores = np.full((1, 133), 0.9, dtype=np.float32)
    scores[0, [5, 6]] = [0.69, 0.71]
    path = tmp_path / "layer.rrd"
    with rr.RecordingStream("test", recording_id="episode", send_properties=False) as recording:
        recording.save(path)
        rr.log(camera, rr.Transform3D(translation=[0.0, 0.0, 0.0], mat3x3=np.eye(3)), static=True, recording=recording)
        rr.log(f"{camera}/pinhole", rr.Pinhole(image_from_camera=np.eye(3), resolution=[1920, 1080]), static=True, recording=recording)
        # The columns dataforge.hands.log_keypoints2d writes; this environment has no simplecv.
        rr.send_columns(
            f"{camera}/pinhole/pred/rtmw/coco133_uv",
            indexes=[rr.TimeColumn("video_time", duration=[1.0])],
            columns=[
                *rr.Points2D.columns(positions=xy.reshape(-1, 2)).partition([133]),
                *rr.DynamicArchetype.columns(archetype="simplecv.KeypointConfidence2D", components={"confidences": scores.reshape(-1)}).partition([133]),
            ],
            recording=recording,
        )
    with Server(port=0, datasets={"selfcap": [path]}) as server:
        view = server.client().get_dataset("selfcap").filter_segments("episode")
        context = catalog.SegmentCatalog(view, view.schema().column_names(), RigPaths(("/world/rig_00",), "/world/rig_04", "/world/rig_05"))
        phones, ego = evaluation_io.read_views(context, "episode", np.array([1.0]))
    assert ego is None
    assert np.isnan(phones[0].pixels[0, 5]).all() and phones[0].scores[0, 5] == 0.0
    np.testing.assert_array_equal(phones[0].pixels[0, 6], [1.0, 1.0])
    assert phones[0].scores[0, 6] == pytest.approx(0.71)
    assert np.isnan(phones[0].pixels[0, HEADSET_COVERED]).all() and (phones[0].scores[0, HEADSET_COVERED] == 0.0).all()
