"""Calibration source is only a named catalog rig value; borrowing never follows paths."""

import json
from pathlib import Path

import numpy as np
import pytest
import rerun as rr
from serde.json import to_json

from selfcap_curate.catalog_io import DATASET_NAME, open_segment, read_factories
from selfcap_curate.device_tables import read_camera50_table
from selfcap_curate.factory import intrinsic_matrix, read_quest_block

FAULTY = "selfcap__2025-12-09__021d2e8b__ep001"


def write_factory_record(path: Path, segment: str, *, faulty: bool, document: bool = True, device: str = "Quest - Mountain-63ADA0FC") -> None:
    intrinsics = []
    for layout, camera in (("left", "50"), ("right", "50" if faulty else "51")):
        intrinsics.append(
            {
                "camera_id": camera,
                "positional_layout": layout,
                "lens_intrinsics": {
                    "focal_length_x": 850.0,
                    "focal_length_y": 851.0,
                    "principal_point_x": 640.0,
                    "principal_point_y": 642.0,
                    "available": layout == "right" or not faulty,
                },
                "sensor_info": {"pixel_array_size": {"width": 1280, "height": 1280}},
                "capture_resolution": {"width": 1280, "height": 960},
                "distortion": {},
            }
        )
    ext = (
        []
        if faulty
        else [
            {"type": f"device_to_{side}", "to": camera, "R": np.eye(3).tolist(), "T": translation}
            for side, camera, translation in (("left", "50", [-0.03, 0.01, -0.07]), ("right", "51", [0.03, 0.01, -0.07]))
        ]
    )
    with rr.RecordingStream("factory-test", recording_id=segment) as recording:
        recording.save(path)
        for rig, kind, cameras in [(0, "exo", 1), (1, "exo", 1), (2, "exo", 1), (3, "ego", 3), (4, "quest", 2)]:
            recording.log(f"/world/rig_{rig:02d}", rr.AnyValues(kind=kind, name=f"phone-{rig}"), static=True)
            if document:
                raw = {
                    "device_id": device if kind == "quest" else f"unused-{rig}",
                    "intrinsics": intrinsics if kind == "quest" else [],
                    "extrinsics": ext if kind == "quest" else [],
                }
                recording.log(f"/world/rig_{rig:02d}", rr.AnyValues(raw_calibration_json=json.dumps(raw)), static=True)
            for camera in range(cameras):
                recording.log(
                    f"/world/rig_{rig:02d}/cam_{camera:02d}/pinhole/video", rr.VideoStream(codec=rr.VideoCodec.AV1, sample=[1]), static=True
                )


@pytest.mark.integration
def test_quest_automatically_resolves_the_healthy_catalog_segment(tmp_path: Path, private_server) -> None:
    healthy = read_camera50_table().sessions["2025-12-09__021d2e8b"].healthy_segment
    a, b = tmp_path / "faulty.rrd", tmp_path / "healthy.rrd"
    write_factory_record(a, FAULTY, faulty=True)
    write_factory_record(b, healthy, faulty=False)
    with private_server({DATASET_NAME: [a, b]}) as server:
        dataset = server.client().get_dataset(DATASET_NAME)
        result = read_quest_block(open_segment(dataset, FAULTY))
        assert result.mode == "faulty-left-borrowed-cam50"
        assert intrinsic_matrix(result.pinholes["1"])[1, 2] == 482.0
        np.testing.assert_allclose(result.rig_T_cam["1"][:3, :3], np.eye(3), atol=0.0)
        np.testing.assert_allclose(result.rig_T_cam["1"][:3, 3], [0.002, 0.01, -0.07], atol=1e-7)


@pytest.mark.integration
def test_missing_catalog_document_does_not_fall_back_to_raw_files(tmp_path: Path, private_server) -> None:
    path = tmp_path / "missing.rrd"
    write_factory_record(path, FAULTY, faulty=True, document=False)
    with private_server({DATASET_NAME: [path]}) as server, pytest.raises(ValueError, match="missing raw_calibration_json"):
        read_factories(open_segment(server.client().get_dataset(DATASET_NAME), FAULTY))


@pytest.mark.integration
def test_borrow_source_must_belong_to_the_same_quest_device(tmp_path: Path, private_server) -> None:
    healthy = read_camera50_table().sessions["2025-12-09__021d2e8b"].healthy_segment
    a, b = tmp_path / "faulty.rrd", tmp_path / "wrong.rrd"
    write_factory_record(a, FAULTY, faulty=True)
    write_factory_record(b, healthy, faulty=False, device="wrong-headset")
    with private_server({DATASET_NAME: [a, b]}) as server, pytest.raises(ValueError, match="healthy segment of"):
        read_quest_block(open_segment(server.client().get_dataset(DATASET_NAME), FAULTY))


def test_borrowing_table_is_complete_and_strict(tmp_path: Path) -> None:

    table = read_camera50_table()
    assert len(table.sessions) == 84
    assert all(":" not in item.healthy_segment and "/" not in item.healthy_segment for item in table.sessions.values())
    path = tmp_path / "unknown.json"
    data = json.loads(to_json(table))
    data["unknown"] = True
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=str(path)):
        read_camera50_table(path)
