"""Polycam camera JSON loads with or without ``timestamp``; exports from 2026-09 onward omit it."""

import json

from serde.json import from_json

from simplecv.data.polycam import PolycamCameraData

CAMERA_JSON: dict[str, float | int | bool] = {
    "blur_score": 0.5,
    "cx": 511.5,
    "cy": 383.5,
    "fx": 760.0,
    "fy": 760.0,
    "height": 768,
    "width": 1024,
    "manual_keyframe": False,
    "t_00": 1.0,
    "t_01": 0.0,
    "t_02": 0.0,
    "t_03": 0.1,
    "t_10": 0.0,
    "t_11": 1.0,
    "t_12": 0.0,
    "t_13": 0.2,
    "t_20": 0.0,
    "t_21": 0.0,
    "t_22": 1.0,
    "t_23": 0.3,
}


def test_camera_json_with_timestamp() -> None:
    camera: PolycamCameraData = from_json(PolycamCameraData, json.dumps({**CAMERA_JSON, "timestamp": 824048931094}))
    assert camera.timestamp == 824048931094
    assert camera.world_T_cam_44.shape == (4, 4)


def test_camera_json_without_timestamp() -> None:
    camera: PolycamCameraData = from_json(PolycamCameraData, json.dumps(CAMERA_JSON))
    assert camera.timestamp == 0
    assert camera.width == 1024
    assert camera.world_T_cam_44.shape == (4, 4)
