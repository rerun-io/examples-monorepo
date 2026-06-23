"""Device-free tests for RerunVideoLogger + create_blueprint.

The dev-env rerun (0.33) exposes no dataframe read API, so instead of reading an
``.rrd`` back we (a) assert the deterministic path dicts directly and (b) capture
``rr.log`` calls via monkeypatch to verify the exact entity layout ``log_static``
produces — including the load-bearing invariants the refactor introduced (the rig
node has NO transform; the green reference tint lands on ``cam_00`` only).
"""

from __future__ import annotations

import numpy as np
import pytest
import rerun as rr

from live_rerun.blueprint import create_blueprint
from live_rerun.calibration import OakCameraCalib, oak_calibration_to_rig
from live_rerun.rerun_video_logger import SCHEMA_VERSION, RerunVideoLogger


def _k() -> np.ndarray:
    return np.array([[800.0, 0.0, 640.0], [0.0, 800.0, 360.0], [0.0, 0.0, 1.0]], dtype=float)


def _rig():
    """A left(reference)/rgb/right rig, as the OAK source yields it (left = cam_00)."""
    rgb_ext = np.eye(4)
    rgb_ext[:3, 3] = [-3.7, 0.0, 0.0]
    right_ext = np.eye(4)
    right_ext[:3, 3] = [-7.5, 0.0, 0.0]
    return oak_calibration_to_rig(
        [
            OakCameraCalib("left", 1280, 720, _k(), [], "grayscale", None),
            OakCameraCalib("rgb", 1280, 720, _k(), [], "rgb", rgb_ext),
            OakCameraCalib("right", 1280, 720, _k(), [], "grayscale", right_ext),
        ]
    )


def _capture(monkeypatch) -> list[tuple[str, list[str]]]:
    """Patch ``rr.log`` to record (entity_path, [component type names]) per call."""
    calls: list[tuple[str, list[str]]] = []

    def fake_log(entity_path, *components, **_):
        calls.append((str(entity_path), [type(c).__name__ for c in components]))

    monkeypatch.setattr(rr, "log", fake_log)
    return calls


def test_video_and_pinhole_path_dicts() -> None:
    logger = RerunVideoLogger(_rig(), "h265")
    # video_paths: keyed by role name (frames route by label), value ends at /video
    assert logger.video_paths == {
        "left": "world/rig_00/cam_00/pinhole/video",
        "rgb": "world/rig_00/cam_01/pinhole/video",
        "right": "world/rig_00/cam_02/pinhole/video",
    }
    # pinhole_paths: keyed by canonical cam_NN (blueprint panels), value ends at /pinhole
    assert logger.pinhole_paths == {
        "cam_00": "world/rig_00/cam_00/pinhole",
        "cam_01": "world/rig_00/cam_01/pinhole",
        "cam_02": "world/rig_00/cam_02/pinhole",
    }


def test_log_static_entity_layout(monkeypatch) -> None:
    calls = _capture(monkeypatch)
    RerunVideoLogger(_rig(), "h265").log_static()

    pairs = [(entity, comp) for entity, comps in calls for comp in comps]
    entities = {entity for entity, _ in calls}

    assert {"world", "world/rig_00"} <= entities
    for cam in ("cam_00", "cam_01", "cam_02"):
        assert {f"world/rig_00/{cam}", f"world/rig_00/{cam}/pinhole", f"world/rig_00/{cam}/pinhole/video"} <= entities

    # The rig node carries metadata but deliberately NO Transform3D (implicit
    # identity, so a SLAM pass can drive world_T_rig temporally without a clash).
    rig_components = [c for e, c in pairs if e == "world/rig_00"]
    assert "AnyValues" in rig_components
    assert not any("Transform3D" in c for c in rig_components)

    # Each camera DOES carry a Transform3D (rig_T_cam) plus name/kind metadata.
    for cam in ("cam_00", "cam_01", "cam_02"):
        cam_components = [c for e, c in pairs if e == f"world/rig_00/{cam}"]
        assert any("Transform3D" in c for c in cam_components)
        assert "AnyValues" in cam_components

    # The green reference tint (a bare Pinhole partial-update, distinct from
    # log_pinhole's PinholeWithDistortion) lands on cam_00 (the reference) ONLY.
    assert [e for e, c in pairs if c == "Pinhole"] == ["world/rig_00/cam_00/pinhole"]

    # Every video entity gets a VideoStream (codec).
    for cam in ("cam_00", "cam_01", "cam_02"):
        assert any("VideoStream" in c for c in (comp for e, comp in pairs if e == f"world/rig_00/{cam}/pinhole/video"))


def test_log_sample_routes_by_name_and_rejects_unknown(monkeypatch) -> None:
    calls = _capture(monkeypatch)
    logger = RerunVideoLogger(_rig(), "h265")
    with rr.RecordingStream("test_live_rerun"):  # active recording so set_time is happy
        logger.log_sample("rgb", b"\x00\x00\x00\x01x", is_keyframe=True, device_time_s=0.1)
        assert calls[-1][0] == "world/rig_00/cam_01/pinhole/video"
        with pytest.raises(KeyError):
            logger.log_sample("bogus", b"", is_keyframe=False, device_time_s=0.0)


def test_schema_version_constant() -> None:
    assert SCHEMA_VERSION == "live-rerun-rig:v1"


def test_create_blueprint_builds_from_pinhole_paths() -> None:
    # Smoke: the wiring oak_live_rerun.py uses (logger.pinhole_paths -> panels) builds without error.
    logger = RerunVideoLogger(_rig(), "h265")
    assert create_blueprint(logger.pinhole_paths) is not None
