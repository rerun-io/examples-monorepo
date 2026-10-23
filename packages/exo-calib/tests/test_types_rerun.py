"""``from_layer`` reads back exactly what ``to_rerun`` wrote (integration: ``rerun_roundtrip`` starts an in-process catalog)."""

import numpy as np
import pytest
import rerun as rr
from conftest import FakeScene
from dagkit.sources import Column, RecordingSource
from dagkit.testing import rerun_roundtrip
from dagkit.time import STATIC
from dagkit.types import log_value
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from test_nodes import kp2d_records

from exo_calib.types import BaseCalibration, BodyPose, BodyShape, MultiViewKeypoints, RefinementDiagnostics

pytestmark = pytest.mark.integration


def test_rig_roundtrip_is_static_and_exact() -> None:
    rig = FakeScene().rig.select(["rig_02/cam_00", "rig_00/cam_00", "rig_03/cam_00", "rig_01/cam_00"])  # not sorted
    ((t_ns, back),) = rerun_roundtrip(rig, "/world/exocalib/init", "video_time", STATIC)
    assert t_ns == STATIC and back.names == rig.names
    np.testing.assert_allclose(back.cam_T_world, rig.cam_T_world, atol=1e-6)  # Rerun stores float32 matrices
    np.testing.assert_allclose(back.intrinsics, rig.intrinsics, atol=1e-4)
    np.testing.assert_array_equal(back.resolution_wh, rig.resolution_wh)


def test_kp2d_record_roundtrip_keeps_every_field() -> None:
    t_ns, record = kp2d_records(FakeScene())[3]
    ((back_t, back),) = rerun_roundtrip(record, "/world/exocalib/kp2d", "video_time", t_ns)
    assert isinstance(back, MultiViewKeypoints) and back_t == t_ns
    assert (back.frame_index, back.views, back.box_source, back.crop_input_wh, back.video_wh) == (
        record.frame_index, record.views, record.box_source, record.crop_input_wh, record.video_wh,
    )  # fmt: skip
    for name in ("cam_t_ns", "kp_xy", "conf", "bbox_xyxy", "crop_origin_xy", "crop_size_wh"):
        np.testing.assert_array_equal(getattr(back, name), getattr(record, name))
        assert getattr(back, name).dtype == getattr(record, name).dtype


def test_diagnostics_and_base_calibration_roundtrip() -> None:
    diag = RefinementDiagnostics(1.25, [0.5, 0.25], [10, 12, 9], 1.0625, [1.0, 0.5, 2.0], depth_scale=1.125, body_scale=None)
    ((t_ns, back),) = rerun_roundtrip(diag, "/world/exocalib/refined", "video_time", STATIC)
    assert t_ns == STATIC and back == diag  # an unmeasured scale stays None (null in eval.json, no column in the layer)
    older = RefinementDiagnostics(1.25, [0.5], [10, 12, 9], 1.0, [1.0, 0.5, 2.0])  # writes no scale column, as layers before them
    ((_, back_older),) = rerun_roundtrip(older, "/world/exocalib/refined", "video_time", STATIC)
    assert back_older == older and back_older.depth_scale is None and back_older.body_scale is None
    base = BaseCalibration(FakeScene().rig.select(["rig_03/cam_00", "rig_01/cam_00"]))
    ((_, back_base),) = rerun_roundtrip(base, "/world", "video_time", STATIC)
    assert isinstance(back_base, BaseCalibration) and back_base.rig.names == base.rig.names
    np.testing.assert_allclose(back_base.rig.cam_T_world, base.rig.cam_T_world, atol=1e-6)


finite32 = st.floats(-10.0, 10.0, width=32, allow_nan=False)


@settings(max_examples=5, deadline=None)
@given(
    betas=arrays(np.float32, 16, elements=finite32),
    faces=arrays(np.int64, st.tuples(st.integers(1, 6), st.just(3)), elements=st.integers(0, 9)),
    pose=arrays(np.float32, 165, elements=finite32),
    trans=arrays(np.float32, 3, elements=finite32),
    vertices=arrays(np.float32, st.tuples(st.just(10), st.just(3)), elements=finite32),
)
def test_body_shape_and_pose_roundtrip_exactly(betas, faces, pose, trans, vertices) -> None:
    shape = BodyShape(betas, faces)
    ((t_ns, back_shape),) = rerun_roundtrip(shape, "/world/exocalib/body", "video_time", STATIC)
    assert t_ns == STATIC
    np.testing.assert_array_equal(back_shape.betas, betas)
    np.testing.assert_array_equal(back_shape.faces, faces)
    body = BodyPose(pose, trans, vertices)
    ((t_ns, back),) = rerun_roundtrip(body, "/world/exocalib/body", "video_time", 1_000)
    assert t_ns == 1_000
    for name in ("pose", "trans", "vertices"):
        np.testing.assert_array_equal(getattr(back, name), getattr(body, name))
        assert getattr(back, name).dtype == np.float32


def test_a_cleared_body_reads_back_as_an_empty_one(tmp_path) -> None:
    """The node clears each frameset's mesh at the next video frame; reading the layer back gives that clear as an empty body."""
    body = BodyPose(np.ones(165, np.float32), np.zeros(3, np.float32), np.ones((10, 3), np.float32))
    recording = rr.RecordingStream("exo-calib-test", recording_id="clear", send_properties=False)
    recording.save(tmp_path / "body.rrd")
    for t_ns, value in ((1_000, body), (2_000, BodyPose.empty())):
        log_value(recording, "video_time", t_ns, value, "/world/exocalib/body")
    recording.flush()
    recording.disconnect()

    back = [(e.t_ns, e.value) for e in RecordingSource(tmp_path / "body.rrd", {"v": Column(BodyPose, "/world/exocalib/body", False)}, "video_time").events()]

    assert [t for t, _ in back] == [1_000, 2_000] and len(back[0][1].vertices) == 10 and len(back[1][1].vertices) == 0  # type: ignore[attr-defined]
