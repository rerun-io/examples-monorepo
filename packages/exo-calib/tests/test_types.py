"""exo-calib Types: serde round-trips and the Rerun mapping structure (unit lane: no catalog)."""

import numpy as np
import pytest
from dagkit.types import Type
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from serde import SerdeError, from_dict, to_dict

from exo_calib.apis.calibrate import INIT_LAYERS, LAYERS, PASS1_LAYERS, PORT_TYPES
from exo_calib.types import Keypoints3dTracks, Kp2dOverlay, MultiViewKeypoints, RefinementDiagnostics, RigAlignment, RigLayout

ROOT = "/world/exocalib/kp2d"


@st.composite
def records(draw: st.DrawFn) -> MultiViewKeypoints:
    v: int = draw(st.integers(1, 4))
    floats = st.floats(-1e4, 1e4, width=32) | st.just(float("nan"))
    return MultiViewKeypoints(
        frame_index=draw(st.integers(0, 10**6)),
        views=tuple(f"rig_{i:02d}/cam_00" for i in range(v)),
        cam_t_ns=draw(arrays(np.int64, (v,), elements=st.integers(0, 2**40))),
        kp_xy=draw(arrays(np.float32, (v, 133, 2), elements=floats)),
        conf=draw(arrays(np.float32, (v, 133), elements=st.floats(0.0, 1.0, width=32))),
        bbox_xyxy=draw(arrays(np.float32, (v, 4), elements=floats)),
        box_source=tuple(draw(st.sampled_from(["yolox", "tracked", "none"])) for _ in range(v)),
        crop_origin_xy=draw(arrays(np.float32, (v, 2), elements=floats)),
        crop_size_wh=draw(arrays(np.float32, (v, 2), elements=floats)),
        crop_input_wh=(192, 256),
        video_wh=(1280, 720),
    )


@settings(max_examples=40, deadline=None)
@given(record=records())
def test_keypoints_serde_roundtrip_keeps_dtype_and_shape(record: MultiViewKeypoints) -> None:
    back: MultiViewKeypoints = from_dict(MultiViewKeypoints, to_dict(record))
    for name in ("cam_t_ns", "kp_xy", "conf", "bbox_xyxy", "crop_origin_xy", "crop_size_wh"):
        np.testing.assert_array_equal(getattr(back, name), getattr(record, name))
        assert getattr(back, name).dtype == getattr(record, name).dtype
    assert (back.views, back.box_source, back.frame_index, back.crop_input_wh) == (record.views, record.box_source, record.frame_index, record.crop_input_wh)


def test_unknown_fields_are_refused() -> None:
    with pytest.raises(SerdeError):
        from_dict(RigLayout, {"exo_camera_names": [], "ego_camera_names": [], "calibrated_camera_names": [], "extra": 1})
    diag = RefinementDiagnostics(1.5, [1.0, 0.9], [10, 12], 1.02, [1.0, 0.99])
    assert from_dict(RefinementDiagnostics, to_dict(diag)) == diag


@settings(max_examples=40, deadline=None)
@given(record=records())
def test_keypoints_to_rerun_writes_one_record_per_camera_under_the_root(record: MultiViewKeypoints) -> None:
    logs = record.to_rerun(ROOT)
    assert [log.path for log in logs] == [f"{ROOT}/{view}" for view in record.views] and not any(log.static for log in logs)
    assert MultiViewKeypoints.contents(ROOT) == [f"{ROOT}/**"]


@settings(max_examples=40, deadline=None)
@given(xyz=arrays(np.float64, (2, 5, 3), elements=st.floats(-5.0, 5.0) | st.just(float("nan"))))
def test_kp3d_logs_each_rig_variant_that_has_a_finite_point(xyz: np.ndarray) -> None:
    tracks = Keypoints3dTracks(np.arange(5, dtype=np.int64), xyz, np.ones((2, 5)))
    paths: list[str] = [log.path for log in tracks.to_rerun("/world/exocalib/kp3d")]
    expected = [f"/world/exocalib/kp3d_{v}" for k, v in enumerate(("init", "refined")) if np.isfinite(xyz[k]).all(axis=1).any()]
    assert paths == expected


def test_overlay_draws_the_box_and_raw_joints_always_the_skeleton_only_where_it_has_keypoints_and_alignment_writes_both_rigs() -> None:
    raw = np.full((133, 2), np.nan, dtype=np.float32)
    box, crop = np.array([1.0, 2.0, 30.0, 40.0], np.float32), np.array([0.0, 0.0, 50.0, 60.0], np.float32)
    leaf = "/world/b/cam_00/pinhole/exocalib_kp2d"
    plain = Kp2dOverlay("b/cam_00", box, "tracked", crop, raw, np.zeros(133, np.float32))
    assert [log.path for log in plain.to_rerun(ROOT)] == [f"{leaf}_bbox", f"{leaf}_crop", f"{leaf}_raw"]  # every frame: no stale raw points
    raw[:3] = 10.0
    drawn = Kp2dOverlay("b/cam_00", box, "yolox", crop, raw, np.ones(133, np.float32), raw.astype(np.float64), np.ones(133))
    assert [log.path for log in drawn.to_rerun(ROOT)] == [f"{leaf}_bbox", f"{leaf}_crop", f"{leaf}_raw", leaf, f"{leaf}_rejected"]
    back = from_dict(Kp2dOverlay, to_dict(drawn))
    assert back.view == drawn.view and back.gated_xy is not None and back.gated_xy.dtype == np.float64 and from_dict(Kp2dOverlay, to_dict(plain)).gated_xy is None
    align = RigAlignment(("a",), np.zeros((1, 3)), np.repeat(np.eye(3)[None], 2, 0), np.zeros((2, 3)), np.zeros((2, 1, 3)))
    logs = align.to_rerun("/world")
    assert {log.path for log in logs} == {f"/world/{p}" for p in ("exocalib/init", "exocalib/kp3d_init", "exocalib_error/init", "exocalib/refined", "exocalib/kp3d_refined", "exocalib_error/refined")}
    assert all(log.static for log in logs)
    assert RefinementDiagnostics(1.0, [], [], 1.0, []).to_rerun("/world/exocalib/refined")[0].static


def test_every_port_the_app_writes_to_a_sink_is_a_dagkit_type() -> None:
    ports = {port for layers in (LAYERS, PASS1_LAYERS, INIT_LAYERS) for layer_ports in layers.values() for port in layer_ports}
    assert all(issubclass(PORT_TYPES[port], Type) for port in ports), [port for port in ports if not issubclass(PORT_TYPES[port], Type)]  # type: ignore[arg-type]
