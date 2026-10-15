"""vistadream Types: serde round-trips, the Rerun mapping, and to_rerun -> from_layer on catalog-shaped rows (unit lane)."""

import dataclasses

import numpy as np
import pytest
import rerun as rr
import torch
from dagkit.testing import layer_table
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from scipy.spatial.transform import Rotation
from serde import SerdeError, from_dict, to_dict

from vistadream.types import CameraView, OutpaintCanvas, RelativeDepth, RgbImage, Scene, SceneViews, Splats

SHAPES = st.tuples(st.integers(2, 6), st.integers(2, 6))  # rr.Image squeezes size-1 axes
UNIT = st.floats(0.0, 1.0, width=32, allow_subnormal=False)


@st.composite
def views(draw: st.DrawFn, depth: bool | None = None) -> CameraView:
    h, w = draw(SHAPES)
    with_depth: bool = draw(st.booleans()) if depth is None else depth
    K = np.array([[draw(st.floats(10.0, 1e3)), 0.0, w / 2], [0.0, draw(st.floats(10.0, 1e3)), h / 2], [0.0, 0.0, 1.0]])
    quat = draw(arrays(np.float64, 4, elements=st.floats(-1.0, 1.0)))
    assume(np.linalg.norm(quat) > 0.1)
    cam_T_world = np.eye(4)
    cam_T_world[:3, :3] = Rotation.from_quat(quat).as_matrix()  # any rotation: a lost or transposed one must show
    cam_T_world[:3, 3] = draw(arrays(np.float64, 3, elements=st.floats(-10.0, 10.0, width=32, allow_subnormal=False)))
    return CameraView(
        draw(arrays(np.uint8, (h, w, 3))),
        K,
        cam_T_world,
        draw(arrays(np.float32, (h, w), elements=st.floats(0.0, 100.0, width=32, allow_subnormal=False))) if with_depth else None,
        draw(arrays(np.bool_, (h, w))) if with_depth else None,
        draw(st.none() | arrays(np.bool_, (h, w))),
    )


@st.composite
def splats(draw: st.DrawFn) -> Splats:
    n: int = draw(st.integers(1, 8))
    quats = torch.nn.functional.normalize(torch.as_tensor(draw(arrays(np.float32, (n, 4), elements=st.floats(0.125, 1.0, width=32, allow_subnormal=False)))), dim=1)
    return Splats(
        torch.as_tensor(draw(arrays(np.float32, (n, 3), elements=st.floats(-10.0, 10.0, width=32, allow_subnormal=False)))),
        torch.as_tensor(draw(arrays(np.float32, (n, 3), elements=st.floats(2**-12, 1.0, width=32, allow_subnormal=False)))),
        quats,
        torch.as_tensor(draw(arrays(np.float32, (n, 3), elements=UNIT))),
        torch.as_tensor(draw(arrays(np.float32, (n,), elements=UNIT))),
    )


def assert_views_equal(a: CameraView, b: CameraView) -> None:
    for name in ("rgb", "K", "cam_T_world", "depth", "valid", "supervise"):
        x, y = getattr(a, name), getattr(b, name)
        assert (x is None) == (y is None), name
        if x is not None:
            np.testing.assert_allclose(x, y, rtol=1e-6, atol=1e-5, err_msg=name)
            assert x.dtype == y.dtype, name


@settings(max_examples=30, deadline=None)
@given(views())
def test_camera_view_serde_and_layer_round_trip(view: CameraView) -> None:
    assert_views_equal(from_dict(CameraView, to_dict(view)), view)
    [(t, back)] = list(CameraView.from_layer(layer_table([(5, view)], "/world/input"), "/world/input", "video_time"))
    assert t == 5
    assert_views_equal(back, view)


def test_camera_view_rerun_paths_and_contract() -> None:
    view = CameraView(np.zeros((4, 6, 3), np.uint8), np.eye(3), np.eye(4), np.ones((4, 6), np.float32), np.ones((4, 6), bool))
    logs = {log.path: log for log in view.to_rerun("/world/input")}
    assert sorted(logs) == ["/world/input", "/world/input/pinhole", "/world/input/pinhole/depth", "/world/input/pinhole/rgb", "/world/input/pinhole/valid"]
    assert isinstance(logs["/world/input/pinhole/depth"].data, rr.DepthImage) and not any(log.static for log in logs.values())


@settings(max_examples=20, deadline=None)
@given(st.lists(views(depth=True), min_size=1, max_size=4), st.data())
def test_scene_views_round_trip_and_names(training: list[CameraView], data: st.DataObject) -> None:
    scene = SceneViews(tuple(training), data.draw(st.integers(0, min(1, len(training) - 1))))
    assert "input" in scene.names() and len(set(scene.names())) == len(training)
    back: SceneViews = from_dict(SceneViews, to_dict(scene))
    [(_, read)] = list(SceneViews.from_layer(layer_table([(0, scene)], "/world"), "/world", "video_time"))
    for other in (back, read):
        assert other.input_index == scene.input_index
        for x, y in zip(other.views, scene.views, strict=True):
            assert_views_equal(x, y)


def test_scene_views_read_back_snapshots_whose_views_change() -> None:
    view = CameraView(np.zeros((2, 3, 3), np.uint8), np.eye(3), np.eye(4), np.ones((2, 3), np.float32))
    snapshots: list[tuple[int, SceneViews]] = [(0, SceneViews((view,), 0)), (1, SceneViews((view, view), 1)), (2, SceneViews((view, view, view), 1))]
    read = list(SceneViews.from_layer(layer_table(snapshots, "/world"), "/world", "video_time"))
    assert [(t, s.input_index, s.names()) for t, s in read] == [(0, 0, ["input"]), (1, 1, ["outpaint", "input"]), (2, 1, ["outpaint", "input", "view_02"])]


@settings(max_examples=30, deadline=None)
@given(SHAPES, st.data())
def test_image_canvas_depth_round_trip(shape: tuple[int, int], data: st.DataObject) -> None:
    h, w = shape
    image = RgbImage(data.draw(arrays(np.uint8, (h, w, 3))))
    canvas = OutpaintCanvas(image.rgb, data.draw(arrays(np.bool_, (h, w))), np.array([0, 0, w, h], dtype=np.int64))
    depth = RelativeDepth(data.draw(arrays(np.float32, (h, w), elements=st.floats(0.0, 50.0, width=32, allow_subnormal=False))), canvas.mask, np.eye(3))
    for value in (image, canvas, depth):
        kind = type(value)
        [(_, read)] = list(kind.from_layer(layer_table([(0, value)], "/x"), "/x", "video_time"))
        for back in (from_dict(kind, to_dict(value)), read):  # serde, then the Rerun mapping: values, shapes and dtypes kept
            for field in dataclasses.fields(kind):
                x, y = getattr(back, field.name), getattr(value, field.name)
                np.testing.assert_array_equal(x, y, err_msg=f"{kind.__name__}.{field.name}")
                assert x.dtype == y.dtype, f"{kind.__name__}.{field.name}"


def test_unknown_fields_are_refused() -> None:
    with pytest.raises(SerdeError):
        from_dict(RgbImage, {"rgb": np.zeros((1, 1, 3), np.uint8), "extra": 1})


@settings(max_examples=30, deadline=None)
@given(splats())
def test_splats_map_to_native_gaussian_splats(value: Splats) -> None:
    [log] = value.to_rerun("/world/splats")
    assert log.path == "/world/splats" and isinstance(log.data, rr.GaussianSplats3D) and not log.static
    [(_, back)] = list(Splats.from_layer(layer_table([(0, value)], "/world/splats"), "/world/splats", "video_time"))
    torch.testing.assert_close(back.centers, value.centers)
    torch.testing.assert_close(back.scales, value.scales)
    torch.testing.assert_close(back.quats_wxyz, value.quats_wxyz)  # Rerun stores xyzw; the Type is wxyz
    torch.testing.assert_close(back.rgb, value.rgb, atol=0.5 / 255 + 1e-6, rtol=0.0)  # RGBA8
    torch.testing.assert_close(back.opacity, value.opacity, atol=0.5 / 255 + 1e-6, rtol=0.0)


@settings(max_examples=20, deadline=None)
@given(splats(), st.lists(views(depth=True), min_size=1, max_size=3))
def test_scene_logs_its_splats_beside_its_views_and_reads_both_back(value: Splats, training: list[CameraView]) -> None:
    scene = Scene(value, SceneViews(tuple(training), 0))
    assert {log.path for log in scene.to_rerun("/world")} >= {"/world/splats", "/world/input"}
    [(_, back)] = list(Scene.from_layer(layer_table([(0, scene)], "/world"), "/world", "video_time"))
    torch.testing.assert_close(back.splats.centers, value.centers)
    assert back.views.input_index == 0
    for x, y in zip(back.views.views, training, strict=True):
        assert_views_equal(x, y)
