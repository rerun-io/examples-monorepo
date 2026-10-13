"""Op properties (unit lane, CPU): camera path, image sizing, depth alignment, Gaussian initialisation and training."""

import numpy as np
import torch
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from PIL import Image
from vistadream_fakes import fake_rasterize

from vistadream.ops.depth_align import align_depth
from vistadream.ops.image import add_border_and_mask, process_image
from vistadream.ops.splats import gaussians, hole_mask, train
from vistadream.ops.trajectory import backproject, spiral
from vistadream.types import CameraView

DEPTH = st.floats(0.25, 20.0, width=32, allow_subnormal=False)


@st.composite
def depth_and_K(draw: st.DrawFn) -> tuple[np.ndarray, np.ndarray]:
    h, w = draw(st.integers(4, 12)), draw(st.integers(4, 12))
    depth = draw(arrays(np.float32, (h, w), elements=DEPTH))
    f: float = draw(st.floats(2.0, 50.0))
    return depth, np.array([[f, 0.0, w / 2], [0.0, f, h / 2], [0.0, 0.0, 1.0]])


@settings(max_examples=50, deadline=None)
@given(depth_and_K(), st.integers(2, 200))
def test_spiral_poses_are_rigid_and_start_and_end_at_the_input_camera(scene: tuple[np.ndarray, np.ndarray], n: int) -> None:
    depth, K = scene
    assume(np.ptp(depth) > 0.01)  # a flat wall still has a lateral spread; a zero radius is degenerate
    poses = spiral(depth, K, n)
    assert poses.shape == (n, 4, 4) and np.isfinite(poses).all()
    rotations = poses[:, :3, :3]
    np.testing.assert_allclose(rotations @ rotations.transpose(0, 2, 1), np.broadcast_to(np.eye(3), (n, 3, 3)), atol=1e-9)
    np.testing.assert_allclose(np.linalg.det(rotations), 1.0, atol=1e-9)
    np.testing.assert_allclose(poses[[0, -1]], np.broadcast_to(np.eye(4), (2, 4, 4)), atol=1e-9)  # sin(0) = sin(2 pi) = 0


@settings(max_examples=50, deadline=None)
@given(depth_and_K())
def test_backproject_reprojects_onto_its_pixels(scene: tuple[np.ndarray, np.ndarray]) -> None:
    depth, K = scene
    points = backproject(depth, K)
    np.testing.assert_allclose(points[..., 2], depth, rtol=1e-6)
    uv = (points @ K.T)[..., :2] / points[..., 2:]
    v, u = np.indices(depth.shape)
    np.testing.assert_allclose(uv, np.stack([u, v], axis=-1), atol=1e-6)


@settings(max_examples=40, deadline=None)
@given(st.integers(16, 2500), st.integers(16, 2500), st.sampled_from([512, 1024, 1920]))
def test_process_image_keeps_aspect_and_caps_the_long_side(width: int, height: int, max_dimension: int) -> None:
    out_w, out_h = process_image(Image.new("RGB", (width, height)), max_dimension=max_dimension).size
    assert out_w % 32 == 0 and out_h % 32 == 0
    if max(width, height) > max_dimension:
        assert max(out_w, out_h) <= max_dimension + 16  # rounding to 32 moves a side by at most 16
    assume(min(out_w, out_h) >= 64)  # rounding a short side to 32 dominates its aspect
    assert abs(out_w / out_h - width / height) / (width / height) < 0.35


@settings(max_examples=40, deadline=None)
@given(st.integers(32, 200), st.integers(32, 200), st.floats(0.05, 0.5))
def test_border_mask_covers_exactly_the_canvas_outside_the_image(width: int, height: int, border: float) -> None:
    rgb = np.full((height, width, 3), 200, np.uint8)
    canvas, mask, (x, y, w, h) = add_border_and_mask(rgb, border)
    assert canvas.shape[0] % 32 == 0 and canvas.shape[1] % 32 == 0 and (w, h) == (width, height)
    assume(x + w <= canvas.shape[1] and y + h <= canvas.shape[0])  # rounding down can crop the image (tiny borders)
    assert mask.sum() == mask.size - width * height
    assert not mask[y : y + h, x : x + w].any() and (canvas[mask] == 0).all() and (canvas[~mask] == 200).all()


@settings(max_examples=40, deadline=None)
@given(depth_and_K(), st.floats(0.2, 5.0), st.floats(-1.0, 1.0), st.data())
def test_align_depth_recovers_an_affine_prediction_where_known(scene: tuple[np.ndarray, np.ndarray], scale: float, shift: float, data: st.DataObject) -> None:
    rendered, _ = scene
    h, w = rendered.shape
    holes = np.zeros((h, w), bool)
    holes[:, : data.draw(st.integers(1, w // 2))] = True
    predicted = ((rendered - shift) / scale).astype(np.float32)
    assume((~holes).sum() > 10 and (predicted < 199).all() and np.ptp(predicted[~holes]) > 1e-3)  # fewer known pixels: identity
    aligned = align_depth(rendered, predicted, holes)
    np.testing.assert_allclose(aligned[~holes], rendered[~holes], rtol=1e-3, atol=1e-3)
    assert align_depth(rendered, predicted, np.zeros_like(holes)) is rendered


def test_hole_mask_is_empty_on_full_coverage_and_grows_holes() -> None:
    alpha = torch.ones(40, 40)
    assert not hole_mask(alpha).any()
    alpha[1:3, 1:3] = 0.0  # specks under the 5x5 opening vanish
    assert not hole_mask(alpha).any()
    alpha[14:26, 14:26] = 0.0
    holes = hole_mask(alpha)
    assert holes[14:26, 14:26].all() and holes.sum() > 144  # a real hole survives and grows


@settings(max_examples=30, deadline=None)
@given(depth_and_K(), st.data())
def test_gaussians_sit_on_the_backprojected_spawn_pixels(scene: tuple[np.ndarray, np.ndarray], data: st.DataObject) -> None:
    depth, K = scene
    spawn = data.draw(arrays(np.bool_, depth.shape))
    cam_T_world = np.eye(4)
    cam_T_world[:3, 3] = data.draw(arrays(np.float64, 3, elements=st.floats(-5.0, 5.0)))
    view = CameraView(np.full((*depth.shape, 3), 64, np.uint8), K, cam_T_world, depth, np.ones(depth.shape, bool))
    params = gaussians(view, spawn, "cpu")
    picked = spawn if spawn.sum() >= 3 else np.zeros_like(spawn) | (np.arange(depth.shape[0]) == 0)[:, None]
    expected = backproject(depth, K)[picked] - cam_T_world[:3, 3]  # world = cam_T_world^-1 applied to camera points
    np.testing.assert_allclose(params.means.detach().numpy(), expected, rtol=1e-5, atol=1e-4)
    splats = params.activated()
    np.testing.assert_allclose(splats.opacity.detach().numpy(), 0.8, rtol=1e-6)
    np.testing.assert_allclose(splats.rgb.detach().numpy(), 64 / 255, rtol=1e-6)


def test_train_moves_only_trainable_colours_towards_the_target() -> None:
    depth = np.full((16, 16), 2.0, np.float32)
    K = np.array([[16.0, 0.0, 8.0], [0.0, 16.0, 8.0], [0.0, 0.0, 1.0]])
    view = CameraView(np.full((16, 16, 3), 200, np.uint8), K, np.eye(4), depth, np.ones((16, 16), bool))
    with torch.no_grad():
        frozen = gaussians(CameraView(np.full((16, 16, 3), 10, np.uint8), K, np.eye(4), depth), np.ones((16, 16), bool), "cpu").activated()
    start = gaussians(CameraView(np.full((16, 16, 3), 50, np.uint8), K, np.eye(4), depth), np.ones((16, 16), bool), "cpu")
    out = train(frozen, [start], [view], 40, np.random.RandomState(0), fake_rasterize)
    n = len(frozen.centers)
    torch.testing.assert_close(out.rgb[:n], frozen.rgb)
    assert out.rgb[n:].mean() > 50 / 255 + 1e-3 and not out.rgb.requires_grad
