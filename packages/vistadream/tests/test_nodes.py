"""Node contracts with fakes (unit lane, CPU): reset idempotent, outputs monotonic per port, flush drains."""

import numpy as np
import torch
from dagkit.node import Event, Fenced, RunContext, Tick
from dagkit.testing import Recorder, check_node_contract
from monopriors.depth_utils import depth_edges_mask
from vistadream_fakes import FakePredictor, Hosted, fake_inpaint, fake_rasterize

from vistadream.nodes.prepare import DepthNode, OutpaintNode
from vistadream.nodes.splats import EDGE_THRESHOLD, ExpandNode, InitNode, RenderNode
from vistadream.types import OutpaintCanvas, RelativeDepth, RgbImage, SceneViews, Splats

IMAGE = RgbImage(np.random.default_rng(0).integers(0, 256, (64, 96, 3), dtype=np.uint8))  # large enough for the border to fit the canvas


def prepared(inpaint: bool) -> tuple[OutpaintCanvas, RelativeDepth]:
    [(_, _, canvas)] = check_node_contract(OutpaintNode(fake_inpaint if inpaint else None), [Tick(0, "image", IMAGE, {})])
    [(_, _, depth)] = check_node_contract(DepthNode(FakePredictor()), [Tick(0, "canvas", canvas, {})])
    assert isinstance(canvas, OutpaintCanvas) and isinstance(depth, RelativeDepth)
    return canvas, depth


def initialised(inpaint: bool) -> tuple[Splats, SceneViews, RelativeDepth]:
    canvas, depth = prepared(inpaint)
    tick = Tick(0, "depth", depth, {"canvas": canvas, "image": IMAGE})
    check_node_contract(Hosted(InitNode(iters=3, device="cpu", rasterize=fake_rasterize)), [tick])
    out = Recorder()
    InitNode(iters=3, device="cpu", rasterize=fake_rasterize).step(tick, out)
    assert [(port, t) for port, t, _ in out.items] == [("scene", 0), ("views", 0)]
    scene, views = out.items[0][2], out.items[1][2]
    assert isinstance(scene, Fenced) and isinstance(scene.value, Splats) and isinstance(views, SceneViews)
    return scene.value, views, depth


def test_outpaint_without_inpainting_is_the_image() -> None:
    canvas, _ = prepared(inpaint=False)
    np.testing.assert_array_equal(canvas.rgb, IMAGE.rgb)
    assert not canvas.mask.any() and canvas.box_xywh.tolist() == [0, 0, 96, 64]


def test_outpaint_fills_only_the_border() -> None:
    canvas, _ = prepared(inpaint=True)
    x, y, w, h = canvas.box_xywh.tolist()
    np.testing.assert_array_equal(canvas.rgb[y : y + h, x : x + w], IMAGE.rgb)
    assert (canvas.rgb[canvas.mask] == 128).all() and canvas.mask.sum() == canvas.mask.size - 96 * 64


def test_depth_zeroes_non_finite_predictions() -> None:
    class Holey(FakePredictor):
        def __call__(self, rgb, K_33=None):  # type: ignore[no-untyped-def]
            prediction = super().__call__(rgb, K_33)
            prediction.depth[0, 0] = np.inf
            return prediction

    [(_, _, depth)] = check_node_contract(DepthNode(Holey()), [Tick(0, "canvas", prepared(False)[0], {})])
    assert isinstance(depth, RelativeDepth) and depth.depth[0, 0] == 0.0 and np.isfinite(depth.depth).all()


def test_init_spawns_one_gaussian_per_pixel_off_a_depth_edge() -> None:
    splats, views, depth = initialised(inpaint=False)
    assert views.input_index == 0 and len(views.views) == 1
    assert len(splats.centers) == (~depth_edges_mask(depth.depth, threshold=EDGE_THRESHOLD)).sum()
    np.testing.assert_array_equal(views.input.rgb, IMAGE.rgb)


def test_init_with_outpainting_puts_the_canvas_first_and_centres_the_input_camera() -> None:
    _, views, depth = initialised(inpaint=True)
    assert views.input_index == 1 and views.views[0].supervise is not None and views.views[0].rgb.shape[:2] == depth.depth.shape
    np.testing.assert_allclose(views.input.K[:2, 2], [48.0, 32.0])
    np.testing.assert_allclose(views.input.K[0, 0], depth.K[0, 0])


def test_render_emits_the_path_on_a_30_fps_clock_from_the_input_camera() -> None:
    splats, views, _ = initialised(inpaint=False)
    outputs = check_node_contract(Hosted(RenderNode(rasterize=fake_rasterize)), [Tick(7, "scene", Fenced(splats, None), {"views": views})])
    assert [t for _, t, _ in outputs] == [7 + i * 1_000_000_000 // 30 for i in range(150)]
    first: dict = outputs[0][2]  # type: ignore[assignment]
    np.testing.assert_allclose(first["cam_T_world"], np.eye(4), atol=1e-9)
    assert first["rgb"].shape == (64, 96, 3) and first["rgb"].dtype == np.uint8


def test_expand_adds_inpainted_views_and_keeps_the_old_gaussians() -> None:
    splats, views, _ = initialised(inpaint=True)
    node = ExpandNode(fake_inpaint, FakePredictor(), n_frames=4, iters=2, rasterize=fake_rasterize)
    node.reset(RunContext("vistadream-single-image", "expand", "lossless"))
    out = Recorder()
    node.run({"scene": [Event(0, "scene", Fenced(splats, None))], "views": [Event(0, "views", views)]}, out)
    assert [(port, t) for port, t, _ in out.items] == [("scene", 0), ("views", 0)]
    grown, grown_views = out.items[0][2], out.items[1][2]
    assert isinstance(grown, Fenced) and isinstance(grown_views, SceneViews)
    assert len(grown_views.views) == 4 and grown_views.input_index == 1
    added = grown_views.views[2]
    assert added.supervise is not None and added.supervise[:, :5].all() and not added.supervise[:, -5:].any()  # the fake's hole: left tenth
    torch.testing.assert_close(grown.value.centers[: len(splats.centers)], splats.centers)
    assert len(grown.value.centers) > len(splats.centers)
