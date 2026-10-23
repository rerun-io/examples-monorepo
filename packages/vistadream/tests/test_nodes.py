"""Node contracts with fakes (unit lane, CPU): reset idempotent, outputs monotonic per port, flush drains."""

import numpy as np
import torch
from dagkit.node import Event, Fenced, RunContext, Tick
from dagkit.testing import Recorder, check_node_contract
from monopriors.depth_utils import depth_edges_mask
from vistadream_fakes import FakePredictor, fake_inpaint, fake_rasterize

from vistadream.nodes.prepare import DepthNode, OutpaintNode
from vistadream.nodes.splats import EDGE_THRESHOLD, ExpandNode, InitNode, RenderNode
from vistadream.types import CameraView, OutpaintCanvas, RelativeDepth, RgbImage, Scene

IMAGE = RgbImage(np.random.default_rng(0).integers(0, 256, (64, 96, 3), dtype=np.uint8))  # large enough for the border to fit the canvas


def prepared(inpaint: bool) -> tuple[OutpaintCanvas, RelativeDepth]:
    [(_, _, canvas)] = check_node_contract(OutpaintNode(fake_inpaint if inpaint else None), [Tick(0, "image", IMAGE, {})])
    [(_, _, depth)] = check_node_contract(DepthNode(FakePredictor()), [Tick(0, "canvas", canvas, {})])
    assert isinstance(canvas, OutpaintCanvas) and isinstance(depth, RelativeDepth)
    return canvas, depth


def initialised(inpaint: bool) -> tuple[Scene, RelativeDepth]:
    canvas, depth = prepared(inpaint)
    tick = Tick(0, "depth", depth, {"canvas": canvas, "image": IMAGE})
    check_node_contract(InitNode(iters=3, device="cpu", rasterize=fake_rasterize), [tick])
    out = Recorder()
    InitNode(iters=3, device="cpu", rasterize=fake_rasterize).step(tick, out)
    [(port, t, scene)] = out.items  # the splats and their views as ONE value: a render never meets one without the other
    assert (port, t) == ("scene", 0) and isinstance(scene, Fenced) and isinstance(scene.value, Scene)
    return scene.value, depth


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
    scene, depth = initialised(inpaint=False)
    assert scene.views.input_index == 0 and len(scene.views.views) == 1
    assert len(scene.splats.centers) == (~depth_edges_mask(depth.depth, threshold=EDGE_THRESHOLD)).sum()
    np.testing.assert_array_equal(scene.views.input.rgb, IMAGE.rgb)
    np.testing.assert_array_equal(scene.views.input.spawn, ~depth_edges_mask(depth.depth, threshold=EDGE_THRESHOLD))  # the mask main drew


def test_each_training_view_carries_the_pixels_that_spawned_its_gaussians() -> None:
    """Main drew every training view's ``inpaint_wo_edges_mask``: the input off the depth edges, the canvas and every added view
    only where they are new (outpainted or a hole), valid and off an edge. The views carry it, and it is what spawned the splats."""
    scene, depth = initialised(inpaint=True)
    canvas, image = scene.views.views
    edges = depth_edges_mask(depth.depth, threshold=EDGE_THRESHOLD)
    assert canvas.supervise is not None and canvas.spawn is not None and image.spawn is not None
    np.testing.assert_array_equal(canvas.spawn, canvas.supervise & ~edges & depth.valid)
    node = ExpandNode(fake_inpaint, FakePredictor(), n_frames=3, iters=2, rasterize=fake_rasterize)
    node.reset(RunContext("vistadream-single-image", "expand", "lossless"))
    out = Recorder()
    node.run({"scene": [Event(0, "scene", Fenced(scene, None))]}, out)
    [(_, _, fenced)] = out.items
    assert isinstance(fenced, Fenced) and isinstance(grown := fenced.value, Scene)
    added = grown.views.views[2]
    assert added.spawn is not None and added.supervise is not None and added.valid is not None and added.depth is not None
    np.testing.assert_array_equal(added.spawn, added.supervise & ~depth_edges_mask(added.depth, threshold=EDGE_THRESHOLD) & added.valid)
    assert len(grown.splats.centers) == sum(int(v.spawn.sum()) for v in grown.views.views if v.spawn is not None) and all(v.spawn is not None for v in grown.views.views)


def test_init_with_outpainting_puts_the_canvas_first_and_centres_the_input_camera() -> None:
    scene, depth = initialised(inpaint=True)
    views = scene.views
    assert views.input_index == 1 and views.views[0].supervise is not None and views.views[0].rgb.shape[:2] == depth.depth.shape
    np.testing.assert_allclose(views.input.K[:2, 2], [48.0, 32.0])
    np.testing.assert_allclose(views.input.K[0, 0], depth.K[0, 0])


def test_render_emits_the_path_on_a_30_fps_clock_from_the_input_camera() -> None:
    scene, _ = initialised(inpaint=False)
    outputs = check_node_contract(RenderNode(rasterize=fake_rasterize), [Tick(7, "scene", Fenced(scene, None), {})])
    assert [t for _, t, _ in outputs] == [7 + i * 1_000_000_000 // 30 for i in range(150)]
    first = outputs[0][2]
    assert isinstance(first, CameraView)
    np.testing.assert_allclose(first.cam_T_world, np.eye(4), atol=1e-9)
    assert first.rgb.shape == (64, 96, 3) and first.rgb.dtype == np.uint8
    assert first.depth is not None and first.depth.shape == (64, 96) and (first.depth == 1.0).all()  # each step's render depth, as main drew it


def test_expand_adds_inpainted_views_and_keeps_the_old_gaussians() -> None:
    scene, _ = initialised(inpaint=True)
    node = ExpandNode(fake_inpaint, FakePredictor(), n_frames=4, iters=2, rasterize=fake_rasterize)
    node.reset(RunContext("vistadream-single-image", "expand", "lossless"))
    out = Recorder()
    node.run({"scene": [Event(0, "scene", Fenced(scene, None))]}, out)
    [(port, t, grown)] = out.items
    assert (port, t) == ("scene", 0) and isinstance(grown, Fenced) and isinstance(grown.value, Scene)
    assert len(grown.value.views.views) == 4 and grown.value.views.input_index == 1
    added = grown.value.views.views[2]
    assert added.supervise is not None and added.supervise[:, :5].all() and not added.supervise[:, -5:].any()  # the fake's hole: left tenth
    torch.testing.assert_close(grown.value.splats.centers[: len(scene.splats.centers)], scene.splats.centers)
    assert len(grown.value.splats.centers) > len(scene.splats.centers)
