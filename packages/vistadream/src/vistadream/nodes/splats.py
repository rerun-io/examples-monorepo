"""The Gaussian scene: build it from the depth, grow it with inpainted views (coarse stage), render it along a path, save it."""

import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
from dagkit.node import Emit, Event, Fenced, RunContext, Tick
from jaxtyping import Bool, Float32, Float64, UInt8
from monopriors.depth_utils import depth_edges_mask
from monopriors.models.relative_depth.base_relative_depth import BaseRelativePredictor, RelativeDepthPrediction
from numpy import ndarray
from plyfile import PlyData, PlyElement
from simplecv.gpu import consume, fence_after
from torch import Tensor

from vistadream.models.flux import Inpaint
from vistadream.ops.depth_align import align_depth
from vistadream.ops.splats import Rasterize, gaussians, hole_mask, render, train
from vistadream.ops.trajectory import spiral
from vistadream.types import CameraView, OutpaintCanvas, RelativeDepth, RgbImage, SceneViews, Splats

EDGE_THRESHOLD: float = 0.01
"""Depth-gradient magnitude above which a pixel is a depth edge and spawns no Gaussian."""


class InitNode:
    """Fit the first Gaussian scene

    The input view (and the outpainted canvas before it) as Gaussians on every pixel off a depth edge (canvas: only the
    outpainted, valid ones), optimised ``iters`` steps against both views."""

    inputs = {"depth": RelativeDepth, "canvas": OutpaintCanvas, "image": RgbImage}
    outputs = {"scene": Fenced[Splats], "views": SceneViews}

    def __init__(self, iters: int = 100, seed: int = 0, device: str = "cuda", rasterize: Rasterize = render) -> None:
        self.iters, self.seed, self.device, self.rasterize = iters, seed, device, rasterize
        self.rng: np.random.RandomState = np.random.RandomState(seed)

    def reset(self, ctx: RunContext) -> None:
        self.rng = np.random.RandomState(self.seed)  # legacy generator: seed 0 draws what np.random.seed(0) did pre-port

    def step(self, tick: Tick, out: Emit) -> None:
        depth, canvas, image = tick.value, tick.asof["canvas"], tick.asof["image"]
        assert isinstance(depth, RelativeDepth) and isinstance(canvas, OutpaintCanvas) and isinstance(image, RgbImage)
        edges: Bool[ndarray, "H W"] = depth_edges_mask(depth.depth, threshold=EDGE_THRESHOLD)
        x, y, width, height = (int(v) for v in canvas.box_xywh)
        crop: tuple[slice, slice] = (slice(y, y + height), slice(x, x + width))
        K: Float64[ndarray, "3 3"] = depth.K.copy()
        K[:2, 2] = width / 2.0, height / 2.0  # the canvas focal length, centred on the input
        views: list[CameraView] = [CameraView(image.rgb, K, np.eye(4), depth.depth[crop], depth.valid[crop])]
        spawn: list[Bool[ndarray, "h w"]] = [~edges[crop]]
        if canvas.mask.any():
            views.insert(0, CameraView(canvas.rgb, depth.K, np.eye(4), depth.depth, depth.valid, supervise=canvas.mask))
            spawn.insert(0, canvas.mask & ~edges & depth.valid)
        splats: Splats = train(None, [gaussians(v, s, self.device) for v, s in zip(views, spawn, strict=True)], views, self.iters, self.rng, self.rasterize)
        out.put("scene", tick.t_ns, Fenced(splats, fence_after(splats.centers)))
        out.put("views", tick.t_ns, SceneViews(tuple(views), input_index=len(views) - 1))

    def flush(self, out: Emit) -> None:
        pass


class ExpandNode:
    """Grow the scene with inpainted views

    Coarse stage only: up to ``n_frames - 2`` views picked on a dense spiral (largest holes under 25 % of the image, never
    next to a picked one), inpainted, given aligned depth, and each optimised ``iters`` steps into the scene (old Gaussians
    fixed)."""

    inputs = {"scene": Fenced[Splats], "views": SceneViews}
    outputs = {"scene": Fenced[Splats], "views": SceneViews}

    def __init__(
        self, inpaint: Inpaint, predictor: BaseRelativePredictor, n_frames: int = 10, iters: int = 500, seed: int = 0, rasterize: Rasterize = render
    ) -> None:
        self.inpaint, self.predictor, self.rasterize = inpaint, predictor, rasterize
        self.n_frames, self.iters, self.seed, self.margin = n_frames, iters, seed, 32
        self.rng: np.random.RandomState = np.random.RandomState(seed)

    def reset(self, ctx: RunContext) -> None:
        self.rng = np.random.RandomState(self.seed)

    def _holes(self, splats: Splats, K: Float64[ndarray, "3 3"], cam_T_world: Float64[ndarray, "4 4"], width: int, height: int) -> tuple[Tensor, Tensor, Bool[ndarray, "h w"]]:
        with torch.no_grad():
            rgb, depth, alpha = self.rasterize(splats, K, cam_T_world, width, height)
        return rgb, depth, hole_mask(alpha)

    def run(self, inputs: Mapping[str, Sequence[Event]], out: Emit) -> None:
        [scene], [views_event] = inputs["scene"], inputs["views"]
        assert isinstance(scene.value, Fenced) and isinstance(views_event.value, SceneViews)
        splats: Splats = consume(scene.value)
        input_index: int = views_event.value.input_index
        views: list[CameraView] = list(views_event.value.views)
        base: CameraView = views[0]
        assert base.depth is not None
        height, width = base.rgb.shape[0] + self.margin, base.rgb.shape[1] + self.margin
        K: Float64[ndarray, "3 3"] = base.K.copy()
        K[:2, 2] = width / 2.0, height / 2.0
        dense: Float64[ndarray, "n 4 4"] = spiral(base.depth, base.K, self.n_frames * 10)
        picked: list[int] = []
        for _ in range(self.n_frames - 2):
            ratios: Float32[ndarray, "n"] = np.array([self._holes(splats, K, pose, width, height)[2].astype(np.float32).mean() for pose in dense])
            ratios[ratios > 0.25] = 0.0
            for i in picked:
                ratios[max(i - 1, 0) : i + 2] = 0.0
            best: int = int(np.argmax(ratios))
            if ratios[best] < 0.05:
                break
            picked.append(best)
            rendered_rgb, rendered_depth, holes = self._holes(splats, K, dense[best], width, height)
            rgb: UInt8[ndarray, "h w 3"] = self.inpaint((rendered_rgb.cpu().numpy() * 255).astype(np.uint8), holes.astype(np.uint8) * 255)
            prediction: RelativeDepthPrediction = self.predictor(rgb=rgb, K_33=K.astype(np.float32))
            assert prediction.confidence is not None
            predicted: Float32[ndarray, "h w"] = np.where(np.isfinite(prediction.depth), prediction.depth, 0.0).astype(np.float32)
            depth: Float32[ndarray, "h w"] = align_depth(rendered_depth.cpu().numpy(), predicted, holes)
            valid: Bool[ndarray, "h w"] = prediction.confidence > 0
            view = CameraView(rgb, K, dense[best], depth, valid, supervise=holes)
            views.append(view)
            spawn: Bool[ndarray, "h w"] = holes & ~depth_edges_mask(depth, threshold=EDGE_THRESHOLD) & valid
            splats = train(splats, [gaussians(view, spawn, str(splats.centers.device))], views, self.iters, self.rng, self.rasterize)
        out.put("scene", scene.t_ns, Fenced(splats, fence_after(splats.centers)))
        out.put("views", scene.t_ns, SceneViews(tuple(views), input_index))


class RenderNode:
    """Render the scene along a camera path

    The final scene at the input camera along the spiral: 150 views (25 per training view when there are more than 2),
    one every ``1 / fps`` s after the scene's time."""

    inputs = {"scene": Fenced[Splats], "views": SceneViews}
    outputs = {"view": CameraView}

    def __init__(self, fps: int = 30, rasterize: Rasterize = render) -> None:
        self.fps, self.rasterize = fps, rasterize

    def reset(self, ctx: RunContext) -> None:
        pass

    def step(self, tick: Tick, out: Emit) -> None:
        assert isinstance(tick.value, Fenced)
        splats: Splats = consume(tick.value)
        scene = tick.asof["views"]
        assert isinstance(scene, SceneViews) and scene.views[0].depth is not None
        n: int = min(len(scene.views) * 25 if len(scene.views) > 2 else 150, 150)
        camera: CameraView = scene.input
        height, width = camera.rgb.shape[:2]
        with torch.no_grad():
            for i, cam_T_world in enumerate(spiral(scene.views[0].depth, scene.views[0].K, n)):
                rgb: Float32[Tensor, "h w 3"] = self.rasterize(splats, camera.K, cam_T_world, width, height)[0]
                rgb8: UInt8[ndarray, "h w 3"] = (rgb.clamp(0, 1) * 255.0).round().byte().cpu().numpy()
                out.put("view", tick.t_ns + i * 1_000_000_000 // self.fps, CameraView(rgb8, camera.K, cam_T_world))

    def flush(self, out: Emit) -> None:
        pass


class PlySink:
    """Save the scene as a 3DGS PLY file

    Degree-0 SH (DC + 45 zero coefficients), logit opacity, log scales."""

    inputs = {"scene": Splats}
    outputs: dict = {}

    def __init__(self, path: Path) -> None:
        self.path: Path = path

    def reset(self, ctx: RunContext) -> None:
        pass

    def step(self, tick: Tick, out: Emit) -> None:
        assert isinstance(tick.value, Splats)
        s: Splats = tick.value
        columns: list[tuple[str, Float32[Tensor, " n"] | float]] = [
            *zip(("x", "y", "z"), s.centers.T, strict=True),
            *((name, 0.0) for name in ("nx", "ny", "nz")),
            *zip(("f_dc_0", "f_dc_1", "f_dc_2"), ((s.rgb - 0.5) / 0.28209479177387814).T, strict=True),
            *((f"f_rest_{i}", 0.0) for i in range(45)),
            ("opacity", torch.logit(s.opacity)),
            *zip(("scale_0", "scale_1", "scale_2"), s.scales.log().T, strict=True),
            *zip(("rot_0", "rot_1", "rot_2", "rot_3"), s.quats_wxyz.T, strict=True),
        ]
        vertices: ndarray = np.empty(len(s.opacity), dtype=[(name, "f4") for name, _ in columns])
        for name, values in columns:
            vertices[name] = values.detach().cpu().numpy() if isinstance(values, Tensor) else values
        self.path.parent.mkdir(parents=True, exist_ok=True)
        partial: Path = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex[:12]}.partial")  # a sibling: the rename stays on one filesystem
        try:
            PlyData([PlyElement.describe(vertices, "vertex")]).write(partial)
            partial.replace(self.path)  # atomic: the path holds the earlier file or the whole new one, never a partial write
        finally:
            partial.unlink(missing_ok=True)

    def flush(self, out: Emit) -> None:
        pass
