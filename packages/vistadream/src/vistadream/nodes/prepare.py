"""The image, its outpainted canvas and its depth: the nodes before any Gaussian exists."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
from dagkit.node import Emit, Event, RunContext, Tick
from dagkit.sources import IterSource
from jaxtyping import Bool, UInt8
from monopriors.models.relative_depth.base_relative_depth import BaseRelativePredictor, RelativeDepthPrediction
from numpy import ndarray
from PIL import Image

from vistadream.models.flux import Inpaint
from vistadream.ops.image import add_border_and_mask, process_image
from vistadream.types import OutpaintCanvas, RelativeDepth, RgbImage


class ImageSource(IterSource):
    """Read the photo and resize it

    One ``image`` event at t = 0: the file as RGB, resized by ``process_image`` (long side ``max_resolution``)."""

    def __init__(self, path: Path, max_resolution: int) -> None:
        def events() -> Iterator[Event]:
            yield Event(0, "image", RgbImage(np.array(process_image(Image.open(path).convert("RGB"), max_dimension=max_resolution))))

        super().__init__({"image": RgbImage}, events)


class OutpaintNode:
    """Outpaint a border around the image

    Pads the image by ``expansion / 2`` per side and fills the border with ``inpaint`` (FLUX.1 Fill); without one, the
    canvas is the image."""

    inputs = {"image": RgbImage}
    outputs = {"canvas": OutpaintCanvas}

    def __init__(self, inpaint: Inpaint | None, expansion: float = 0.3) -> None:
        self.inpaint: Inpaint | None = inpaint
        self.expansion: float = expansion

    def reset(self, ctx: RunContext) -> None:
        pass

    def step(self, tick: Tick, out: Emit) -> None:
        assert isinstance(tick.value, RgbImage)
        rgb: UInt8[ndarray, "h w 3"] = tick.value.rgb
        if self.inpaint is None:
            height, width = rgb.shape[:2]
            canvas = OutpaintCanvas(rgb, np.zeros((height, width), dtype=bool), np.array([0, 0, width, height], dtype=np.int64))
        else:
            bordered, mask, box_xywh = add_border_and_mask(rgb, self.expansion / 2.0)
            canvas = OutpaintCanvas(self.inpaint(bordered, mask.astype(np.uint8) * 255), mask, box_xywh)
        out.put("canvas", tick.t_ns, canvas)

    def flush(self, out: Emit) -> None:
        pass


class DepthNode:
    """Predict the depth of the canvas

    Relative depth of the canvas from a predictor with a validity mask (MoGe v1); non-finite depth becomes 0."""

    inputs = {"canvas": OutpaintCanvas}
    outputs = {"depth": RelativeDepth}

    def __init__(self, predictor: BaseRelativePredictor) -> None:
        self.predictor: BaseRelativePredictor = predictor

    def reset(self, ctx: RunContext) -> None:
        pass

    def step(self, tick: Tick, out: Emit) -> None:
        assert isinstance(tick.value, OutpaintCanvas)
        prediction: RelativeDepthPrediction = self.predictor(rgb=tick.value.rgb, K_33=None)
        if prediction.confidence is None:
            raise ValueError("vistadream needs a depth predictor with a validity mask (MoGe)")
        finite: Bool[ndarray, "h w"] = np.isfinite(prediction.depth)
        depth = RelativeDepth(np.where(finite, prediction.depth, 0.0).astype(np.float32), prediction.confidence > 0, prediction.K_33.astype(np.float64))
        out.put("depth", tick.t_ns, depth)

    def flush(self, out: Emit) -> None:
        pass
