"""CPU fakes for the unit lane (no GPU, no weights) and a catalog-shaped table builder."""

import dataclasses
from collections import defaultdict
from collections.abc import Mapping

import numpy as np
import pyarrow as pa
import torch
from dagkit.node import Emit, Fenced, PortType, RunContext, Tick
from jaxtyping import Float32, Float64, UInt8
from monopriors.models.relative_depth.base_relative_depth import BaseRelativePredictor, RelativeDepthPrediction
from numpy import ndarray

from vistadream.ops.splats import Render
from vistadream.types import Splats


def layer_table(values: list[tuple[int, object]], root: str) -> pa.Table:
    """What a catalog reader returns for these values: one row per time, ``<entity>:<component>`` columns."""
    cells: dict[str, dict[int, list]] = defaultdict(dict)
    for t_ns, value in values:
        for log in value.to_rerun(root):  # type: ignore[attr-defined]
            for batch in log.data.as_component_batches():
                cells[f"{log.path}:{batch.component_descriptor()}"][t_ns] = batch.as_arrow_array().to_pylist()
    times: list[int] = sorted({t for t, _ in values})
    return pa.table({"video_time": pa.array(times, type=pa.int64())} | {name: [c.get(t) for t in times] for name, c in cells.items()})


def fake_rasterize(splats: Splats, K: Float64[ndarray, "3 3"], cam_T_world: Float64[ndarray, "4 4"], width: int, height: int) -> Render:
    """Differentiable in colour: every pixel shows the mean splat colour; a camera off the origin leaves its left tenth uncovered."""
    rgb: Float32[torch.Tensor, "h w 3"] = splats.rgb.mean(0).expand(height, width, 3)
    alpha: Float32[torch.Tensor, "h w"] = torch.ones(height, width)
    if np.abs(cam_T_world[:3, 3]).max() > 1e-6:
        alpha[:, : width // 10] = 0.0
    return rgb, torch.ones(height, width), alpha


class FakePredictor(BaseRelativePredictor):
    """A gently tilted plane (no depth edges) 1-1.4 units away with every pixel valid."""

    def __call__(self, rgb: UInt8[ndarray, "h w 3"], K_33: Float32[ndarray, "3 3"] | None = None) -> RelativeDepthPrediction:
        height, width = rgb.shape[:2]
        depth: Float32[ndarray, "h w"] = (1.0 + np.linspace(0.0, 0.4, width, dtype=np.float32)[None, :].repeat(height, 0)).astype(np.float32)
        K: Float32[ndarray, "3 3"] = np.array([[width, 0, width / 2], [0, width, height / 2], [0, 0, 1]], dtype=np.float32)
        return RelativeDepthPrediction(1.0 / depth, depth, K, np.ones_like(depth))


def fake_inpaint(rgb: UInt8[ndarray, "h w 3"], mask: UInt8[ndarray, "h w"]) -> UInt8[ndarray, "h w 3"]:
    return np.where(mask[..., None] > 0, np.uint8(128), rgb).astype(np.uint8)


def host(value: object) -> object:
    """A value with every tensor as a numpy array (fenced values unwrapped, dataclasses as dicts), for equality checks."""
    value = value.value if isinstance(value, Fenced) else value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: host(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, tuple):
        return [host(v) for v in value]
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value


class HostEmit:
    def __init__(self, out: Emit) -> None:
        self.out = out

    def put(self, port: str, t_ns: int, value: object) -> None:
        self.out.put(port, t_ns, host(value))

    def watermark(self, t_ns: int) -> None:
        self.out.watermark(t_ns)


class Hosted:
    """A node whose outputs reach ``check_node_contract`` as host values (it compares runs with numpy)."""

    def __init__(self, node: object) -> None:
        self.node = node
        self.inputs, self.outputs = node.inputs, node.outputs  # type: ignore[attr-defined]

    def reset(self, ctx: RunContext) -> None:
        self.node.reset(ctx)  # type: ignore[attr-defined]

    def step(self, tick: Tick, out: Emit) -> None:
        self.node.step(tick, HostEmit(out))  # type: ignore[attr-defined]

    def flush(self, out: Emit) -> None:
        self.node.flush(HostEmit(out))  # type: ignore[attr-defined]


class Keep:
    """Sink keeping a host copy of every value it receives."""

    outputs: Mapping[str, PortType] = {}

    def __init__(self, inputs: Mapping[str, PortType]) -> None:
        self.inputs: dict[str, PortType] = dict(inputs)
        self.items: list[tuple[str, int, object]] = []

    def reset(self, ctx: RunContext) -> None:
        self.items = []

    def step(self, tick: Tick, out: Emit) -> None:
        self.items.append((tick.port, tick.t_ns, host(tick.value)))

    def flush(self, out: Emit) -> None:
        pass
