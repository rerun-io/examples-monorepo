"""handtrack's DetNet-F (MEgATrack's two-slot hand detector) as a posekit ``HandDetector``.

The graph is handtrack's ``detnet_full.onnx`` (``handtrack.apis.export_detnet``): a 480 x 640 grey frame valued in [0, 1] in, one
circle (centre / 640 and / 480, radius / 640) and one presence logit per slot out, left slot first. This detector builds that input as
the training stream does (BT.709 luma, ``quarter_turns`` clockwise turns for a camera mounted on its side, the largest fit into
640 x 480, centred bars) and maps each present slot's circle back to a square box in the source frame. One box per side, so no
identity tracking is needed. Train-time parity: the fit equals handtrack's for 1920 x 1080 (RoboCap, GSI) and 640 x 480 frames.
"""

from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float32, Int64, UInt8
from torch import Tensor

from posekit.models.base import HandDetector
from posekit.predictions import BoxDetections, HandDetections, HandSide, validate_frames_rgb
from posekit.runtimes import OnnxBackendConfig, OnnxOrTrtBackendConfig, TensorRuntime, create_runtime_from_onnx, run_chunked

NET_WIDTH: int = 640
NET_HEIGHT: int = 480
BT709: tuple[float, float, float] = (0.2126, 0.7152, 0.0722)
SLOT_SIDES: tuple[HandSide, HandSide] = ("left", "right")


@dataclass(frozen=True, slots=True)
class Fit:
    """How one frame size lands in the net frame: turned, scaled, then padded."""

    width: int
    """Source width before turning."""
    height: int
    quarter_turns: int
    scale: float
    pad_x: int
    pad_y: int


def detnet_input(frames_rgb: UInt8[Tensor, "b h w 3"], quarter_turns: int) -> tuple[Float32[Tensor, "b 1 480 640"], Fit]:
    """BT.709 luma, turned ``quarter_turns`` times clockwise, resized (antialiased bilinear) to the largest fit and centred."""
    if quarter_turns not in (0, 1, 2, 3):
        raise ValueError("quarter_turns must be 0, 1, 2 or 3")
    height, width = int(frames_rgb.shape[1]), int(frames_rgb.shape[2])
    weights: Float32[Tensor, "3"] = torch.tensor(BT709, device=frames_rgb.device)
    grey: Float32[Tensor, "b h w"] = torch.einsum("bhwc,c->bhw", frames_rgb.float(), weights).round().clamp(0.0, 255.0)
    turned: Float32[Tensor, "b H W"] = torch.rot90(grey, -quarter_turns, (1, 2)) if quarter_turns else grey
    scale: float = min(NET_WIDTH / turned.shape[2], NET_HEIGHT / turned.shape[1])
    size: tuple[int, int] = (round(turned.shape[1] * scale), round(turned.shape[2] * scale))
    resized: Float32[Tensor, "b 1 y x"] = F.interpolate(turned[:, None], size=size, mode="bilinear", antialias=True, align_corners=False).round()
    pad_x, pad_y = (NET_WIDTH - size[1]) // 2, (NET_HEIGHT - size[0]) // 2
    net: Float32[Tensor, "b 1 480 640"] = F.pad(resized, (pad_x, NET_WIDTH - size[1] - pad_x, pad_y, NET_HEIGHT - size[0] - pad_y)) / 255.0
    return net, Fit(width, height, quarter_turns, scale, pad_x, pad_y)


def to_source(points: Float32[Tensor, "n 2"], fit: Fit) -> Float32[Tensor, "n 2"]:
    """Net pixel centres back to source pixel centres (undo the pad and scale, then the turns)."""
    uv: Float32[Tensor, "n 2"] = (points - torch.tensor([fit.pad_x, fit.pad_y], device=points.device) + 0.5) / fit.scale - 0.5
    width, height = (fit.height, fit.width) if fit.quarter_turns % 2 else (fit.width, fit.height)
    for _ in range(fit.quarter_turns):  # undo one clockwise turn: (u, v) of a W x H image came from (v, W - 1 - u)
        uv = torch.stack((uv[:, 1], width - 1 - uv[:, 0]), dim=1)
        width, height = height, width
    return uv


def detnet_boxes(center: Float32[Tensor, "b 2 2"], radius: Float32[Tensor, "b 2"], presence_logit: Float32[Tensor, "b 2"], fit: Fit,
                 threshold: float) -> HandDetections:
    """Present slots as source-frame squares around their circles (scored by presence probability), left slot first."""
    probability: Float32[Tensor, "b 2"] = torch.sigmoid(presence_logit.float())
    keep: Bool[Tensor, "b 2"] = probability > threshold
    if not bool(keep.any()):
        return HandDetections.empty(center.device)
    frame, slot = torch.where(keep)
    centre: Float32[Tensor, "n 2"] = center[frame, slot].float() * torch.tensor([NET_WIDTH, NET_HEIGHT], device=center.device)
    half: Float32[Tensor, "n"] = radius[frame, slot].float() * NET_WIDTH / fit.scale
    source: Float32[Tensor, "n 2"] = to_source(centre, fit)
    xyxy: Float32[Tensor, "n 4"] = torch.cat((source - half[:, None], source + half[:, None]), dim=1)
    indices: Int64[Tensor, "n"] = frame.to(torch.int64)
    return HandDetections(BoxDetections(xyxy, probability[frame, slot], indices), tuple(SLOT_SIDES[int(s)] for s in slot.tolist()))


@dataclass
class DetNetHandDetectorConfig:
    """handtrack's DetNet-F hand detector (exported ``detnet_full.onnx``)."""

    weights: Path = Path("detnet_full.onnx")
    """``detnet_full.onnx`` from ``handtrack/tools/export_detnet.py``."""
    threshold: float = 0.5
    """Presence probability above which a slot is a hand."""
    quarter_turns: int = 0
    """Clockwise quarter turns that make this camera upright (the model is trained on upright frames; RoboCap cams 2-3: 2)."""
    backend: OnnxOrTrtBackendConfig = field(default_factory=OnnxBackendConfig)
    """ONNX Runtime reference or a cached TensorRT engine, dynamic batch."""

    def setup(self) -> "DetNetHandDetector":
        return DetNetHandDetector(self)


class DetNetHandDetector(HandDetector):
    """At most one left and one right hand per frame."""

    def __init__(self, config: DetNetHandDetectorConfig) -> None:
        self.config: DetNetHandDetectorConfig = config
        self.runtime: TensorRuntime = create_runtime_from_onnx(config.weights, config.backend)

    def __call__(self, frames_rgb: UInt8[Tensor, "b h w 3"]) -> HandDetections:
        validate_frames_rgb(frames_rgb)
        net, fit = detnet_input(frames_rgb, self.config.quarter_turns)
        outputs: dict[str, Tensor] = run_chunked(self.runtime, {"image": net.contiguous()})
        return detnet_boxes(outputs["center"], outputs["radius"], outputs["presence_logit"], fit, self.config.threshold)
