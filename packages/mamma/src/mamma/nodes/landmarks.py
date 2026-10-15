"""Dense 2D landmarks of every tracked person in every view (MammaNet), cropped from the frames that travel with the tracks.

With the hi-res preset those are the source's ``frames_native``: crops sample the decoded size while masks and outputs
stay in engine pixels (the scale is the width ratio, as the original pipeline used).
"""

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

import numpy as np
import torch
from dagkit.node import Emit, Fenced, PortType, RunContext, Tick
from jaxtyping import Bool, Float32, Int64, UInt8
from numpy import ndarray
from simplecv.gpu import consume, fence_after
from simplecv.types.frames import MultiViewFrames
from torch import Tensor

from mamma.models.landmarks.estimator import DenseLandmarks
from mamma.types import MultiViewLandmarks, MultiViewTracks, TrackedFrameset


@runtime_checkable
class DenseLandmarker(Protocol):
    """What the node needs from ``models.landmarks.estimator.LandmarkEstimator`` (tests inject a fake)."""

    def __call__(
        self, frames: UInt8[Tensor, "v 3 hh ww"], view: Int64[ndarray, " k"], masks: Bool[Tensor, "k h w"], boxes_xyxy: Float32[ndarray, "k 4"], rgb_scale: float = 1.0
    ) -> DenseLandmarks: ...


class LandmarksNode:
    """Find each tracked person's 2D landmarks.

    MammaNet's dense landmarks (with log-variance, visibility and floor contact) per (view, person) track, cropped around
    the track's mask from the frames of the same tick.
    """

    inputs: Mapping[str, PortType] = {"tracked": Fenced[TrackedFrameset]}
    outputs: Mapping[str, PortType] = {"landmarks": Fenced[MultiViewLandmarks]}

    def __init__(self, estimator: DenseLandmarker) -> None:
        self.estimator: DenseLandmarker = estimator

    def reset(self, ctx: RunContext) -> None:
        pass

    def step(self, tick: Tick, out: Emit) -> None:
        assert isinstance(tick.value, Fenced) and isinstance(tick.value.value, TrackedFrameset)
        tracked: TrackedFrameset = consume(tick.value)
        tracks: MultiViewTracks = tracked.tracks
        frames: MultiViewFrames = tracked.crops
        rows: Int64[ndarray, " r"] = np.flatnonzero(np.isfinite(tracks.boxes_xyxy).all(axis=1))  # an empty mask has no crop
        device: torch.device = tracks.masks.device
        if rows.size:
            dense: DenseLandmarks = self.estimator(
                frames.rgb, tracks.view[rows], tracks.masks[torch.as_tensor(rows, device=device)], tracks.boxes_xyxy[rows], frames.rgb.shape[3] / tracks.masks.shape[2]
            )
        else:
            empty: Float32[Tensor, "0 0"] = torch.zeros((0, 0), device=device)
            dense = DenseLandmarks(empty[..., None].expand(0, 0, 2), empty, empty, empty)
        value = MultiViewLandmarks(
            tracks.frame_index, tracks.views, tracks.view[rows], tracks.person[rows], dense.xy, dense.log_variance, dense.visibility, dense.floor_contact, tracks.scale_xy
        )  # fmt: skip
        out.put("landmarks", tick.t_ns, Fenced(value, fence_after(value.xy) if rows.size else fence_after(tracks.masks)))

    def flush(self, out: Emit) -> None:
        pass
