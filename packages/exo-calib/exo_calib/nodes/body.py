"""Stage F (whole segment, last pass): the person's full SMPL-X body through the refined rig, the capture the calibration is for.

The node gates and undistorts the Stage B keypoints as the refinement does, keeps the framesets that see the body in at least
two views, and fits one body to them (resource ``smplx``: ``ops.body.SmplxFitter``); each frameset's mesh shows on its video
frame and clears at the next (resource ``frames``).
"""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, cast

import numpy as np
from dagkit.node import Emit, Event, PortType, RunContext
from dagkit.time import STATIC
from jaxtyping import Float64, Int64
from numpy import ndarray
from simplecv.ops.framesets import clear_times
from simplecv.types.rig import RigCameras

from exo_calib.ops.body import BodyFit
from exo_calib.ops.confidence import sees_body
from exo_calib.ops.keypoints import GatedKeypoints, stack_postprocessed, undistort_views
from exo_calib.types import BodyPose, BodyShape, MultiViewKeypoints, RigLayout

if TYPE_CHECKING:
    from simplecv.sources.catalog_video import SegmentFrames

    from exo_calib.ops.body import SmplxFitter


class BodyNode:
    """Fit the person's SMPL-X body."""

    inputs: Mapping[str, PortType] = {"kp2d": MultiViewKeypoints, "rig": RigCameras}
    outputs: Mapping[str, PortType] = {"body_shape": BodyShape, "body": BodyPose}

    def __init__(self, layout: RigLayout) -> None:
        self.layout: RigLayout = layout
        """The cameras' lenses: the fit works in each camera's pinhole pixels."""

    def reset(self, ctx: RunContext) -> None:
        self.fitter, self.frames = cast("SmplxFitter", ctx.resources["smplx"]), cast("SegmentFrames", ctx.resources["frames"])

    def run(self, inputs: Mapping[str, Sequence[Event]], out: Emit) -> None:
        records: list[MultiViewKeypoints] = [e.value for e in inputs["kp2d"]]  # type: ignore[misc]
        times: list[int] = [e.t_ns for e in inputs["kp2d"]]
        rig = inputs["rig"][-1].value
        assert isinstance(rig, RigCameras) and records, "the body needs Stage B records and the refined rig"
        rig = rig.select(records[0].views)
        gated: GatedKeypoints = stack_postprocessed(records)
        xy: Float64[ndarray, "v t 133 2"] = undistort_views([self.layout.lens(name) for name in rig.names], gated.xy, rig.intrinsics)
        fitted: Int64[ndarray, " f"] = np.flatnonzero(sees_body(gated.conf).sum(axis=0) >= 2)
        if not fitted.size:
            print(f"body: none of {len(records)} framesets sees the body in two views; nothing fitted")
            return
        fit: BodyFit = self.fitter(xy[:, fitted], gated.conf[:, fitted], rig.intrinsics, rig.cam_T_world)
        print(f"body: fitted {fitted.size} of {len(records)} framesets")
        out.put("body_shape", STATIC, BodyShape(fit.betas, self.fitter.faces))
        # A frameset's time and frame_index are the first camera's; only a fitted frameset replaces the body.
        clears: list[int | None] = clear_times(self.frames.times_ns(0), np.array([records[t].frame_index for t in fitted.tolist()], dtype=np.int64))
        for k, (t, clear_ns) in enumerate(zip(fitted.tolist(), clears, strict=True)):
            out.put("body", times[t], BodyPose(fit.pose[k], fit.transl[k], fit.vertices[k]))
            if clear_ns is not None:
                out.put("body", clear_ns, BodyPose.empty())
