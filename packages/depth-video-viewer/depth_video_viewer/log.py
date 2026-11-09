"""Log a `DepthVideoStream` with the stock rerun-sdk.

The archetype exists only in the depth-video-viewer, so it is logged as built-in component types under its own
descriptors. They are the descriptors of the in-tree prototype (`rerun.archetypes.DepthVideoStream`), so one
recording works in both viewers. Out-of-tree scaffolding: a generated SDK archetype replaces this module once
`DepthVideoStream` ships in Rerun.
"""

from dataclasses import dataclass

import numpy as np
import rerun as rr
from jaxtyping import Int64
from numpy import ndarray

ARCHETYPE = "rerun.archetypes.DepthVideoStream"


def _described(batch: rr.ComponentBatchMixin, field: str) -> rr.DescribedComponentBatch:
    descriptor = rr.ComponentDescriptor(f"DepthVideoStream:{field}", archetype=ARCHETYPE, component_type=batch.component_type())
    return batch.described(descriptor)


@dataclass(frozen=True, slots=True)
class DepthVideoProperties:
    """The static part of a `DepthVideoStream`: how decoded codes map to depth and color."""

    meter: float
    """Decoded codes per meter. For codes in units of `u` millimetres this is `1000 / u`."""
    colormap: rr.components.Colormap | None = None
    """The viewer uses Rerun's depth colormap without it."""
    depth_range: tuple[float, float] | None = None
    """Visible range in codes; the viewer uses [0, 2^bit_depth - 1] without it."""
    point_fill_ratio: float | None = None


def static_components(props: DepthVideoProperties) -> list[rr.DescribedComponentBatch]:
    """Codec (AV1) and the depth properties, to log once with `static=True`."""
    batches = [
        _described(rr.components.VideoCodecBatch([rr.components.VideoCodec.AV1]), "codec"),
        _described(rr.components.DepthMeterBatch([props.meter]), "meter"),
    ]
    if props.colormap is not None:
        batches.append(_described(rr.components.ColormapBatch([props.colormap]), "colormap"))
    if props.depth_range is not None:
        batches.append(_described(rr.components.ValueRangeBatch([list(props.depth_range)]), "depth_range"))
    if props.point_fill_ratio is not None:
        batches.append(_described(rr.components.FillRatioBatch([props.point_fill_ratio]), "point_fill_ratio"))
    return batches


def log_depth_video(
    entity: str,
    samples: list[bytes],
    times_ns: Int64[ndarray, "n"],
    props: DepthVideoProperties,
    timeline: str = "video_time",
    recording: rr.RecordingStream | None = None,
) -> None:
    """Log one AV1 sample per frame at `times_ns` on a duration timeline, plus the static components.

    The viewer decodes samples in time order, so the timestamps must not decrease (equal ones keep their order).
    """
    if len(samples) != len(times_ns):
        raise ValueError(f"{len(samples)} samples for {len(times_ns)} timestamps")
    if (np.diff(times_ns) < 0).any():
        raise ValueError("timestamps decrease: the viewer would decode the samples out of order")
    if not props.meter > 0:
        raise ValueError(f"meter must be positive, got {props.meter}")
    rr.log(entity, static_components(props), static=True, recording=recording)
    sample = _described(rr.components.VideoSampleBatch(samples), "sample")
    rr.send_columns(
        entity,
        indexes=[rr.TimeColumn(timeline, duration=times_ns.astype("timedelta64[ns]"))],
        columns=[sample.partition()],
        recording=recording,
    )
