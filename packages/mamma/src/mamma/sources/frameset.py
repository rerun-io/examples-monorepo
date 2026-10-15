"""The capture graph's source: a ``VideoSource`` whose framesets carry the frames landmarks crop from."""

from collections.abc import Iterator

from dagkit.node import Event, Fenced, PortType
from simplecv.sources.catalog_video import VideoSource

from mamma.types import Frameset


class FramesetSource:
    """Decode the cameras' videos.

    The video source's STATIC values, then one ``Frameset`` per sampled index: the engine-size frames and the frames
    landmarks crop from (the full-size ``frames_native`` when the source keeps it, else the same frames), so the crops travel
    with their tick through tracking.
    """

    def __init__(self, video: VideoSource) -> None:
        self.video, self.clock, self.cameras = video, video.clock, video.cameras
        self.outputs: dict[str, PortType] = {"frames": Fenced[Frameset]} | {p: t for p, t in video.outputs.items() if p not in ("frames", "frames_native")}

    def prefetch(self) -> None:
        """Read every camera's packets now (a catalog read), so a run's clock starts at the first decode, as the pre-port
        export's did."""
        for view in range(len(self.cameras)):
            self.video.frames.times_ns(view)

    def events(self) -> Iterator[Event]:
        native: Fenced | None = None
        for event in self.video.events():
            if event.port not in ("frames", "frames_native"):
                yield event
                continue
            assert isinstance(event.value, Fenced), "mamma decodes on the GPU (fenced frames)"
            if event.port == "frames_native":  # the video source emits it just before the frames of the same time
                native = event.value
            else:
                yield Event(event.t_ns, "frames", Fenced(Frameset(event.value.value, (native or event.value).value), event.value.fence))

    def close(self) -> None:
        self.video.close()
