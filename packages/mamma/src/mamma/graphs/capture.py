"""The mamma capture graph (SPEC 5.2): decode -> track -> landmarks -> fit, every output to the sinks.

```
video ─frames (Block 2)──────────────▶ track ─tracks (Block 2)─▶ landmarks ─landmarks (Block 4)─▶ fit ─fit, kp3d, topology─▶ sinks
  ├─rig (AsOf, STATIC)──────────────────▶ track, fit                 ▲                                  (tracks, landmarks too)
  └─frames_native | frames (AsOf, keep 4)────────────────────────────┘
```
Threaded runner: lossless for eval and layers, realtime for live viewing. Models are built by the App and passed to the nodes.
"""

from collections.abc import Mapping

from dagkit.graph import AsOf, Block, Graph, PortRef
from dagkit.node import Node, Source

from mamma.nodes.fit import FitNode
from mamma.nodes.landmarks import LandmarksNode
from mamma.nodes.track import TrackNode

FRAMES_KEPT: int = 4
"""Realtime AsOf history of the crop frames: a native 4-view frameset is ~37 MB of GPU memory."""


def capture_graph(video: Source, track: TrackNode, landmarks: LandmarksNode, fit: FitNode, sinks: Mapping[str, Node]) -> Graph:
    """``video`` provides ``frames``, ``rig`` and, for hi-res crops, ``frames_native``; every sink input names a producer port."""
    graph = Graph("mamma-capture")
    source = graph.source("video", video)
    tracked = graph.node("track", track)
    landmarked = graph.node("landmarks", landmarks)
    fitted = graph.node("fit", fit)
    graph.edge(source["frames"], tracked["frames"], Block(2))
    graph.edge(source["rig"], tracked["rig"], AsOf())
    graph.edge(tracked["tracks"], landmarked["tracks"], Block(2))
    graph.edge(source["frames_native" if "frames_native" in video.outputs else "frames"], landmarked["frames"], AsOf(keep=FRAMES_KEPT))
    graph.edge(landmarked["landmarks"], fitted["landmarks"], Block(4))
    graph.edge(source["rig"], fitted["rig"], AsOf())
    ports: dict[str, PortRef] = {"tracks": tracked["tracks"], "landmarks": landmarked["landmarks"]} | {port: fitted[port] for port in FitNode.outputs}
    for name, sink in sinks.items():
        ref = graph.sink(name, sink)
        for port in sink.inputs:
            graph.edge(ports[port], ref[port])
    return graph
