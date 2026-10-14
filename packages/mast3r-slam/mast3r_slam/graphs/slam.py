"""The SLAM graph (SPEC 5.5).

```
video ─frames─▶ frontend (main thread) ─task─▶ backend (own process) ─map, keyframes─▶ sinks
  └─rig (AsOf, calibrated only)─▶ ▲  └─tracked──────────────────────────────────────────▶ sinks
```
``frames`` is Block(2) (DropOldest(1) for a live camera), ``task`` Block(16). Resources: ``mast3r`` (the model, shared
with the backend by CUDA IPC) and ``shared`` (keyframe buffer and current-frame state: the feedback path of the SLAM loop,
so the graph stays acyclic). Lossless determinism comes from the frontend's acknowledgement wait, not from the runner:
this graph is outside dagkit's runner-equivalence promise; the verify band covers it.
"""

from collections.abc import Mapping

from dagkit.graph import AsOf, Block, DropOldest, Graph, NodeRef, Placement, PortRef, Spawn
from dagkit.node import Node, Source
from mast3r.model import AsymmetricMASt3R

from mast3r_slam.models.mast3r import load_mast3r
from mast3r_slam.nodes.backend import Backend
from mast3r_slam.nodes.frontend import Frontend
from mast3r_slam.nodes.nerfstudio import NerfstudioSink
from mast3r_slam.nodes.shared import SharedSlam
from mast3r_slam.types import SlamConfig

DEVICE: str = "cuda:0"


def shared_model() -> AsymmetricMASt3R:
    """The ``mast3r`` resource: weights on the GPU, shared with the backend process."""
    model: AsymmetricMASt3R = load_mast3r(device=DEVICE)
    model.share_memory()
    return model


def slam_graph(
    video: Source, cfg: SlamConfig, frame_hw: tuple[int, int], sinks: Mapping[str, Node], *, live: bool = False, nerfstudio: NerfstudioSink | None = None
) -> Graph:
    """Sinks name the ports they take: ``tracked`` and ``keyframes`` (TrackedFrame), ``map`` (MapUpdate).

    ``frame_hw`` is the MASt3R frame size of the input (it sizes the shared buffers before the first frame arrives).
    """
    graph = Graph("mast3r-slam")
    graph.resource("mast3r", shared_model)
    graph.resource("shared", lambda: SharedSlam(*frame_hw, cfg.keyframe_buffer, cfg.use_calib, device=DEVICE))
    source: NodeRef = graph.source("video", video)
    frontend_node = Frontend(cfg, device=DEVICE)
    frontend: NodeRef = graph.node("frontend", frontend_node, placement=Placement("main"), needs=("mast3r", "shared"))
    backend: NodeRef = graph.node("backend", Spawn(Backend, cfg), placement=Placement("process"), needs=("mast3r", "shared"))
    graph.edge(source["frames"], frontend["frames"], DropOldest(1) if live else Block(2))
    if "rig" in frontend_node.inputs:
        graph.edge(source["rig"], frontend["rig"], AsOf())
    graph.edge(frontend["task"], backend["task"], Block(16))
    ports: dict[str, PortRef] = {"tracked": frontend["tracked"], "map": backend["map"], "keyframes": backend["keyframes"]}
    for name, sink in sinks.items():
        ref: NodeRef = graph.sink(name, sink)
        for port in sink.inputs:
            graph.edge(ports[port], ref[port])
    if nerfstudio is not None:  # a plain node, not a sink: it reads the final keyframes from the `shared` resource
        graph.edge(backend["keyframes"], graph.node("nerfstudio", nerfstudio, needs=("shared",))["keyframes"])
    return graph
