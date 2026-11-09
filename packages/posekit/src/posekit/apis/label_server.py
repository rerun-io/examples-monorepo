"""Hand labeler backend: serve the review UI and the labeling API for the segments of one catalog dataset.

Frames come from ``posekit.labeler.frames.FrameSource``, proposals from ``posekit.labeler.proposals.read_proposals``
(read once per segment when it opens), labels from and to the ``hand_boxes`` layer (``posekit.labeler.store``).

SIGTERM (``systemctl stop``) and Ctrl-C stop the server the same way: no new writes, then every unsaved label goes
to its segment's autosave file before the process exits.
"""

from __future__ import annotations

import signal
import threading
from dataclasses import dataclass, field
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import FrameType
from typing import Literal, TypeAlias

import cv2
from rerun.catalog import DatasetEntry

from posekit.labeler.carry import OPENCV_THREADS, VisualCarrier, VisualTrackerKind, opencv_tracker
from posekit.labeler.frames import FrameSource
from posekit.labeler.proposals import ProposalsByTime, read_proposals
from posekit.labeler.schema import Proposal
from posekit.labeler.server import LabelApp, make_server
from posekit.labeler.store import LAYER_DIR, CatalogLayers
from posekit.labeler.tracker import Propagator

PropagatorKind: TypeAlias = Literal["chain", "megatrack"]
"""Machine propagation: ``chain`` (EgoHand chains, ``posekit.labeler.tracker``; the default) or ``megatrack`` (a 3D hand per side
from KeyNet-F in every view, ``posekit.labeler.megatrack``; needs ``PYTHONPATH=packages/handtrack`` and a GPU)."""
KEYNET_WEIGHTS: Path = Path.home() / "hand-labeler-data" / "keynet" / "keynet-strong-pinch.weights.pt"
"""RoboCap's deployed KeyNet-F (sha256 067cf7f6...), copied from pablo-dl-server ``handtrack-data/runs/keynet-strong-pinch``."""

STATIC_DIR: Path = Path(__file__).resolve().parents[1] / "labeler" / "static"
"""The frontend shipped with the package."""


@dataclass
class Config:
    """Hand labeler backend."""

    catalog_url: str = "rerun+http://127.0.0.1:9897"
    """Catalog that holds the dataset (and receives the ``hand_boxes`` layer)."""
    dataset: str = "robocap-labeling"
    """Catalog dataset."""
    host: str = "0.0.0.0"
    """Interface to listen on."""
    port: int = 9898
    """Port to listen on."""
    static_dir: Path = STATIC_DIR
    """Frontend directory (``index.html``, ``app.js``, ``app.css``)."""
    reviewer: str = "pablo"
    """Default reviewer name the browser starts with."""
    layer_dir: Path = LAYER_DIR
    """Where committed and autosaved ``hand_boxes`` layer files go."""
    segments: tuple[str, ...] = ()
    """Segments to offer; empty = every segment of the dataset."""
    eval_segments: tuple[str, ...] = ("robocap__f408193e6447b3b0__s00000016", "robocap__fe6fede545c972fa__s00000013")
    """Held-out eval segments; all others are train."""
    viewer_catalog_url: str | None = None
    """Catalog URL the browser's web viewer uses (e.g. a tailnet name); None = ``catalog_url``."""
    autosave_every: int = 50
    """New labels between autosaves."""
    warm: bool = True
    """Open every segment in the background at start, so the first requests do not wait on the catalog."""
    visual_tracker: VisualTrackerKind = "off"
    """OpenCV tracker that carries reviewed boxes when EgoHand misses (origin ``tracked``, ``posekit.labeler.carry``);
    ``off`` = chains hold one miss and end."""
    propagator: PropagatorKind = "chain"
    """``chain`` (default, uses ``visual_tracker``) or ``megatrack`` (``visual_tracker`` is then unused)."""
    keynet_weights: Path = KEYNET_WEIGHTS
    """KeyNet-F weights for ``--propagator megatrack``."""
    keynet_device: str = "cuda"
    """Torch device of KeyNet for ``--propagator megatrack``."""


@dataclass
class CatalogProposals:
    """Proposals per segment, read from the catalog when the segment's frames open."""

    dataset: DatasetEntry
    """Labeling dataset."""
    by_segment: dict[str, ProposalsByTime] = field(default_factory=dict)
    """Proposals by grid time, per opened segment."""

    def load(self, source: FrameSource) -> None:
        """Read every proposal of the source's segment (one catalog round trip)."""
        self.by_segment[source.segment_id] = read_proposals(self.dataset, source.segment_id, source.cams, source.times_ns)

    def proposals(self, segment_id: str, t_ns: int) -> list[Proposal]:
        """Every proposal at one grid time (empty for an unopened segment or a time without any)."""
        return self.by_segment.get(segment_id, {}).get(t_ns, [])


def main(config: Config) -> None:
    """Open the catalog, start the server and serve until interrupted."""
    layers: CatalogLayers = CatalogLayers(config.catalog_url, config.dataset, config.layer_dir)
    dataset: DatasetEntry = layers.dataset
    segment_ids: list[str] = list(config.segments) or sorted(dataset.segment_ids())
    proposals: CatalogProposals = CatalogProposals(dataset)
    if config.visual_tracker != "off":
        cv2.setNumThreads(OPENCV_THREADS)
    sources: dict[str, FrameSource] = {}

    def open_frames(segment_id: str) -> FrameSource:
        source: FrameSource = FrameSource.from_catalog(dataset, segment_id)
        proposals.load(source)
        sources[segment_id] = source
        return source

    def open_carrier(segment_id: str) -> VisualCarrier:
        source: FrameSource = sources[segment_id]
        return VisualCarrier(source, source.times_ns, opencv_tracker(config.visual_tracker))

    def open_megatrack(segment_id: str) -> Propagator:
        # Imported here: posekit.labeler.keynet needs handtrack on PYTHONPATH, which the default propagator does not.
        from posekit.labeler.keynet import KeyNetEstimator
        from posekit.labeler.megatrack import MegaTrackPropagator
        from posekit.labeler.rig import Kb4Rig, read_rig

        rig: Kb4Rig = read_rig(dataset, segment_id)
        return MegaTrackPropagator(rig, sources[segment_id], KeyNetEstimator(rig, config.keynet_weights, device=config.keynet_device))

    def viewer_url(segment_id: str) -> str:
        url: str = dataset.segment_url(segment_id)
        if config.viewer_catalog_url is not None:
            url = url.replace(config.catalog_url.rstrip("/"), config.viewer_catalog_url.rstrip("/"), 1)
        return url

    app: LabelApp = LabelApp(
        segment_ids=segment_ids,
        open_frames=open_frames,
        proposals=proposals,
        layers=layers,
        viewer_url=viewer_url,
        eval_segments=frozenset(config.eval_segments),
        static_dir=config.static_dir,
        reviewer=config.reviewer,
        dataset=config.dataset,
        autosave_every=config.autosave_every,
        open_carrier=None if config.visual_tracker == "off" or config.propagator == "megatrack" else open_carrier,
        open_propagator=open_megatrack if config.propagator == "megatrack" else None,
    )
    if config.warm:
        threading.Thread(target=app.summaries, name="labeler-warm", daemon=True).start()
    server: ThreadingHTTPServer = make_server(app, config.host, config.port)

    def stop_on_sigterm(signum: int, frame: FrameType | None) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop_on_sigterm)
    print(
        f"[labeler] {len(segment_ids)} segments of {config.dataset} on http://{config.host}:{server.server_address[1]}/"
        f" (propagator: {config.propagator}, visual tracker: {config.visual_tracker if config.propagator == 'chain' else 'unused'})"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[labeler] stopping")
    finally:
        # A second signal (systemd and pixi may both deliver one) must not interrupt the final save.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        server.server_close()
        app.close()
        print("[labeler] stopped")
