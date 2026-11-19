"""KeyNet-F on KB4 perspective crops against the RoboCap ``hands`` layer (the same network, run by robocap-live) on Cap B s13."""

from __future__ import annotations

import socket
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pytest

pytest.importorskip("handtrack", reason="posekit.labeler.keynet imports handtrack: run with PYTHONPATH=packages/handtrack")

CATALOG_URL: str = "rerun+http://127.0.0.1:9897"
SEGMENT: str = "robocap__fe6fede545c972fa__s00000013"
WEIGHTS: Path = Path.home() / "hand-labeler-data" / "keynet" / "keynet-strong-pinch.weights.pt"


def _catalog_reachable(url: str) -> bool:
    parsed = urlparse(url.replace("rerun+", ""))
    try:
        with socket.create_connection((parsed.hostname, parsed.port), timeout=1.0):
            return True
    except OSError:
        return False


@pytest.mark.integration
def test_keynet_on_kb4_crops_matches_the_hands_layer_and_triangulates() -> None:
    if not WEIGHTS.exists():
        pytest.skip(f"KeyNet-F weights {WEIGHTS} are missing (copy keynet-strong-pinch/best.weights.pt from pablo-dl-server)")
    if not _catalog_reachable(CATALOG_URL):
        pytest.skip(f"labeling catalog {CATALOG_URL} (dataset robocap-labeling) is not reachable")
    import torch
    from rerun.catalog import CatalogClient

    from posekit.labeler.frames import FrameSource
    from posekit.labeler.keynet import KeyNetEstimator
    from posekit.labeler.megatrack import CropRequest
    from posekit.labeler.proposals import read_proposals
    from posekit.labeler.rig import fuse_views, read_rig
    from posekit.labeler.schema import Proposal

    if not torch.cuda.is_available():
        pytest.skip("KeyNet runs on the GPU")
    dataset = CatalogClient(CATALOG_URL).get_dataset("robocap-labeling")
    rig = read_rig(dataset, SEGMENT)
    assert rig.names == ("left_front", "right_front", "left_eye", "right_eye", "left", "right")
    source = FrameSource.from_catalog(dataset, SEGMENT)
    proposals = read_proposals(dataset, SEGMENT, source.cams, source.times_ns)
    estimator = KeyNetEstimator(rig, WEIGHTS)
    errors: list[float] = []
    reprojection: list[float] = []
    for t_ns in source.times_ns[100:150:7]:
        # The hands layer's KeyNet boxes (keypoint bounding box + margin) as the crops' seeds, side by side.
        layer: list[Proposal] = [p for p in proposals[t_ns] if p.model == "hands_layer"]
        for side in ("left", "right"):
            seeds: list[Proposal] = [p for p in layer if p.side == side]
            if len(seeds) < 2:
                continue
            frames = {p.cam: source.frame(p.cam, t_ns) for p in seeds}
            views = [v for v in estimator.estimate(frames, [CropRequest(p.cam, side, box=p.box) for p in seeds]) if v is not None]
            for v, p in zip(views, seeds, strict=False):
                centre = np.array([(p.box.x0 + p.box.x1) / 2.0, (p.box.y0 + p.box.y1) / 2.0])
                errors.append(float(np.linalg.norm(v.uv.mean(axis=0) - centre)))
            fused = fuse_views(rig, [v for v in views if v.presence >= 0.5])
            if fused is not None:
                reprojection.extend(fused.errors_px[cam] for cam in fused.inliers)
    assert len(errors) >= 8 and len(reprojection) >= 4
    # Same network, other crops and no prior: the keypoints' centre stays on the hands layer's hand, and two KB4 views agree.
    assert float(np.median(errors)) < 25.0
    assert float(np.median(reprojection)) < 8.0
