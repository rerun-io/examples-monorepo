"""Replay ViPE's frozen DINOv3 backbone from CUDA graphs during one MetricTrail run.

DPVO's own encoders and recurrent update replay inside :func:`dpvo.graph_replay.graph_replay`; the DINOv3 backbone of
ViPE's DINO-fused patchifier is ViPE's class, so this wraps its tensor extractor (``FrozenDinoV3._extract_chunk``; the
forward around it does PCA and host logging, which stay outside the graph) and routes it through the same bounded cache.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from functools import wraps

from dpvo.graph_replay import replay_features
from jaxtyping import Float
from torch import Tensor
from vipe.metrictrail.dino_fusion import FrozenDinoV3


@contextmanager
def dino_cuda_graphs() -> Iterator[None]:
    """Replay DINOv3's feature extraction while a :func:`~dpvo.graph_replay.graph_replay` context is open (else eager),
    restoring ViPE's method on exit."""
    original = FrozenDinoV3._extract_chunk

    @wraps(original)
    def extract(self: FrozenDinoV3, images: Float[Tensor, "b 3 h w"]) -> Float[Tensor, "b c ph pw"]:
        replayed: Tensor | None = replay_features(self, images, original)
        return replayed if replayed is not None else original(self, images)

    FrozenDinoV3._extract_chunk = extract
    try:
        yield
    finally:
        FrozenDinoV3._extract_chunk = original
