"""The NVS methods of the run layout (docs/run-layout.md): one typed row per method, so no code parses a method name."""

from dataclasses import dataclass
from typing import Literal, TypeAlias

from hexels_bench.hexel_ply import HexelVariant, property_names

Method: TypeAlias = Literal[
    "brush-7k",
    "brush-7k-wgsl",
    "hexels-v1-7k",
    "hexels-v3-7k",
    "hexels-v1-30k-pub-cuda",
    "hexels-v2-30k-pub-cuda",
    "hexels-v3-30k-pub-cuda",
    "hexels-v1-30k-pub-wgsl",
    "hexels-v2-30k-pub-wgsl",
    "hexels-v3-30k-pub-wgsl",
    "brush-smoke",
]
SPLAT_FLOATS: int = 59
"""Learned floats of a 3DGS splat with SH degree 3: position 3, scale 3, rotation 4, opacity 1, SH 48."""


def primitive_bytes(variant: HexelVariant | None) -> int:
    """FP32 learned bytes per primitive (Deploy MB): one hexel PLY row of `variant`, or a 3DGS splat for None."""
    return 4 * (SPLAT_FLOATS if variant is None else len(property_names(variant)))


@dataclass(frozen=True, slots=True)
class MethodSpec:
    """What a method name stands for."""

    variant: HexelVariant | None
    """Hexels appearance variant; None for Brush 3DGS."""
    steps: int
    """Optimisation steps of the model."""
    published: bool
    """The model is a published 30k checkpoint (HF jhu-ccvl/Hexels), not one we trained."""
    renders_from: Method | None
    """WGSL re-render: the run whose model.ply and cameras.json it renders; None when the run renders its own model."""

    @property
    def trains(self) -> bool:
        """We train this model, so the run also holds `train.json`."""
        return not self.published and self.renders_from is None


METHODS: dict[Method, MethodSpec] = {
    "brush-7k": MethodSpec(variant=None, steps=7000, published=False, renders_from=None),
    "brush-7k-wgsl": MethodSpec(variant=None, steps=7000, published=False, renders_from="brush-7k"),
    "hexels-v1-7k": MethodSpec(variant="v1", steps=7000, published=False, renders_from=None),
    "hexels-v3-7k": MethodSpec(variant="v3", steps=7000, published=False, renders_from=None),
    "hexels-v1-30k-pub-cuda": MethodSpec(variant="v1", steps=30000, published=True, renders_from=None),
    "hexels-v2-30k-pub-cuda": MethodSpec(variant="v2", steps=30000, published=True, renders_from=None),
    "hexels-v3-30k-pub-cuda": MethodSpec(variant="v3", steps=30000, published=True, renders_from=None),
    "hexels-v1-30k-pub-wgsl": MethodSpec(variant="v1", steps=30000, published=True, renders_from="hexels-v1-30k-pub-cuda"),
    "hexels-v2-30k-pub-wgsl": MethodSpec(variant="v2", steps=30000, published=True, renders_from="hexels-v2-30k-pub-cuda"),
    "hexels-v3-30k-pub-wgsl": MethodSpec(variant="v3", steps=30000, published=True, renders_from="hexels-v3-30k-pub-cuda"),
    "brush-smoke": MethodSpec(variant=None, steps=30, published=False, renders_from=None),
}
"""In queue order within a scene: a WGSL re-render follows the run it re-renders. `brush-smoke` is a 30-step check of
the Brush protocol path, outside the benchmark."""
BENCHMARK_METHODS: tuple[Method, ...] = tuple(method for method in METHODS if method != "brush-smoke")
"""The methods the queue runs and the table reports by default."""
