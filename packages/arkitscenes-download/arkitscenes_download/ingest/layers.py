"""Declarative definitions of the published RRD layers."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Layer:
    """One independently published RRD layer."""

    name: str
    """Layer file stem and catalog layer name."""
    send_properties: bool
    """Whether recording properties belong in this layer."""


# The base layer also embeds the per-sequence default blueprint (see ingest.cli).
LAYERS: tuple[Layer, ...] = (
    Layer("base", True),
    Layer("calibration", False),
    Layer("video_wide", False),
    Layer("video_ultrawide", False),
    Layer("depth", False),
    Layer("imu", False),
    Layer("gt", False),
)
LAYER_NAMES: tuple[str, ...] = tuple(layer.name for layer in LAYERS)
LAYERS_BY_NAME: dict[str, Layer] = {layer.name: layer for layer in LAYERS}
