"""Declarative definitions of the published RRD layers."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Layer:
    """One independently published RRD layer."""

    name: str
    """Layer file stem and catalog layer name."""
    send_properties: bool
    """Whether recording properties belong in this layer."""
    embed_blueprint: bool
    """Whether the default blueprint belongs in this layer."""


LAYERS: tuple[Layer, ...] = (
    Layer("base", True, True),
    Layer("calibration", False, False),
    Layer("video_wide", False, False),
    Layer("video_ultrawide", False, False),
    Layer("depth", False, False),
    Layer("imu", False, False),
    Layer("gt", False, False),
)
LAYER_NAMES: tuple[str, ...] = tuple(layer.name for layer in LAYERS)
LAYERS_BY_NAME: dict[str, Layer] = {layer.name: layer for layer in LAYERS}
