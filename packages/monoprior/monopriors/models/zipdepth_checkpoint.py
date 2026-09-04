"""Shared ZipDepth checkpoint decoding."""

from pathlib import Path

import torch
from torch import Tensor

from monopriors.third_party.zipdepth.model_utils import StateDict, strip_state_dict_prefixes

RANGE_MARGIN_M_KEY: str = "range_margin_m"
"""Trainer-checkpoint key holding the prompted head's output-range margin, in metres."""

RANGE_MARGIN_COVERAGE_MAX_KEY: str = "range_margin_coverage_max"
"""Trainer-checkpoint key holding the prompt coverage at or below which the margin applies."""


def load_zipdepth_checkpoint(checkpoint: Path) -> dict[str, object]:
    """Load a bare state dict or a trainer checkpoint as a raw mapping.

    Args:
        checkpoint: Path to a ``.pth`` written by ``torch.save``.

    Returns:
        The deserialized top-level mapping.
    """
    loaded: dict[str, object] = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(loaded, dict):
        raise ValueError(f"checkpoint must contain a mapping: {checkpoint}")
    return loaded


def narrow_zipdepth_state_dict(raw_state: object, checkpoint: Path) -> dict[str, Tensor]:
    """Validate and type a serialized ZipDepth model state mapping."""
    if not isinstance(raw_state, dict):
        raise ValueError(f"checkpoint model state must contain a mapping: {checkpoint}")
    state_dict: dict[str, Tensor] = {}
    for key, value in raw_state.items():
        if not isinstance(key, str):
            raise ValueError(f"checkpoint model state has non-string key {key!r}: {checkpoint}")
        if not isinstance(value, Tensor):
            raise ValueError(f"checkpoint model state key {key!r} must contain a Tensor: {checkpoint}")
        state_dict[key] = value
    return state_dict


def state_dict_from_checkpoint(loaded: dict[str, object], checkpoint: Path) -> StateDict:
    """Unwrap and clean the model state held by an already-loaded checkpoint."""
    raw_state: object = loaded.get("model_state_dict", loaded)
    state_dict: dict[str, Tensor] = narrow_zipdepth_state_dict(raw_state, checkpoint)
    return strip_state_dict_prefixes(state_dict)


def _float_from_checkpoint(loaded: dict[str, object], checkpoint: Path, key: str, default: float) -> float:
    """Read one optional numeric scalar recorded beside the model state.

    Args:
        loaded: Mapping returned by :func:`load_zipdepth_checkpoint`.
        checkpoint: Source path, used only for error messages.
        key: Top-level key to read.
        default: Value used when the key is absent.

    Returns:
        The recorded number as a float, or ``default``.

    Raises:
        ValueError: If the key holds anything but a non-boolean number.
    """
    raw_value: object = loaded.get(key, default)
    if isinstance(raw_value, bool) or not isinstance(raw_value, float | int):
        raise ValueError(f"checkpoint {key} must be a number: {checkpoint}")
    return float(raw_value)


def range_margin_m_from_checkpoint(loaded: dict[str, object], checkpoint: Path) -> float:
    """Read the prompted output-range margin, in metres, recorded by training.

    Args:
        loaded: Mapping returned by :func:`load_zipdepth_checkpoint`.
        checkpoint: Source path, used only for error messages.

    Returns:
        The recorded margin in metres, or ``0.0`` for bare state dicts and for
        checkpoints written before the margin existed (for example
        ``zdpda-v4``).
    """
    return _float_from_checkpoint(loaded, checkpoint, RANGE_MARGIN_M_KEY, 0.0)


def range_margin_coverage_max_from_checkpoint(loaded: dict[str, object], checkpoint: Path) -> float:
    """Read the prompt-coverage ceiling that gates the output-range margin.

    Args:
        loaded: Mapping returned by :func:`load_zipdepth_checkpoint`.
        checkpoint: Source path, used only for error messages.

    Returns:
        The recorded ceiling, or ``1.0`` for bare state dicts and for checkpoints
        written before the gate existed. ``1.0`` is the whole canvas, so an
        ungated checkpoint keeps applying its margin to every image, exactly as
        it did before the gate.
    """
    return _float_from_checkpoint(loaded, checkpoint, RANGE_MARGIN_COVERAGE_MAX_KEY, 1.0)


def load_zipdepth_state_dict(checkpoint: Path) -> StateDict:
    """Load, unwrap, and clean a bare or trainer ZipDepth state dictionary."""
    return state_dict_from_checkpoint(load_zipdepth_checkpoint(checkpoint), checkpoint)


__all__ = (
    "RANGE_MARGIN_COVERAGE_MAX_KEY",
    "RANGE_MARGIN_M_KEY",
    "load_zipdepth_checkpoint",
    "load_zipdepth_state_dict",
    "narrow_zipdepth_state_dict",
    "range_margin_coverage_max_from_checkpoint",
    "range_margin_m_from_checkpoint",
    "state_dict_from_checkpoint",
)
