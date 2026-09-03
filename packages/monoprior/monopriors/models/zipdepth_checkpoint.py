"""Shared ZipDepth checkpoint decoding."""

from pathlib import Path

import torch
from torch import Tensor

from monopriors.third_party.zipdepth.model_utils import StateDict, strip_state_dict_prefixes


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


def load_zipdepth_state_dict(checkpoint: Path) -> StateDict:
    """Load, unwrap, and clean a bare or trainer ZipDepth state dictionary."""
    loaded: dict[str, object] = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(loaded, dict):
        raise ValueError(f"checkpoint must contain a mapping: {checkpoint}")
    raw_state: object = loaded.get("model_state_dict", loaded)
    state_dict: dict[str, Tensor] = narrow_zipdepth_state_dict(raw_state, checkpoint)
    return strip_state_dict_prefixes(state_dict)


__all__ = ("load_zipdepth_state_dict", "narrow_zipdepth_state_dict")
