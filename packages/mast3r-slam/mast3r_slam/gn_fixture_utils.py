from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


GN_CAPTURE_DIR_ENV = "MAST3R_SLAM_GN_CAPTURE_DIR"
GN_CAPTURE_LIMIT_ENV = "MAST3R_SLAM_GN_CAPTURE_LIMIT"


def _clone_for_capture(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().cpu().contiguous()
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return type(value)(_clone_for_capture(item) for item in value)
    if isinstance(value, dict):
        return {key: _clone_for_capture(item) for key, item in value.items()}
    return value


def _capture_dir() -> Path | None:
    capture_dir = os.environ.get(GN_CAPTURE_DIR_ENV)
    if not capture_dir:
        return None
    return Path(capture_dir)


def _capture_limit() -> int:
    raw_limit = os.environ.get(GN_CAPTURE_LIMIT_ENV, "1")
    try:
        return max(int(raw_limit), 0)
    except ValueError:
        return 1


def _next_capture_path(capture_dir: Path, kind: str) -> Path | None:
    capture_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(capture_dir.glob(f"{kind}-*.pt"))
    limit = _capture_limit()
    if limit and len(existing) >= limit:
        return None
    return capture_dir / f"{kind}-{len(existing):03d}.pt"


def maybe_capture_gn_fixture(kind: str, inputs: dict[str, Any], metadata: dict[str, Any] | None = None) -> Path | None:
    capture_dir = _capture_dir()
    if capture_dir is None:
        return None

    output_path = _next_capture_path(capture_dir, kind)
    if output_path is None:
        return None

    payload = {
        "kind": kind,
        "captured_at_unix_s": time.time(),
        "metadata": _clone_for_capture(metadata or {}),
        "inputs": _clone_for_capture(inputs),
    }
    torch.save(payload, output_path)
    return output_path


def load_gn_fixture(path: str | Path) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict payload in fixture file, got {type(payload)!r}")
    return payload
