"""Owned headless viewers and settled screenshots: the pixel evidence for the canonical viewer's goldens."""

from __future__ import annotations

import os
import socket
import sys
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import UInt8
from PIL import Image
from rerun.experimental import ViewerClient

PACKAGE_DIR: Path = Path(__file__).resolve().parents[1]
CANONICAL_VIEWER: Path = PACKAGE_DIR / "target" / "release" / "rerun-canonical"
"""The release binary that ``canonical-viewer-build`` writes."""
STOCK_VIEWER: Path = Path(sys.executable).parent / "rerun"
"""This environment's stock viewer, from the conda rerun-sdk at the same Rerun release."""


def free_port() -> int:
    """A free loopback port for an owned viewer."""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def spawn_viewer(binary: Path) -> ViewerClient:
    """Start an owned headless viewer on a free port; the client is a context manager that stops it.

    The viewer must report the SDK's Rerun release, or its pixels prove nothing about this release. It runs with
    ``RUST_LOG=warn``: every INFO line is also a toast for about 4 s, and the startup one ("Headless viewer running at
    ...") lies over the top-right view, so two otherwise equal viewers differed by when they were captured.
    """
    with patch.dict(os.environ, {"RUST_LOG": "warn"}):
        viewer: ViewerClient = ViewerClient.spawn(headless=True, port=free_port(), executable_path=str(binary), hide_welcome_screen=True)
    version: str | None = viewer.get_state().viewer_version
    if version != rr.__version__:
        viewer.close()
        raise RuntimeError(f"{binary} is Rerun {version}; the SDK is {rr.__version__}")
    return viewer


def capture(viewer: ViewerClient, view: rrb.View, path: Path, timeout_s: float = 60.0) -> UInt8[np.ndarray, "h w 3"]:
    """Screenshot one view until two captures in a row are equal and not blank; return the settled pixels.

    Args:
        viewer: An owned viewer that already has the recording open.
        view: The blueprint view to capture.
        path: Where the screenshot PNG is written (overwritten on each attempt).
        timeout_s: How long to wait for the pixels to settle.

    Returns:
        The settled RGB pixels, ``(h, w, 3)`` uint8.
    """
    deadline: float = time.monotonic() + timeout_s
    previous: UInt8[np.ndarray, "h w 3"] | None = None
    while time.monotonic() < deadline:
        if not any(state.view_id == view.id for state in viewer.get_state().views):
            time.sleep(0.1)
            continue
        path.unlink(missing_ok=True)
        try:
            viewer.save_screenshot(str(path), view_id=view.id)
        except ConnectionError as error:
            if "not found" not in str(error):
                raise
            time.sleep(0.1)
            continue
        pixels: UInt8[np.ndarray, "h w 3"] | None = None
        file_deadline: float = min(deadline, time.monotonic() + 2.0)
        while pixels is None and time.monotonic() < file_deadline:
            try:
                with Image.open(path) as screenshot:
                    pixels = np.asarray(screenshot.convert("RGB"), dtype=np.uint8)
            except (FileNotFoundError, OSError):
                time.sleep(0.1)
        if pixels is not None and int(pixels.max()) - int(pixels.min()) > 20:
            if previous is not None and np.array_equal(previous, pixels):
                return pixels
            previous = pixels
        time.sleep(0.1)
    raise RuntimeError(f"Viewer pixels did not settle: {path}")
