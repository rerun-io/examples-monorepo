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

STOCK_VIEWER: Path = Path(sys.executable).parent / "rerun"
"""This environment's stock viewer, from the conda rerun-sdk at the same Rerun release."""


def free_port() -> int:
    """A free loopback port for an owned viewer."""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def spawn_viewer(binary: Path) -> ViewerClient:
    """Start an owned headless viewer on a free port; the client is a context manager that stops it.

    The viewer must report the SDK's Rerun release, or its pixels prove nothing about this release. Its startup line
    "Headless viewer running at ..." is also a toast for about 4 s, over the top-right view, so a capture would depend
    on timing; ``re_viewer::headless=warn`` silences that one line and keeps every other INFO line.
    """
    with patch.dict(os.environ, {"RUST_LOG": "info,re_viewer::headless=warn"}):
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
        # save_screenshot returns once the viewer has written the file.
        with Image.open(path) as screenshot:
            pixels: UInt8[np.ndarray, "h w 3"] = np.asarray(screenshot.convert("RGB"), dtype=np.uint8)
        if int(pixels.max()) - int(pixels.min()) > 20:
            if previous is not None and np.array_equal(previous, pixels):
                return pixels
            previous = pixels
        time.sleep(0.1)
    raise RuntimeError(f"Viewer pixels did not settle: {path}")
