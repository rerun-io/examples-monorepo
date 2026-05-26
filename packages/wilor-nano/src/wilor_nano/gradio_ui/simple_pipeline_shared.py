from __future__ import annotations

import socket
import tempfile
import time
import uuid
from pathlib import Path
from typing import Final

import gradio as gr
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import rerun.experimental
from jaxtyping import UInt8
from numpy import ndarray
from PIL import Image

TEST_INPUT_DIR: Final[Path] = Path(__file__).resolve().parents[3] / "assets"
DEFAULT_IMAGE_PATH: Final[Path] = TEST_INPUT_DIR / "img.png"
DEFAULT_OUTPUT_DIR: Final[Path] = Path(tempfile.gettempdir()) / "wilor_nano_simple_pipeline"
DEFAULT_VIEWER_HOST: Final[str] = "127.0.0.1"
DEFAULT_VIEWER_PORT: Final[int] = 9877
DEFAULT_APPLICATION_ID: Final[str] = "wilor_nano_simple_pipeline"
assert DEFAULT_IMAGE_PATH.exists(), f"{DEFAULT_IMAGE_PATH} does not exist"


def switch_to_outputs() -> object:
    """Switch a Gradio Tabs component to its outputs tab."""
    return gr.update(selected="outputs")


def ensure_recording_id(recording_id: uuid.UUID | str | None) -> str:
    """Normalize a recording id into a stable string."""
    if recording_id is None:
        return str(uuid.uuid4())
    if isinstance(recording_id, uuid.UUID):
        return str(recording_id)
    return str(uuid.UUID(str(recording_id)))


def layer_rrd_path(output_dir: Path, *, recording_id: str, layer: str) -> Path:
    """Return the saved RRD path for one simple pipeline layer."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rrd_path: Path = output_dir / f"simple_pipeline_{recording_id}_{layer}.rrd"
    return rrd_path


def viewer_url(port: int = DEFAULT_VIEWER_PORT) -> str:
    """Return the Rerun gRPC proxy URL for a local native viewer."""
    url: str = f"rerun+http://{DEFAULT_VIEWER_HOST}:{port}/proxy"
    return url


def is_port_listening(host: str, port: int, timeout_sec: float = 0.2) -> bool:
    """Return whether a TCP listener is reachable."""
    try:
        with socket.create_connection((host, port), timeout=timeout_sec):
            return True
    except OSError:
        return False


def ensure_native_viewer(port: int = DEFAULT_VIEWER_PORT, timeout_sec: float = 10.0) -> str:
    """Start a local native viewer when the configured port is not already listening."""
    if not is_port_listening(DEFAULT_VIEWER_HOST, port):
        rr.spawn(port=port, connect=False, detach_process=True, hide_welcome_screen=True)

    deadline_sec: float = time.monotonic() + timeout_sec
    while time.monotonic() < deadline_sec:
        if is_port_listening(DEFAULT_VIEWER_HOST, port):
            return viewer_url(port=port)
        time.sleep(0.1)

    raise TimeoutError(f"Rerun viewer did not start on {DEFAULT_VIEWER_HOST}:{port}")


def wait_for_file(path: Path, timeout_sec: float = 5.0) -> bool:
    """Poll until a file exists and has content."""
    deadline_sec: float = time.monotonic() + timeout_sec
    while time.monotonic() < deadline_sec:
        if path.exists() and path.stat().st_size > 0:
            return True
        time.sleep(0.05)
    return path.exists() and path.stat().st_size > 0


def save_native_screenshot(
    output_dir: Path,
    *,
    prefix: str,
    port: int = DEFAULT_VIEWER_PORT,
    settle_sec: float = 0.5,
) -> Path:
    """Ask the native Rerun viewer to save a screenshot and wait for the file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if settle_sec > 0.0:
        time.sleep(settle_sec)
    screenshot_path: Path = output_dir / f"{prefix}_{uuid.uuid4()}.png"
    viewer_client = rerun.experimental.ViewerClient(f"{DEFAULT_VIEWER_HOST}:{port}")
    viewer_client.save_screenshot(str(screenshot_path))

    if not wait_for_file(screenshot_path):
        raise TimeoutError(f"Rerun viewer did not finish writing {screenshot_path}")
    return screenshot_path


def load_rgb_image(image_path: Path = DEFAULT_IMAGE_PATH) -> UInt8[ndarray, "h w 3"]:
    """Load an RGB uint8 image for the simple pipeline demos.

    Args:
        image_path: Path to an image file.

    Returns:
        UInt8[ndarray, "h w 3"]: RGB image loaded from disk.
    """
    image: Image.Image = Image.open(image_path).convert("RGB")
    rgb_hw3: UInt8[ndarray, "h w 3"] = np.asarray(image, dtype=np.uint8)
    return rgb_hw3


def resolve_rgb_image(
    rgb_hw3: UInt8[ndarray, "h w 3"] | None,
    image_path: Path = DEFAULT_IMAGE_PATH,
) -> tuple[UInt8[ndarray, "h w 3"], str]:
    """Resolve an optional uploaded image, falling back to the default example image."""
    if rgb_hw3 is None:
        default_rgb_hw3: UInt8[ndarray, "h w 3"] = load_rgb_image(image_path)
        return default_rgb_hw3, "default"

    resolved_rgb_hw3: UInt8[ndarray, "h w 3"] = np.asarray(rgb_hw3, dtype=np.uint8)
    if resolved_rgb_hw3.ndim != 3 or resolved_rgb_hw3.shape[2] != 3:
        raise ValueError(f"Expected RGB image with shape (h, w, 3), got {resolved_rgb_hw3.shape}")
    return resolved_rgb_hw3, "uploaded"


def build_simple_blueprint() -> rrb.Blueprint:
    """Build the stable 2D layout shared by the simple viewers."""
    blueprint: rrb.Blueprint = rrb.Blueprint(
        rrb.Spatial2DView(name="Simple Pipeline", origin="image"),
        collapse_panels=True,
    )
    return blueprint
