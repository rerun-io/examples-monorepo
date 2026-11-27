"""With no extension data, rerun-canonical draws the same pixels as the stock viewer of the same Rerun release.

This is the proof that the hooks patch and the build change nothing else.
"""

from pathlib import Path

import numpy as np
import pytest
import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import Float32, UInt8

from canonical_viewer.pixels import STOCK_VIEWER, capture, spawn_viewer

pytestmark = pytest.mark.golden


def write_scene(path: Path) -> list[rrb.View]:
    """Save a small 2D + 3D recording with a fixed eye; return its views."""
    generator: np.random.Generator = np.random.default_rng(0)
    points: Float32[np.ndarray, "n 3"] = generator.normal(size=(2000, 3)).astype(np.float32)
    colors: UInt8[np.ndarray, "n 3"] = generator.integers(0, 256, size=(2000, 3), dtype=np.uint8)
    image: UInt8[np.ndarray, "h w 3"] = np.zeros((120, 160, 3), dtype=np.uint8)
    image[:, :, 0] = np.linspace(0, 255, 160, dtype=np.uint8)[None, :]
    image[:, :, 1] = np.linspace(0, 255, 120, dtype=np.uint8)[:, None]
    views: list[rrb.View] = [
        rrb.Spatial3DView(
            origin="/world",
            name="3d",
            eye_controls=rrb.EyeControls3D(position=(6.0, -6.0, 4.0), look_target=(0.0, 0.0, 0.0), eye_up=(0.0, 0.0, 1.0)),
        ),
        rrb.Spatial2DView(origin="/image", name="2d"),
    ]
    recording: rr.RecordingStream = rr.RecordingStream("canonical-viewer-stock-parity", recording_id="stock-parity")
    recording.save(path)
    recording.send_blueprint(rrb.Blueprint(rrb.Horizontal(*views), collapse_panels=True))
    recording.log("/world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    recording.log("/world/points", rr.Points3D(points, colors=colors, radii=0.03), static=True)
    recording.log("/world/boxes", rr.Boxes3D(centers=[[0, 0, 0], [2, 1, 0]], half_sizes=[[1, 1, 1], [0.5, 0.5, 1.5]], colors=[[255, 80, 40], [40, 200, 255]]), static=True)
    recording.log("/world/arrows", rr.Arrows3D(vectors=[[2, 0, 0], [0, 2, 0], [0, 0, 2]], colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]]), static=True)
    recording.log("/image", rr.Image(image), static=True)
    recording.log("/image/boxes", rr.Boxes2D(mins=[[20, 20]], sizes=[[60, 40]], colors=[[255, 255, 0]]), static=True)
    recording.flush()
    recording.disconnect()
    return views


def test_canonical_draws_what_stock_draws(canonical_viewer: Path, tmp_path: Path) -> None:
    recording_path: Path = tmp_path / "stock-parity.rrd"
    views: list[rrb.View] = write_scene(recording_path)
    pixels: dict[str, list[UInt8[np.ndarray, "h w 3"]]] = {}
    for name, binary in (("canonical", canonical_viewer), ("stock", STOCK_VIEWER)):
        with spawn_viewer(binary) as viewer:
            viewer.open(str(recording_path))
            pixels[name] = [capture(viewer, view, tmp_path / f"{name}-{view.name}.png") for view in views]
    for view, canonical, stock in zip(views, pixels["canonical"], pixels["stock"], strict=True):
        assert canonical.shape == stock.shape, view.name
        assert np.array_equal(canonical, stock), f"{view.name}: canonical differs from stock in {float((canonical != stock).mean()):.4%} of values"
