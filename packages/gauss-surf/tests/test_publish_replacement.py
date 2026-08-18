"""Republish contracts for fused direct-gsplat publication."""

from pathlib import Path
from typing import Any

import numpy as np
import torch
from jaxtyping import UInt8, UInt16
from numpy import ndarray
from rerun.catalog import DatasetEntry, DatasetView

from gauss_surf.catalog import SegmentReader
from gauss_surf.render_io import RenderCamera
from gauss_surf.train_gsplat.publish import (
    PublishStats,
    UltrawideQuantizedRender,
    WideQuantizedRender,
    publish_in_process,
)


def _camera(camera: str, index: int) -> RenderCamera:
    """Build one tiny synthetic full-grid camera."""
    world_from_camera_34: ndarray = np.eye(4, dtype=np.float32)[:3]
    return RenderCamera(
        stem=f"{camera}_{index:06d}",
        camera=camera,  # pyrefly: ignore  # validated fixture literal
        timestamp_ns=index if camera == "wide" else 1_000 + index,
        width=1,
        height=1,
        fx=1.0,
        fy=1.0,
        cx=0.5,
        cy=0.5,
        world_from_camera_34=world_from_camera_34,
    )


def test_publish_replaces_existing_canonical_rrds(tmp_path: Path, monkeypatch: Any) -> None:
    """A rerun atomically replaces all three existing canonical layer files."""
    cameras: list[RenderCamera] = [
        *[_camera("wide", index) for index in range(2)],
        *[_camera("uw", index) for index in range(3)],
    ]
    registered_layers: list[str] = []

    def fake_quantized_wide_render(
        _splats: torch.nn.ParameterDict | dict[str, torch.Tensor],
        camera: RenderCamera,
        _background_3: torch.Tensor,
    ) -> WideQuantizedRender:
        depth_mm_hw: UInt16[ndarray, "h=1 w=1"] = np.ones((1, 1), dtype=np.uint16)
        normal_rgb_hw3: UInt8[ndarray, "h=1 w=1 3"] = np.full((1, 1, 3), 128, dtype=np.uint8)
        return WideQuantizedRender(camera, depth_mm_hw, normal_rgb_hw3)

    def fake_quantized_ultrawide_render(
        _splats: torch.nn.ParameterDict | dict[str, torch.Tensor],
        camera: RenderCamera,
        _background_3: torch.Tensor,
    ) -> UltrawideQuantizedRender:
        depth_mm_hw: UInt16[ndarray, "h=1 w=1"] = np.ones((1, 1), dtype=np.uint16)
        normal_rgb_hw3: UInt8[ndarray, "h=1 w=1 3"] = np.full((1, 1, 3), 128, dtype=np.uint8)
        rgb_hw3: UInt8[ndarray, "h=1 w=1 3"] = np.zeros((1, 1, 3), dtype=np.uint8)
        return UltrawideQuantizedRender(camera, depth_mm_hw, normal_rgb_hw3, rgb_hw3)

    monkeypatch.setattr("gauss_surf.train_gsplat.publish.load_render_cameras", lambda _path: cameras)
    monkeypatch.setattr("gauss_surf.train_gsplat.publish.reference_blobs_at_component_timestamps", lambda _view, _column: {})
    monkeypatch.setattr("gauss_surf.train_gsplat.publish.load_reference_blobs", lambda _view: [])
    monkeypatch.setattr("gauss_surf.train_gsplat.publish._quantized_wide_render", fake_quantized_wide_render)
    monkeypatch.setattr("gauss_surf.train_gsplat.publish._quantized_ultrawide_render", fake_quantized_ultrawide_render)
    monkeypatch.setattr(SegmentReader, "require_layers", lambda _reader, _layers: None)
    segment_view: DatasetView = object.__new__(DatasetView)
    monkeypatch.setattr(SegmentReader, "segment_view", lambda _reader: segment_view)
    monkeypatch.setattr(
        "gauss_surf.train_gsplat.publish.register_layer",
        lambda _dataset, _path, layer_name: registered_layers.append(layer_name),
    )

    splats: dict[str, torch.Tensor] = {
        "means": torch.tensor([[0.0, 0.0, 1.0]]),
        "scales": torch.log(torch.tensor([[0.1, 0.1, 0.1]])),
        "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        "opacities": torch.tensor([[0.0]]),
        "sh0": torch.zeros((1, 1, 3)),
        "shN": torch.zeros((1, 15, 3)),
    }
    output_dirs: tuple[Path, Path, Path] = tuple(tmp_path / name for name in ("splat", "splat_depth", "splat_triage"))
    for output_dir in output_dirs:
        output_dir.mkdir()
        (output_dir / "segment.rrd").write_bytes(b"old canonical layer")

    dataset: DatasetEntry = object.__new__(DatasetEntry)
    reader: SegmentReader = SegmentReader(dataset=dataset, video_id="segment")
    stats: PublishStats = publish_in_process(
        splats,
        reader=reader,
        bundle_dir=tmp_path,
        video_id="segment",
        splat_output_dir=output_dirs[0],
        depth_output_dir=output_dirs[1],
        triage_output_dir=output_dirs[2],
        batch_size=3,
        encoder_workers=1,
    )

    assert registered_layers == ["splat", "splat_depth", "splat_triage"]
    assert stats.frame_count == 5
    assert all((output_dir / "segment.rrd").read_bytes() != b"old canonical layer" for output_dir in output_dirs)
