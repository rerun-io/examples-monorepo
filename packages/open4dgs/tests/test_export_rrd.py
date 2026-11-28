"""The exporters' Rerun recordings on a tiny synthetic run: the viewer contracts come from gsplat_rust_renderer.

``open4dgs-export`` (``apis.export``, default ``--format asog``) puts the ``.asog`` verbatim in
``AnimatedSogSplats3D`` with a one-pose stock preview (one shared writer, ``splat_export.write_asog_rrd``);
``--format per-frame`` logs one ``GaussianSplats3D`` row per frame.
The contracts themselves are pinned in packages/gsplat-rust-renderer/tests.
"""

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

pytest.importorskip("asog", reason="asog-src is in the open4dgs environments only")
pytest.importorskip("trigs", reason="open4dgs-trigs-src is in the open4dgs environments only")

from test_asog import FPS, NUM_FRAMES, synthetic_params  # noqa: E402

if TYPE_CHECKING:
    from open4dgs.apis.export import ExportFormat

FRAME_START: int = 100


def static_columns(path: Path, entity: str) -> dict[str, list[object]]:
    """First-row values of every component column of ``entity``'s static chunks."""
    from rerun.chunk import RrdReader

    out: dict[str, list[object]] = {}
    for chunk in RrdReader(path).stream().filter(content=entity, is_static=True).to_chunks():
        batch = chunk.to_record_batch()
        for name in batch.schema.names:
            if ":" in name:
                out[name] = batch.column(name).to_pylist()[0]
    return out


def tiny_run(root: Path) -> Path:
    """A run directory: ``run.rrd`` (the ``run`` property + one training camera) and ``trigs_ckpt.pt`` (synthetic params)."""
    import rerun as rr

    from open4dgs.properties import any_values
    from open4dgs.run_log import RunProperties

    run: Path = root / "tiny-run"
    run.mkdir()
    recording = rr.RecordingStream("open4dgs-runs", recording_id=run.name)
    recording.save(run / "run.rrd")
    properties = RunProperties(
        dataset="d", segment_id=run.name, catalog_url="rerun+http://localhost:1", frame_start=FRAME_START, frame_end=FRAME_START + NUM_FRAMES - 1,
        num_frames=NUM_FRAMES, cameras=("rig_00/cam_00",), train_cameras=("rig_00/cam_00",), holdout_cameras=(),
    )  # fmt: skip
    recording.send_property("run", any_values(properties.to_properties()))
    recording.log("/world/cameras/train/rig_00/cam_00", rr.Transform3D(translation=[0.0, 0.0, -2.0], mat3x3=np.eye(3)), static=True)
    recording.flush()
    recording.disconnect()
    params = synthetic_params()
    torch.save({"params": params, "sh_degree": 3, "fps": FPS, "num_frames": NUM_FRAMES}, run / "trigs_ckpt.pt")
    return run


def test_asog_rrd_carries_the_file_and_a_one_pose_preview(tmp_path: Path) -> None:
    from open4dgs.apis.export import Config, main
    from open4dgs.asog import AsogSettings, encode_asog
    from open4dgs.splat_export import SPLAT_ENTITY

    run: Path = tiny_run(tmp_path)
    asog_path: Path = tmp_path / "clip.asog"
    asog_path.write_bytes(encode_asog(synthetic_params(), FPS, NUM_FRAMES, AsogSettings(preset="upstream", prune=0.0))[0])
    out: Path = tmp_path / "asog.rrd"
    summary = main(Config(run=run, register=False, preview_rows=64, asog=asog_path, output=out))
    columns = static_columns(out, SPLAT_ENTITY)

    assert bytes(columns["AnimatedSogSplats3D:asog"]) == asog_path.read_bytes()
    assert columns["AnimatedSogSplats3D:frame_origin"] == [float(FRAME_START)]
    assert columns["AnimatedSogSplats3D:active_threshold"] == [pytest.approx(0.05)]
    assert 0 < len(columns["GaussianSplats3D:centers"]) <= 64  # one pose, every k-th live row
    assert summary["frame_origin"] == FRAME_START


STOCK_VIEWER_LINE: str = "a stock Rerun viewer shows one still pose; use --format per-frame for playback in stock viewers"


def entity_rows(path: Path) -> dict[str, int]:
    """Rows per entity path over every chunk of ``path``."""
    from rerun.chunk import RrdReader

    rows: dict[str, int] = {}
    for chunk in RrdReader(path).stream().to_chunks():
        rows[chunk.entity_path] = rows.get(chunk.entity_path, 0) + chunk.to_record_batch().num_rows
    return rows


def test_default_export_is_the_asog_payload_with_a_preview(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from open4dgs.apis.export import Config, main
    from open4dgs.asog import AsogSettings, encode_asog
    from open4dgs.splat_export import SPLAT_ENTITY

    run: Path = tiny_run(tmp_path)
    asog_path: Path = tmp_path / "clip.asog"
    asog_path.write_bytes(encode_asog(synthetic_params(), FPS, NUM_FRAMES, AsogSettings(preset="upstream", prune=0.0))[0])
    summary = main(Config(run=run, register=False, asog=asog_path))
    columns = static_columns(run / "asog.rrd", SPLAT_ENTITY)

    assert bytes(columns["AnimatedSogSplats3D:asog"]) == asog_path.read_bytes()
    assert columns["AnimatedSogSplats3D:frame_origin"] == [float(FRAME_START)]
    assert columns["AnimatedSogSplats3D:active_threshold"] == [pytest.approx(0.05)]
    assert len(columns["GaussianSplats3D:centers"]) > 0  # the one-pose stock preview
    assert summary["asog_rrd"] == str(run / "asog.rrd")
    assert STOCK_VIEWER_LINE in capsys.readouterr().out


def test_default_export_encodes_the_checkpoint_into_the_recording_only(tmp_path: Path) -> None:
    """Without ``--asog`` the checkpoint is encoded in memory: the recording carries the ASOG and no ``.asog`` file is written."""
    from gsplat_rust_renderer.asog_decode import parse_asog

    from open4dgs.apis.export import Config, main
    from open4dgs.asog import AsogSettings, encode_asog
    from open4dgs.splat_export import SPLAT_ENTITY

    run: Path = tiny_run(tmp_path)
    main(Config(run=run, register=False))
    checkpoint = torch.load(run / "trigs_ckpt.pt", map_location="cpu", weights_only=True)
    expected = encode_asog(checkpoint["params"], FPS, NUM_FRAMES, AsogSettings())[1]
    payload: bytes = bytes(static_columns(run / "asog.rrd", SPLAT_ENTITY)["AnimatedSogSplats3D:asog"])

    assert parse_asog(payload, "asog.rrd").manifest.count == expected.count  # the band-safe encode of the checkpoint
    assert not list(tmp_path.rglob("*.asog"))


def test_per_frame_export_is_the_old_default(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from open4dgs.apis.export import Config, main

    run: Path = tiny_run(tmp_path)
    summary = main(Config(run=run, register=False, format="per-frame"))

    # Pinned from the per-frame default before --format asog existed (34adf304e): same entities, one row per frame, plus
    # the layer's run record.
    assert entity_rows(run / "splats.rrd") == {
        "/__properties": 1,
        "/__properties/run_splats": 1,
        "/world": 1,
        "/world/cameras/train/rig_00/cam_00": 1,
        "/world/person/splats": NUM_FRAMES,
    }
    assert (summary["frames"], summary["gaussians_min"], summary["gaussians_max"], summary["rows"]) == (30, 478, 512, 512)
    assert STOCK_VIEWER_LINE not in capsys.readouterr().out


@pytest.mark.parametrize(("format", "file", "layer"), [("asog", "asog.rrd", "asog"), ("per-frame", "splats.rrd", "splats")])
def test_each_export_layer_carries_the_run_record_of_this_export(tmp_path: Path, format: "ExportFormat", file: str, layer: str) -> None:
    """A flow runner reads the layer's record from the catalog to tell this export from an older one."""
    from rerun.chunk import RrdReader

    from open4dgs.apis.export import Config, main

    run: Path = tiny_run(tmp_path)
    main(Config(run=run, register=False, format=format))
    cells: dict[str, object] = {}
    for chunk in RrdReader(run / file).stream().filter(content=f"/__properties/run_{layer}", is_static=True).to_chunks():
        batch = chunk.to_record_batch()
        cells |= {name: batch.column(name).to_pylist()[0] for name in batch.schema.names if not name.startswith("rerun.")}
    assert (cells["layer"], cells["segment"]) == ([layer], [run.name]) and "export" in str(cells["command"])
