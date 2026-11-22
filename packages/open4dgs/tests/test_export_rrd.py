"""The exporters' Rerun recordings on a tiny synthetic run: the viewer contracts come from gsplat_rust_renderer.

``export_asog.write_rrd`` and ``export_splats`` (default ``--format asog``) put the ``.asog`` verbatim in
``AnimatedSogSplats3D`` with a one-pose stock preview (one shared writer, ``splat_export.write_asog_rrd``);
``--format per-frame`` logs one ``GaussianSplats3D`` row per frame; ``--format trigs-motion`` logs the checkpoint once
as canonical ``GaussianSplats3D`` + ``TrigsMotion3D``.
The contracts themselves are pinned in packages/gsplat-rust-renderer/tests.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("asog", reason="asog-src is in the open4dgs environments only")
pytest.importorskip("trigs", reason="open4dgs-trigs-src is in the open4dgs environments only")

from test_asog import FPS, NUM_FRAMES, synthetic_params  # noqa: E402

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

    run: Path = root / "tiny-run"
    run.mkdir()
    recording = rr.RecordingStream("open4dgs-runs", recording_id=run.name)
    recording.save(run / "run.rrd")
    recording.send_property("run", rr.AnyValues(frame_start=FRAME_START))
    recording.log("/world/cameras/train/rig_00/cam_00", rr.Transform3D(translation=[0.0, 0.0, -2.0], mat3x3=np.eye(3)), static=True)
    recording.flush()
    recording.disconnect()
    params = synthetic_params()
    torch.save({"params": params, "sh_degree": 3, "fps": FPS, "num_frames": NUM_FRAMES}, run / "trigs_ckpt.pt")
    return run


def test_asog_rrd_carries_the_file_and_a_one_pose_preview(tmp_path: Path) -> None:
    from open4dgs.apis.export_asog import Config, write_rrd
    from open4dgs.asog import AsogSettings, encode_params
    from open4dgs.splat_export import SPLAT_ENTITY

    run: Path = tiny_run(tmp_path)
    asog_path: Path = tmp_path / "clip.asog"
    encode_params(synthetic_params(), FPS, NUM_FRAMES, asog_path, AsogSettings(preset="upstream", prune=0.0))
    out: Path = tmp_path / "asog.rrd"
    summary = write_rrd(Config(run=run, register=False, preview_rows=64), asog_path, out)
    columns = static_columns(out, SPLAT_ENTITY)

    assert bytes(columns["AnimatedSogSplats3D:asog"]) == asog_path.read_bytes()
    assert columns["AnimatedSogSplats3D:frame_origin"] == [float(FRAME_START)]
    assert columns["AnimatedSogSplats3D:active_threshold"] == [pytest.approx(0.05)]
    assert 0 < len(columns["GaussianSplats3D:centers"]) <= 64  # one pose, every k-th live row
    assert summary["frame_origin"] == FRAME_START


def test_trigs_motion_export_logs_the_checkpoint_once(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from open4dgs.apis.export_splats import Config, main
    from open4dgs.splat_export import SPLAT_ENTITY

    run: Path = tiny_run(tmp_path)
    summary = main(Config(run=run, register=False, format="trigs-motion"))
    out: Path = run / "trigs_motion.rrd"
    columns = static_columns(out, SPLAT_ENTITY)
    params = synthetic_params()

    assert summary["rows"] == len(params["means"])
    assert len(columns["GaussianSplats3D:centers"]) == len(params["means"])
    np.testing.assert_allclose(np.asarray(columns["GaussianSplats3D:centers"]), params["means"].numpy())
    np.testing.assert_allclose(np.asarray(columns["TrigsMotion3D:times"]), params["times"].numpy().reshape(-1))
    assert columns["TrigsMotion3D:fps"] == [FPS] and columns["TrigsMotion3D:frame_origin"] == [float(FRAME_START)]
    assert sorted(set(columns["TrigsMotion3D:is_static"])) == [0.0, 1.0]
    assert STOCK_VIEWER_LINE not in capsys.readouterr().out


STOCK_VIEWER_LINE: str = "a stock Rerun viewer shows one still pose; use --format per-frame for playback in stock viewers"


def entity_rows(path: Path) -> dict[str, int]:
    """Rows per entity path over every chunk of ``path``."""
    from rerun.chunk import RrdReader

    rows: dict[str, int] = {}
    for chunk in RrdReader(path).stream().to_chunks():
        rows[chunk.entity_path] = rows.get(chunk.entity_path, 0) + chunk.to_record_batch().num_rows
    return rows


def test_default_export_is_the_asog_payload_with_a_preview(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from open4dgs.apis.export_splats import Config, main
    from open4dgs.asog import AsogSettings, encode_params
    from open4dgs.splat_export import SPLAT_ENTITY

    run: Path = tiny_run(tmp_path)
    asog_path: Path = tmp_path / "clip.asog"
    encode_params(synthetic_params(), FPS, NUM_FRAMES, asog_path, AsogSettings(preset="upstream", prune=0.0))
    summary = main(Config(run=run, register=False, asog=asog_path))
    columns = static_columns(run / "asog.rrd", SPLAT_ENTITY)

    assert bytes(columns["AnimatedSogSplats3D:asog"]) == asog_path.read_bytes()
    assert columns["AnimatedSogSplats3D:frame_origin"] == [float(FRAME_START)]
    assert columns["AnimatedSogSplats3D:active_threshold"] == [pytest.approx(0.05)]
    assert len(columns["GaussianSplats3D:centers"]) > 0  # the one-pose stock preview
    assert summary["asog_rrd"] == str(run / "asog.rrd")
    assert STOCK_VIEWER_LINE in capsys.readouterr().out


def test_per_frame_export_is_the_old_default(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from open4dgs.apis.export_splats import Config, main

    run: Path = tiny_run(tmp_path)
    summary = main(Config(run=run, register=False, format="per-frame"))

    # Pinned from the per-frame default before --format asog existed (34adf304e): same entities, one row per frame.
    assert entity_rows(run / "splats.rrd") == {
        "/__properties": 1,
        "/world": 1,
        "/world/cameras/train/rig_00/cam_00": 1,
        "/world/person/splats": NUM_FRAMES,
    }
    assert (summary["frames"], summary["gaussians_min"], summary["gaussians_max"], summary["rows"]) == (30, 478, 512, 512)
    assert STOCK_VIEWER_LINE not in capsys.readouterr().out
