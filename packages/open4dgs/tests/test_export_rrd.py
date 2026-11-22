"""The exporters' Rerun recordings on a tiny synthetic run: the viewer contracts come from gsplat_rust_renderer.

``export_asog.write_rrd`` puts the ``.asog`` verbatim in ``AnimatedSogSplats3D`` with a one-pose stock preview.
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

