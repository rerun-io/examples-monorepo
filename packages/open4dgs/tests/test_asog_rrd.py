"""The open4dgs copy of the gsplat viewer's ``AnimatedSogSplats3D`` writer against a fixture its own writer logged.

``fixtures/asog_contract.rrd`` (32 KB): a 96-row synthetic clip (band-safe ``.asog``, 30 fps, 40 frames) logged by
``gsplat_rust_renderer.sog3d.log_asog`` (branch gsplat/asog @ 4fa286974) on ``/world/person/splats`` with
frame_origin 100, active_threshold 0.05 and 32 preview rows. Regenerate with the two scripts in
/mnt/nas/ssd-8tb/scratch/open4dgs/runs/sog/C1-open4dgs-asog-rrd/ (make_tiny_asog.py in the open4dgs env, then
make_contract_rrd.py in the gsplat-rust-renderer env).
"""

from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

pytest.importorskip("asog", reason="asog-src is in the open4dgs environments only")

FIXTURE: Path = Path(__file__).parent / "fixtures" / "asog_contract.rrd"
ENTITY: str = "/world/person/splats"


def entity_columns(path: Path) -> dict[str, tuple[pa.Field, pa.Array]]:
    """Every component column of the static ``ENTITY`` chunks: name -> (field, first-row values)."""
    from rerun.chunk import RrdReader

    out: dict[str, tuple[pa.Field, pa.Array]] = {}
    for chunk in RrdReader(path).stream().filter(content=ENTITY, is_static=True).to_chunks():
        batch = chunk.to_record_batch()
        for field in batch.schema:
            if field.name.startswith(("AnimatedSogSplats3D:", "GaussianSplats3D:")):
                out[field.name] = (field, batch.column(field.name)[0].values)
    return out


def test_writer_matches_viewer_fixture(tmp_path: Path) -> None:
    import rerun as rr
    from asog import read

    from open4dgs.asog_rrd import log_asog

    expected = entity_columns(FIXTURE)
    data: bytes = np.asarray(expected["AnimatedSogSplats3D:asog"][1], dtype=np.uint8).tobytes()
    (tmp_path / "clip.asog").write_bytes(data)
    recording = rr.RecordingStream("asog-contract-test", recording_id="asog-contract-test")
    recording.save(tmp_path / "ours.rrd")
    rows: int = log_asog(recording, ENTITY, data, read(tmp_path / "clip.asog"), frame_origin=100.0, active_threshold=0.05, preview_rows=32)
    recording.flush()
    recording.disconnect()
    ours = entity_columns(tmp_path / "ours.rrd")

    assert sorted(ours) == sorted(expected)
    for name, (field, _) in expected.items():
        assert ours[name][0].type == field.type, name
        assert ours[name][0].metadata == field.metadata, name  # archetype, component, component type
    for name in (
        "AnimatedSogSplats3D:format",
        "AnimatedSogSplats3D:asog",
        "AnimatedSogSplats3D:frame_origin",
        "AnimatedSogSplats3D:active_threshold",
    ):
        assert ours[name][1].to_pylist() == expected[name][1].to_pylist(), name
    assert rows == len(expected["GaussianSplats3D:centers"][1])
    # The preview: the reference decoder here, the viewer package's numpy port there.
    for name in ("GaussianSplats3D:centers", "GaussianSplats3D:quaternions", "GaussianSplats3D:scales"):
        a = np.asarray(ours[name][1].to_pylist(), dtype=np.float64)
        b = np.asarray(expected[name][1].to_pylist(), dtype=np.float64)
        np.testing.assert_allclose(a, b, atol=1e-6, err_msg=name)
    colors = np.asarray(ours["GaussianSplats3D:colors"][1].to_pylist(), dtype=np.uint32).view(np.uint8).astype(np.int16)
    np.testing.assert_array_less(np.abs(colors - np.asarray(expected["GaussianSplats3D:colors"][1].to_pylist(), dtype=np.uint32).view(np.uint8)), 2)


def test_evaluate_at_format_cut_is_the_reference(tmp_path: Path) -> None:
    from asog import read

    from open4dgs.asog_rrd import evaluate

    (tmp_path / "clip.asog").write_bytes(np.asarray(entity_columns(FIXTURE)["AnimatedSogSplats3D:asog"][1], dtype=np.uint8).tobytes())
    clip = read(tmp_path / "clip.asog")
    for t in (0.0, 0.4, 0.9, 1.3):
        ref, ours = clip.evaluate(t), evaluate(clip, t, 0.05)
        np.testing.assert_array_equal(ours.positions, ref.positions)
        np.testing.assert_array_equal(ours.rotations, ref.rotations)
        np.testing.assert_array_equal(ours.weights, ref.weights)
    looser = evaluate(clip, 0.9, 0.01)
    assert (looser.weights > 0).sum() >= (clip.evaluate(0.9).weights > 0).sum()


def test_payload_refuses_a_cut_outside_unit_interval() -> None:
    from open4dgs.asog_rrd import payload

    with pytest.raises(ValueError, match="active_threshold"):
        payload(b"", 1, 0.0, 1.0)
