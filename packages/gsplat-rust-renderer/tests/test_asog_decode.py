"""The numpy `.asog` port (`gsplat_rust_renderer.asog_decode`) == the animated-sog reference decoder, and the
`AnimatedSogSplats3D` writer.

The reference values come from `crates/gsplat-viewer/tests/fixtures/asog_reference.rrd` (2,048 rows of the run C
band `.asog`, decoded and evaluated by `asog.decode` in the open4dgs env; `tools/asog_reference_fixture.py`), the
same fixture the Rust decoder is checked on (`crates/gsplat-viewer/tests/asog.rs`).
"""

from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
import rerun as rr
from jaxtyping import Float64
from rerun.chunk import RrdReader

from gsplat_rust_renderer.asog_decode import AsogFile, AsogFrame, DecodedAsog, decode, decode_sh, evaluate, parse_asog
from gsplat_rust_renderer.sog3d import ANIMATED_ARCHETYPE, log_asog

FIXTURE: Path = Path(__file__).resolve().parents[1] / "crates/gsplat-viewer/tests/fixtures/asog_reference.rrd"


def columns(path: Path, entity: str) -> dict[str, np.ndarray]:
    """Row 0 of every `<archetype>:<field>` column of a static entity, as numpy."""
    out: dict[str, np.ndarray] = {}
    for chunk in RrdReader(path).stream().filter(content=entity).to_chunks():
        batch: pa.RecordBatch = chunk.to_record_batch()
        for field in batch.schema:
            if ":" in field.name:
                values: pa.Array = batch.column(field.name)[0].values
                if pa.types.is_fixed_size_list(values.type):
                    out[field.name] = values.flatten().to_numpy().reshape(-1, values.type.list_size)
                elif pa.types.is_string(values.type):
                    out[field.name] = np.array(values.to_pylist())
                else:
                    out[field.name] = values.to_numpy()
    return out


def fixture_asog() -> AsogFile:
    return parse_asog(columns(FIXTURE, "/asog")[f"{ANIMATED_ARCHETYPE}:asog"].tobytes(), str(FIXTURE))


def test_decode_and_evaluate_match_the_reference() -> None:
    asog: AsogFile = fixture_asog()
    decoded: DecodedAsog = decode(asog)
    canonical: dict[str, np.ndarray] = columns(FIXTURE, "/expected/canonical")
    np.testing.assert_allclose(decoded.means.ravel(), canonical["Expected:means"], rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(decoded.omega.transpose(1, 0, 2).ravel(), canonical["Expected:omega"], rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(decoded.g.T.ravel(), canonical["Expected:g"], rtol=1e-6, atol=1e-7)
    # The fixture stores SH for its first rows only.
    sh_rest: Float64[np.ndarray, "n c 3"] = decode_sh(asog)
    np.testing.assert_allclose(
        sh_rest[: len(canonical["Expected:sh_rest"]) // (3 * sh_rest.shape[1])].ravel(), canonical["Expected:sh_rest"], rtol=1e-6, atol=1e-7
    )
    for k, t in enumerate(canonical["Expected:times"]):
        expected: dict[str, np.ndarray] = columns(FIXTURE, f"/expected/t{k}")
        frame: AsogFrame = evaluate(decoded, asog.manifest, float(t))
        np.testing.assert_array_equal(frame.weights > 0, expected["Expected:weights"] > 0)
        np.testing.assert_allclose(frame.weights, expected["Expected:weights"], atol=1e-12)
        np.testing.assert_allclose(frame.positions.ravel(), expected["Expected:positions"], atol=1e-12)
        np.testing.assert_allclose(frame.rotations.ravel(), expected["Expected:rotations"], atol=1e-12)


def test_writer_logs_the_file_verbatim_and_a_mid_clip_preview(tmp_path: Path) -> None:
    asog: AsogFile = fixture_asog()
    out: Path = tmp_path / "asog.rrd"
    recording = rr.RecordingStream("asog-writer-test")
    recording.save(out)
    log_asog("/clip", asog, frame_origin=10.0, preview_rows=64, fps=asog.manifest.timeline.frame_rate, recording=recording)
    recording.flush()
    recording.disconnect()
    got: dict[str, np.ndarray] = columns(out, "/clip")
    assert got[f"{ANIMATED_ARCHETYPE}:asog"].tobytes() == asog.data
    assert list(got[f"{ANIMATED_ARCHETYPE}:format"]) == ["asog/1"]
    assert got[f"{ANIMATED_ARCHETYPE}:frame_origin"].tolist() == [10.0]
    assert got[f"{ANIMATED_ARCHETYPE}:active_threshold"].tolist() == [pytest.approx(0.05)]
    decoded: DecodedAsog = decode(asog)
    mid: AsogFrame = evaluate(decoded, asog.manifest, asog.manifest.duration / 2)
    centers: np.ndarray = got["GaussianSplats3D:centers"]
    alive: np.ndarray = np.nonzero(mid.weights > 0)[0]
    assert 0 < len(centers) <= 64
    # Preview rows are the evaluated mid-clip pose of live rows, not the canonical means.
    np.testing.assert_allclose(centers[0], mid.positions[alive[0]], atol=1e-6)


def test_writer_refuses_another_timeline_rate() -> None:
    asog: AsogFile = fixture_asog()
    with pytest.raises(ValueError, match="frameRate"):
        log_asog("/clip", asog, fps=30.0, recording=rr.RecordingStream("asog-fps-test"))


def test_parse_rejects_other_files() -> None:
    with pytest.raises(ValueError, match="not an .asog"):
        parse_asog(b"PK\x03\x04" + bytes(60), "x.sog")
