"""TrigsMotion3D writer contract: the Rust reference fixture round-trips, checkpoint parameters map to the contract."""

import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
import rerun as rr
from rerun.chunk import RrdReader

from gsplat_rust_renderer.trigs_motion3d import ARCHETYPE, TrigsMotion3D, TrigsParams, log_trigs_motion, trigs_motion

# Written from 1,000 rows of a trained 4DAnyone run; the Rust test checks the viewer's evaluator against the trainer on it.
FIXTURE: Path = Path(__file__).resolve().parents[1] / "crates/gsplat-viewer/tests/fixtures/trigs_motion_1000.rrd"


def read_fields(path: Path, entity: str) -> tuple[dict[str, np.ndarray], dict[str, pa.DataType]]:
    """Every `<archetype>:<field>` column of a static entity: values (rows flattened per fixed-size list) and Arrow types."""
    values: dict[str, np.ndarray] = {}
    types: dict[str, pa.DataType] = {}
    for chunk in RrdReader(path).stream().filter(content=entity).to_chunks():
        batch: pa.RecordBatch = chunk.to_record_batch()
        for field in batch.schema:
            if ":" not in field.name:
                continue
            rows: pa.Array = batch.column(field.name).flatten()
            width: int | None = rows.type.list_size if pa.types.is_fixed_size_list(rows.type) else None
            flat: np.ndarray = rows.flatten().to_numpy() if width is not None else rows.to_numpy()
            values[field.name] = flat.reshape(-1, width) if width is not None else flat
            types[field.name] = field.type
    return values, types


def motion_from_fields(values: dict[str, np.ndarray]) -> TrigsMotion3D:
    def field(name: str) -> np.ndarray:
        return values[f"{ARCHETYPE}:{name}"]

    return TrigsMotion3D(
        opacities=field("opacities"), times=field("times"), log_durations=field("log_durations"), twist_base=field("twist_base"),
        twist_c0=field("twist_c0"), twist_c1=field("twist_c1"), twist_c2=field("twist_c2"), anchors=field("anchors"), is_static=field("is_static"),
        fps=float(field("fps")[0]), frame_origin=float(field("frame_origin")[0]), active_threshold=float(field("active_threshold")[0]),
    )


def test_fixture_round_trips(tmp_path: Path) -> None:
    """Writing the fixture's motion back gives the same columns, Arrow types and untyped descriptors."""
    expected, expected_types = read_fields(FIXTURE, "/model")
    motion: TrigsMotion3D = motion_from_fields(expected)
    assert len(motion.opacities) == 1000
    out: Path = tmp_path / "round-trip.rrd"
    recording = rr.RecordingStream("trigs-motion-round-trip")
    recording.save(out)
    recording.log("/model", motion, static=True)
    recording.flush()
    recording.disconnect()
    actual, actual_types = read_fields(out, "/model")
    motion_names: set[str] = {name for name in expected if name.startswith(f"{ARCHETYPE}:")}
    assert set(actual) == motion_names
    for name in motion_names:
        assert actual_types[name] == expected_types[name], name
        np.testing.assert_array_equal(actual[name], expected[name], err_msg=name)
    assert [d.component_type for d in (b.component_descriptor() for b in motion.as_component_batches())] == [None] * 12


def sample_params(rows: int = 3) -> TrigsParams:
    rng: np.random.Generator = np.random.default_rng(3)

    def f32(*shape: int) -> np.ndarray:
        return rng.standard_normal(shape).astype(np.float32)

    return TrigsParams(
        means=f32(rows, 3), quats=f32(rows, 4) * 3.0, scales=f32(rows, 3), opacities=f32(rows), sh0=f32(rows, 3), times=f32(rows), log_durations=f32(rows),
        zeta_base_t=f32(rows, 3), zeta_base_r=f32(rows, 3), zeta_c0_t=f32(rows, 3), zeta_c0_r=f32(rows, 3), zeta_c1_t=f32(rows, 3), zeta_c1_r=f32(rows, 3),
        zeta_c2_t=f32(rows, 3), zeta_c2_r=f32(rows, 3), anchors=f32(rows, 3), is_static=np.array([0.0, 1.0, 0.7], np.float32)[:rows],
    )


def test_params_map_to_contract(tmp_path: Path) -> None:
    """Twists stack (nu, omega); opacity and alpha are sigmoid(logit); rotations go wxyz -> unit xyzw; scales exp; static > 0.5."""
    params: TrigsParams = sample_params()
    motion: TrigsMotion3D = trigs_motion(params, fps=30.0, frame_origin=99.0, active_threshold=0.01)
    opacity: np.ndarray = 1.0 / (1.0 + np.exp(-params.opacities.astype(np.float64)))
    np.testing.assert_allclose(motion.opacities, opacity, rtol=1e-6)
    np.testing.assert_array_equal(motion.twist_c1, np.concatenate([params.zeta_c1_t, params.zeta_c1_r], axis=1))
    np.testing.assert_array_equal(motion.is_static, [0.0, 1.0, 1.0])
    assert (motion.fps, motion.frame_origin, motion.active_threshold) == (30.0, 99.0, 0.01)
    assert trigs_motion(replace(params, is_static=None), 30.0, 0.0, 0.0).is_static.tolist() == [0.0, 0.0, 0.0]

    out: Path = tmp_path / "compact.rrd"
    recording = rr.RecordingStream("trigs-motion-params")
    recording.save(out)
    log_trigs_motion("/world/person/splats", params, 30.0, 99.0, 0.01, recording=recording)
    recording.flush()
    recording.disconnect()
    values, _ = read_fields(out, "/world/person/splats")
    unit: np.ndarray = params.quats / np.linalg.norm(params.quats, axis=1, keepdims=True)
    np.testing.assert_allclose(values["GaussianSplats3D:quaternions"], unit[:, [1, 2, 3, 0]], rtol=1e-6)
    np.testing.assert_allclose(values["GaussianSplats3D:scales"], np.exp(params.scales), rtol=1e-6)
    alpha: np.ndarray = values["GaussianSplats3D:colors"].astype(np.uint32) & 0xFF
    np.testing.assert_array_equal(alpha, np.floor(opacity * 255.0 + 0.5))
    np.testing.assert_array_equal(values[f"{ARCHETYPE}:frame_origin"], [99.0])


def test_rejects_ragged_rows() -> None:
    motion: TrigsMotion3D = trigs_motion(sample_params(), 30.0, 0.0, 0.01)
    with pytest.raises(ValueError, match="rows"):
        replace(motion, anchors=motion.anchors[:2])
    with pytest.raises(ValueError, match="fps"):
        replace(motion, fps=0.0)


def test_writer_needs_no_torch() -> None:
    """The writer is numpy only: importing it loads neither torch nor open4dgs."""
    code: str = "import sys, gsplat_rust_renderer.trigs_motion3d; assert not {'torch', 'open4dgs'} & set(sys.modules), sorted(sys.modules)"
    subprocess.run([sys.executable, "-c", code], check=True, timeout=120)
