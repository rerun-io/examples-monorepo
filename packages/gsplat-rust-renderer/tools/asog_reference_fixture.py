"""Write the `.asog` reference fixture for `crates/gsplat-viewer/tests/asog.rs` (tests E1 + E2 of the SOG/ASOG design).

Runs in an environment with the asog reference decoder (the `asog-src` recipe, e.g. the open4dgs env), not in the
gsplat env: `python -I tools/asog_reference_fixture.py <src.asog> <out.rrd>`.

It takes 2,048 rows of a real `.asog` (every row whose omega gets near zero, every row whose pinv mapping onto the
twist form is rank-deficient or inexact, then random rows), re-encodes them as a smaller `.asog` with the same
manifest (the SH palette cut to the labels in use; every other quantity is per row, so each row decodes exactly as
in the source) and logs:
  /asog                 AnimatedSogSplats3D:format "asog/1", :asog (the subset file bytes), :active_threshold 0.05
  /expected/canonical   Expected:means, :quats (wxyz), :log_scales, :opacity, :sh0, :sh_rest (first 64 rows), :mu, :sigma,
                        :anchor, :omega (O0..O2 per row), :g (G0..G4 per row), :static (1/0), :times; f32 flat batches
  /expected/t<k>        Expected:positions, :rotations (wxyz), :weights from `Asog.evaluate(times[k])`, f64 flat
  /expected/twist       Expected:nu (N0..N2 per row, the min-norm pinv of the glue's `nu_control_points`, f32),
                        :well_conditioned (1/0: relative G residual < 1e-9, rank 5, |omega(tau)|^2 >= 1e-8 on 33 taus)
"""

import importlib
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import rerun as rr
from numpy import ndarray

TIMES: tuple[float, ...] = tuple(float(np.float32(t)) for t in (0.0, 0.97, 2.42, 3.71, 4.8))
"""Evaluation times, rounded to f32 so the Rust f32 time input is the same instant."""
ROWS: int = 2048
THRESHOLD: float = 0.05
SH_ROWS: int = 64
"""Rows whose SH is stored in the fixture (keeps the file under 1 MB)."""


def batch(archetype: str, field: str, values: ndarray | list[str]) -> rr.DescribedComponentBatch:
    descriptor = rr.ComponentDescriptor(f"{archetype}:{field}", archetype=archetype, component_type=None)
    array: pa.Array = pa.array(values, pa.utf8()) if isinstance(values, list) else pa.array(np.ascontiguousarray(values).reshape(-1))
    return rr.DescribedComponentBatch(rr.AnyBatchValue(descriptor, array), descriptor)


def expected(dtype: type, **fields: ndarray) -> list[rr.DescribedComponentBatch]:
    return [batch("Expected", k, np.asarray(v, dtype)) for k, v in fields.items()]


def nu_control_points(omega: ndarray, g: ndarray) -> tuple[ndarray, ndarray, ndarray]:
    """open4dgs glue `nu_control_points` (asog.py:202-220): min-norm N with quartic(N . O) = G; also residual and rank."""
    o0, o1, o2 = omega
    z = np.zeros_like(o0)
    rows = [[o0, z, z], [o1 / 2, o0 / 2, z], [o2 / 6, 4 * o1 / 6, o0 / 6], [z, o2 / 2, o1 / 2], [z, z, o2]]
    system = np.stack([np.concatenate(r, axis=1) for r in rows], axis=1)
    solution = (np.linalg.pinv(system) @ g.T[:, :, None])[:, :, 0]
    residual = np.abs((system @ solution[:, :, None])[:, :, 0] - g.T).max(1) / (np.abs(g).max(0) + 1e-30)
    return solution.reshape(-1, 3, 3).transpose(1, 0, 2), residual, np.linalg.matrix_rank(system)


def main(src: Path, out: Path) -> None:
    container = importlib.import_module("asog.container")
    decode = importlib.import_module("asog.decode")
    motion = importlib.import_module("asog.motion")
    manifest, streams, _ = container.read(src)
    full = decode.read(src)
    n: int = full.count
    nu, residual, rank = nu_control_points(full.omega, full.g)
    tiny = np.zeros(n, bool)
    for tau in np.linspace(0, 1, 33):
        om = motion.bezier2(np.full(n, tau), full.omega)
        tiny |= (om * om).sum(1) < 1e-8
    dynamic = ~full.static
    special = np.nonzero(dynamic & (tiny | (rank < 5) | (residual > 1e-6)))[0]
    rng = np.random.default_rng(0)
    rest = rng.choice(np.setdiff1d(np.arange(n), special), ROWS - len(special), replace=False)
    idx = np.sort(np.concatenate([special, rest]))
    print(f"{len(special)} special rows (tiny omega {int((tiny & dynamic).sum())}, rank<5 {int(((rank < 5) & dynamic).sum())}, "
          f"residual>1e-6 {int(((residual > 1e-6) & dynamic).sum())}), {len(rest)} random")

    sub = dict(manifest)
    sub.pop("streams")
    sub["count"] = int(len(idx))
    encoded = []
    labels = streams["shlabels"][idx, 0] if "shlabels" in streams else None
    used = np.unique(labels) if labels is not None else None
    for desc in manifest["streams"]:
        name = desc["name"]
        if name == "shpalette" and used is not None:
            values = streams[name][used]
        elif name == "shlabels" and used is not None and labels is not None:
            values = np.searchsorted(used, labels)[:, None]
        else:
            values = streams[name][idx]
        encoded.append(container.encode_stream(name, values))
    if used is not None:
        sub["shN"] = dict(manifest["shN"], count=int(len(used)))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "subset.asog"
        container.write(path, sub, encoded)
        data: bytes = path.read_bytes()
        small = decode.read(path)
    # Per-row quantities must survive the cut exactly.
    for name in ("means", "quats", "mu", "sigma", "anchor", "opacity", "log_scales", "sh0", "shN"):
        assert np.array_equal(getattr(small, name), getattr(full, name)[idx]), name
    assert np.array_equal(small.omega, full.omega[:, idx]) and np.array_equal(small.g, full.g[:, idx])

    rec = rr.RecordingStream("asog_reference_fixture")
    rec.save(out)
    rec.log("/asog", [batch("AnimatedSogSplats3D", "format", ["asog/1"]),
                      batch("AnimatedSogSplats3D", "asog", np.frombuffer(data, np.uint8)),
                      batch("AnimatedSogSplats3D", "active_threshold", np.array([THRESHOLD], np.float32))], static=True)
    rec.log("/expected/canonical", expected(
        np.float32,
        means=small.means, quats=small.quats, log_scales=small.log_scales, opacity=small.opacity, sh0=small.sh0,
        sh_rest=small.shN[:SH_ROWS] if small.shN is not None else np.zeros(0), mu=small.mu, sigma=small.sigma, anchor=small.anchor,
        omega=small.omega.transpose(1, 0, 2), g=small.g.T, static=small.static.astype(np.float64), times=np.array(TIMES),
    ), static=True)
    for k, t in enumerate(TIMES):
        frame = small.evaluate(t)
        ref = full.evaluate(t)
        assert np.array_equal(frame.positions, ref.positions[idx]) and np.array_equal(frame.weights, ref.weights[idx])
        rec.log(f"/expected/t{k}", expected(np.float64, positions=frame.positions, rotations=frame.rotations, weights=frame.weights), static=True)
        print(f"t={t}: {int((frame.weights > 0).sum())} of {len(idx)} rows visible")
    well = (rank[idx] == 5) & (residual[idx] < 1e-9) & ~tiny[idx] & dynamic[idx]
    rec.log("/expected/twist", expected(np.float32, nu=nu[:, idx].transpose(1, 0, 2), well_conditioned=well.astype(np.float64)), static=True)
    rec.flush()
    print(f"{out}: {len(data):,} B subset .asog, {int(well.sum())} well-conditioned rows")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
