"""SOG v2/v3: the numpy oracle against a tiny spec-following encoder's inputs, the SogSplats3D writer contract, and
Rust `gsplat_core::sog` == oracle (the Rust side: `crates/gsplat-viewer/tests/sog.rs`; integration: PlayCanvas skull.sog).

`python tests/test_sog_decode.py` rewrites the Rust fixture `crates/gsplat-viewer/tests/fixtures/sog_synthetic.rrd`:
/v2 and /v3 hold `SogSplats3D` + preview as `log_sog` writes them; /expected/v2 and /expected/v3 hold the numpy oracle
decode in the gsplat-core layout (`Expected:transforms` f32x10, `:raw_opacities` f32, `:sh` f32x3 per coefficient, DC first).
"""

import io
import json
import os
import subprocess
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
import rerun as rr
from jaxtyping import Float32, Float64, Int64, UInt8
from numpy import ndarray
from PIL import Image
from rerun.chunk import RrdReader

from gsplat_rust_renderer.sog3d import ARCHETYPE, PLANES, SogSplats3D, log_sog, preview_splats
from gsplat_rust_renderer.sog_decode import CoreSplats, DecodedSog, SogFile, codebook, core_layout, decode, parse_meta, read_sog

FIXTURE: Path = Path(__file__).resolve().parents[1] / "crates/gsplat-viewer/tests/fixtures/sog_synthetic.rrd"


@dataclass(frozen=True, slots=True)
class Truth:
    """What the encoder was given (before quantisation)."""

    means: Float64[ndarray, "n 3"]
    quats_wxyz: Float64[ndarray, "n 4"]
    """Unit, w >= 0 not required."""
    scale_index: Int64[ndarray, "n 3"]
    sh0_index: Int64[ndarray, "n 3"]
    alpha_byte: Int64[ndarray, "n"]
    labels: Int64[ndarray, "n"]
    """Palette entry per splat as written (may be out of range)."""


def texture_size(n: int) -> tuple[int, int]:
    """splat-transform write-sog.ts: W = ceil(sqrt(n) / 4) * 4, H = ceil(n / W / 4) * 4."""
    width: int = int(np.ceil(np.sqrt(n) / 4.0)) * 4
    return width, int(np.ceil(n / width / 4.0)) * 4


def webp(texels: UInt8[ndarray, "n 4"], width: int, height: int) -> bytes:
    """Lossless RGBA WebP; unused texels RGB 0, A 255. `exact` keeps RGB under A = 0."""
    image: UInt8[ndarray, "m 4"] = np.zeros((width * height, 4), np.uint8)
    image[:, 3] = 255
    image[: len(texels)] = texels
    out = io.BytesIO()
    Image.fromarray(image.reshape(height, width, 4), "RGBA").save(out, "WEBP", lossless=True, exact=True, quality=100, method=0)
    return out.getvalue()


def smallest_three(q: Float64[ndarray, "n 4"]) -> UInt8[ndarray, "n 4"]:
    """v2: drop the largest |component| (made positive), store the rest as (c sqrt2 + 1) / 2 * 255, A = 252 + index."""
    largest: Int64[ndarray, "n"] = np.argmax(np.abs(q), axis=1)
    q = q * np.where(q[np.arange(len(q)), largest] < 0.0, -1.0, 1.0)[:, None]
    out: UInt8[ndarray, "n 4"] = np.zeros((len(q), 4), np.uint8)
    for i, m in enumerate(largest):
        rest: Float64[ndarray, "3"] = np.delete(q[i], m)
        out[i, :3] = np.clip(np.round((rest * np.sqrt(2.0) + 1.0) / 2.0 * 255.0), 0, 255)
        out[i, 3] = 252 + m
    return out


def exp_map(q: Float64[ndarray, "n 4"]) -> UInt8[ndarray, "n 4"]:
    """v3 (quat-expmap.ts encodeQuatExpMap): w >= 0, p = (2/pi) atan2(|v|, w) v/|v|, bytes round(p 127 + 128)."""
    q = q * np.where(q[:, 0] < 0.0, -1.0, 1.0)[:, None]
    v: Float64[ndarray, "n 3"] = q[:, 1:]
    sv: Float64[ndarray, "n"] = np.linalg.norm(v, axis=1)
    k: Float64[ndarray, "n"] = np.where(sv > 1e-12, np.arctan2(sv, q[:, 0]) / np.where(sv > 1e-12, sv, 1.0), 1.0) / (np.pi / 2.0)
    out: UInt8[ndarray, "n 4"] = np.full((len(q), 4), 255, np.uint8)
    out[:, :3] = np.clip(np.round(v * k[:, None] * 127.0 + 128.0), 0, 255)
    return out


def encode(version: int, n: int, bands: int, palette: int, seed: int) -> tuple[dict[str, bytes], str, Truth]:
    """Random splats with special rows: identity/reserved quats, alpha 0/255, null scales codebook[0] (v2), labels
    past the palette (v2 B byte set, v3 24-bit). v2 omits shN.count/bands (bands inferred); v3 writes both."""
    rng: np.random.Generator = np.random.default_rng(seed)
    means: Float64[ndarray, "n 3"] = rng.normal(0.0, 2.0, (n, 3)) * np.array([1.0, 0.5, 3.0])
    means[0] = 0.0
    quats: Float64[ndarray, "n 4"] = rng.normal(size=(n, 4))
    quats[1] = [1.0, 0.0, 0.0, 0.0]
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    logs: Float64[ndarray, "n 3"] = np.sign(means) * np.log1p(np.abs(means))
    mins: Float64[ndarray, "3"] = logs.min(axis=0)
    maxs: Float64[ndarray, "3"] = logs.max(axis=0)
    q16: Int64[ndarray, "n 3"] = np.round((logs - mins) / (maxs - mins) * 65535.0).astype(np.int64)
    width, height = texture_size(n)

    def rgba(rgb: ndarray, alpha: ndarray | int = 255) -> UInt8[ndarray, "n 4"]:
        out: UInt8[ndarray, "n 4"] = np.zeros((n, 4), np.uint8)
        out[:, :3] = rgb
        out[:, 3] = alpha
        return out

    quat_texels: UInt8[ndarray, "n 4"] = exp_map(quats) if version == 3 else smallest_three(quats)
    if version == 2:
        quat_texels[2] = [17, 99, 200, 251]  # reserved tag: identity
    scale_index: Int64[ndarray, "n 3"] = rng.integers(0, 256, (n, 3))
    scale_index[:4] = 0  # rows that read the (v2: null) codebook[0]
    sh0_index: Int64[ndarray, "n 3"] = rng.integers(0, 256, (n, 3))
    alpha_byte: Int64[ndarray, "n"] = rng.integers(0, 256, n)
    alpha_byte[:3] = [0, 255, 204]
    coeffs: int = (bands + 1) ** 2 - 1
    labels: Int64[ndarray, "n"] = rng.integers(0, palette, n)
    labels[:2] = [2 * palette, 2 * palette + 5]  # past the palette and its padded rows (v2 without count) -> zero SH
    label_texels: UInt8[ndarray, "n 4"] = rgba(np.stack([labels & 255, (labels >> 8) & 255, (labels >> 16) & 255], axis=1))
    if version == 2:
        label_texels[5, 2] = 7  # v2 ignores the B byte
    else:
        label_texels[5, 2] = 1  # v3: label + 65536 -> out of range
    centroid_width: int = 64 * coeffs
    centroid_rows: int = -(-palette // 64)
    centroids: UInt8[ndarray, "m 4"] = np.zeros((palette * coeffs, 4), np.uint8)
    centroids[:, :3] = rng.integers(0, 256, (palette * coeffs, 3))
    centroids[:, 3] = 255

    planes: dict[str, bytes] = {
        "means_l": webp(rgba(q16 & 255), width, height),
        "means_u": webp(rgba(q16 >> 8), width, height),
        "quats": webp(quat_texels, width, height),
        "scales": webp(rgba(scale_index), width, height),
        "sh0": webp(rgba(sh0_index, alpha_byte), width, height),
        "shN_centroids": webp(centroids, centroid_width, centroid_rows),
        "shN_labels": webp(label_texels, width, height),
    }
    scales_cb: list[float | None] = [float(v) for v in np.sort(rng.uniform(-9.0, 0.0, 256))]
    if version == 2:
        scales_cb[0] = None
    sh_n: dict[str, object] = {"codebook": sorted(float(v) for v in rng.uniform(-1.5, 1.5, 256)), "files": ["shN_centroids.webp", "shN_labels.webp"]}
    if version == 3:
        sh_n |= {"count": palette, "bands": bands}
    meta: dict[str, object] = {
        "version": version,
        "asset": {"generator": "gsplat-rust-renderer tests/test_sog_decode.py"},
        "count": n,
        "unknown_field": [1, 2, 3],
        "means": {"mins": mins.tolist(), "maxs": maxs.tolist(), "files": ["means_l.webp", "means_u.webp"]},
        "scales": {"codebook": scales_cb, "files": ["scales.webp"]},
        "quats": {"files": ["quats.webp"]},
        "sh0": {"codebook": sorted(float(v) for v in rng.uniform(-3.0, 3.0, 256)), "files": ["sh0.webp"]},
        "shN": sh_n,
    }
    return planes, json.dumps(meta), Truth(means, quats, scale_index, sh0_index, alpha_byte, labels)


def write_sog_dir(directory: Path, planes: dict[str, bytes], meta_json: str) -> Path:
    """An unbundled SOG directory (meta.json + <role>.webp)."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "meta.json").write_text(meta_json)
    for role, data in planes.items():
        (directory / f"{role}.webp").write_bytes(data)
    return directory


def expected_batches(transforms: Float32[ndarray, "n 10"], raw_opacities: Float32[ndarray, "n"], sh: Float32[ndarray, "n k 3"]) -> list[rr.DescribedComponentBatch]:
    """`Expected:*` untyped f32 batches (fixed-size lists per splat / coefficient)."""

    def described(field: str, values: ndarray, width: int | None) -> rr.DescribedComponentBatch:
        flat: pa.Array = pa.array(np.ascontiguousarray(values, np.float32).reshape(-1), pa.float32())
        array: pa.Array = flat if width is None else pa.FixedSizeListArray.from_arrays(flat, width)
        descriptor: rr.ComponentDescriptor = rr.ComponentDescriptor(f"Expected:{field}", archetype="Expected", component_type=None)
        return rr.DescribedComponentBatch(rr.AnyBatchValue(descriptor, array), descriptor)

    return [described("transforms", transforms, 10), described("raw_opacities", raw_opacities, None), described("sh", sh, 3)]


def write_oracle_rrd(out: Path, sogs: dict[str, Path], preview_rows: int) -> None:
    """Each SOG under /<name> (writer contract) and its numpy oracle decode under /expected/<name>."""
    recording = rr.RecordingStream("sog-oracle")
    recording.save(out)
    for name, path in sogs.items():
        sog: SogFile = read_sog(path)
        log_sog(f"/{name}", sog, preview_rows, recording=recording)
        core: CoreSplats = core_layout(decode(sog))
        recording.log(f"/expected/{name}", expected_batches(core.transforms, core.raw_opacities, core.sh), static=True, strict=True)
    recording.flush()
    recording.disconnect()


def write_fixture(out: Path, scratch: Path) -> None:
    """The committed fixture: v2 (bands 3 inferred, null codebook[0]) and v3 (bands 2, count/bands given), 1,000 splats."""
    v2_planes, v2_meta, _ = encode(2, 1000, 3, 300, seed=2)
    v3_planes, v3_meta, _ = encode(3, 1000, 2, 300, seed=3)
    write_oracle_rrd(out, {"v2": write_sog_dir(scratch / "v2", v2_planes, v2_meta), "v3": write_sog_dir(scratch / "v3", v3_planes, v3_meta)}, preview_rows=256)

ROOT: Path = Path(__file__).resolve().parents[1]
SKULL_CANDIDATES: tuple[Path, ...] = (ROOT / "data/sog/skull.sog", Path("/mnt/nas/ssd-8tb/scratch/open4dgs/sog/samples/skull.sog"))


def columns(path: Path, entity: str) -> dict[str, pa.Array]:
    """Row 0 of every `<archetype>:<field>` column of a static entity."""
    out: dict[str, pa.Array] = {}
    for chunk in RrdReader(path).stream().filter(content=entity).to_chunks():
        batch: pa.RecordBatch = chunk.to_record_batch()
        for field in batch.schema:
            if ":" in field.name:
                out[field.name] = batch.column(field.name)[0].values
    return out


@pytest.mark.parametrize("version", [2, 3])
def test_oracle_recovers_encoder_inputs(version: int, tmp_path: Path) -> None:
    bands: int = 3 if version == 2 else 2
    planes, meta_json, truth = encode(version, 1000, bands, 300, seed=11 + version)
    sog: SogFile = read_sog(write_sog_dir(tmp_path, planes, meta_json))
    decoded: DecodedSog = decode(sog)
    meta = sog.meta
    # Positions: within half a 16-bit step of the log-domain span.
    logs: np.ndarray = np.sign(truth.means) * np.log1p(np.abs(truth.means))
    got: np.ndarray = np.sign(decoded.means) * np.log1p(np.abs(decoded.means.astype(np.float64)))
    step: np.ndarray = (np.array(meta.means.maxs) - np.array(meta.means.mins)) / 65535.0
    assert (np.abs(got - logs) <= 0.5 * step + 1e-6).all()
    assert np.allclose(decoded.means[0], 0.0, atol=1e-3)
    # Rotations: same rotation (sign-free) within the format's quantisation (v2 ~2 deg, v3 ~1.2 deg max).
    q: np.ndarray = decoded.quats_wxyz.astype(np.float64)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    angle: np.ndarray = 2.0 * np.degrees(np.arccos(np.clip(np.abs((q * truth.quats_wxyz).sum(axis=1)), 0.0, 1.0)))
    if version == 2:
        assert np.array_equal(decoded.quats_wxyz[2], [1.0, 0.0, 0.0, 0.0])  # reserved tag 251
        angle = np.delete(angle, 2)
        assert angle.max() < 2.1
    else:
        assert angle.max() < 1.3
    if version == 3:
        assert np.array_equal(decoded.quats_wxyz[1], [1.0, 0.0, 0.0, 0.0])  # the exp-map lattice holds identity exactly
    # Codebook lookups (v2 scales codebook[0] is null: engine patch).
    scales_cb: np.ndarray = codebook(meta.scales.codebook, "scales")
    assert np.array_equal(decoded.log_scales, scales_cb[truth.scale_index])
    if version == 2:
        cb = meta.scales.codebook
        assert cb[0] is None and cb[1] is not None and cb[255] is not None
        assert scales_cb[0] == np.float32(cb[1] + (cb[1] - cb[255]) / 255.0)
    assert np.array_equal(decoded.sh0, codebook(meta.sh0.codebook, "sh0")[truth.sh0_index])
    # Opacity: v2 linear, v3 squared (unclamped here, clamped in the core layout).
    a: np.ndarray = truth.alpha_byte.astype(np.float64)
    assert np.allclose(decoded.opacity, (a / 204.0) ** 2 if version == 3 else a / 255.0, rtol=0, atol=1e-15)
    core = core_layout(decoded)
    if version == 3:
        assert decoded.opacity[1] == pytest.approx(1.5625)
        assert core.raw_opacities[1] == core.raw_opacities[2]  # 255 and 204 both clamp to 1
    assert core.raw_opacities[0] == pytest.approx(np.log(1e-6 / (1.0 - 1e-6)), rel=1e-6)
    # SH: bands inferred (v2) / given (v3); palette lookups; out-of-range labels give zeros.
    assert decoded.sh_bands == bands
    coeffs: int = (bands + 1) ** 2 - 1
    assert decoded.sh_rest.shape == (1000, coeffs, 3)
    assert not decoded.sh_rest[:2].any()
    assert decoded.sh_rest[5].any() == (version == 2)  # v2 ignores the B byte; v3 reads label + 65536
    assert core.sh.shape == (1000, coeffs + 1, 3) and np.array_equal(core.sh[:, 0], decoded.sh0)


def test_bundled_and_unbundled_read_the_same(tmp_path: Path) -> None:
    planes, meta_json, _ = encode(3, 200, 1, 64, seed=5)
    directory: Path = write_sog_dir(tmp_path / "dir", planes, meta_json)
    bundle: Path = tmp_path / "scene.sog"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_STORED) as archive:
        for file in sorted(directory.iterdir()):
            archive.write(file, file.name)
    a, b = read_sog(directory), read_sog(bundle)
    assert a.planes == b.planes and a.meta_json == b.meta_json
    assert read_sog(directory / "meta.json").planes == a.planes


def test_meta_rejects_v1_and_short_planes(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not a SOG v2/v3"):
        parse_meta('{"means": {"shape": [1, 3]}}', "v1.json")
    with pytest.raises(ValueError, match="not a SOG v2/v3"):
        parse_meta('{"version": 4, "count": 1, "means": {"mins": [0,0,0], "maxs": [1,1,1], "files": ["a", "b"]}, '
                   '"scales": {"codebook": [], "files": ["s"]}, "quats": {"files": ["q"]}, "sh0": {"codebook": [], "files": ["c"]}}', "v4.json")
    planes, meta_json, _ = encode(2, 100, 1, 64, seed=6)
    sog: SogFile = read_sog(write_sog_dir(tmp_path, planes, meta_json.replace('"count": 100', '"count": 100000')))
    with pytest.raises(ValueError, match="fewer than the 100000 splats"):
        decode(sog)


def test_writer_stores_bytes_verbatim_with_preview(tmp_path: Path) -> None:
    planes, meta_json, _ = encode(2, 1000, 3, 300, seed=7)
    sog: SogFile = read_sog(write_sog_dir(tmp_path / "sog", planes, meta_json))
    out: Path = tmp_path / "sog.rrd"
    recording = rr.RecordingStream("sog-writer")
    recording.save(out)
    log_sog("/world/splats", sog, preview_rows=300, recording=recording)
    recording.flush()
    recording.disconnect()
    stored: dict[str, pa.Array] = columns(out, "/world/splats")
    assert stored[f"{ARCHETYPE}:format"].to_pylist() == ["sog/2"]
    assert stored[f"{ARCHETYPE}:meta"].to_pylist() == [meta_json]
    for plane in PLANES:
        assert stored[f"{ARCHETYPE}:{plane}"].type == pa.uint8()
        assert stored[f"{ARCHETYPE}:{plane}"].to_numpy().tobytes() == planes[plane]
    # Preview: every 4th splat (ceil(1000 / 300) = 4) -> 250 rows, unit xyzw quaternions.
    centers: np.ndarray = stored["GaussianSplats3D:centers"].flatten().to_numpy().reshape(-1, 3)
    decoded: DecodedSog = decode(sog)
    assert np.array_equal(centers, decoded.means[::4])
    quats: np.ndarray = stored["GaussianSplats3D:quaternions"].flatten().to_numpy().reshape(-1, 4)
    assert np.allclose(np.linalg.norm(quats, axis=1), 1.0, atol=1e-6)
    expected: np.ndarray = decoded.quats_wxyz[::4] / np.linalg.norm(decoded.quats_wxyz[::4], axis=1, keepdims=True)
    assert np.allclose(quats, np.roll(expected, -1, axis=1), atol=1e-6)
    assert preview_splats(decoded, 0) is None
    with pytest.raises(ValueError, match="missing"):
        SogSplats3D(SogFile(sog.meta_json, sog.meta, {k: v for k, v in sog.planes.items() if k != "quats"}))


def test_committed_fixture_is_the_oracle_of_its_planes() -> None:
    """The Rust fixture's /expected rows are the oracle decode of the stored planes (regenerate: python tests/test_sog_decode.py)."""
    for name in ("v2", "v3"):
        stored: dict[str, pa.Array] = columns(FIXTURE, f"/{name}")
        meta_json: str = stored[f"{ARCHETYPE}:meta"].to_pylist()[0]
        planes: dict[str, bytes] = {p: stored[f"{ARCHETYPE}:{p}"].to_numpy().tobytes() for p in PLANES if f"{ARCHETYPE}:{p}" in stored}
        core = core_layout(decode(SogFile(meta_json, parse_meta(meta_json, name), planes)))
        reference: dict[str, pa.Array] = columns(FIXTURE, f"/expected/{name}")
        assert np.array_equal(reference["Expected:transforms"].flatten().to_numpy().reshape(-1, 10), core.transforms)
        assert np.array_equal(reference["Expected:raw_opacities"].to_numpy(), core.raw_opacities)
        assert np.array_equal(reference["Expected:sh"].flatten().to_numpy().reshape(core.sh.shape), core.sh)


@pytest.mark.integration
def test_rust_decode_matches_oracle_on_skull(tmp_path: Path) -> None:
    """PlayCanvas skull.sog (v2, SH bands inferred, 246,821 splats): Rust decode == numpy oracle per splat (<= 1e-6)."""
    skull: Path | None = next((p for p in SKULL_CANDIDATES if p.exists()), None)
    if skull is None:
        pytest.skip(f"skull.sog missing (fetch: pixi run -e gsplat-rust-renderer gsplat-log-sog, or {SKULL_CANDIDATES[1]})")
    sog: SogFile = read_sog(skull)
    start: float = time.perf_counter()
    decoded: DecodedSog = decode(sog)
    print(f"numpy oracle decode of {skull.name}: {(time.perf_counter() - start) * 1e3:.0f} ms")
    assert len(decoded.means) == 246_821 and decoded.sh_bands == 3
    out: Path = tmp_path / "skull-oracle.rrd"
    write_oracle_rrd(out, {"skull": skull}, preview_rows=16_384)
    env: dict[str, str] = {**os.environ, "GSPLAT_SOG_ORACLE_RRD": str(out)}
    result = subprocess.run(
        ["cargo", "test", "--locked", "-p", "gsplat-viewer", "--test", "sog", "oracle_rrd_from_env_matches", "--", "--nocapture"],
        cwd=ROOT, env=env, text=True, capture_output=True, check=False,
    )
    print(result.stdout[-2000:])
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    assert "/skull (sog/2): 246821 splats, SH degree 3" in result.stdout


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as scratch:
        write_fixture(FIXTURE, Path(scratch))
    print(f"wrote {FIXTURE} ({FIXTURE.stat().st_size} B)")
