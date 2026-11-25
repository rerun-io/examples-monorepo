"""Streamed SOG (lod-meta.json + SOG files) read and logged as `SogLodSplats3D`."""

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
import rerun as rr
from test_sog_decode import columns, encode, write_sog_dir  # pyrefly: ignore[missing-import]  (pytest rootdir import)

from gsplat_rust_renderer.sog3d import LOD_ARCHETYPE, PLANES, log_sog_lod
from gsplat_rust_renderer.sog_decode import decode, read_sog
from gsplat_rust_renderer.sog_lod import SogLodSet, read_sog_lod

SKATEPARK: Path = Path("/mnt/nas/ssd-8tb/gsplat-public-scenes/skatepark/raw")


def dc_only(planes: dict[str, bytes], meta_json: str, bands: int) -> tuple[dict[str, bytes], str]:
    """The SOG without its shN planes and meta when `bands` is 0, else unchanged."""
    if bands:
        return planes, meta_json
    meta: dict[str, object] = json.loads(meta_json)
    del meta["shN"]
    return {k: v for k, v in planes.items() if not k.startswith("shN")}, json.dumps(meta)


def write_lod_set(root: Path, *, environment: bool = False, bands: tuple[int, int, int] = (3, 3, 3)) -> Path:
    """Two levels: level 0 = file 0_0 (300 splats, two leaves of 150), level 1 = file 1_0 (80 splats, 40 per leaf).

    `bands` = SH bands of 0_0, 1_0 and the environment (0 = DC only: no shN planes)."""
    planes0, meta0, _ = encode(2, 300, bands[0] or 3, 64, seed=21)
    planes1, meta1, _ = encode(2, 80, bands[1] or 3, 64, seed=22)
    planes0, meta0 = dc_only(planes0, meta0, bands[0])
    planes1, meta1 = dc_only(planes1, meta1, bands[1])
    write_sog_dir(root / "0_0", planes0, meta0)
    write_sog_dir(root / "1_0", planes1, meta1)

    def leaf(lo: float, hi: float, offset0: int, offset1: int) -> dict[str, object]:
        return {
            "bound": {"min": [lo, -1.0, -1.0], "max": [hi, 1.0, 1.0]},
            "lods": {"0": {"file": 0, "offset": offset0, "count": 150}, "1": {"file": 1, "offset": offset1, "count": 40}},
        }

    meta: dict[str, object] = {
        "version": 1,
        "asset": {"generator": "tests/test_sog_lod.py"},
        "count": 380,
        "counts": [300, 80],
        "lodLevels": 2,
        "filenames": ["0_0/meta.json", "1_0/meta.json"],
        "tree": {"bound": {"min": [-2.0, -1.0, -1.0], "max": [2.0, 1.0, 1.0]}, "children": [leaf(-2.0, 0.0, 0, 0), leaf(0.0, 2.0, 150, 40)]},
    }
    if environment:
        planes_env, meta_env, _ = encode(2, 20, bands[2] or 3, 64, seed=23)
        planes_env, meta_env = dc_only(planes_env, meta_env, bands[2])
        write_sog_dir(root / "env", planes_env, meta_env)
        meta["environment"] = "env/meta.json"
    (root / "lod-meta.json").write_text(json.dumps(meta))
    return root


def test_reader_lists_files_levels_and_leaves(tmp_path: Path) -> None:
    root: Path = write_lod_set(tmp_path / "set", environment=True)
    lod: SogLodSet = read_sog_lod(root)
    assert lod.files == ["0_0/meta.json", "1_0/meta.json", "env/meta.json"]
    assert lod.file_levels == [0, 1, None]
    assert lod.levels == 2 and lod.counts == [300, 80]
    assert [len(leaf.runs) for leaf in lod.leaves] == [2, 2]
    assert lod.leaves[1].runs[0].offset == 150 and lod.leaves[1].runs[1].count == 40
    assert read_sog_lod(root / "lod-meta.json").files == lod.files


def test_reader_rejects_runs_past_their_file(tmp_path: Path) -> None:
    root: Path = write_lod_set(tmp_path / "set")
    text: str = (root / "lod-meta.json").read_text().replace('"offset": 150, "count": 150', '"offset": 200, "count": 150')
    (root / "lod-meta.json").write_text(text)
    with pytest.raises(ValueError, match="past the 300 splats"):
        read_sog_lod(root)


def test_writer_stores_every_file_verbatim_with_preview(tmp_path: Path) -> None:
    root: Path = write_lod_set(tmp_path / "set", environment=True)
    lod: SogLodSet = read_sog_lod(root)
    out: Path = tmp_path / "lod.rrd"
    recording = rr.RecordingStream("sog-lod-writer")
    recording.save(out)
    log_sog_lod("/world/splats", lod, preview_rows=50, recording=recording)
    recording.flush()
    recording.disconnect()
    stored: dict[str, pa.Array] = columns(out, "/world/splats")
    assert stored[f"{LOD_ARCHETYPE}:format"].to_pylist() == ["sog-lod/1"]
    assert stored[f"{LOD_ARCHETYPE}:lod_meta"].to_pylist() == [(root / "lod-meta.json").read_text()]
    assert stored[f"{LOD_ARCHETYPE}:files"].to_pylist() == ["0_0/meta.json", "1_0/meta.json", "env/meta.json"]
    assert stored[f"{LOD_ARCHETYPE}:metas"].to_pylist() == [(root / f).read_text() for f in lod.files]
    sizes: np.ndarray = stored[f"{LOD_ARCHETYPE}:plane_sizes"].to_numpy().reshape(3, len(PLANES))
    for plane_index, plane in enumerate(PLANES):
        flat: bytes = stored[f"{LOD_ARCHETYPE}:{plane}"].to_numpy().tobytes()
        expected: bytes = b"".join((root / Path(f).parent / f"{plane}.webp").read_bytes() for f in lod.files)
        assert flat == expected
        assert sizes[:, plane_index].tolist() == [len((root / Path(f).parent / f"{plane}.webp").read_bytes()) for f in lod.files]
    # No SogSplats3D on the entity: a SOG-aware viewer must not take the LOD set for one SOG.
    assert not any(name.startswith("SogSplats3D:") for name in stored)
    # Preview: the coarsest level with >= 50 rows is level 1 (80 rows) -> every 2nd splat of 1_0: 40 rows.
    centers: np.ndarray = stored["GaussianSplats3D:centers"].flatten().to_numpy().reshape(-1, 3)
    assert np.array_equal(centers, decode(read_sog(root / "1_0")).means[::2])


@pytest.mark.parametrize("bands", [(3, 0, 0), (0, 3, 3)])
def test_writer_accepts_files_with_and_without_sh_in_either_order(tmp_path: Path, bands: tuple[int, int, int]) -> None:
    root: Path = write_lod_set(tmp_path / "set", environment=True, bands=bands)
    lod: SogLodSet = read_sog_lod(root)
    out: Path = tmp_path / "lod.rrd"
    recording = rr.RecordingStream("sog-lod-mixed-sh")
    recording.save(out)
    log_sog_lod("/world/splats", lod, preview_rows=50, recording=recording)
    recording.flush()
    recording.disconnect()
    sizes: np.ndarray = columns(out, "/world/splats")[f"{LOD_ARCHETYPE}:plane_sizes"].to_numpy().reshape(3, len(PLANES))
    sh_planes: list[int] = [PLANES.index("shN_centroids"), PLANES.index("shN_labels")]
    assert [bool(sizes[f, sh_planes].all()) for f in range(3)] == [b > 0 for b in bands]


@pytest.mark.integration
def test_skatepark_set_reads() -> None:
    """The public Streamed SOG (splat-transform v3.3.3, 6 levels, 30 files + env) parses and every run fits its file."""
    if not (SKATEPARK / "lod-meta.json").exists():
        pytest.skip(f"{SKATEPARK} is not mounted")
    lod: SogLodSet = read_sog_lod(SKATEPARK)
    assert lod.levels == 6 and len(lod.files) == 31 and lod.file_levels[-1] is None
    assert sum(lod.file_counts[:-1]) == sum(lod.counts)
