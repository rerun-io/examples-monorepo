"""Unit tests for moving an INRIA 3DGS splat by a similarity."""

from pathlib import Path

import numpy as np
import pytest
from jaxtyping import Float64
from numpy import ndarray
from scipy.spatial.transform import Rotation

from monopriors.splat_similarity import Similarity, gaussian_covariances, read_ply, sh_colour, transform_gaussians, transform_ply

N: int = 64


def _table(rng: np.random.Generator, sh_degree: int = 3) -> np.ndarray:
    rest: int = 3 * ((sh_degree + 1) ** 2 - 1)
    names: list[str] = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", *(f"f_rest_{i}" for i in range(rest))]
    names += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    table: np.ndarray = np.zeros(N, dtype=[(name, "<f4") for name in names])
    for name in names:
        table[name] = rng.normal(size=N).astype(np.float32)
    return table


def _similarity(rng: np.random.Generator) -> Similarity:
    return Similarity(scale=0.8, rotation=Rotation.random(random_state=rng).as_matrix(), translation=rng.normal(size=3))


def test_sh_colour_follows_the_rotation() -> None:
    rng: np.random.Generator = np.random.default_rng(1)
    table: np.ndarray = _table(rng)
    similarity: Similarity = _similarity(rng)
    moved: np.ndarray = transform_gaussians(table, similarity)
    directions: Float64[ndarray, "n 3"] = rng.normal(size=(N, 3))
    np.testing.assert_allclose(sh_colour(moved, directions @ similarity.rotation.T), sh_colour(table, directions), atol=2e-5)


def test_covariance_is_scaled_and_rotated() -> None:
    rng: np.random.Generator = np.random.default_rng(2)
    table: np.ndarray = _table(rng)
    similarity: Similarity = _similarity(rng)
    expected: Float64[ndarray, "n 3 3"] = similarity.scale**2 * similarity.rotation @ gaussian_covariances(table) @ similarity.rotation.T
    np.testing.assert_allclose(gaussian_covariances(transform_gaussians(table, similarity)), expected, rtol=1e-4, atol=1e-5)


def test_positions_and_round_trip() -> None:
    rng: np.random.Generator = np.random.default_rng(3)
    table: np.ndarray = _table(rng)
    similarity: Similarity = _similarity(rng)
    moved: np.ndarray = transform_gaussians(table, similarity)
    positions: Float64[ndarray, "n 3"] = np.stack([table[c] for c in "xyz"], 1).astype(np.float64)
    np.testing.assert_allclose(np.stack([moved[c] for c in "xyz"], 1), similarity.apply(positions), atol=1e-5)
    back: np.ndarray = transform_gaussians(moved, similarity.inverse())
    for name in table.dtype.names or ():
        if name.startswith("rot_"):
            continue
        np.testing.assert_allclose(back[name], table[name], atol=2e-4, err_msg=name)
    quat: Float64[ndarray, "n 4"] = np.stack([table[f"rot_{i}"] for i in range(4)], 1).astype(np.float64)
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    quat_back: Float64[ndarray, "n 4"] = np.stack([back[f"rot_{i}"] for i in range(4)], 1).astype(np.float64)
    np.testing.assert_allclose(np.abs(np.sum(quat * quat_back, axis=1)), 1.0, atol=1e-5)  # same rotation up to sign
    np.testing.assert_array_equal(back["opacity"], table["opacity"])


@pytest.mark.parametrize("sh_degree", [1, 3])
def test_ply_keeps_header_and_extra_properties(tmp_path: Path, sh_degree: int) -> None:
    rng: np.random.Generator = np.random.default_rng(4)
    table: np.ndarray = _table(rng, sh_degree)
    names: list[str] = list(table.dtype.names or ())
    header: str = (
        f"ply\nformat binary_little_endian 1.0\ncomment SplatRenderMode: mip\nelement vertex {N}\n"
        + "".join(f"property float {n}\n" for n in names)
        + "end_header\n"
    )
    source: Path = tmp_path / "in.ply"
    source.write_bytes(header.encode() + table.tobytes())
    similarity: Similarity = Similarity(scale=1.0, rotation=np.eye(3), translation=np.zeros(3))
    assert transform_ply(source, tmp_path / "out.ply", similarity) == N
    out_header, out_table = read_ply(tmp_path / "out.ply")
    assert out_header == header.encode()
    for name in names:
        if not name.startswith("rot_"):
            np.testing.assert_allclose(out_table[name], table[name], atol=1e-6, err_msg=name)


def test_load_prior_alignment(tmp_path: Path) -> None:
    from monopriors.apis.splat_similarity import load_prior_alignment

    path: Path = tmp_path / "summary.json"
    path.write_text('{"x": 1, "prior_alignment": {"scale": 0.8, "rotation_xyzw": [0, 0, 0, 1], "translation_m": [1, 2, 3], "num_fitted": 5}}')
    similarity: Similarity = load_prior_alignment(path).similarity()
    np.testing.assert_allclose(similarity.apply(np.array([[1.0, 0.0, 0.0]])), [[1.8, 2.0, 3.0]])
    path.write_text('{"x": 1}')
    with pytest.raises(ValueError, match="prior_alignment"):
        load_prior_alignment(path)
