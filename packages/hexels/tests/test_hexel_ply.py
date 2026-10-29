import dataclasses
from pathlib import Path

import numpy as np
import pytest

from hexels_bench.hexel_ply import HexelCloud, HexelPly, HexelVariant, property_names, read_hexel_ply, write_hexel_ply


def random_cloud(variant: HexelVariant, count: int, seed: int = 0) -> HexelCloud:
    rng: np.random.Generator = np.random.default_rng(seed)

    def floats(*shape: int) -> np.ndarray:
        return rng.standard_normal(shape).astype(np.float32)

    quats: np.ndarray = floats(count, 4)
    return HexelCloud(
        variant=variant,
        centers_xyz=floats(count, 3),
        quats_wxyz=quats / np.linalg.norm(quats, axis=1, keepdims=True),
        scales_uv=np.abs(floats(count, 2)),
        opacities=rng.uniform(size=count).astype(np.float32),
        sharpness=np.abs(floats(count)) + np.float32(1.0),
        base_rgb=floats(count, 3),
        grad_u_rgb=floats(count, 3),
        grad_v_rgb=floats(count, 3),
        view_linear=floats(count, 3, 3) if variant == "v2" else None,
        sh_coeffs=floats(count, 15, 3) if variant == "v3" else None,
    )


@pytest.mark.parametrize("variant", ["v1", "v2", "v3"])
def test_round_trip_is_bit_exact(tmp_path: Path, variant: HexelVariant) -> None:
    cloud: HexelCloud = random_cloud(variant, 17)
    path: Path = tmp_path / "model.ply"
    write_hexel_ply(path, HexelPly(cloud=cloud, source="runs/x/ckpt_030000.pt", step=30000))
    loaded: HexelPly = read_hexel_ply(path)
    assert loaded.step == 30000
    assert loaded.source == "runs/x/ckpt_030000.pt"
    np.testing.assert_array_equal(loaded.cloud.to_rows(), cloud.to_rows())


@pytest.mark.parametrize(("variant", "width"), [("v1", 20), ("v2", 29), ("v3", 65)])
def test_header_follows_contract(tmp_path: Path, variant: HexelVariant, width: int) -> None:
    path: Path = tmp_path / "model.ply"
    write_hexel_ply(path, HexelPly(cloud=random_cloud(variant, 3), source="s", step=7))
    header: list[str] = path.read_bytes().split(b"end_header\n")[0].decode().splitlines()
    assert header[:6] == [
        "ply",
        "format binary_little_endian 1.0",
        "comment hexels_format 1",
        f"comment hexels_variant {variant}",
        "comment hexels_source s",
        "comment hexels_step 7",
    ]
    assert header[6] == "element vertex 3"
    assert [line.split()[2] for line in header[7:]] == list(property_names(variant))
    assert all(line.split()[1] == "float" for line in header[7:])
    assert len(property_names(variant)) == width
    assert path.stat().st_size == len("\n".join(header)) + len("\nend_header\n") + 3 * width * 4


def test_sh_layout_is_coefficient_major_rgb_innermost() -> None:
    cloud: HexelCloud = random_cloud("v3", 2)
    rows: np.ndarray = cloud.to_rows()
    assert cloud.sh_coeffs is not None
    # sh_(3k+c) = coeffs[k][c]; property sh_7 is coefficient 2, green.
    assert rows[1, 20 + 7] == cloud.sh_coeffs[1, 2, 1]


def test_linear_view_layout_is_row_major() -> None:
    cloud: HexelCloud = random_cloud("v2", 2)
    assert cloud.view_linear is not None
    # a_ij = A[i][j]: a_12 is property 20 + 5.
    assert cloud.to_rows()[0, 20 + 5] == cloud.view_linear[0, 1, 2]


def test_variant_payload_must_match() -> None:
    cloud: HexelCloud = random_cloud("v1", 2)
    with pytest.raises(ValueError, match="sh_coeffs"):
        dataclasses.replace(cloud, variant="v3")


def test_reader_rejects_wrong_property_order(tmp_path: Path) -> None:
    path: Path = tmp_path / "model.ply"
    write_hexel_ply(path, HexelPly(cloud=random_cloud("v1", 2), source="s", step=1))
    data: bytes = path.read_bytes().replace(b"property float su\nproperty float sv\n", b"property float sv\nproperty float su\n")
    path.write_bytes(data)
    with pytest.raises(ValueError, match="properties"):
        read_hexel_ply(path)
