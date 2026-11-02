import json
from pathlib import Path

import numpy as np
import pytest
import torch

from hexels_bench.apis.export_published_images import stage_published_image
from hexels_bench.image2d import PROPERTIES_2D, HexelImage, ImageFit, hexel_image_from_fields, load_image_fit, write_hexel_image_ply
from hexels_bench.layout import Roots
from hexels_bench.png16 import read_png16


def save_fit(path: Path, layout: str) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    fields: dict[str, torch.Tensor] = {
        "positions": torch.rand(4, 2, generator=generator), "log_scales": torch.randn(4, 2, generator=generator),
        "rotations": torch.randn(4, 1, generator=generator), "features": torch.rand(4, 3, generator=generator),
    }  # fmt: skip
    if layout == "dense15":
        fields["color_gradients"] = torch.randn(4, 2, 3, generator=generator)
        fields["softness"] = torch.rand(4, generator=generator) * 20
    torch.save({"step": 5100, "method": "hexel", "layout": layout, "scale_mode": "log", "height": 30, "width": 40, "parameters": fields}, path)
    return fields


def test_dense_fit_maps_to_pixel_space(tmp_path: Path) -> None:
    fields: dict[str, torch.Tensor] = save_fit(tmp_path / "fit.pt", "dense15")
    fit: ImageFit = load_image_fit(tmp_path / "fit.pt")
    assert fit.step == 5100
    image: HexelImage = fit.image
    np.testing.assert_allclose(image.centers_xy, fields["positions"].numpy() * [40, 30], rtol=1e-6)
    np.testing.assert_allclose(image.scales_uv, np.exp(fields["log_scales"].numpy()), rtol=1e-6)
    np.testing.assert_array_equal(image.angles, fields["rotations"].numpy()[:, 0])
    np.testing.assert_array_equal(image.grad_y_rgb, fields["color_gradients"].numpy()[:, 1])
    np.testing.assert_array_equal(image.sharpness, fields["softness"].numpy())


def test_flat_fit_gets_default_softness_and_zero_gradients(tmp_path: Path) -> None:
    save_fit(tmp_path / "fit.pt", "flat8")
    image: HexelImage = load_image_fit(tmp_path / "fit.pt").image
    assert np.all(image.sharpness == 10.0)
    assert not image.grad_x_rgb.any()
    write_hexel_image_ply(tmp_path / "fit.ply", image, "fit.pt", 5100)
    header: str = (tmp_path / "fit.ply").read_bytes().split(b"end_header\n")[0].decode()
    assert "comment hexels_variant 2d" in header and "comment hexels_image 40 30" in header
    assert [line.split()[2] for line in header.splitlines() if line.startswith("property")] == list(PROPERTIES_2D)


def test_decoded_fields_map_like_upstreams_decoder() -> None:
    """Stored-file fields (numpy, as upstream's .hxm/.hxc decoders return them) take the same mapping."""
    rng = np.random.default_rng(1)
    fields: dict[str, np.ndarray] = {
        "positions": rng.random((3, 2), dtype=np.float32), "log_scales": rng.standard_normal((3, 2), dtype=np.float32),
        "rotations": rng.standard_normal((3, 1), dtype=np.float32), "features": rng.random((3, 3), dtype=np.float32),
        "color_gradients": rng.standard_normal((3, 2, 3), dtype=np.float32), "softness": rng.random(3, dtype=np.float32) * 20,
    }  # fmt: skip
    image: HexelImage = hexel_image_from_fields(2048, 1024, fields)
    rows: np.ndarray = image.to_rows()
    assert rows.shape == (3, 15)
    np.testing.assert_array_equal(rows[:, 0], fields["positions"][:, 0] * np.float32(2048))
    np.testing.assert_array_equal(rows[:, 1], fields["positions"][:, 1] * np.float32(1024))
    np.testing.assert_array_equal(rows[:, 2:4], np.exp(fields["log_scales"]))
    np.testing.assert_array_equal(rows[:, 5], fields["softness"])
    np.testing.assert_array_equal(rows[:, 9:12], fields["color_gradients"][:, 0])
    np.testing.assert_array_equal(rows[:, 12:15], fields["color_gradients"][:, 1])


@pytest.mark.integration
def test_published_packed_fit_decodes_through_upstream() -> None:
    pytest.importorskip("scripts.decode_image", reason="needs upstream hexels (hexels env)")
    folder: Path = Roots().data / "hexels-pretrained/images_2d/anime-1_2k/hexels_spatial-rgb_pruned-20000-to-10000_seed123_step005100"
    if not folder.is_dir():
        pytest.skip(f"needs {folder} (READY/pretrained-2d)")
    packed: HexelImage = load_image_fit(folder / "hexels_spatial-rgb_mixed14.hxm").image
    raw: HexelImage = load_image_fit(folder / "hexels_raw-fp32.pt").image
    assert packed.centers_xy.shape == raw.centers_xy.shape == (10000, 2)
    # mixed14 stores positions with 16 bits over the field's range: within 2 ulp of a code step.
    assert np.max(np.abs(packed.centers_xy - raw.centers_xy)) < 2048 * 1.1 / 65535 * 2
    assert np.max(np.abs(packed.sharpness - raw.sharpness)) < 0.05


@pytest.mark.integration
def test_published_anime_fit_exports() -> None:
    fit: Path = Roots().data / "hexels-pretrained/images_2d/anime-1_2k/hexels_spatial-rgb_pruned-20000-to-10000_seed123_step005100/hexels_raw-fp32.pt"
    if not fit.is_file():
        pytest.skip(f"needs {fit} (READY/pretrained-2d)")
    loaded: ImageFit = load_image_fit(fit)
    image: HexelImage = loaded.image
    assert (image.width, image.height, loaded.step) == (2048, 2048, 5100)
    assert image.centers_xy.shape == (10000, 2)
    assert np.isfinite(image.to_rows()).all()


@pytest.mark.integration
def test_stage_published_image_writes_ply_and_lossless_gt(tmp_path: Path) -> None:
    data: Path = Roots().data
    for marker in ("pretrained-2d", "imagegs"):
        if not (data / "READY" / marker).is_file():
            pytest.skip(f"needs {data}/READY/{marker}")
    roots = Roots(data=data, runs=tmp_path)
    ply: Path = stage_published_image(roots, "anime-1_2k", "spatial-rgb")
    assert ply.stat().st_size > 10000 * 15 * 4
    gt: np.ndarray = read_png16(tmp_path / "imagegs" / "anime-1_2k" / "gt" / "test" / "anime-1_2k.png")
    assert gt.shape == (2048, 2048, 3)
    assert np.all(gt % 257 == 0)


def test_upstream_fit_records_read_the_fields_the_tools_use(tmp_path: Path) -> None:
    """`scripts.fit_image` metrics.json and a published fit's metadata.json, as upstream writes them (extra fields kept out)."""
    from hexels_bench.image2d import FitMetrics, PublishedFitMetadata
    from hexels_bench.layout import read_json

    checkpoints: list[dict] = [{"step": 10000, "bits": bits, "psnr_db": 30.0 + bits / 100, "file": "x", "bytes": 1} for bits in (32, 16)]
    fit: dict = {"complete": True, "job": {"max_steps": 10000, "image": "a.png"}, "training_seconds": 521.9, "gpu": "GPU", "checkpoints": checkpoints}
    (tmp_path / "metrics.json").write_text(json.dumps(fit))
    metrics: FitMetrics = read_json(FitMetrics, tmp_path / "metrics.json")
    assert (metrics.job.max_steps, metrics.training_seconds, metrics.gpu) == (10000, 521.9, "GPU")
    assert {checkpoint.bits: checkpoint.psnr_db for checkpoint in metrics.checkpoints} == {32: 30.32, 16: 30.16}
    representations: list[dict] = [{"format": "mixed14", "path": "p", "expected_metrics": {"psnr_db": 35.8, "ssim": 0.9}}]
    (tmp_path / "metadata.json").write_text(json.dumps({"id": "x", "representations": representations}))
    metadata: PublishedFitMetadata = read_json(PublishedFitMetadata, tmp_path / "metadata.json")
    assert [(item.format, item.expected_metrics.psnr_db) for item in metadata.representations] == [("mixed14", 35.8)]
