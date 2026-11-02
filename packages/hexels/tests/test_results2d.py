import json
from pathlib import Path

import numpy as np
import pytest
from serde import from_dict

from hexels_bench.paper import PaperNumbers
from hexels_bench.results2d import (
    IDENTICAL_PSNR,
    Condition,
    ConditionScores,
    ImageRun,
    TableRow,
    image_category,
    load_image_run,
    mae_map,
    psnr_between,
    table_rows,
    worst_window,
    write_image_run,
)


def scores(condition: Condition, psnr: float = 40.0) -> ConditionScores:
    return ConditionScores(
        condition=condition, source="fit.hxm", render="renders/test/anime-1_2k.png", psnr=psnr, ssim=0.98, lpips_alex=0.04,
        fps=250.0, trial_ms=[4.0] * 5, reference_psnr=None, vs_cuda_psnr=None,
    )  # fmt: skip


def run(conditions: list[ConditionScores], primary: Condition = "decoded_hxm") -> ImageRun:
    return ImageRun(
        method="hexels-2d-pub-wgsl", image="anime-1_2k", category="anime", decoder="wgsl", gpu="RTX", host="h",
        primitives=10000, bytes_fp32=600000, bytes_packed=253524, primary=primary, conditions=conditions,
    )  # fmt: skip


def test_image_run_round_trips_and_selects_the_primary(tmp_path: Path) -> None:
    original: ImageRun = run([scores("decoded_hxm", 43.5), scores("raw_fp32", 43.6)])
    write_image_run(tmp_path / "metrics.json", original)
    loaded: ImageRun = load_image_run(tmp_path / "metrics.json")
    assert loaded == original
    assert loaded.scores().psnr == 43.5
    assert loaded.scores("raw_fp32").psnr == 43.6


def test_image_run_rejects_missing_primary_and_duplicates() -> None:
    with pytest.raises(ValueError):
        run([scores("raw_fp32")])
    with pytest.raises(ValueError):
        run([scores("decoded_hxm"), scores("decoded_hxm")])


def test_category_is_the_stem_prefix() -> None:
    assert image_category("painting-7_2k") == "painting"
    with pytest.raises(ValueError):
        image_category("garden")


def test_psnr_between_renders_caps_identical_images() -> None:
    first: np.ndarray = np.full((4, 4, 3), 1000, dtype=np.uint16)
    assert psnr_between(first, first) == IDENTICAL_PSNR
    second: np.ndarray = first.copy()
    second[0, 0, 0] += 655  # 0.01 in one of 48 elements
    expected: float = -10.0 * np.log10((655 / 65535) ** 2 / 48)
    assert psnr_between(first, second) == pytest.approx(expected)


def test_mae_map_clamps_the_prediction_and_averages_channels() -> None:
    prediction: np.ndarray = np.array([[[1.5, 0.2, -0.1]]], dtype=np.float32)
    target: np.ndarray = np.array([[[1.0, 0.5, 0.0]]], dtype=np.float32)
    np.testing.assert_allclose(mae_map(prediction, target), [[0.1]], rtol=1e-6)


def test_worst_window_finds_the_hot_patch() -> None:
    gain: np.ndarray = np.zeros((64, 96), dtype=np.float32)
    gain[40:56, 70:86] = 1.0
    assert worst_window(gain, size=16, stride=2) == (70, 40)
    assert worst_window(gain, size=16, stride=8) == (72, 40)


def test_table_rows_give_per_image_rows_and_a_mean_with_published_and_table1_numbers(tmp_path: Path) -> None:
    published: dict = {"psnr_hxm": 0.0, "ssim_hxm": 0.9, "lpips_alex_hxm": 0.1, "psnr_uint16": 0.0, "median_fps_uint16_a5000": 200.0}
    paper: PaperNumbers = from_dict(
        PaperNumbers,
        {
            "source": {"hardware_paper": "RTX A5000 24 GB"},
            "hf_nvs_per_scene": {},
            "sfm_points": {},
            "table1_2d": [{"method": "Hexels", "psnr": 35.82, "ssim": 0.9359, "lpips_alex": 0.0924}],
            "hf_images_2d_per_image": {
                "anime-1_2k": {"spatial-rgb": published | {"psnr_hxm": 41.0, "psnr_uint16": 41.1}},
                "art-1_2k": {"spatial-rgb": published | {"psnr_hxm": 21.0, "psnr_uint16": 21.1}},
            },
            "hf_surface_main_per_scene": {},
        },
    )
    for stem, psnr, seconds in (("anime-1_2k", 40.0, 500.0), ("art-1_2k", 20.0, 700.0)):
        image = ImageRun(
            method="hexels-2d", image=stem, category=image_category(stem), decoder="cuda", gpu="RTX 3060", host="h",
            primitives=10000, bytes_fp32=600000, bytes_packed=253524, primary="decoded_hxm",
            conditions=[scores("decoded_hxm", psnr), scores("raw_fp32", psnr + 0.1)],
        )  # fmt: skip
        write_image_run(tmp_path / stem / "hexels-2d" / "metrics.json", image)
        train: dict = {"method": "hexels-2d", "dataset": "imagegs", "scene": stem, "steps": 5100, "loop_seconds": seconds, "total_seconds": seconds + 9.0,
                       "peak_vram_gib": 2.0, "primitives": 10000, "sfm_points": None, "gpu": "RTX 3060", "host": "h", "source_sha": "x", "command": "fit"}  # fmt: skip
        (tmp_path / stem / "hexels-2d" / "train.json").write_text(json.dumps(train))
    rows: list[TableRow] = table_rows(tmp_path, ["anime-1_2k", "art-1_2k"], paper)
    by_key: dict[tuple[str, str], TableRow] = {(row.condition, row.group): row for row in rows}
    assert {row.method for row in rows} == {"hexels-2d"}
    assert by_key["decoded_hxm", "anime-1_2k"].psnr == 40.0 and by_key["decoded_hxm", "anime-1_2k"].published_psnr == 41.0
    assert by_key["decoded_hxm", "all"].psnr == pytest.approx(30.0) and by_key["decoded_hxm", "all"].published_psnr == pytest.approx(31.0)
    assert by_key["decoded_hxm", "all"].images == 2 and by_key["decoded_hxm", "all"].train_seconds == pytest.approx(600.0)
    assert by_key["decoded_hxm", "all"].table1_psnr == 35.82 and by_key["decoded_hxm", "art-1_2k"].table1_psnr is None
    assert by_key["raw_fp32", "all"].psnr == pytest.approx(30.1) and by_key["raw_fp32", "all"].published_psnr is None
