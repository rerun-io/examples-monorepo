"""``open4dgs-train``'s flags become typed inputs once (clip source, init, prior), and the run report keeps the summary
layout the runs on open4dgs-runs already have."""

from pathlib import Path

import pytest

pytest.importorskip("trigs", reason="the train API imports the TRiGS adapter (open4dgs envs)")

from open4dgs.apis.train import Config  # noqa: E402
from open4dgs.compensation import BrightnessSplit  # noqa: E402
from open4dgs.run_inputs import BodyInit, CatalogWindow, LayerInit, RrdInit, SelfCapWindow  # noqa: E402
from open4dgs.run_report import RunReport  # noqa: E402
from open4dgs.trigs_adapter import CameraCompensation  # noqa: E402


def test_flags_choose_one_clip_source() -> None:
    catalog = Config(segment_id="s", frame_start=5, roi="people", roi_pad=0.2).clip_source()
    assert isinstance(catalog, CatalogWindow) and catalog.frame_start == 5 and catalog.roi_pad == 0.2
    assert Config().clip_source().roi_pad is None  # type: ignore[union-attr]

    selfcap = Config(segment_id="s", selfcap_time_start_s=3.0, selfcap_roi=True, selfcap_capture_times=True).clip_source()
    assert isinstance(selfcap, SelfCapWindow) and selfcap.time_start_s == 3.0 and selfcap.roi and selfcap.capture_times


@pytest.mark.parametrize("flag", ["selfcap_capture_times", "selfcap_roi"])
def test_selfcap_flags_without_a_selfcap_window_are_refused(flag: str) -> None:
    with pytest.raises(ValueError, match="selfcap-time-start-s"):
        Config(**{flag: True})  # type: ignore[arg-type]


def test_flags_choose_one_init() -> None:
    assert Config(samples_per_person=7, keyframe_step=2).init_source() == BodyInit("fdanyone-views", Config().segment_id, 7, 2)
    layer = Config(init="layer", mask_dataset="companion", init_layer="init_roma").init_source()
    assert isinstance(layer, LayerInit) and layer.dataset == "companion" and layer.layer == "init_roma"
    assert Config(init="layer", init_rrd=Path("x.rrd")).init_source() == RrdInit(Path("x.rrd"))  # the recording wins, as before


def report(**overrides: object) -> RunReport:
    means = {"crop_raw_psnr": 20.0, "crop_raw_ssim": 0.8, "crop_raw_lpips": 0.2, "fg_psnr": 15.0, "alpha_iou": 0.9}
    values: dict[str, object] = {
        "holdout": means,
        "holdout_init": means,
        "holdout_floors": {"black": means, "mean": means},
        "train": {},
        "geometry_holdout": {"median_abs_cm": 2.0},
        "brightness": {"rig_00/cam_00": BrightnessSplit(1.0, 0.9, 1.1, 0.8)},
        "brightness_train": {},
        "learned": {"rig_00/cam_00": CameraCompensation()},
        "holdout_fitted": {},
        "it_per_s": 50.0,
        "train_s": 30.0,
        "peak_gpu_gb": 10.0,
        "init_rows": 100,
        "final_gaussians": 90,
        "loss_rgb_first": 0.3,
        "loss_rgb_last": None,
        "prior_scalars": {"prior_depth_first": 0.1},
        "timings": {"read_clip_s": 1.0},
        **overrides,
    }
    return RunReport(**values)  # type: ignore[arg-type]


def test_report_summary_keeps_the_layout_of_the_runs_on_the_catalog() -> None:
    summary = report().summary()
    assert list(summary)[:6] == ["holdout", "holdout_init", "holdout_floor_black", "holdout_floor_mean", "train", "geometry_holdout"]
    assert "train_appearance" not in summary and "loss_rgb_last" not in summary  # optional groups and unscored values are absent
    assert summary["brightness"] == {"rig_00/cam_00": {"ratio": 1.0, "alpha_part": 0.9, "colour_part": 1.1, "alpha_mean": 0.8}}

    flat = report(holdout_odd_fit={"crop_raw_psnr": 21.0}).summary_properties()
    assert flat["holdout_crop_raw_psnr"] == 20.0 and flat["holdout_floor_mean_alpha_iou"] == 0.9 and flat["holdout_odd_fit_crop_raw_psnr"] == 21.0
    assert flat["brightness_rig_00_cam_00_ratio"] == 1.0 and flat["init_rows"] == 100.0 and flat["prior_depth_first"] == 0.1
    assert flat["read_clip_s"] == 1.0 and not any(k.startswith("learned") for k in flat)


def test_headline_reads_each_group() -> None:
    line: str = report().headline()
    assert line.startswith("[open4dgs] crop_raw PSNR/SSIM/LPIPS holdout 20.00/0.800/0.200 fg 15.00 iou 0.900;") and line.endswith("train n/a")
