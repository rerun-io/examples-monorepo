from pathlib import Path

import pytest

from metrictrail.benchmarks import BENCHMARKS, BenchmarkPreset


def test_tum_overrides_match_upstream_command() -> None:
    # configs/slam/README_metrictrail.md, "tum" (ViPE @ ead2e00).
    overrides: list[str] = BENCHMARKS["tum"].run_overrides(Path("data/TUM/rgb/fr1_desk"), Path("vipe_results/tum"))
    assert overrides == [
        "pipeline=metrictrail",
        "slam@pipeline.slam=metrictrail_gba_refine",
        "streams=frame_dir_stream",
        "streams.base_path=data/TUM/rgb/fr1_desk",
        "pipeline.init.input_raster.policy=unchanged",
        "pipeline.init.calibration.allow_unstable=true",
        "pipeline.output.path=vipe_results/tum",
    ]


def test_sevenscenes_overrides_match_upstream_command() -> None:
    overrides: list[str] = BENCHMARKS["7scenes"].run_overrides(Path("data/7-Scenes/rgb/chess__seq-01"), Path("vipe_results/sevenscenes"))
    assert overrides == [
        "pipeline=metrictrail",
        "slam@pipeline.slam=metrictrail_gba",
        "streams=frame_dir_stream",
        "streams.base_path=data/7-Scenes/rgb/chess__seq-01",
        "pipeline.init.input_raster.policy=resize",
        "pipeline.init.input_raster.height=434",
        "pipeline.init.input_raster.width=574",
        "pipeline.init.tracking_stride=5",
        "pipeline.init.calibration.allow_unstable=true",
        "pipeline.output.path=vipe_results/sevenscenes",
    ]


def test_lamaria_sets_seed_null_unless_a_seed_is_given() -> None:
    lamaria: BenchmarkPreset = BENCHMARKS["lamaria"]
    default_run: list[str] = lamaria.run_overrides(Path("frames/seq"), Path("out"))
    assert "pipeline.slam.seed=null" in default_run
    assert "pipeline.init.input_raster.max_frames=100000" in default_run
    seeded_run: list[str] = lamaria.run_overrides(Path("frames/seq"), Path("out"), seed=1235)
    assert "pipeline.slam.seed=1235" in seeded_run
    assert "pipeline.slam.seed=null" not in seeded_run


def test_known_intrinsics_replaces_allow_unstable() -> None:
    overrides: list[str] = BENCHMARKS["kitti_calibrated"].run_overrides(Path("rgb/00"), Path("out"), intrinsics_path=Path("prepared/00.npz"))
    assert "pipeline.optional.intrinsics={path: prepared/00.npz}" in overrides
    assert "pipeline.init.calibration.allow_unstable=true" not in overrides
    with pytest.raises(ValueError, match="intrinsics_path"):
        BENCHMARKS["kitti_calibrated"].run_overrides(Path("rgb/00"), Path("out"))
    with pytest.raises(ValueError, match="intrinsics_path"):
        BENCHMARKS["tum"].run_overrides(Path("rgb/x"), Path("out"), intrinsics_path=Path("K.txt"))


def test_preset_rejects_inconsistent_raster() -> None:
    with pytest.raises(ValueError, match="raster_hw"):
        BenchmarkPreset("tum", "metrictrail_gba_refine", "cover_crop")
    with pytest.raises(ValueError, match="raster_hw"):
        BenchmarkPreset("tum", "metrictrail_gba_refine", "unchanged", raster_hw=(10, 10))


def test_video_preset_is_the_readme_quick_start() -> None:
    # README.md quick start (ViPE @ ead2e00): python run.py pipeline=metrictrail streams.base_path=<clip>.mp4 pipeline.output.path=...
    overrides: list[str] = BENCHMARKS["video"].run_overrides(Path("clips/demo.mp4"), Path("out"))
    assert overrides == ["pipeline=metrictrail", "streams.base_path=clips/demo.mp4", "pipeline.output.path=out"]


def test_seed_and_seed_name_overrides() -> None:
    overrides: list[str] = BENCHMARKS["tum"].run_overrides(Path("frames/rgbd_dataset_freiburg1_desk"), Path("out"), seed=1235, seed_name="desk")
    assert overrides[-3:] == ["pipeline.slam.seed=1235", "pipeline.slam.seed_name=desk", "pipeline.output.path=out"]


def test_every_requested_benchmark_has_a_preset() -> None:
    assert {"tum", "7scenes", "kitti", "kitti_calibrated", "wild_slam", "lamaria", "video"} <= set(BENCHMARKS)
    assert BENCHMARKS["wild_slam"].run_overrides(Path("rgb/ball"), Path("out"))[4:7] == [
        "pipeline.init.input_raster.policy=cover_crop",
        "pipeline.init.input_raster.height=288",
        "pipeline.init.input_raster.width=512",
    ]


def test_only_the_video_preset_keeps_the_pipeline_raster() -> None:
    with pytest.raises(ValueError, match="pipeline's default raster"):
        BenchmarkPreset("tum", "metrictrail_local", "pipeline_default")
    with pytest.raises(ValueError, match="pipeline's default raster"):
        BenchmarkPreset("video", "metrictrail_local", "unchanged", stream="video")


def test_vslamlab_raw_tum_frames_get_the_paper_raster_at_read_time() -> None:
    from metrictrail.benchmarks import VSLAMLAB_RASTER_OVERRIDES

    assert VSLAMLAB_RASTER_OVERRIDES["tum"] == [
        "pipeline.init.input_raster.policy=resize",
        "pipeline.init.input_raster.height=388",
        "pipeline.init.input_raster.width=518",
        "pipeline.init.input_raster.resampler=bilinear",
    ]
