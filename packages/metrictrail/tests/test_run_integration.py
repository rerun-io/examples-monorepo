"""Integration: our logging on a REAL finished upstream run (NAS assets), and the full pipeline run (needs lietorch_mojo + GPU)."""

import json
import os
from pathlib import Path

import pytest
from rerun.chunk import RrdReader

from metrictrail import entities

pytestmark = pytest.mark.integration

UPSTREAM_TUM: Path = Path("/mnt/nas/ssd-8tb/metrictrail/results/upstream/tum/seed1234")
"""Unmodified upstream ViPE @ ead2e00, seed 1234, on the paper's prepared TUM frames (the baseline sweep)."""
TUM_FRAMES: Path = Path("/mnt/nas/ssd-8tb/metrictrail/data/TUM/rgb/desk")
BENCHMARK: Path = Path("/mnt/nas/ssd-8tb/datasets/VSLAM-LAB-Benchmark")
DESK: Path = BENCHMARK / "RGBDTUM" / "rgbd_dataset_freiburg1_desk"
STAIRS: Path = BENCHMARK / "7SCENES" / "stairs_seq-01"
WEIGHTS: Path = Path("/mnt/nas/ssd-8tb/metrictrail/weights")


def require(*assets: Path) -> None:
    for asset in assets:
        if not asset.exists():
            pytest.skip(f"missing asset {asset}")


def test_log_run_on_a_real_upstream_output_matches_the_harness(tmp_path: Path) -> None:
    require(UPSTREAM_TUM / "pose" / "desk.npz", UPSTREAM_TUM / "depth" / "desk.zip", TUM_FRAMES, DESK / "groundtruth.csv")
    from simplecv.rerun_log_utils import RerunTyroConfig

    from metrictrail.apis.log_run import Config, main
    from metrictrail.eval.protocols import TUM
    from metrictrail.eval.score import PaperScore, paper_score
    from metrictrail.eval.vipe_output import read_trajectory
    from metrictrail.frames import list_frames
    from metrictrail.ground_truth import load_gt_context

    rrd: Path = tmp_path / "desk.rrd"
    config = Config(
        run_dir=UPSTREAM_TUM,
        input=TUM_FRAMES,
        benchmark="tum",
        name="desk",
        gt_sequence=DESK,
        label="upstream",
        depth_every=50,
        rr_config=RerunTyroConfig(application_id="metrictrail-test", save=rrd, headless=True),
    )
    metrics = main(config)
    config.rr_config.rec_stream.disconnect()  # finishes the .rrd (footer)
    trajectory = read_trajectory(UPSTREAM_TUM, "desk")
    context = load_gt_context(DESK, TUM, list_frames(TUM_FRAMES))
    paper: PaperScore = paper_score(trajectory.world_T_cam, trajectory.source_frame_ids, context.frames, context.gt, TUM)
    assert metrics.ate_sim3_m == pytest.approx(paper.sim3.rmse_m, rel=1e-12)
    assert metrics.num_gt_pairs == paper.sim3.num_pairs
    schema = RrdReader(rrd).store().schema()
    assert {
        "/" + entities.CAM_RGB,
        "/" + entities.CAM_DEPTH,
        "/" + entities.world_trajectory("upstream"),
        "/" + entities.WORLD_GT_TRAJECTORY,
    } <= set(schema.entity_paths())
    assert f"/{entities.CAM_DEPTH}:EncodedDepthImage:blob" in set(schema.column_names())  # millimetre PNG


def test_full_run_on_a_short_real_sequence(tmp_path: Path) -> None:
    # ViPE's three lietorch copies import lietorch_mojo, whose Mojo extension the `lietorch-mojo-build` task compiles.
    pytest.importorskip("lietorch_mojo._lietorch_mojo_backends", reason="asset: the lietorch-mojo extension (run lietorch-mojo-build)")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device (asset: a GPU)")
    require(STAIRS / "rgb_0", WEIGHTS / "metrictrail" / "dino_dpvo.pth", WEIGHTS / "dinov3", WEIGHTS / "moge2" / "moge-2-vitl" / "model.pt")
    os.environ.setdefault("VIPE_METRICTRAIL_DPVO_CHECKPOINT", str(WEIGHTS / "metrictrail" / "dino_dpvo.pth"))
    os.environ.setdefault("VIPE_DINOV3_VITS16_CHECKPOINT", str(WEIGHTS / "dinov3" / "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"))
    os.environ.setdefault("VIPE_MOGE2_CHECKPOINT", str(WEIGHTS / "moge2" / "moge-2-vitl" / "model.pt"))
    os.environ.setdefault("TORCH_HOME", str(WEIGHTS / "torch_home"))
    from simplecv.rerun_log_utils import RerunTyroConfig

    from metrictrail.apis.run import Config, main

    rrd: Path = tmp_path / "stairs.rrd"
    config = Config(
        input=STAIRS,
        output=tmp_path / "run",
        benchmark="7scenes",
        seed=1234,
        rr_config=RerunTyroConfig(application_id="metrictrail-test", save=rrd, headless=True),
    )
    metrics = main(config)
    config.rr_config.rec_stream.disconnect()
    assert metrics.num_tracked == 100  # 500 frames, every 5th tracked
    assert metrics.ate_sim3_m is not None and metrics.ate_sim3_m < 0.5
    paths: set[str] = set(RrdReader(rrd).store().schema().entity_paths())
    assert {
        "/" + entities.CAM_DINO_PCA,
        "/" + entities.CAM_PATCHES,
        "/" + entities.CAM_NORMALS,
        "/" + entities.DPVO_WINDOW,
        "/" + entities.WORLD_MESH,
        "/" + entities.WORLD_MESH_NORMALS,
        "/" + entities.METRIC_MESH,
    } <= paths
    assert metrics.mesh_triangles is not None and metrics.mesh_triangles > 0
    assert "/" + entities.WORLD_CLOUD not in paths  # upstream's cloud is opt-in (--fuse-cloud)


def test_a_second_run_reuses_the_models_of_the_first(tmp_path: Path) -> None:
    # The app's path: one model cache for the life of the process, so the second click builds no network.
    pytest.importorskip("lietorch_mojo._lietorch_mojo_backends", reason="asset: the lietorch-mojo extension (run lietorch-mojo-build)")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device (asset: a GPU)")
    require(STAIRS / "rgb_0", WEIGHTS / "metrictrail" / "dino_dpvo.pth", WEIGHTS / "dinov3", WEIGHTS / "moge2" / "moge-2-vitl" / "model.pt")
    os.environ.setdefault("VIPE_METRICTRAIL_DPVO_CHECKPOINT", str(WEIGHTS / "metrictrail" / "dino_dpvo.pth"))
    os.environ.setdefault("VIPE_DINOV3_VITS16_CHECKPOINT", str(WEIGHTS / "dinov3" / "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"))
    os.environ.setdefault("VIPE_MOGE2_CHECKPOINT", str(WEIGHTS / "moge2" / "moge-2-vitl" / "model.pt"))
    os.environ.setdefault("TORCH_HOME", str(WEIGHTS / "torch_home"))
    import rerun as rr
    from vipe.utils.model_cache import ModelCache

    from metrictrail.apis.run import RunSettings, run_and_log

    class CountingCache(ModelCache):
        def __init__(self) -> None:
            super().__init__()
            self.built: list[str] = []

        def get(self, key, builder):  # noqa: ANN001, ANN202 - ViPE's untyped signature
            return super().get(key, lambda: (self.built.append(key), builder())[1])

    models = CountingCache()
    settings = [RunSettings(input=STAIRS, output=tmp_path / f"run{index}", benchmark="7scenes", seed=1234, fuse_mesh=False) for index in range(2)]
    rec = rr.RecordingStream("metrictrail-test", recording_id="models-once")
    rec.save(tmp_path / "first.rrd")
    first = run_and_log(settings[0], rec, live=True, models=models)
    built_first: list[str] = list(models.built)
    assert any("moge" in key for key in built_first) and any("dino_dpvo" in key for key in built_first) and any("droid" in key for key in built_first)
    rec.save(tmp_path / "second.rrd")
    second = run_and_log(settings[1], rec, live=True, models=models)
    assert models.built == built_first  # nothing built anew
    assert second.logged.metrics.num_tracked == first.logged.metrics.num_tracked == 100
    for run in settings:
        # The live calibration view ran inside SharedK's sweep; ViPE turns any exception there into its MoGe-2 focal.
        record: dict[str, object] = json.loads(next((run.output / "vipe_out" / "vipe").glob("*_metrictrail.json")).read_text())["calibration"]
        fallback = record.get("calibration_fallback")
        assert not (isinstance(fallback, dict) and fallback.get("reason") == "vipe_stage_raised"), fallback
    rec.disconnect()  # finishes the .rrd (footer)
    paths: set[str] = set(RrdReader(tmp_path / "second.rrd").store().schema().entity_paths())
    assert {"/" + entities.CALIBRATION_PATH, "/" + entities.CALIBRATION_FOCAL, "/" + entities.CALIBRATION_NEWEST_DEPTH} <= paths
