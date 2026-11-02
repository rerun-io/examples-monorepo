import json
from pathlib import Path

from hexels_bench.upstream import absolute_job


def test_absolute_job_resolves_upstream_relative_paths(tmp_path: Path) -> None:
    data = tmp_path / "data-root"
    (data / "dtu" / "scan24").mkdir(parents=True)
    cwd = tmp_path / "cwd"
    (cwd / "pretrained" / "Hexels" / "cameras").mkdir(parents=True)
    (cwd / "data").symlink_to(data, target_is_directory=True)
    settings = cwd / "pretrained" / "Hexels" / "cameras" / "settings.json"
    settings.write_text(json.dumps({"source_path": "data/dtu/scan24", "llffhold": 8}))
    job = tmp_path / "evaluation_job.json"
    job.write_text(
        json.dumps(
            {
                "model_path": "pretrained/Hexels/nvs/dtu/scan24/m",
                "camera_contract": "pretrained/Hexels/cameras/contract.json",
                "camera_settings": "pretrained/Hexels/cameras/settings.json",
                "result": "outputs/x/metrics.json",
                "iteration": 30000,
            }
        )
    )

    out = absolute_job(job, cwd, tmp_path / "out")

    rewritten = json.loads(out.read_text())
    assert Path(rewritten["model_path"]) == (cwd / "pretrained/Hexels/nvs/dtu/scan24/m").resolve()
    assert Path(rewritten["camera_contract"]) == (cwd / "pretrained/Hexels/cameras/contract.json").resolve()
    assert rewritten["iteration"] == 30000
    assert rewritten["result"] == "outputs/x/metrics.json"
    rewritten_settings = json.loads(Path(rewritten["camera_settings"]).read_text())
    assert Path(rewritten_settings["source_path"]) == (data / "dtu" / "scan24").resolve()
    assert rewritten_settings["llffhold"] == 8


def test_published_evaluation_reads_hf_files_and_evaluator_results(tmp_path: Path) -> None:
    """HF evaluation.json has no benchmark; the evaluator's result has one; both type the per-view rows."""
    from hexels_bench.apis.render_published import upstream_float_metrics
    from hexels_bench.layout import read_json
    from hexels_bench.metrics import MetricsFile
    from hexels_bench.upstream import PublishedEvaluation

    views: list[dict] = [
        {"image_name": name, "psnr_db": 30.0 + index, "ssim": 0.9, "lpips_vgg": 0.1, "rgb_sse": 1.0} for index, name in enumerate(("0000", "0008"))
    ]
    published: PublishedEvaluation = read_json(PublishedEvaluation, write(tmp_path / "evaluation.json", {"complete": True, "per_view": views}))
    assert published.benchmark is None and [view.image_name for view in published.per_view] == ["0000", "0008"]
    result: PublishedEvaluation = read_json(
        PublishedEvaluation, write(tmp_path / "result.json", {"per_view": views, "benchmark": {"median_fps": 1599.8, "trials": []}})
    )
    shared = MetricsFile(
        method="hexels-v1-30k-pub-cuda", dataset="dtu", scene="scan24", views=2, psnr=30.0, ssim=0.9, lpips_vgg=0.1,
        fps=1600.0, deploy_mb=1.0, ply_mb=2.0, scorer="shared-png16", per_view=[],
    )  # fmt: skip
    metrics: MetricsFile = upstream_float_metrics(result, "hexels-v1-30k-pub-cuda", "dtu", "scan24", shared)
    assert (metrics.scorer, metrics.fps, metrics.psnr, metrics.deploy_mb) == ("upstream-float", 1599.8, 30.5, 1.0)


def write(path: Path, record: dict) -> Path:
    path.write_text(json.dumps(record))
    return path
