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
