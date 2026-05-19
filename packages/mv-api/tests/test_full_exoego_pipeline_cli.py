import subprocess
import sys
from pathlib import Path


def test_node_app_cli_help_exposes_raw_hocap_and_max_frames() -> None:
    package_root: Path = Path(__file__).resolve().parents[1]
    result: subprocess.CompletedProcess[str] = subprocess.run(
        [sys.executable, "tools/apps/full_exoego_app.py", "--help"],
        cwd=package_root,
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert "hocap" in result.stdout
    assert "synced-videos" in result.stdout
    assert "--max-frames" in result.stdout
    assert "--calib-config.device" in result.stdout
    assert "--tracker-config.device" in result.stdout
    assert "--calib-ts-nano" in result.stdout


def test_legacy_full_pipeline_script_has_been_removed() -> None:
    package_root: Path = Path(__file__).resolve().parents[1]
    assert not (package_root / "tools" / "run_exoego_full.py").exists()
