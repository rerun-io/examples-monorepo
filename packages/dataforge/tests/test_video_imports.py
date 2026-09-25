"""The shared encoders and the dataset registry import without PyTurboJPEG (slam-rs envs lack it)."""

import subprocess
import sys


def test_registry_imports_do_not_load_turbojpeg() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import dataforge.datasets, dataforge.logging_toolkit, dataforge.video_encoding; assert 'turbojpeg' not in sys.modules",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
