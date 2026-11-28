"""fdanyone-run start-up: the entry module stays light."""

import subprocess
import sys


def test_the_cli_entry_does_not_import_torchmetrics() -> None:
    # 2 s of fdanyone-run's start-up (with torchvision.models); the run itself never scores (fused SSIM, no holdout)
    probe: str = "import sys, fdanyone_rerun.apis.run; print(sorted(m for m in ('torchmetrics', 'open4dgs.evaluate') if m in sys.modules))"
    done = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "[]"
