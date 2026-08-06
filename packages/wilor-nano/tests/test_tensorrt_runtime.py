from __future__ import annotations

import subprocess
import sys


def test_tensorrt_runtime_import_does_not_load_pytorch_vit_stack() -> None:
    command: str = (
        "import sys; "
        "import wilor_nano.api.tensorrt_runtime; "
        "assert 'wilor_nano.models.vit' not in sys.modules, sorted(sys.modules); "
        "assert 'timm' not in sys.modules, sorted(sys.modules)"
    )
    completed_process: subprocess.CompletedProcess[str] = subprocess.run(
        [sys.executable, "-c", command],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed_process.returncode == 0, completed_process.stderr
