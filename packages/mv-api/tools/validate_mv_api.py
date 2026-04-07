from __future__ import annotations

import importlib
import runpy
import subprocess
import sys
from pathlib import Path

PACKAGE_DIR: Path = Path(__file__).resolve().parent.parent

MODULES: tuple[str, ...] = (
    "mv_api",
    "mv_api.api.full_exoego_pipeline",
    "mv_api.api.exo_only_calibration",
    "mv_api.api.batch_calibration",
    "mv_api.gradio_ui.full_pipeline_rrd_ui",
)

TOOLS_WITH_HELP: tuple[str, ...] = (
    "tools/run_rrd_client_example.py",
    "tools/run_exo_only_calib.py",
    "tools/batch_exo_calib_client.py",
)


def _run_checked(args: list[str]) -> None:
    subprocess.run(args, cwd=PACKAGE_DIR, check=True)


def main() -> None:
    for module_name in MODULES:
        importlib.import_module(module_name)

    for rel_path in TOOLS_WITH_HELP:
        _run_checked([sys.executable, rel_path, "--help"])

    runpy.run_path(str(PACKAGE_DIR / "tools" / "app_full_pipeline_rrd.py"), run_name="mv_api_app_validation")
    print("mv-api validation completed successfully.")


if __name__ == "__main__":
    main()
