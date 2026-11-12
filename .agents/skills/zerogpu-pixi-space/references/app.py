"""Bootstrap the locked Pixi environment used by this Space.

This file is the Hugging Face entrypoint and deliberately imports nothing but
the standard library: the application's dependencies live in ``pixi.lock`` and
must not leak into the Space's base image.
"""

import os
import subprocess
from pathlib import Path

PIXI_VERSION: str = "v0.83.0"

def main() -> None:
    """Install the pinned Pixi CLI when absent, then become the locked app task."""
    project_dir: Path = Path(__file__).parent
    pixi_home: Path = Path(os.environ.get("PIXI_HOME", Path.home() / ".pixi"))
    pixi_path: Path = pixi_home / "bin" / "pixi"
    environment: dict[str, str] = dict(os.environ)
    environment.update(
        {
            "CONDA_OVERRIDE_CUDA": "13.0",
            "PIXI_HOME": str(pixi_home),
            "PIXI_VERSION": PIXI_VERSION,
        }
    )
    if not pixi_path.exists():
        subprocess.run(
            ["bash", "-c", "curl -fsSL https://pixi.sh/install.sh | bash"],
            check=True,
            cwd=project_dir,
            env=environment,
        )
    # Replace this process: a Space restart (and Dev Mode's Refresh) then signals pixi and the app, not an orphaned parent.
    os.chdir(project_dir)
    os.execve(pixi_path, [str(pixi_path), "run", "--frozen", "app"], environment)


if __name__ == "__main__":
    main()
