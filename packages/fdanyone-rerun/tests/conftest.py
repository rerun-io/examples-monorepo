"""One GPU run of the example clip per session, shared by the integration and golden lanes."""

import os
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def example_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Run the Space's policy (six views, one ring at 15 degrees, Turbo, seed 42) on the example clip, or reuse
    ``$FDANYONE_TEST_RUN`` (a finished run directory of the same policy)."""
    reuse: str | None = os.environ.get("FDANYONE_TEST_RUN")
    if reuse:
        return Path(reuse)
    # Imported here, not at module level: pytest-beartype hooks imports after conftest.py loads (AGENTS.md, Beartype).
    from fdanyone_rerun.assets import EXAMPLE_VIDEO, model_dir

    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    if not EXAMPLE_VIDEO.is_file() or not (model_dir() / "4danyone" / "model.safetensors").is_file():
        pytest.skip("needs the weights and the example clip: pixi run -e fdanyone-rerun-dev fdanyone-download")
    from fdanyone_rerun.core import RunConfig, run_views

    output_dir: Path = tmp_path_factory.mktemp("fdanyone") / "example"
    run_views(RunConfig(video=EXAMPLE_VIDEO, output_dir=output_dir))
    return output_dir
