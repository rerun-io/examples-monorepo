"""The suite skips cleanly, module by module, when the Mojo extension is not built (a fresh checkout)."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

TESTS_DIR: Path = Path(__file__).resolve().parent
INNER_RUN: str = "LIETORCH_MOJO_INNER_COLLECTION_RUN"

BLOCK_EXTENSION_PLUGIN: str = '''"""Make the compiled Mojo extension unimportable."""
import sys

sys.modules["lietorch_mojo._lietorch_mojo_backends"] = None
'''


@pytest.mark.skipif(os.environ.get(INNER_RUN) == "1", reason="this is the inner run")
def test_unbuilt_extension_skips_instead_of_crashing(tmp_path: Path) -> None:
    """Regression (verify-api r2): a conftest-level importorskip aborted collection with exit 1."""
    (tmp_path / "block_mojo_extension.py").write_text(BLOCK_EXTENSION_PLUGIN)
    env: dict[str, str] = {
        **os.environ,
        INNER_RUN: "1",
        "PYTHONPATH": os.pathsep.join([str(tmp_path), str(TESTS_DIR.parent), os.environ.get("PYTHONPATH", "")]),
    }
    cmd: list[str] = [sys.executable, "-m", "pytest", "-p", "block_mojo_extension", "-p", "no:cacheprovider", "-q", "-rs"]
    # test_math.py needs no extension (it builds its own probe), so it is left out to keep this fast.
    cmd += ["--ignore", str(TESTS_DIR / "test_math.py"), str(TESTS_DIR)]
    result = subprocess.run(cmd, cwd=TESTS_DIR.parent, env=env, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "build the Mojo extension first" in result.stdout, result.stdout
