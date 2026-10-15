"""dagkit imports no torch (decision 10): every module imports in a fresh interpreter that blocks torch."""

import pkgutil
import subprocess
import sys

import dagkit


def test_every_module_imports_without_torch() -> None:
    modules: list[str] = ["dagkit"] + [m.name for m in pkgutil.walk_packages(dagkit.__path__, "dagkit.")]
    code: str = (
        "import sys\n"
        "sys.modules['torch'] = None  # any `import torch` raises ImportError\n"
        f"for name in {modules!r}:\n"
        "    __import__(name)\n"
        "assert sys.modules['torch'] is None, 'torch was imported'\n"
        "print(len(sys.modules))\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert len(modules) >= 12
