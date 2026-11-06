"""Freeze upstream lietorch's outputs into tests/fixtures/lietorch_references.rrd (see tests/freeze_references.py).

Needs lietorch and a CUDA GPU; `pixi run -e lietorch-mojo-dev lietorch-mojo-freeze-references` runs it in a `pixi exec`
environment so lietorch never enters the workspace lock.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from freeze_references import main  # noqa: E402

if __name__ == "__main__":
    main()
