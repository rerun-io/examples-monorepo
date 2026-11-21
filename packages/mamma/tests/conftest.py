"""mamma test configuration. Imports no workspace package at module level (beartype claw ordering)."""

import os

# beartype runs in this process (pytest-beartype): torch.compile paths (the fitter's compiled iteration)
# stay eager, see AGENTS.md "Beartype".
os.environ.setdefault("PIXI_DEV_MODE", "1")
