"""The canonical Rerun viewer: one build (native, headless, web) with every viewer extension and Rerun patch.

The Rust workspace is in ``rust/``; this package holds the tools that prepare and check its builds.
"""

from pathlib import Path

PACKAGE_DIR: Path = Path(__file__).resolve().parents[1]
CANONICAL_VIEWER: Path = PACKAGE_DIR / "target" / "release" / "rerun-canonical"
"""The release binary that ``canonical-viewer-build`` writes."""
