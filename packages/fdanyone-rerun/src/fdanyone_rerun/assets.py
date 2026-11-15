"""Where the weights live and the GVHMR run root that upstream 4DAnyone expects.

The weights come from the ``_download-fdanyone-models`` pixi task into ``$FDANYONE_MODEL_DIR`` (the package's
``data/models`` by default). Upstream runs GVHMR as a working tree: it imports ``hmr4d`` from the root and reads
``inputs/checkpoints/...`` relative to it. ``recipes/gvhmr-src`` installs that tree read-only under
``$PREFIX/share/gvhmr-src``, so :func:`gvhmr_root` copies it (10 MB) into a writable root next to the weights; upstream's
``ensure_models`` then links the checkpoints into its ``inputs/``. A copy, not links: GVHMR finds ``inputs/`` from
``Path(hmr4d.__file__).resolve()``, which would follow a link back into the read-only tree.
"""

import os
import shutil
import sys
from pathlib import Path

PACKAGE_ROOT: Path = Path(__file__).resolve().parents[2]
"""``packages/fdanyone-rerun``."""
EXAMPLE_VIDEO: Path = PACKAGE_ROOT / "data" / "source" / "pexels" / "2785536-uhd_2160_3840_25fps.mp4"
"""The README and Space example clip (``_download-fdanyone-examples``)."""
GVHMR_TREE_ENTRIES: tuple[str, ...] = ("hmr4d", "tools", "LICENSE", "REVISION")
"""What the gvhmr-src tree holds, in copy order: REVISION last, so it marks a complete copy."""


def model_dir() -> Path:
    """The weights directory: ``$FDANYONE_MODEL_DIR``, else the package's ``data/models``."""
    configured: str | None = os.environ.get("FDANYONE_MODEL_DIR")
    return Path(configured).expanduser().resolve() if configured else PACKAGE_ROOT / "data" / "models"


def gvhmr_root(models: Path, tree: Path | None = None) -> Path:
    """Build (or refresh) ``<models>/gvhmr-root``: a copy of the installed GVHMR tree.

    The copy is refreshed when its ``REVISION`` differs from the installed tree's; ``inputs/`` (upstream's checkpoint
    links) is kept.

    Args:
        models: The weights directory; the root lives inside it so its checkpoint links stay beside the weights.
        tree: The installed GVHMR tree; defaults to this environment's ``share/gvhmr-src``.

    Returns:
        The root to pass to upstream as ``gvhmr_root``.

    Raises:
        FileNotFoundError: When the GVHMR tree is not installed (the environment lacks ``gvhmr-src``).
    """
    source: Path = tree if tree is not None else Path(sys.prefix) / "share" / "gvhmr-src"
    if not (source / "hmr4d" / "__init__.py").is_file():
        raise FileNotFoundError(f"GVHMR is not installed at {source}; use an environment with recipes/gvhmr-src")
    root: Path = models / "gvhmr-root"
    revision: Path = root / "REVISION"
    if revision.is_file() and revision.read_text() == (source / "REVISION").read_text():
        return root
    root.mkdir(parents=True, exist_ok=True)
    for name in GVHMR_TREE_ENTRIES:
        target: Path = root / name
        if target.is_symlink() or target.is_file():
            target.unlink()
        elif target.is_dir():
            shutil.rmtree(target)
        if (source / name).is_dir():
            shutil.copytree(source / name, target)
        else:
            shutil.copy2(source / name, target)
    return root
