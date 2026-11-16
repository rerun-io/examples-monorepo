"""The GVHMR compatibility links and the pinned TAEW2.2 fetch.

In the monorepo the weights come from the pixi download tasks (``fdanyone-download``); upstream's on-demand
Hugging Face, example-clip and interactive SMPL-X downloads are not copied.
"""

from __future__ import annotations

import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from fdanyone.assets import (
    TAEW2_2,
    TAEW2_2_SHA256,
    TAEW2_2_URL,
    GVHMR_LINKS,
    SMPLX_MODEL,
)
from fdanyone.errors import AssetError

LOGGER = logging.getLogger("fdanyone")



def require_gvhmr_checkout(gvhmr_root: str | Path) -> Path:
    root = Path(gvhmr_root).expanduser().resolve()
    if not (root / "hmr4d/__init__.py").is_file():
        raise AssetError(
            f"GVHMR is not initialized at {root}. Run `git submodule update --init third_party/GVHMR` first."
        )
    return root


def _ensure_link(source: Path, destination: Path) -> None:
    source = source.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        try:
            if destination.resolve(strict=True).samefile(source):
                return
        except FileNotFoundError:
            pass
        destination.unlink()
    elif destination.exists():
        if destination.samefile(source):
            return
        raise AssetError(
            f"GVHMR asset location is occupied by an unrelated file: {destination}. "
            f"Move it away so the downloaded {source.name} can be linked."
        )
    relative = os.path.relpath(source, start=destination.parent)
    destination.symlink_to(relative)


def create_classic_gvhmr_links(
    model_dir: str | Path = "models",
    gvhmr_root: str | Path = "third_party/GVHMR",
    *,
    require_models: bool = True,
    require_smplx: bool = True,
) -> Path:
    """Create the ignored compatibility links expected by upstream GVHMR."""

    root = require_gvhmr_checkout(gvhmr_root)
    models = Path(model_dir).expanduser().resolve()
    for relative, target in GVHMR_LINKS:
        source = models / relative
        if not source.is_file():
            required = require_smplx if relative == SMPLX_MODEL else require_models
            if required:
                command = "scripts/download_smplx.py" if relative == SMPLX_MODEL else "scripts/download_model.py"
                raise AssetError(f"Model file is missing: {source}. Run `python {command}` first.")
            continue
        _ensure_link(source, root / target)
    return root


def ensure_tiny_decoder(model_dir: str | Path = "models") -> Path | None:
    """Download the pinned TAEW2.2 decoder (target decode) when the config uses it."""

    from fdanyone.config import INFERENCE
    from fdanyone.io import sha256_file

    if INFERENCE.target_decoder != "taew2_2":
        return None
    path = Path(model_dir).expanduser().resolve() / TAEW2_2
    if not path.is_file():
        LOGGER.info("Downloading the TAEW2.2 tiny decoder (first run only)")
        path.parent.mkdir(parents=True, exist_ok=True)
        staged = path.with_suffix(".part")
        try:
            urllib.request.urlretrieve(TAEW2_2_URL, staged)
        except urllib.error.URLError as exc:
            raise AssetError(f"Could not download the TAEW2.2 decoder from {TAEW2_2_URL}.") from exc
        digest = sha256_file(staged)
        if digest != TAEW2_2_SHA256:
            staged.unlink(missing_ok=True)
            raise AssetError(f"TAEW2.2 SHA-256 mismatch: {digest} != {TAEW2_2_SHA256}.")
        staged.replace(path)
    return path


