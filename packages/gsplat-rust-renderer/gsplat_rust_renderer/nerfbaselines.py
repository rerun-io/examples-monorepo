"""nerfbaselines test-data layout + zip extraction for the gsplat demos.

Two HuggingFace sources back the blender scenes (lego by default):

- pretrained models: repo ``nerfbaselines/nerfbaselines`` (model), file
  ``3dgs-mcmc/blender/<scene>.zip`` — an INRIA-layout ``point_cloud.ply`` under
  ``checkpoint/point_cloud/iteration_30000/`` plus reference renders
  (``predictions/color`` / ``predictions/gt-color``) and ``results.json``.
- datasets: repo ``nerfbaselines/nerfbaselines-data`` (dataset), file
  ``blender/<scene>.zip`` — standard nerf-synthetic ``<scene>/transforms_*.json``
  + ``<scene>/{train,test,val}/*.png``.

The pixi ``_gsplat-rust-renderer-download-*`` tasks run this module as a CLI
(``python -m gsplat_rust_renderer.nerfbaselines {data,pretrained} <scene>``),
which ``hf download``s the zip and unpacks it (stdlib zipfile) atomically
behind an idempotence guard; the CLIs resolve their default paths through the
helpers here. Replaces the old ``pablovela5620/splat-dataset`` +
``pablovela5620/nerf-synthetic-mirror`` HuggingFace mirrors.
"""

from __future__ import annotations

import shutil
import tempfile
import zipfile
from pathlib import Path

PRETRAINED_REPO: str = "nerfbaselines/nerfbaselines"
"""HuggingFace model repo holding the pretrained 3dgs-mcmc checkpoints."""
DATA_REPO: str = "nerfbaselines/nerfbaselines-data"
"""HuggingFace dataset repo holding the nerf-synthetic scene captures."""
DEFAULT_SCENE: str = "lego"
"""Default blender scene used by the demos."""
BLENDER_SCENES: tuple[str, ...] = ("lego", "hotdog", "chair", "drums", "ficus", "materials", "mic", "ship")
"""All scenes in the NeRF Synthetic Blender benchmark."""

# Module lives at <package>/gsplat_rust_renderer/nerfbaselines.py, so parents[1]
# is the package root where the pixi tasks (cwd = packages/gsplat-rust-renderer)
# create ``data/``.
DATA_ROOT: Path = Path(__file__).resolve().parents[1] / "data" / "nerfbaselines"
"""Root under which the pretrained/ and data/ trees are extracted."""


def scene_data_dir(scene: str = DEFAULT_SCENE, root: Path = DATA_ROOT) -> Path:
    """Return the nerf-synthetic scene directory (``transforms_*.json`` + split dirs)."""
    return root / "data" / scene


def scene_pretrained_dir(scene: str = DEFAULT_SCENE, root: Path = DATA_ROOT) -> Path:
    """Return the extracted pretrained-model directory for *scene*."""
    return root / "pretrained" / scene


def scene_ply_path(scene: str = DEFAULT_SCENE, root: Path = DATA_ROOT) -> Path:
    """Return the INRIA-layout ``point_cloud.ply`` for the pretrained *scene*."""
    return scene_pretrained_dir(scene, root) / "checkpoint" / "point_cloud" / "iteration_30000" / "point_cloud.ply"


def extract_zip(zip_path: Path, dest_dir: Path, inner: str | None = None) -> None:
    """Extract *zip_path* (or its *inner* top-level dir) to *dest_dir* atomically.

    Extraction goes to a temp sibling directory; the result is published with a
    single ``rename`` — never a merge into a shared parent — so an interrupted
    or concurrent run can neither leave a partial tree that the idempotence
    guards mistake for complete data nor nest content into a directory another
    process just published. If *dest_dir* appears concurrently (a racing task
    won), this extraction is discarded and the winner's tree is kept.

    Args:
        zip_path: Path to the downloaded ``.zip`` archive.
        dest_dir: Final directory the content is published at.
        inner: Optional top-level dir inside the archive to publish as
            *dest_dir* (for zips that nest everything under ``<scene>/``).
    """
    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{dest_dir.name}-extract-", dir=dest_dir.parent))
    try:
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(tmp)
        source = tmp / inner if inner is not None else tmp
        try:
            source.rename(dest_dir)
        except OSError:
            if not dest_dir.exists():
                raise
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def download_and_extract(kind: str, scene: str, root: Path = DATA_ROOT) -> Path:
    """Fetch + unpack one nerfbaselines zip if its target dir is missing.

    Args:
        kind: ``"data"`` (nerf-synthetic capture) or ``"pretrained"`` (3dgs-mcmc checkpoint).
        scene: Blender scene name, e.g. ``"lego"``.
        root: Data root (overridable for tests).

    Returns:
        The extracted target directory.
    """
    from huggingface_hub import hf_hub_download

    if kind == "data":
        target = scene_data_dir(scene, root)
        if not target.exists():
            zip_path = hf_hub_download(DATA_REPO, f"blender/{scene}.zip", repo_type="dataset")
            # The data zip nests everything under "<scene>/"; publish exactly
            # that dir as the target (never merge into the shared data/ parent).
            extract_zip(Path(zip_path), target, inner=scene)
    elif kind == "pretrained":
        target = scene_pretrained_dir(scene, root)
        if not target.exists():
            zip_path = hf_hub_download(PRETRAINED_REPO, f"3dgs-mcmc/blender/{scene}.zip")
            extract_zip(Path(zip_path), target)
    else:
        raise ValueError(f"unknown kind {kind!r}, expected 'data' or 'pretrained'")
    print(f"{kind}/{scene} ready at {target}")
    return target


if __name__ == "__main__":
    import sys

    download_and_extract(sys.argv[1], sys.argv[2])
