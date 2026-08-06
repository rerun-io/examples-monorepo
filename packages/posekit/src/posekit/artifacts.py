"""ONNX artifact acquisition and caching.

ONNX files are the portable interchange for the accelerated backends: the ONNX
Runtime backend loads them directly and the TensorRT backend builds machine-local
engines from them. posekit's artifacts come from OpenMMLab deploy zips (the
rtmlib convention for RTMPose/RTMW/YOLOX) or from a torch export done by a model
family. Generic ONNX graph surgery lives in ``trtkit.onnx_graph``.
"""

import os
import zipfile
from pathlib import Path

from simplecv.apis.download_utils import download_file

__all__ = ("DEFAULT_ONNX_CACHE_DIR", "fetch_openmmlab_onnx")

DEFAULT_ONNX_CACHE_DIR: Path = Path(os.environ.get("POSEKIT_ONNX_CACHE", "~/.cache/posekit/onnx")).expanduser()
"""Portable ONNX artifact cache; override with the ``POSEKIT_ONNX_CACHE`` env var."""


def fetch_openmmlab_onnx(zip_url: str, *, cache_dir: Path = DEFAULT_ONNX_CACHE_DIR) -> Path:
    """Download an OpenMMLab SDK deploy zip and return its cached ``end2end.onnx``.

    Args:
        zip_url: ``download.openmmlab.com`` deploy-zip URL (contains ``end2end.onnx``).
        cache_dir: ONNX artifact cache root.

    Returns:
        Path to the extracted ONNX file, named after the zip stem.

    Raises:
        RuntimeError: If the zip does not contain an ONNX file.
    """
    stem: str = Path(zip_url).stem
    onnx_path: Path = cache_dir / f"{stem}.onnx"
    if onnx_path.exists():
        return onnx_path
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path: Path = cache_dir / f"{stem}.zip"
    if not zip_path.exists():
        print(f"[posekit] downloading ONNX artifact: {zip_url}")
        tmp_path: Path = zip_path.with_suffix(".zip.part")
        download_file(zip_url, tmp_path)
        tmp_path.rename(zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        onnx_members: list[str] = [name for name in archive.namelist() if name.endswith(".onnx")]
        if not onnx_members:
            raise RuntimeError(f"No .onnx file inside {zip_url}.")
        with archive.open(onnx_members[0]) as src:
            onnx_path.write_bytes(src.read())
    zip_path.unlink()
    return onnx_path
