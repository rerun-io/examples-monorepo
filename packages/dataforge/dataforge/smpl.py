"""The body models dataforge poses or takes triangles from: where it gets them (one pinned private Hub copy) and SMPL's faces.

EPFL and Harmony4D need the official neutral SMPL only for its triangle list, Ego-Exo4D SMPL-H and SMPL-X, MAMMA its
SMPL-X files; anyone without access to the copy places the files at the destination instead.
"""

from collections.abc import Sequence
from functools import cache
from pathlib import Path

import numpy as np
from huggingface_hub.errors import HfHubHTTPError
from jaxtyping import UInt32
from numpy import ndarray
from simplecv.ops.smplx.smplx_torch import SmplxLayerTorch

from dataforge.transports import HfFileInfo, hf_fetch_verified, hf_lfs_files

BODY_MODELS_REPO: str = "pablovela5620/mamma-streaming-data"
"""Private dataset holding the body models; contacted only when they are not already in place."""
BODY_MODELS_REVISION: str = "0a961eedf02e2fb3d32929b11cef625661dffc60"
SMPL_FILE: str = "body_models/smpl/SMPL_NEUTRAL.pkl"


def fetch_body_models(local_dir: Path, files: Sequence[str]) -> int:
    """Fetch ``files`` (repo paths) under ``local_dir`` unless all are already there; return how many were fetched."""
    if all((local_dir / name).is_file() for name in files):
        return 0
    try:
        found: list[HfFileInfo] = hf_lfs_files(BODY_MODELS_REPO, files, revision=BODY_MODELS_REVISION)
    except HfHubHTTPError as error:
        raise FileNotFoundError(f"{BODY_MODELS_REPO}: HTTP {error.response.status_code}; log in with access or place {', '.join(files)} under {local_dir}") from error
    return sum(hf_fetch_verified(BODY_MODELS_REPO, file, local_dir=local_dir, revision=BODY_MODELS_REVISION) for file in found)


def fetch_smpl_neutral(root: Path, smpl_model_root: Path | None) -> None:
    """Fetch the official neutral SMPL under ``root`` (``root/body_models/smpl/SMPL_NEUTRAL.pkl``), or check that an explicit
    ``smpl_model_root`` holds it: an explicit root is never fetched into."""
    if smpl_model_root is None:
        fetch_body_models(root, [SMPL_FILE])
    elif not (smpl_model_root / "smpl/SMPL_NEUTRAL.pkl").is_file():
        raise FileNotFoundError(f"no official neutral SMPL at {smpl_model_root}/smpl/SMPL_NEUTRAL.pkl")


@cache
def smpl_faces(model_root: Path) -> UInt32[ndarray, "f 3"]:
    """SMPL's triangle list (shape-independent), from the official neutral model under ``model_root/smpl``; loaded once per process."""
    model: SmplxLayerTorch = SmplxLayerTorch(betas=np.zeros(10, dtype=np.float32), model_type="smpl", gender="neutral", model_root_dir=model_root)
    return np.asarray(model.faces, dtype=np.uint32)
