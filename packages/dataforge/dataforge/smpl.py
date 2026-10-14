"""The official neutral SMPL model: where dataforge gets it (a pinned private Hub copy) and how a dataset puts it in place.

EPFL, Harmony4D and EgoHumans need it only for SMPL's triangle list; anyone without access to the copy places the official
``SMPL_NEUTRAL.pkl`` at the destination instead.
"""

from pathlib import Path

from huggingface_hub.errors import HfHubHTTPError

from dataforge.transports import hf_fetch_verified, hf_lfs_files

SMPL_REPO: str = "pablovela5620/mamma-streaming-data"
"""Private dataset holding the official neutral SMPL model; contacted only when the model is not already in place."""
SMPL_REVISION: str = "e05c6b723e24e2c0fc79201e06e8c8fbaa6cedfd"
SMPL_FILE: str = "body_models/smpl/SMPL_NEUTRAL.pkl"


def fetch_smpl_neutral(local_dir: Path, smpl_root: Path, *, explicit: bool) -> None:
    """Put the official neutral SMPL at ``smpl_root/smpl/SMPL_NEUTRAL.pkl`` unless it is there.

    ``SMPL_FILE`` lands under ``local_dir`` (whose ``body_models`` is the default ``smpl_root``); an ``explicit`` root is
    never fetched into.
    """
    dest: Path = smpl_root / "smpl/SMPL_NEUTRAL.pkl"
    if dest.is_file():
        return
    if explicit:
        raise FileNotFoundError(f"no official neutral SMPL at {dest}")
    try:
        (smpl,) = hf_lfs_files(SMPL_REPO, [SMPL_FILE], revision=SMPL_REVISION)
    except HfHubHTTPError as error:
        raise FileNotFoundError(f"{SMPL_REPO}: HTTP {error.response.status_code}; log in with access or place the official SMPL_NEUTRAL.pkl at {dest}") from error
    hf_fetch_verified(SMPL_REPO, smpl, local_dir=local_dir, revision=SMPL_REVISION)
