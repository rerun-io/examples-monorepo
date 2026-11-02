"""The SOMA-X body files, from a pinned Hugging Face revision unless a local directory overrides them."""

from functools import cache
from pathlib import Path

from huggingface_hub import snapshot_download

SOMA_X_REPO_ID: str = "nvidia/SOMA-X"
SOMA_X_REVISION: str = "f81a01796f652260ff930d5318d6c6b845a505ed"
SOMA_BODY_FILES: tuple[str, ...] = ("SOMA_neutral.npz", "SOMA_procedural_transforms.json", "SOMA_template_rig.usda", "LICENSE")


@cache
def soma_assets(assets: Path | None = None) -> Path:
    """The directory SOMALayer reads: ``assets``, or the pinned download in the Hugging Face cache."""
    if assets is not None:
        return assets
    return Path(snapshot_download(repo_id=SOMA_X_REPO_ID, revision=SOMA_X_REVISION, allow_patterns=list(SOMA_BODY_FILES)))
